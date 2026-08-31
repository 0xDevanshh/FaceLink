"""Reverse-image-search adapter contract + engine-agnostic result harvesting.

Design note (this is the load-bearing decision of the search layer):

Search engines rewrite their DOM constantly, so we do NOT depend on their CSS
class names. Every adapter drives the real engine UI, then hands the settled
page to `harvest_anchors`, which scrapes *every* outbound link and filters out
the engine's own chrome. When Google reshuffles its markup, a class-name
scraper returns zero results; an outbound-link harvester keeps working.

Per-engine "preferred container" selectors are tried first as an optimisation;
whole-page harvesting is the fallback, never the other way round.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse

from ..config import SOCIAL_DOMAINS, settings
from ..models import SearchCandidate

log = logging.getLogger(__name__)

# Engine chrome, CDNs and legal boilerplate — never a "result".
JUNK_DOMAINS = {
    "google.com", "google.co.in", "gstatic.com", "googleusercontent.com",
    "googleapis.com", "google.co.uk", "policies.google.com", "support.google.com",
    "accounts.google.com", "myactivity.google.com", "maps.google.com",
    "play.google.com", "news.google.com", "books.google.com", "lens.google.com",
    "yandex.com", "yandex.ru", "yandex.net", "ya.ru", "yastatic.net",
    "bing.com", "microsoft.com", "msn.com", "live.com", "microsoftonline.com",
    "tineye.com", "ideeinc.com",
    "w3.org", "schema.org", "gmpg.org", "chromewebstore.google.com",
    "duckduckgo.com", "ecosia.org",
}

# Redirect wrappers that hide the real destination in a query parameter.
REDIRECT_PARAMS = ("q", "u", "url", "imgurl", "mediaurl", "target", "to", "r")

IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp|avif)(\?|$)", re.I)


@dataclass
class EngineResult:
    engine: str
    candidates: list[SearchCandidate] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    query_mode: str = ""  # "upload" | "by-url" | "api"


def normalise_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify_social(url: str) -> tuple[bool, str | None]:
    """Is this URL a social-media post, and on which platform?"""
    host = normalise_domain(url)
    if not host:
        return False, None
    for domain, platform in SOCIAL_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return True, platform
    return False, None


def is_junk(url: str) -> bool:
    host = normalise_domain(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in JUNK_DOMAINS)


def unwrap_redirect(url: str) -> str:
    """Follow engine redirect wrappers (`/url?q=…`, `&mediaurl=…`) statically."""
    for _ in range(3):
        try:
            parts = urlparse(url)
        except ValueError:
            return url
        qs = parse_qs(parts.query)
        nxt = None
        for key in REDIRECT_PARAMS:
            vals = qs.get(key)
            if vals and vals[0].startswith(("http://", "https://")):
                nxt = unquote(vals[0])
                break
        if not nxt or nxt == url:
            return url
        url = nxt
    return url


def looks_like_post(url: str) -> bool:
    """Prefer a specific post over a bare profile or the platform's homepage."""
    path = urlparse(url).path.rstrip("/")
    if not path or path.count("/") == 0:
        return False
    post_markers = ("/p/", "/reel/", "/status/", "/posts/", "/photo/", "/video/",
                    "/watch", "/comments/", "/pin/", "/post/", "/media/")
    return any(m in url for m in post_markers) or path.count("/") >= 2


def build_candidates(engine: str, raw: list[dict], limit: int | None = None) -> list[SearchCandidate]:
    """Normalise, de-junk and de-duplicate raw `{href,text,thumb}` harvest rows."""
    limit = limit or settings.max_candidates_per_engine
    seen: set[str] = set()
    out: list[SearchCandidate] = []

    for row in raw:
        href = (row.get("href") or "").strip()
        if not href.startswith(("http://", "https://")):
            continue
        href = unwrap_redirect(href)
        if is_junk(href):
            continue
        # Direct image files are useful as evidence but are not "posts".
        key = href.split("#")[0]
        if key in seen:
            continue
        seen.add(key)

        is_social, platform = classify_social(href)
        out.append(
            SearchCandidate(
                engine=engine,
                url=href,
                domain=normalise_domain(href),
                title=(row.get("text") or "")[:200],
                thumbnail=(row.get("thumb") or "")[:500],
                is_social=is_social,
                platform=platform,
            )
        )

    # Social posts first, then social profiles, then everything else —
    # the task asks specifically for a social media post.
    out.sort(key=lambda c: (not c.is_social, not looks_like_post(c.url)))
    return out[:limit]


class SearchEngineAdapter:
    """One reverse-image engine. Adapters own their own fragility."""

    name = "abstract"
    supports_upload = True
    supports_by_url = False

    def search(self, image_path: str, image_url: str | None = None) -> EngineResult:
        raise NotImplementedError

    # -- helpers shared by the browser-driven adapters ----------------------

    @staticmethod
    def harvest_anchors(page, containers: tuple[str, ...] = ()) -> list[dict]:
        """Scrape outbound links, preferring known result containers."""
        js = """
        (root) => {
          const scope = root || document;
          return Array.from(scope.querySelectorAll('a[href]')).map(a => {
            const img = a.querySelector('img') || (a.parentElement && a.parentElement.querySelector('img'));
            return {
              href: a.href,
              text: (a.innerText || a.getAttribute('aria-label') || '').trim().slice(0, 200),
              thumb: img ? (img.currentSrc || img.src || '') : ''
            };
          });
        }
        """
        for sel in containers:
            try:
                node = page.query_selector(sel)
                if node:
                    rows = page.evaluate(js, node)
                    if rows and len(rows) > 3:
                        log.debug("harvested %d anchors from %s", len(rows), sel)
                        return rows
            except Exception as exc:  # noqa: BLE001
                log.debug("container %s failed: %s", sel, exc)
        try:
            return page.evaluate(js, None)
        except Exception as exc:  # noqa: BLE001
            log.warning("anchor harvest failed: %s", exc)
            return []
