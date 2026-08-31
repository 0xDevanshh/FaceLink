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
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlparse, urlsplit, urlunsplit

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


# Tracking parameters engines bolt on. They must be stripped before the URL is
# hashed on-chain: the same post reached via two engines would otherwise produce
# two different `matchedUrlHash` values, making the record non-reproducible.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_referrer", "fbclid", "gclid", "dclid", "msclkid", "yclid",
    "igshid", "ref_src", "ref_url", "si", "feature", "_ga", "mc_cid", "mc_eid",
}


def canonicalise_url(url: str) -> str:
    """Drop tracking cruft and the fragment, keeping every meaningful parameter.

    Only a known-tracking allowlist is removed — parameters that identify the
    content itself (YouTube's `v`, for instance) are always preserved.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))


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
        href = canonicalise_url(unwrap_redirect(href))
        if is_junk(href):
            continue
        key = href.rstrip("/")
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


def collect_results(
    page,
    *,
    engine: str,
    containers: tuple[str, ...],
    markers: tuple[str, ...],
    view_labels: tuple[str, ...] = (),
    limit: int | None = None,
) -> EngineResult:
    """Shared post-query flow: prove we reached results, then extract them.

    Two hard-won rules are encoded here.

    1. **Prove it first.** Every engine must show one of its `markers` (e.g.
       Bing's "pages with this image") before we believe the page holds
       results. Without this guard, an upload that silently failed to submit
       leaves us on the engine's *homepage*, and a whole-page link harvest
       happily returns dozens of trending-topic links that look like real hits.
       That is precisely how a pipeline ends up "finding" matches that were
       never search results at all, so a missing marker is a hard failure.

    2. **Harvest before clicking.** Switching to a source-pages view can
       navigate the page out from under us, so we only click when the current
       view yielded nothing.
    """
    from .browser import click_text, detect_block, scroll_through, settle

    blocked = detect_block(page)
    if blocked:
        return EngineResult(engine, ok=False, error=blocked)

    def body_text() -> str:
        try:
            return (page.inner_text("body") or "").lower()
        except Exception:  # noqa: BLE001
            return ""

    def marker_present() -> bool:
        if not markers:
            return True
        text = body_text()
        return any(m.lower() in text for m in markers)

    if not marker_present():
        # Try to switch into the results view the engine hides behind a tab.
        for label in view_labels:
            if click_text(page, label, timeout_ms=3500):
                log.debug("%s: switched view via %r", engine, label)
                settle(page, 2500)
                if marker_present():
                    break
        if not marker_present():
            return EngineResult(
                engine,
                ok=False,
                error="results view not reached (query may not have been submitted, "
                      "or the engine changed its layout)",
            )

    scroll_through(page, rounds=3)

    rows = SearchEngineAdapter.harvest_anchors(page, containers)
    candidates = build_candidates(engine, rows, limit)
    if not candidates:
        return EngineResult(engine, ok=False, error="no outbound result links found")
    return EngineResult(engine, candidates=candidates)


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
