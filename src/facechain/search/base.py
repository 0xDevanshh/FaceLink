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

from ..config import (
    GITHUB_MEDIA_HOSTS,
    OTHER_WEB_PRIORITY,
    PLATFORM_MEDIA_DOMAINS,
    SOCIAL_DOMAINS,
    platform_priority,
    settings,
)
from ..models import CandidateType, ProviderStatus, SearchCandidate

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
    # Set by an adapter that knows precisely what happened (a detected CAPTCHA,
    # a missing API key). Left None when only the error text is available, in
    # which case the orchestrator classifies it.
    status: ProviderStatus | None = None


def normalise_domain(url: str) -> str:
    """Lowercased hostname with a leading `www.` removed, in punycode.

    Non-ASCII hosts are IDNA-encoded before anything compares them, so a
    homoglyph domain (Cyrillic 'а' in "аpple.com") becomes its `xn--` form and
    cannot collide with the ASCII platform names in the table below. A host we
    cannot encode is treated as having no domain, i.e. unrecognised.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii").lower()
        except (UnicodeError, UnicodeDecodeError):
            return ""
    return host[4:] if host.startswith("www.") else host


def host_matches(host: str, domain: str) -> bool:
    """Exact host, or a genuine subdomain of it.

    The `.`-anchored suffix test is what stops `linkedin.com.attacker.com` and
    `notlinkedin.com` from being read as LinkedIn.
    """
    return bool(host) and (host == domain or host.endswith("." + domain))


def classify_platform(url: str) -> tuple[bool, str | None, int]:
    """`(recognised, platform_name, discovery_priority)` for a URL.

    Recognition is hostname-exact. Priority orders *what we look at first*; it
    never contributes to whether a candidate verifies.
    """
    host = normalise_domain(url)
    if not host:
        return False, None, OTHER_WEB_PRIORITY
    for domain, platform in SOCIAL_DOMAINS.items():
        if host_matches(host, domain):
            return True, platform, platform_priority(platform)
    # A platform's media CDN is still that platform. Checked second so a site
    # domain always wins, and matched with the same `.`-anchored rule so a
    # lookalike CDN name cannot borrow the attribution.
    for domain, platform in PLATFORM_MEDIA_DOMAINS.items():
        if host_matches(host, domain):
            return True, platform, platform_priority(platform)
    return False, None, OTHER_WEB_PRIORITY


def classify_social(url: str) -> tuple[bool, str | None]:
    """Is this URL on a platform we can name, and which one?

    Retained as the search layer's public helper; `classify_platform` adds the
    discovery priority for callers that rank.
    """
    recognised, platform, _ = classify_platform(url)
    return recognised, platform


def is_github_media(url: str) -> bool:
    """A GitHub host that serves image bytes rather than an HTML page."""
    host = normalise_domain(url)
    return any(host_matches(host, h) for h in GITHUB_MEDIA_HOSTS)


# Path shapes that mark an article rather than a generic page.
_ARTICLE_MARKERS = ("/blog/", "/news/", "/article/", "/articles/", "/story/",
                    "/press/", "/post/", "/posts/", "/pulse/")

# Profile paths that the generic "two or more path segments means a post"
# heuristic in `looks_like_post` gets wrong: `linkedin.com/in/someone` is a
# person's profile, not a post, and counting slashes cannot tell the difference.
_PROFILE_PATH_MARKERS = ("/in/", "/company/", "/school/", "/user/", "/users/",
                         "/channel/", "/profile/", "/people/", "/author/")


def initial_candidate_type(url: str, platform: str | None) -> CandidateType:
    """Type a candidate from its URL alone, before we have measured anything.

    This is a provisional label for ranking and display. Once the image has
    actually been fetched and compared, `scorer.score_candidate` upgrades it to
    EXACT_IMAGE / SAME_FACE, which are claims about measurements rather than
    about URL shape.
    """
    lowered = url.lower()
    if platform == "GitHub":
        # A raw avatar file is still evidence about a developer identity.
        return CandidateType.DEVELOPER_PROFILE
    if platform:
        # An explicit profile path beats the slash-counting heuristic.
        if any(m in lowered for m in _PROFILE_PATH_MARKERS):
            return CandidateType.SOCIAL_PROFILE
        return CandidateType.SOCIAL_POST if looks_like_post(url) else CandidateType.SOCIAL_PROFILE
    if any(m in lowered for m in _ARTICLE_MARKERS):
        return CandidateType.PUBLIC_ARTICLE
    return CandidateType.PUBLIC_WEB_PAGE


def classify_block(reason: str) -> ProviderStatus:
    """Map a detected interstitial to a provider status.

    Rate limiting and bot challenges are both refusals, but they are different
    facts: one clears on its own, the other needs a human. Reporting them
    separately is the difference between "retry later" and "this provider is
    unavailable to automation right now".
    """
    lowered = reason.lower()
    if "rate limit" in lowered or "too many requests" in lowered:
        return ProviderStatus.RATE_LIMITED
    return ProviderStatus.CHALLENGED


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

        is_social, platform, priority = classify_platform(href)
        out.append(
            SearchCandidate(
                engine=engine,
                url=href,
                domain=normalise_domain(href),
                title=(row.get("text") or "")[:200],
                thumbnail=(row.get("thumb") or "")[:500],
                is_social=is_social,
                platform=platform,
                platform_priority=priority,
                candidate_type=initial_candidate_type(href, platform),
            )
        )

    # Priority platforms first (LinkedIn → Instagram → X → GitHub → YouTube →
    # other recognised → wider web), and within a platform a specific post
    # ahead of a bare profile. This orders which leads get fetch budget; it does
    # not decide which of them verify.
    out.sort(key=lambda c: (c.platform_priority, not looks_like_post(c.url)))
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
        return EngineResult(engine, ok=False, error=blocked, status=classify_block(blocked))

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
        # We reached the results view (the marker proved it) and it held nothing
        # we could use. That is an empty search, not a broken provider.
        return EngineResult(engine, ok=False, error="no outbound result links found",
                            status=ProviderStatus.NO_RESULTS)
    return EngineResult(engine, candidates=candidates, status=ProviderStatus.COMPLETED)


class SearchEngineAdapter:
    """One reverse-image engine. Adapters own their own fragility."""

    name = "abstract"
    supports_upload = True
    supports_by_url = False
    # True only for an adapter that has no way to search a local file at all —
    # it can only ever run given a public URL, so the orchestrator knows a
    # temporary-hosting attempt is genuinely necessary (not just helpful) for
    # this adapter, and its failure state should be reported as a real failure
    # rather than a generic "not configured".
    requires_public_url = False
    # True only when an adapter's own local-upload path is *just as reliable*
    # as a centrally hosted public URL — e.g. a first-party API upload
    # endpoint, as opposed to a browser adapter's in-page upload flow, which a
    # public URL measurably rescues from bot-detection/selector fragility.
    # Lets the orchestrator skip a central hosting attempt only when nothing
    # selected would actually benefit from one, rather than skipping whenever
    # nothing strictly *requires* one (which browser adapters never do, yet
    # clearly benefit from).
    has_reliable_upload_alternative = False

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
