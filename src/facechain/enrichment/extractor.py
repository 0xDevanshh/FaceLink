"""Profile metadata extractor.

Fetches a public profile page and extracts structured identity information:
avatar URL, display name, bio, links to other platforms.

Design rules:
- Reuses candidate.py's _safe_get/_client so SSRF protection is never bypassed.
- Reuses candidate.py's extract_image_urls for image extraction.
- Never follows login walls, CAPTCHA pages, or robots-disallowed paths.
- Every extracted field records its source URL and extraction method.
- Platform-specific logic is isolated in _PLATFORM_EXTRACTORS so adding a new
  platform is a matter of adding one entry, not touching shared logic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import settings
from ..search.base import classify_platform, normalise_domain
from ..security.ssrf import safe_url_or_none
from ..verification.candidate import _client, _safe_get, extract_image_urls, MAX_PAGE_BYTES
from .profile import DiscoveredProfile, EvidenceLevel, ProfileField

log = logging.getLogger(__name__)

# Platform-specific profile URL patterns used to extract the username/handle
# from a canonical URL.
_USERNAME_PATTERNS: dict[str, re.Pattern] = {
    "GitHub":       re.compile(r"github\.com/([A-Za-z0-9_.-]+)(?:/|$)"),
    "X/Twitter":    re.compile(r"(?:twitter|x)\.com/([A-Za-z0-9_]+)(?:/|$)"),
    "Instagram":    re.compile(r"instagram\.com/([A-Za-z0-9_.]+)(?:/|$)"),
    "LinkedIn":     re.compile(r"linkedin\.com/in/([A-Za-z0-9_.-]+)(?:/|$)"),
    "YouTube":      re.compile(r"youtube\.com/(?:@|channel/|user/)([A-Za-z0-9_.-]+)"),
    "Medium":       re.compile(r"medium\.com/@?([A-Za-z0-9_.-]+)(?:/|$)"),
    "Devfolio":     re.compile(r"devfolio\.co/@([A-Za-z0-9_.-]+)(?:/|$)"),
    "LeetCode":     re.compile(r"leetcode\.com/(?:u/)?([A-Za-z0-9_.-]+)(?:/|$)"),
    "HackerRank":   re.compile(r"hackerrank\.com/([A-Za-z0-9_.-]+)(?:/|$)"),
    "Kaggle":       re.compile(r"kaggle\.com/([A-Za-z0-9_.-]+)(?:/|$)"),
    "StackOverflow": re.compile(r"stackoverflow\.com/users/\d+/([A-Za-z0-9_.-]+)"),
}

# Social domains we consider developer/professional platforms (enrichment targets)
ENRICHMENT_PLATFORMS: frozenset[str] = frozenset({
    "LinkedIn", "GitHub", "X/Twitter", "Instagram", "YouTube",
    "Medium", "Devfolio", "LeetCode", "HackerRank", "Kaggle", "StackOverflow",
})

# Domains for the enrichment platforms (for link detection)
_ENRICHMENT_DOMAINS: dict[str, str] = {
    "github.com": "GitHub",
    "twitter.com": "X/Twitter",
    "x.com": "X/Twitter",
    "instagram.com": "Instagram",
    "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube",
    "medium.com": "Medium",
    "devfolio.co": "Devfolio",
    "leetcode.com": "LeetCode",
    "hackerrank.com": "HackerRank",
    "kaggle.com": "Kaggle",
    "stackoverflow.com": "StackOverflow",
}


def _classify_enrichment_platform(url: str) -> Optional[str]:
    """Return the enrichment platform name for a URL, or None."""
    host = normalise_domain(url)
    for domain, platform in _ENRICHMENT_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return platform
    return None


def _extract_username(url: str, platform: str) -> Optional[str]:
    pattern = _USERNAME_PATTERNS.get(platform)
    if not pattern:
        return None
    m = pattern.search(url)
    return m.group(1) if m else None


def _profile_id(platform: str, username: str) -> str:
    return f"{platform.lower().replace('/', '-').replace(' ', '-')}:{username.lower()}"


def _extract_links(soup: BeautifulSoup, page_url: str) -> list[tuple[str, str]]:
    """Return (absolute_url, platform_name) for all enrichment-platform links on the page."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if not href:
            continue
        if not href.startswith(("http://", "https://")):
            href = urljoin(page_url, href)
        if href in seen:
            continue
        platform = _classify_enrichment_platform(href)
        if platform:
            seen.add(href)
            found.append((href, platform))
    return found


def _extract_jsonld_identity(soup: BeautifulSoup) -> dict:
    """Extract Person / ProfilePage data from JSON-LD blocks."""
    out: dict = {}
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for node in nodes:
            if not isinstance(node, dict):
                continue
            t = node.get("@type", "")
            if t in ("Person", "ProfilePage", "Organization"):
                if "name" in node:
                    out.setdefault("name", node["name"])
                if "description" in node:
                    out.setdefault("description", node["description"])
                if "url" in node:
                    out.setdefault("url", node["url"])
                sameAs = node.get("sameAs", [])
                if isinstance(sameAs, str):
                    sameAs = [sameAs]
                out.setdefault("sameAs", [])
                out["sameAs"].extend(s for s in sameAs if isinstance(s, str))
    return out


def _og_text(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = (soup.find("meta", attrs={"property": prop}) or
           soup.find("meta", attrs={"name": prop}))
    return tag.get("content", "").strip() if tag else None


def extract_profile(url: str, platform: str) -> DiscoveredProfile:
    """Fetch *url* and extract profile metadata.

    Returns a ``DiscoveredProfile`` regardless of success — fetch_note records
    what happened so callers always have a complete audit trail.
    """
    username_raw = _extract_username(url, platform)
    pid = _profile_id(platform, username_raw or urlparse(url).path.strip("/").replace("/", "-"))

    profile = DiscoveredProfile(
        profile_id=pid,
        platform=platform,
        canonical_url=url,
        discovery_method="direct",
    )

    safe = safe_url_or_none(url)
    if safe is None:
        profile.fetch_note = "SSRF: rejected"
        profile.fetch_status = 0
        return profile

    try:
        with _client() as client:
            resp, final_url = _safe_get(client, url)
    except Exception as exc:  # noqa: BLE001
        profile.fetch_note = f"fetch failed: {type(exc).__name__}"
        return profile

    if resp is None:
        profile.fetch_note = "fetch failed (SSRF block or network error)"
        return profile

    profile.fetch_status = resp.status_code

    if resp.status_code != 200:
        profile.fetch_note = f"HTTP {resp.status_code} (login wall or bot block)"
        return profile

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype:
        profile.fetch_note = f"HTTP 200 but content-type={ctype!r}"
        return profile

    profile.fetched = True
    html = resp.content[:MAX_PAGE_BYTES].decode("utf-8", "replace")
    soup = BeautifulSoup(html, "lxml")
    page_url = final_url or url

    # ---- username from URL ------------------------------------------------
    if username_raw:
        profile.username = ProfileField(
            value=username_raw,
            source_url=url,
            extraction_method="url-pattern",
        )

    # ---- display name ------------------------------------------------------
    name_val = (
        _og_text(soup, "og:title") or
        _og_text(soup, "profile:username") or
        (soup.title.string.strip() if soup.title else None)
    )
    if name_val:
        # Strip common suffixes like " | LinkedIn", " (@handle) / X"
        for suffix in (" | LinkedIn", " | GitHub", " | Instagram", " - YouTube",
                       " | Stack Overflow", " | Medium", " | Kaggle", " | HackerRank",
                       " | LeetCode", " | Devfolio"):
            if name_val.endswith(suffix):
                name_val = name_val[: -len(suffix)].strip()
        profile.display_name = ProfileField(
            value=name_val,
            source_url=page_url,
            extraction_method="og:title",
        )

    # ---- bio / description ------------------------------------------------
    bio_val = (
        _og_text(soup, "og:description") or
        _og_text(soup, "description") or
        _og_text(soup, "twitter:description")
    )
    if bio_val:
        profile.bio = ProfileField(
            value=bio_val[:500],
            source_url=page_url,
            extraction_method="og:description",
        )

    # ---- avatar image -----------------------------------------------------
    image_pairs = extract_image_urls(html, page_url)
    # Prefer trusted sources; discard obvious non-profile images
    for img_url, img_label in image_pairs[:5]:
        if any(h in img_url.lower() for h in ("logo", "banner", "cover", "badge")):
            continue
        profile.avatar_url = ProfileField(
            value=img_url,
            source_url=page_url,
            extraction_method=img_label,
        )
        profile.candidate_image_url = img_url
        profile.candidate_image_source = img_label
        break

    # ---- JSON-LD identity fields ------------------------------------------
    jld = _extract_jsonld_identity(soup)
    if jld:
        profile.raw_metadata["json_ld"] = jld
        if not profile.display_name and jld.get("name"):
            profile.display_name = ProfileField(
                value=jld["name"],
                source_url=page_url,
                extraction_method="json-ld",
            )
        if not profile.bio and jld.get("description"):
            profile.bio = ProfileField(
                value=jld["description"][:500],
                source_url=page_url,
                extraction_method="json-ld",
            )

    # ---- linked profiles --------------------------------------------------
    links = _extract_links(soup, page_url)
    for link_url, link_platform in links:
        # Don't link back to the same platform
        if link_platform == platform:
            continue
        profile.linked_profiles.append(ProfileField(
            value=link_url,
            source_url=page_url,
            extraction_method="html-link",
        ))
        if link_platform not in profile.linked_platforms:
            profile.linked_platforms.append(link_platform)

    # sameAs links from JSON-LD
    for same_url in jld.get("sameAs", []):
        lp = _classify_enrichment_platform(same_url)
        if lp and lp != platform:
            profile.linked_profiles.append(ProfileField(
                value=same_url,
                source_url=page_url,
                extraction_method="json-ld:sameAs",
            ))
            if lp not in profile.linked_platforms:
                profile.linked_platforms.append(lp)

    profile.fetch_note = (
        f"HTTP 200, name={bool(profile.display_name)}, "
        f"bio={bool(profile.bio)}, avatar={bool(profile.avatar_url)}, "
        f"links={len(profile.linked_profiles)}"
    )
    return profile
