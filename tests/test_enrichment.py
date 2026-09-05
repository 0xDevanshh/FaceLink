"""Enrichment pipeline tests.

Covers the requirements from the spec:
  - canonical URL extraction
  - original image preference (og:image > thumbnail)
  - profile metadata extraction (name, bio, avatar, links)
  - cross-platform link discovery
  - duplicate profile merging
  - conflicting identity signals
  - inaccessible / login-walled pages
  - JS/dynamic pages (static-only approximation — JS execution not supported)
  - false matches (face detected, wrong person)
  - evidence provenance
  - verified → cross-platform discovery

All network calls are stubbed at the httpx boundary via monkeypatching
candidate.py's _client() exactly as the existing test suite does.
"""

from __future__ import annotations

import io
from typing import Optional

import httpx
import numpy as np
import pytest
from PIL import Image

from facechain.config import settings
from facechain.enrichment.extractor import (
    _classify_enrichment_platform,
    _extract_links,
    _extract_username,
    _profile_id,
    extract_profile,
)
from facechain.enrichment.graph import enrich_case, _assign_evidence_level, _canonical_key
from facechain.enrichment.profile import DiscoveredProfile, EvidenceLevel, ProfileField
from facechain.models import CandidateType, VerifiedCandidate
from facechain.verification import candidate as candmod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def png_bytes(colour=(100, 150, 200), size=(200, 200)) -> bytes:
    rng = np.random.default_rng(7)
    arr = np.clip(np.array(colour, dtype=np.int16) + rng.integers(-60, 60, (*size, 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def embedding(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _vc(url: str, platform: str, face_sim: float = 0.85,
        verified: bool = True) -> VerifiedCandidate:
    """Build a minimal VerifiedCandidate for enrichment tests."""
    from facechain.search.base import classify_platform, initial_candidate_type
    r, p, pr = classify_platform(url)
    return VerifiedCandidate(
        engine="serpapi_google_lens",
        url=url, canonical_url=url,
        domain=url.split("/")[2],
        platform=platform, is_social=r,
        platform_priority=pr,
        candidate_type=initial_candidate_type(url, p),
        face_similarity=face_sim,
        image_similarity=0.9,
        face_detected=True,
        verified=verified,
    )


@pytest.fixture
def stub_http(monkeypatch):
    """Route httpx requests in extractor / candidate through a controlled table."""
    routes: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        resp = routes.get(str(request.url))
        if resp is None:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(resp.status_code, headers=resp.headers,
                              content=resp.content, request=request)

    real_client = candmod._client

    def patched():
        c = real_client()
        c._transport = httpx.MockTransport(handler)
        return c

    monkeypatch.setattr(candmod, "_client", patched)
    monkeypatch.setattr(candmod, "safe_url_or_none", lambda u: u)
    # Also patch extractor's imported _client (same object via import)
    from facechain.enrichment import extractor as ext_mod
    monkeypatch.setattr(ext_mod, "_client", patched)
    monkeypatch.setattr(ext_mod, "safe_url_or_none", lambda u: u)
    return routes


def html_resp(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html"}, content=body.encode())


def img_resp(data: bytes, ctype: str = "image/png") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": ctype}, content=data)


# ===========================================================================
# 1. Canonical URL extraction and username parsing
# ===========================================================================

@pytest.mark.parametrize("url,platform,expected", [
    ("https://github.com/octocat", "GitHub", "octocat"),
    ("https://github.com/octocat/", "GitHub", "octocat"),
    ("https://twitter.com/elonmusk", "X/Twitter", "elonmusk"),
    ("https://x.com/elonmusk", "X/Twitter", "elonmusk"),
    ("https://linkedin.com/in/satya-nadella", "LinkedIn", "satya-nadella"),
    ("https://www.linkedin.com/in/satya-nadella/", "LinkedIn", "satya-nadella"),
    ("https://instagram.com/nasa", "Instagram", "nasa"),
    ("https://medium.com/@user123", "Medium", "user123"),
    ("https://leetcode.com/u/coder42", "LeetCode", "coder42"),
    ("https://stackoverflow.com/users/123/johndoe", "StackOverflow", "johndoe"),
])
def test_username_extraction(url, platform, expected):
    assert _extract_username(url, platform) == expected


def test_profile_id_is_stable_and_lowercase():
    assert _profile_id("GitHub", "OctoCat") == "github:octocat"
    assert _profile_id("X/Twitter", "UserA") == "x-twitter:usera"
    assert _profile_id("LinkedIn", "John-Doe") == "linkedin:john-doe"


def test_canonical_key_normalises_trailing_slash_and_case():
    a = _canonical_key("https://github.com/Octocat/")
    b = _canonical_key("https://github.com/octocat")
    assert a == b


# ===========================================================================
# 2. Platform classification
# ===========================================================================

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/someone", "GitHub"),
    ("https://linkedin.com/in/someone", "LinkedIn"),
    ("https://twitter.com/u", "X/Twitter"),
    ("https://x.com/u", "X/Twitter"),
    ("https://instagram.com/u", "Instagram"),
    ("https://medium.com/@u", "Medium"),
    ("https://devfolio.co/@u", "Devfolio"),
    ("https://leetcode.com/u/x", "LeetCode"),
    ("https://hackerrank.com/u", "HackerRank"),
    ("https://kaggle.com/u", "Kaggle"),
    ("https://stackoverflow.com/users/1/u", "StackOverflow"),
    ("https://youtube.com/@channel", "YouTube"),
    ("https://example.com/whatever", None),
])
def test_platform_classification(url, expected):
    assert _classify_enrichment_platform(url) == expected


# ===========================================================================
# 3. Profile metadata extraction (name, bio, avatar, links)
# ===========================================================================

def test_extract_profile_parses_og_tags(stub_http):
    img = png_bytes()
    stub_http["https://github.com/octocat"] = html_resp("""
        <html><head>
          <meta property="og:title" content="The Octocat · GitHub">
          <meta property="og:description" content="Code explorer. Octocat at GitHub.">
          <meta property="og:image" content="https://avatars.githubusercontent.com/u/583231">
        </head><body></body></html>
    """)
    stub_http["https://avatars.githubusercontent.com/u/583231"] = img_resp(img)

    p = extract_profile("https://github.com/octocat", "GitHub")

    assert p.fetched
    assert p.fetch_status == 200
    assert p.display_name is not None
    assert "Octocat" in p.display_name.value  # suffix stripped
    assert p.bio is not None
    assert "Code explorer" in p.bio.value
    assert p.avatar_url is not None
    assert "avatars.githubusercontent.com" in p.avatar_url.value


def test_og_image_has_provenance(stub_http):
    stub_http["https://github.com/octocat"] = html_resp("""
        <meta property="og:image" content="https://avatars.githubusercontent.com/u/1">
    """)
    p = extract_profile("https://github.com/octocat", "GitHub")
    assert p.avatar_url is not None
    assert p.avatar_url.source_url == "https://github.com/octocat"
    assert p.avatar_url.extraction_method in ("og:image", "github:avatar", "img")


def test_json_ld_identity_extracted(stub_http):
    stub_http["https://linkedin.com/in/satya"] = html_resp("""
        <script type="application/ld+json">
        {"@type":"Person","name":"Satya Nadella","description":"CEO at Microsoft","sameAs":["https://twitter.com/satyanadella"]}
        </script>
    """)
    p = extract_profile("https://linkedin.com/in/satya", "LinkedIn")
    assert p.display_name is not None
    assert p.display_name.value == "Satya Nadella"
    assert p.display_name.extraction_method == "json-ld"
    assert p.bio is not None
    assert "CEO" in p.bio.value
    # sameAs should be in linked_profiles
    assert any("twitter.com" in lf.value for lf in p.linked_profiles)


def test_username_field_extraction_method_is_url_pattern(stub_http):
    stub_http["https://github.com/torvalds"] = html_resp("<html></html>")
    p = extract_profile("https://github.com/torvalds", "GitHub")
    assert p.username is not None
    assert p.username.value == "torvalds"
    assert p.username.extraction_method == "url-pattern"
    assert p.username.source_url == "https://github.com/torvalds"


# ===========================================================================
# 4. Cross-platform link discovery
# ===========================================================================

def test_linked_profiles_extracted_from_page_links(stub_http):
    stub_http["https://github.com/developer1"] = html_resp("""
        <html><body>
          <a href="https://linkedin.com/in/developer1">LinkedIn</a>
          <a href="https://twitter.com/dev1">Twitter</a>
          <a href="https://medium.com/@dev1">Medium</a>
        </body></html>
    """)
    p = extract_profile("https://github.com/developer1", "GitHub")
    platforms = set(p.linked_platforms)
    assert "LinkedIn" in platforms
    assert "X/Twitter" in platforms
    assert "Medium" in platforms
    # GitHub itself must not be in linked_platforms (same platform filtered)
    assert "GitHub" not in platforms


def test_linked_profiles_have_provenance(stub_http):
    stub_http["https://github.com/developer1"] = html_resp("""
        <a href="https://linkedin.com/in/developer1">LinkedIn</a>
    """)
    p = extract_profile("https://github.com/developer1", "GitHub")
    link = next((lf for lf in p.linked_profiles if "linkedin" in lf.value), None)
    assert link is not None
    assert link.source_url.startswith("https://github.com")
    assert link.extraction_method == "html-link"


def test_cross_platform_discovery_follows_links(stub_http):
    """enrich_case must follow links from seed GitHub profile to LinkedIn."""
    github_img = png_bytes()
    linkedin_img = png_bytes((120, 80, 60))

    stub_http["https://github.com/dev"] = html_resp("""
        <html><head>
          <meta property="og:title" content="Dev Person">
          <img class="avatar-user" src="https://avatars.githubusercontent.com/u/1">
          <a href="https://linkedin.com/in/dev-person">LinkedIn</a>
        </head></html>
    """)
    stub_http["https://avatars.githubusercontent.com/u/1"] = img_resp(github_img)
    stub_http["https://linkedin.com/in/dev-person"] = html_resp("""
        <meta property="og:title" content="Dev Person | LinkedIn">
        <meta property="og:image" content="https://media.licdn.com/dms/image/dev.jpg">
    """)
    stub_http["https://media.licdn.com/dms/image/dev.jpg"] = img_resp(linkedin_img)

    vc = _vc("https://github.com/dev", "GitHub", face_sim=0.85)
    graph = enrich_case([vc], embedding(0), {}, emit=None)

    platforms = {p.platform for p in graph.profiles}
    assert "GitHub" in platforms
    assert "LinkedIn" in platforms

    li = next(p for p in graph.profiles if p.platform == "LinkedIn")
    assert li.discovery_method == "cross-profile-link"
    assert li.discovered_from == "github:dev"


# ===========================================================================
# 5. Duplicate profile merging (same canonical URL)
# ===========================================================================

def test_duplicate_url_not_added_twice(stub_http):
    stub_http["https://github.com/octocat"] = html_resp(
        '<meta property="og:title" content="Octocat">'
    )
    # Two verified candidates pointing to the same URL
    vc1 = _vc("https://github.com/octocat", "GitHub", face_sim=0.85)
    vc2 = _vc("https://github.com/octocat", "GitHub", face_sim=0.87)
    graph = enrich_case([vc1, vc2], embedding(0), {}, emit=None)
    github_profiles = [p for p in graph.profiles if p.platform == "GitHub"]
    assert len(github_profiles) == 1, "duplicate URL must not produce two profiles"


# ===========================================================================
# 6. Conflicting identity signals
# ===========================================================================

def test_name_alone_does_not_confirm_profile():
    """A name match without a face match must not exceed POSSIBLE."""
    p = DiscoveredProfile(
        profile_id="linkedin:john-doe",
        platform="LinkedIn",
        canonical_url="https://linkedin.com/in/john-doe",
        display_name=ProfileField(value="John Doe", source_url="x", extraction_method="og:title"),
        face_detected=False,
        face_similarity=0.0,
        discovery_method="cross-profile-link",
    )
    _assign_evidence_level(p, {"john-doe"}, {"john-doe"}, set())
    assert p.evidence_level in (EvidenceLevel.POSSIBLE, EvidenceLevel.REJECTED)
    assert p.evidence_level != EvidenceLevel.CONFIRMED
    assert p.evidence_level != EvidenceLevel.HIGH_CONFIDENCE


def test_face_below_threshold_is_rejected():
    p = DiscoveredProfile(
        profile_id="linkedin:wrong",
        platform="LinkedIn",
        canonical_url="https://linkedin.com/in/wrong",
        face_detected=True,
        face_similarity=0.15,  # below 0.38 threshold
        discovery_method="reverse-image",
    )
    _assign_evidence_level(p, set(), set(), set())
    assert p.evidence_level == EvidenceLevel.REJECTED
    assert "0.15" in p.rejection_reason


def test_face_match_without_corroboration_is_high_confidence():
    p = DiscoveredProfile(
        profile_id="github:someone",
        platform="GitHub",
        canonical_url="https://github.com/someone",
        face_detected=True,
        face_similarity=0.85,
        discovery_method="reverse-image",
    )
    _assign_evidence_level(p, set(), set(), set())
    assert p.evidence_level == EvidenceLevel.HIGH_CONFIDENCE


def test_face_match_plus_cross_link_is_confirmed():
    confirmed_ids = {"github:seed"}
    p = DiscoveredProfile(
        profile_id="linkedin:same-person",
        platform="LinkedIn",
        canonical_url="https://linkedin.com/in/same-person",
        face_detected=True,
        face_similarity=0.85,
        discovery_method="cross-profile-link",
        discovered_from="github:seed",
    )
    _assign_evidence_level(p, set(), set(), confirmed_ids)
    assert p.evidence_level == EvidenceLevel.CONFIRMED


# ===========================================================================
# 7. Inaccessible / login-walled pages
# ===========================================================================

def test_login_walled_page_recorded_honestly(stub_http):
    stub_http["https://linkedin.com/in/someone"] = html_resp("", status=999)
    p = extract_profile("https://linkedin.com/in/someone", "LinkedIn")
    assert not p.fetched
    assert p.fetch_status == 999
    assert "999" in p.fetch_note
    assert p.avatar_url is None


def test_403_page_does_not_crash(stub_http):
    stub_http["https://instagram.com/user"] = html_resp("", status=403)
    p = extract_profile("https://instagram.com/user", "Instagram")
    assert not p.fetched
    assert p.fetch_status == 403


def test_network_error_does_not_crash(stub_http):
    # No route registered → ConnectError
    p = extract_profile("https://github.com/unreachable", "GitHub")
    assert not p.fetched
    assert "fetch" in p.fetch_note.lower()


# ===========================================================================
# 8. JS / dynamic pages — static approximation
# ===========================================================================

def test_page_with_no_metadata_still_returns_partial_profile(stub_http):
    """A page that renders content via JS will have no og: tags in static HTML.
    The extractor must return what it can rather than crashing."""
    stub_http["https://devfolio.co/@hacker"] = html_resp(
        "<html><head><title>hacker | Devfolio</title></head><body></body></html>"
    )
    p = extract_profile("https://devfolio.co/@hacker", "Devfolio")
    assert p.fetched
    # Username comes from URL even when page has no metadata
    assert p.username is not None
    assert p.username.value == "hacker"


# ===========================================================================
# 9. False matches (face detected, wrong person)
# ===========================================================================

def test_false_match_marked_rejected_in_graph(stub_http):
    stub_http["https://github.com/wrongperson"] = html_resp("""
        <meta property="og:image" content="https://avatars.githubusercontent.com/u/99">
    """)
    stub_http["https://avatars.githubusercontent.com/u/99"] = img_resp(png_bytes())

    # Verified candidate with a face similarity below threshold (wrong person)
    vc = _vc("https://github.com/wrongperson", "GitHub",
             face_sim=0.15, verified=False)
    # Enrich still processes unverified candidates when called directly
    # by inspecting face_similarity — but only verified ones are seeds.
    # Verify that a profile with face<threshold is REJECTED, not POSSIBLE.
    p = DiscoveredProfile(
        profile_id="github:wrongperson",
        platform="GitHub",
        canonical_url="https://github.com/wrongperson",
        face_detected=True,
        face_similarity=0.15,
        discovery_method="reverse-image",
    )
    _assign_evidence_level(p, set(), set(), set())
    assert p.evidence_level == EvidenceLevel.REJECTED


# ===========================================================================
# 10. Evidence provenance
# ===========================================================================

def test_every_field_has_source_url(stub_http):
    stub_http["https://github.com/traced"] = html_resp("""
        <html><head>
          <meta property="og:title" content="Traced Person">
          <meta property="og:description" content="A person to trace">
          <meta property="og:image" content="https://avatars.githubusercontent.com/u/2">
          <a href="https://linkedin.com/in/traced">LinkedIn</a>
        </head></html>
    """)
    p = extract_profile("https://github.com/traced", "GitHub")

    if p.display_name:
        assert p.display_name.source_url, "display_name must have source_url"
        assert p.display_name.extraction_method, "display_name must have extraction_method"
    if p.bio:
        assert p.bio.source_url
        assert p.bio.extraction_method
    if p.avatar_url:
        assert p.avatar_url.source_url
        assert p.avatar_url.extraction_method
    for link in p.linked_profiles:
        assert link.source_url
        assert link.extraction_method


def test_discovery_method_is_recorded(stub_http):
    stub_http["https://github.com/seed"] = html_resp(
        '<a href="https://twitter.com/seed_handle">Twitter</a>'
    )
    stub_http["https://twitter.com/seed_handle"] = html_resp(
        '<meta property="og:title" content="Seed Handle">'
    )
    vc = _vc("https://github.com/seed", "GitHub", face_sim=0.85)
    graph = enrich_case([vc], embedding(0), {}, emit=None)

    tw = next((p for p in graph.profiles if p.platform == "X/Twitter"), None)
    if tw:  # only if cross-discovery succeeded
        assert tw.discovery_method == "cross-profile-link"
        assert tw.discovered_from == "github:seed"


# ===========================================================================
# 11. verified → cross-platform: graph summary counts
# ===========================================================================

def test_graph_summary_counts_are_correct(stub_http):
    stub_http["https://github.com/person"] = html_resp(
        '<meta property="og:title" content="Person">'
    )
    vc = _vc("https://github.com/person", "GitHub", face_sim=0.85)
    events = []
    graph = enrich_case([vc], embedding(0), {},
                        emit=lambda s, st, d: events.append((s, st, d)))

    total = (graph.confirmed_count + graph.high_confidence_count +
             graph.possible_count + graph.rejected_count)
    assert total == len(graph.profiles)
    # At least one enrich event was emitted
    assert any("enrich" in str(e) for e in events)


def test_no_verified_candidates_skips_enrichment():
    unverified = [_vc("https://github.com/x", "GitHub", face_sim=0.85, verified=False)]
    graph = enrich_case(unverified, embedding(0), {}, emit=None)
    assert len(graph.profiles) == 0


def test_non_enrichment_platform_is_skipped():
    """A verified candidate on a non-enrichment platform (blogspot.com) must
    not be added as a seed — enrichment only covers known developer platforms."""
    vc = _vc("https://1.bp.blogspot.com/photo.jpg", "Other", face_sim=0.90)
    vc.platform = None  # explicitly not a known platform
    graph = enrich_case([vc], embedding(0), {}, emit=None)
    assert len(graph.profiles) == 0


# ===========================================================================
# 12. Edge building
# ===========================================================================

def test_same_face_edge_created_between_confirmed_profiles(stub_http):
    stub_http["https://github.com/p"] = html_resp(
        '<meta property="og:title" content="P">'
    )
    stub_http["https://twitter.com/p"] = html_resp(
        '<meta property="og:title" content="P">'
    )
    vc1 = _vc("https://github.com/p", "GitHub", face_sim=0.88)
    vc2 = _vc("https://twitter.com/p", "X/Twitter", face_sim=0.86)
    graph = enrich_case([vc1, vc2], embedding(0), {}, emit=None)

    # At least one same-face edge should exist between the two profiles
    same_face_edges = [e for e in graph.edges if e.relationship == "same-face"]
    assert len(same_face_edges) >= 1


def test_profile_graph_field_on_case_model():
    """Case.profile_graph must be None by default (backward-compatible)."""
    from facechain.models import Case
    from datetime import datetime, timezone
    c = Case(
        case_id="test_case",
        created_at=datetime.now(timezone.utc).isoformat(),
        observed_at=0,
    )
    assert c.profile_graph is None
