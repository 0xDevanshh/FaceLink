"""Search-layer parsing: classification, junk filtering, URL canonicalisation.

These are the pure functions between the browser and the verifier, so they are
fully testable without touching the network.
"""

import pytest

from facechain.search.base import (
    build_candidates,
    canonicalise_url,
    classify_social,
    is_junk,
    looks_like_post,
    normalise_domain,
    unwrap_redirect,
)


@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://www.instagram.com/p/ABC123/", "Instagram"),
        ("https://instagram.com/reel/XYZ/", "Instagram"),
        ("https://x.com/someone/status/123", "X/Twitter"),
        ("https://twitter.com/someone/status/123", "X/Twitter"),
        ("https://www.linkedin.com/posts/abc-activity-123", "LinkedIn"),
        ("https://www.youtube.com/shorts/abc", "YouTube"),
        ("https://m.facebook.com/story.php?id=1", "Facebook"),
        ("https://64.media.tumblr.com/abc.jpg", "Tumblr"),
    ],
)
def test_social_classification(url, platform):
    is_social, found = classify_social(url)
    assert is_social and found == platform


@pytest.mark.parametrize(
    "url", ["https://example.com/news", "https://en.wikipedia.org/wiki/X", "https://cdn.foo.net/a.jpg"]
)
def test_non_social_urls(url):
    assert classify_social(url) == (False, None)


def test_lookalike_domain_is_not_social():
    """`instagram.com.evil.net` must not be classified as Instagram."""
    assert classify_social("https://instagram.com.evil.net/p/ABC/") == (False, None)


def test_normalise_domain_strips_www():
    assert normalise_domain("https://www.example.com/x") == "example.com"
    assert normalise_domain("not a url") == ""


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com/search?q=x",
        "https://support.google.com/websearch",
        "https://yandex.com/images/",
        "https://www.bing.com/images",
        "https://tineye.com/search",
        "https://www.w3.org/2000/svg",
    ],
)
def test_engine_chrome_is_junk(url):
    assert is_junk(url)


def test_real_results_are_not_junk():
    assert not is_junk("https://www.instagram.com/p/ABC/")
    assert not is_junk("https://news.example.com/article")


def test_unwrap_google_redirect():
    wrapped = "https://www.google.com/url?q=https%3A%2F%2Finstagram.com%2Fp%2FABC%2F&sa=U"
    assert unwrap_redirect(wrapped) == "https://instagram.com/p/ABC/"


def test_unwrap_bing_mediaurl():
    wrapped = "https://www.bing.com/images/search?mediaurl=https%3A%2F%2Fexample.com%2Fa.jpg"
    assert unwrap_redirect(wrapped) == "https://example.com/a.jpg"


def test_unwrap_leaves_plain_urls_alone():
    assert unwrap_redirect("https://instagram.com/p/ABC/") == "https://instagram.com/p/ABC/"


# ---- URL canonicalisation: this URL's hash goes on-chain ------------------

def test_canonicalise_strips_engine_tracking_params():
    dirty = "https://www.youtube.com/shorts/665OKw6IkEM?utm_medium=organic&utm_source=yandexsmartcamera"
    assert canonicalise_url(dirty) == "https://www.youtube.com/shorts/665OKw6IkEM"


def test_canonicalise_keeps_content_identifying_params():
    url = "https://www.youtube.com/watch?v=abc123&utm_source=x&t=30"
    out = canonicalise_url(url)
    assert "v=abc123" in out and "t=30" in out and "utm_source" not in out


def test_canonicalise_drops_fragment():
    assert canonicalise_url("https://instagram.com/p/ABC/#c") == "https://instagram.com/p/ABC/"


def test_canonicalisation_makes_the_same_post_hash_identically():
    """Two engines returning the same post must produce one on-chain URL hash."""
    from facechain.evidence.hashing import sha256_text

    via_yandex = "https://www.youtube.com/shorts/X?utm_source=yandexsmartcamera"
    via_bing = "https://www.youtube.com/shorts/X?utm_source=bing&msclkid=9"
    assert sha256_text(canonicalise_url(via_yandex)) == sha256_text(canonicalise_url(via_bing))


def test_looks_like_post_distinguishes_posts_from_profiles():
    assert looks_like_post("https://instagram.com/p/ABC123/")
    assert looks_like_post("https://x.com/user/status/1")
    assert not looks_like_post("https://instagram.com/username")
    assert not looks_like_post("https://instagram.com/")


# ---- candidate building --------------------------------------------------

def test_build_candidates_filters_dedupes_and_sorts_social_first():
    rows = [
        {"href": "https://news.example.com/a", "text": "news"},
        {"href": "https://www.google.com/policies", "text": "junk"},
        {"href": "not-a-url", "text": ""},
        {"href": "https://instagram.com/p/ABC/?utm_source=y", "text": "insta post"},
        {"href": "https://instagram.com/p/ABC/", "text": "same post again"},
        {"href": "https://instagram.com/someprofile", "text": "profile"},
    ]
    out = build_candidates("yandex", rows)
    urls = [c.url for c in out]

    assert "https://www.google.com/policies" not in urls
    assert not any(u == "not-a-url" for u in urls)
    assert urls.count("https://instagram.com/p/ABC/") == 1  # deduped after canonicalisation
    assert out[0].is_social and looks_like_post(out[0].url)  # post ranked first
    assert out[0].engine == "yandex"


def test_build_candidates_respects_limit():
    rows = [{"href": f"https://example{i}.com/x", "text": ""} for i in range(50)]
    assert len(build_candidates("bing", rows, limit=5)) == 5


def test_build_candidates_captures_thumbnail_and_title():
    rows = [{"href": "https://instagram.com/p/A/", "text": "caption", "thumb": "https://cdn/x.jpg"}]
    c = build_candidates("yandex", rows)[0]
    assert c.title == "caption" and c.thumbnail == "https://cdn/x.jpg"
    assert c.platform == "Instagram"
