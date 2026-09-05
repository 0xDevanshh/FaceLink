"""Platform classification, discovery priority, and candidate typing.

These are the functions that decide *where we look first* and *how a match is
labelled*. Two properties matter enough to test hard:

  1. Recognition is hostname-exact, so a lookalike domain cannot borrow a
     platform's name (and with it, its priority).
  2. Priority orders discovery only. Nothing here may make a candidate more or
     less verifiable — that is `scorer`'s job, and `test_scoring.py` pins it.
"""

import pytest

from facechain.config import (
    OTHER_WEB_PRIORITY,
    PLATFORM_PRIORITY,
    RECOGNISED_PLATFORM_PRIORITY,
    platform_priority,
)
from facechain.models import CandidateType
from facechain.search.base import (
    build_candidates,
    classify_platform,
    classify_social,
    host_matches,
    initial_candidate_type,
    is_github_media,
    normalise_domain,
)


# ---- the four priority platforms + YouTube -------------------------------

@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://linkedin.com/in/someone", "LinkedIn"),
        ("https://www.linkedin.com/in/someone", "LinkedIn"),
        ("https://in.linkedin.com/in/someone", "LinkedIn"),
        ("https://instagram.com/someone", "Instagram"),
        ("https://www.instagram.com/p/ABC/", "Instagram"),
        ("https://x.com/someone", "X/Twitter"),
        ("https://twitter.com/someone/status/1", "X/Twitter"),
        ("https://github.com/torvalds", "GitHub"),
        ("https://www.github.com/torvalds", "GitHub"),
        ("https://avatars.githubusercontent.com/u/1024025?v=4", "GitHub"),
        ("https://user-images.githubusercontent.com/1/a.png", "GitHub"),
        ("https://raw.githubusercontent.com/o/r/main/a.png", "GitHub"),
        ("https://youtube.com/watch?v=abc", "YouTube"),
        ("https://youtu.be/abc", "YouTube"),
        ("https://www.youtube.com/@someone", "YouTube"),
    ],
)
def test_priority_platforms_are_recognised(url, platform):
    recognised, found, _ = classify_platform(url)
    assert recognised and found == platform


def test_x_and_twitter_are_one_logical_platform():
    """The same account reached by either hostname is the same platform…"""
    assert classify_social("https://x.com/u/status/1")[1] == \
           classify_social("https://twitter.com/u/status/1")[1]


def test_but_the_original_hostname_is_preserved():
    """…while the candidate still records the URL it was actually found at."""
    cands = build_candidates("yandex", [
        {"href": "https://twitter.com/u/status/1", "text": ""},
        {"href": "https://x.com/u/status/2", "text": ""},
    ])
    hosts = {c.domain for c in cands}
    assert hosts == {"twitter.com", "x.com"}


# ---- lookalike / hostile domains ----------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://linkedin.com.attacker.com/in/x",
        "https://github.com.fake-site.com/torvalds",
        "https://instagram.com.example.org/p/A/",
        "https://x.com.bad-domain.net/u",
        "https://notlinkedin.com/in/x",
        "https://mylinkedin.com/in/x",
        "https://github.com.evil/torvalds",
        "https://fakegithubusercontent.com/a.png",
    ],
)
def test_lookalike_domains_are_not_recognised(url):
    recognised, platform, priority = classify_platform(url)
    assert not recognised
    assert platform is None
    assert priority == OTHER_WEB_PRIORITY


def test_homoglyph_domain_cannot_impersonate_a_platform():
    """A Cyrillic 'а' in "аpple.com" must not compare equal to ASCII text.

    Hostnames are IDNA-encoded before matching, so a deceptive-Unicode domain
    becomes its `xn--` form and can only match another punycode name.
    """
    cyrillic_x = "https://х.com/someone"   # U+0445 CYRILLIC SMALL LETTER HA
    assert normalise_domain(cyrillic_x).startswith("xn--")
    assert classify_platform(cyrillic_x)[0] is False


def test_host_matches_requires_a_dot_boundary():
    assert host_matches("linkedin.com", "linkedin.com")
    assert host_matches("in.linkedin.com", "linkedin.com")
    assert not host_matches("linkedin.com.evil.net", "linkedin.com")
    assert not host_matches("notlinkedin.com", "linkedin.com")
    assert not host_matches("", "linkedin.com")


def test_unparseable_url_is_not_a_platform():
    assert classify_platform("not a url at all") == (False, None, OTHER_WEB_PRIORITY)
    assert classify_platform("") == (False, None, OTHER_WEB_PRIORITY)


# ---- discovery priority --------------------------------------------------

def test_priority_order_is_linkedin_instagram_x_github_youtube():
    order = ["LinkedIn", "Instagram", "X/Twitter", "GitHub", "YouTube"]
    priorities = [platform_priority(p) for p in order]
    assert priorities == sorted(priorities), priorities
    assert len(set(priorities)) == len(priorities), "priorities must be distinct"


def test_recognised_but_unlisted_platform_sits_between_youtube_and_the_web():
    facebook = platform_priority("Facebook")
    assert platform_priority("YouTube") < facebook < OTHER_WEB_PRIORITY
    assert facebook == RECOGNISED_PLATFORM_PRIORITY


def test_unrecognised_web_is_last_but_still_present():
    assert platform_priority(None) == OTHER_WEB_PRIORITY
    assert platform_priority("Nowhere") == RECOGNISED_PLATFORM_PRIORITY


def test_candidates_are_ordered_by_discovery_priority():
    rows = [
        {"href": "https://example.com/page", "text": "wider web"},
        {"href": "https://www.youtube.com/watch?v=a", "text": "yt"},
        {"href": "https://github.com/someone", "text": "gh"},
        {"href": "https://x.com/someone/status/1", "text": "x"},
        {"href": "https://instagram.com/p/A/", "text": "ig"},
        {"href": "https://linkedin.com/in/someone", "text": "li"},
    ]
    got = [c.platform for c in build_candidates("yandex", rows)]
    assert got == ["LinkedIn", "Instagram", "X/Twitter", "GitHub", "YouTube", None]


def test_wider_web_candidates_are_kept_not_discarded():
    """Priority is not a filter — an unlisted domain still becomes a candidate."""
    rows = [{"href": "https://some-conference.org/speakers/x", "text": "speaker"}]
    cands = build_candidates("yandex", rows)
    assert len(cands) == 1
    assert cands[0].platform is None
    assert cands[0].platform_priority == OTHER_WEB_PRIORITY


# ---- candidate typing ----------------------------------------------------

def test_github_urls_type_as_developer_profiles():
    assert initial_candidate_type("https://github.com/torvalds", "GitHub") == \
        CandidateType.DEVELOPER_PROFILE
    assert initial_candidate_type("https://avatars.githubusercontent.com/u/1", "GitHub") == \
        CandidateType.DEVELOPER_PROFILE


def test_social_posts_and_profiles_are_typed_apart():
    assert initial_candidate_type("https://instagram.com/p/ABC/", "Instagram") == \
        CandidateType.SOCIAL_POST
    assert initial_candidate_type("https://instagram.com/someone", "Instagram") == \
        CandidateType.SOCIAL_PROFILE


def test_articles_and_plain_pages_are_typed_apart():
    assert initial_candidate_type("https://example.com/blog/a-post", None) == \
        CandidateType.PUBLIC_ARTICLE
    assert initial_candidate_type("https://example.com/team", None) == \
        CandidateType.PUBLIC_WEB_PAGE


def test_github_media_hosts_are_identified():
    assert is_github_media("https://avatars.githubusercontent.com/u/1?v=4")
    assert is_github_media("https://raw.githubusercontent.com/o/r/main/a.png")
    assert not is_github_media("https://github.com/torvalds")
    assert not is_github_media("https://fakegithubusercontent.com/a.png")


def test_every_priority_platform_has_an_entry_in_the_table():
    """Guards against adding a platform to one table and not the other."""
    for name in PLATFORM_PRIORITY:
        assert platform_priority(name) == PLATFORM_PRIORITY[name]


# ---- professional/developer platforms -------------------------------------
#
# `enrichment/extractor.py` already recognises these for profile enrichment;
# the search layer must agree, or a reverse-image hit landing directly on one
# of these domains gets filed as unrecognised "Other Web" at the lowest
# priority instead of being attributed to its actual platform.

@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://medium.com/@someone/a-post-1234", "Medium"),
        ("https://devfolio.co/@someone", "Devfolio"),
        ("https://leetcode.com/someone", "LeetCode"),
        ("https://www.hackerrank.com/someone", "HackerRank"),
        ("https://kaggle.com/someone", "Kaggle"),
        ("https://stackoverflow.com/users/12345/someone", "StackOverflow"),
    ],
)
def test_professional_platforms_are_recognised(url, platform):
    recognised, found, priority = classify_platform(url)
    assert recognised and found == platform
    # Named-but-not-priority tier — same as Facebook/Reddit today, not a
    # reordering of the LinkedIn/Instagram/X/GitHub/YouTube priority list.
    assert priority == RECOGNISED_PLATFORM_PRIORITY


# ---- platform media CDNs -------------------------------------------------
#
# These matter more than they look. A reverse-image engine that finds someone's
# LinkedIn profile photo returns `media.licdn.com/...`, not `linkedin.com/in/...`
# — the CDN is where the pixels live, and it is a different registrable domain.
# Matching only the site domain filed every such hit as unrecognised "Other Web"
# at the lowest discovery priority, so the platform was mis-reported *and* the
# lead was de-prioritised for verification.

@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://media.licdn.com/dms/image/v2/D56/profile-photo.jpg", "LinkedIn"),
        ("https://static.licdn.com/aero-v1/sc/h/abc.png", "LinkedIn"),
        ("https://scontent-iad3-1.cdninstagram.com/v/t51.2885-19/abc.jpg", "Instagram"),
        ("https://pbs.twimg.com/profile_images/123456/abc_400x400.jpg", "X/Twitter"),
        ("https://abs.twimg.com/sticky/abc.png", "X/Twitter"),
        ("https://avatars.githubusercontent.com/u/1024025?v=4", "GitHub"),
        ("https://yt3.ggpht.com/ytc/abc=s900-c-k-c0x00ffffff-no-rj", "YouTube"),
        ("https://i.ytimg.com/vi/abc/maxresdefault.jpg", "YouTube"),
        ("https://i.pinimg.com/736x/ab/cd/ef.jpg", "Pinterest"),
        ("https://p16.tiktokcdn.com/img/abc~c5_100x100.jpeg", "TikTok"),
        ("https://i.redd.it/abc.jpg", "Reddit"),
    ],
)
def test_platform_media_cdns_are_attributed_to_their_platform(url, platform):
    recognised, found, priority = classify_platform(url)
    assert recognised and found == platform
    # And they inherit the platform's discovery priority, not the web's.
    assert priority == platform_priority(platform) < OTHER_WEB_PRIORITY


@pytest.mark.parametrize(
    "url",
    [
        "https://licdn.com.attacker.net/profile.jpg",
        "https://notlicdn.com/profile.jpg",
        "https://mytwimg.com/a.jpg",
        "https://pinimg.com.evil.org/a.jpg",
        "https://faketiktokcdn.com/a.jpg",
    ],
)
def test_lookalike_cdn_domains_cannot_borrow_the_attribution(url):
    assert classify_platform(url)[0] is False


def test_a_shared_cdn_is_left_unattributed():
    """`fbcdn.net` serves both Facebook and Instagram. Guessing which would put
    a claim in the evidence that the URL does not support, so it stays Other Web.
    """
    recognised, platform, priority = classify_platform("https://scontent.fbcdn.net/v/t1.jpg")
    assert not recognised and platform is None
    assert priority == OTHER_WEB_PRIORITY


def test_the_site_domain_wins_over_the_cdn_table():
    """A real page URL must never be typed as a media asset."""
    assert classify_platform("https://www.linkedin.com/in/someone")[1] == "LinkedIn"
    assert initial_candidate_type("https://www.linkedin.com/in/someone", "LinkedIn") == \
        CandidateType.SOCIAL_PROFILE


def test_media_cdn_domains_do_not_overlap_the_site_table():
    """A host in both tables would make attribution order-dependent."""
    from facechain.config import PLATFORM_MEDIA_DOMAINS, SOCIAL_DOMAINS

    overlap = set(PLATFORM_MEDIA_DOMAINS) & set(SOCIAL_DOMAINS)
    assert overlap == {"githubusercontent.com"}, overlap
    # The one intentional overlap must agree on the platform.
    assert PLATFORM_MEDIA_DOMAINS["githubusercontent.com"] == \
        SOCIAL_DOMAINS["githubusercontent.com"] == "GitHub"


def test_every_media_cdn_maps_to_a_platform_we_can_rank():
    from facechain.config import PLATFORM_MEDIA_DOMAINS

    for domain, platform in PLATFORM_MEDIA_DOMAINS.items():
        assert platform_priority(platform) < OTHER_WEB_PRIORITY, domain


@pytest.mark.parametrize(
    "url,expected",
    [
        # `looks_like_post` only counts slashes, so these all looked like posts.
        ("https://www.linkedin.com/in/someone", CandidateType.SOCIAL_PROFILE),
        ("https://www.linkedin.com/company/acme", CandidateType.SOCIAL_PROFILE),
        ("https://www.youtube.com/channel/UCabc", CandidateType.SOCIAL_PROFILE),
        # …while genuine posts still type as posts.
        ("https://www.linkedin.com/posts/someone-activity-123", CandidateType.SOCIAL_POST),
        ("https://x.com/someone/status/123", CandidateType.SOCIAL_POST),
        ("https://instagram.com/p/ABC/", CandidateType.SOCIAL_POST),
    ],
)
def test_profile_paths_are_not_mistyped_as_posts(url, expected):
    platform = classify_platform(url)[1]
    assert initial_candidate_type(url, platform) == expected
