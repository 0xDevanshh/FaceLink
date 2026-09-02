"""How the finite fetch-and-measure budget is spent.

The queue decides which leads get looked at, and the single property it must
never violate is that **priority is not exclusivity**. Priority platforms go
first because they are the likeliest leads, but the wider public web keeps a
reserved share of the budget — otherwise a run that happens to surface twelve
LinkedIn pages would never look anywhere else, and a strong match on a personal
site would be invisible for reasons that have nothing to do with the face.
"""

from __future__ import annotations

from facechain.config import OTHER_WEB_PRIORITY
from facechain.runner import MAX_PER_DOMAIN, _spread_domains, _verification_queue
from facechain.search.base import build_candidates


def cands(*urls: str):
    return build_candidates("yandex", [{"href": u, "text": ""} for u in urls], limit=200)


def test_priority_platforms_are_queued_first():
    queue = _verification_queue(cands(
        "https://example.com/a",
        "https://github.com/someone",
        "https://linkedin.com/in/someone",
    ), limit=3)
    assert [c.platform for c in queue][:2] == ["LinkedIn", "GitHub"]


def test_the_wider_web_keeps_a_reserved_share_of_the_budget():
    """Twelve social hits must not crowd the rest of the web out entirely."""
    social = [f"https://instagram.com/p/{i}/" for i in range(20)]
    web = [f"https://site{i}.example/page" for i in range(20)]
    queue = _verification_queue(cands(*social, *web), limit=12)

    assert len(queue) == 12
    wider = [c for c in queue if c.platform_priority >= OTHER_WEB_PRIORITY]
    assert wider, "the wider web was crowded out of the queue entirely"


def test_a_run_with_no_priority_hits_still_fills_the_budget():
    """The reservation must not waste capacity when there is nothing to reserve
    it against."""
    queue = _verification_queue(cands(*[f"https://site{i}.example/p" for i in range(20)]), limit=8)
    assert len(queue) == 8


def test_a_run_with_only_priority_hits_still_fills_the_budget():
    urls = [f"https://linkedin.com/in/person{i}" for i in range(20)]
    queue = _verification_queue(cands(*urls), limit=8)
    assert len(queue) == 8


def test_one_domain_cannot_monopolise_the_budget():
    """A real run spent four of twelve slots on four near-identical pages from
    one site, which adds no independent evidence."""
    many = [f"https://onesite.example/page{i}" for i in range(10)]
    others = [f"https://other{i}.example/page" for i in range(6)]
    queue = _verification_queue(cands(*many, *others), limit=8)

    from collections import Counter
    counts = Counter(c.domain for c in queue)
    assert counts["onesite.example"] <= MAX_PER_DOMAIN
    assert len(counts) > 1


def test_domain_overflow_is_deferred_not_discarded():
    """Spare budget should still reach a busy domain's extra pages."""
    many = cands(*[f"https://onesite.example/page{i}" for i in range(6)])
    spread = _spread_domains(many, cap=2)
    assert len(spread) == len(many), "no candidate may be dropped outright"
    assert [c.url for c in spread[:2]] == [c.url for c in many[:2]]


def test_a_zero_budget_queues_nothing():
    assert _verification_queue(cands("https://linkedin.com/in/x"), limit=0) == []
    assert _verification_queue([], limit=10) == []


def test_the_queue_never_exceeds_its_limit():
    urls = [f"https://site{i}.example/p" for i in range(50)]
    for limit in (1, 3, 12, 40):
        assert len(_verification_queue(cands(*urls), limit=limit)) == limit
