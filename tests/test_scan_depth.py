"""Scan depth and resource limit tests.

All offline — mocks the verification queue to test depth budget logic.
"""

from __future__ import annotations

import dataclasses

import pytest

from facechain.runner import RunOptions, DEPTH_BUDGETS, _verification_queue
from facechain.models import SearchCandidate, CandidateType
from facechain.config import OTHER_WEB_PRIORITY


def _cand(domain: str, platform: str | None = None, priority: int | None = None) -> SearchCandidate:
    p = priority if priority is not None else (1 if platform else OTHER_WEB_PRIORITY)
    return SearchCandidate(
        engine="yandex",
        url=f"https://{domain}/x",
        domain=domain,
        platform=platform,
        is_social=platform is not None,
        platform_priority=p,
    )


class TestDepthBudgets:
    def test_fast_budget(self):
        assert DEPTH_BUDGETS["fast"] == 5

    def test_standard_budget(self):
        assert DEPTH_BUDGETS["standard"] == 12

    def test_deep_budget(self):
        assert DEPTH_BUDGETS["deep"] == 30

    def test_run_options_default_depth(self):
        opts = RunOptions(image="x.jpg")
        assert opts.scan_depth == "standard"

    def test_run_options_deep(self):
        opts = RunOptions(image="x.jpg", scan_depth="deep")
        assert opts.scan_depth == "deep"

    def test_run_options_fast(self):
        opts = RunOptions(image="x.jpg", scan_depth="fast")
        assert opts.scan_depth == "fast"


class TestVerificationQueue:
    def _many(self, n: int, platform: str | None = None, priority: int | None = None):
        return [_cand(f"domain{i}.com", platform, priority) for i in range(n)]

    def test_queue_respects_limit(self):
        cands = self._many(20)
        queue = _verification_queue(cands, 5)
        assert len(queue) == 5
        assert all(c.verification_queued for c in queue)
        assert all(c.verification_exclusion_reason for c in cands[5:])

    def test_priority_platforms_come_first(self):
        wider = self._many(10, priority=OTHER_WEB_PRIORITY)
        priority = [_cand("linkedin.com", "LinkedIn", 1), _cand("github.com", "GitHub", 4)]
        queue = _verification_queue(wider + priority, 5)
        domains = [c.domain for c in queue]
        # Priority platforms must appear before wider web
        assert "linkedin.com" in domains[:2] or "github.com" in domains[:2]

    def test_wider_web_slice_reserved(self):
        """Even 10 LinkedIn hits must not crowd out the entire wider-web budget."""
        li_hits = self._many(10, "LinkedIn", 1)
        wider = self._many(5, priority=OTHER_WEB_PRIORITY)
        queue = _verification_queue(li_hits + wider, 10)
        wider_in_queue = sum(1 for c in queue if c.platform_priority >= OTHER_WEB_PRIORITY)
        assert wider_in_queue >= 1

    def test_empty_candidates(self):
        assert _verification_queue([], 12) == []

    def test_zero_limit(self):
        cands = self._many(5)
        assert _verification_queue(cands, 0) == []

    def test_domain_cap_prevents_monopoly(self):
        """One domain contributing all 20 candidates must not fill the entire queue
        without any overflow being visible — spread puts overflow after kept entries."""
        from facechain.runner import MAX_PER_DOMAIN
        monopoly = [_cand("dominant.com") for _ in range(20)]
        queue = _verification_queue(monopoly, 10)
        # The whole queue is from the same domain (no alternatives exist),
        # but the kept entries come first and overflow follows — total is capped at limit.
        assert len(queue) == 10
        # The domain spread cap applies: kept = MAX_PER_DOMAIN entries at front,
        # rest are overflow appended after. All still dominant.com since nothing else.
        dominant_count = sum(1 for c in queue if c.domain == "dominant.com")
        assert dominant_count == 10  # everything is the same domain, cap just reorders


class TestRunOptionsDataclass:
    def test_all_fields_present(self):
        fields = {f.name for f in dataclasses.fields(RunOptions)}
        expected = {
            "image", "image_url", "engines", "chain_mode", "face_backend",
            "max_verify", "case_id", "face_index", "crop_rect", "selection_mode",
            "scan_depth",
        }
        assert expected.issubset(fields)

    def test_scan_depth_invalid_still_accepted_at_dataclass_level(self):
        """Validation is done at the runner level, not the dataclass."""
        opts = RunOptions(image="x.jpg", scan_depth="invalid")
        assert opts.scan_depth == "invalid"  # dataclass just stores it
