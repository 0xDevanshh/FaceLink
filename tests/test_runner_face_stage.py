"""The runner's face stage, driven through `run()` on real photographs.

These tests go through the real entry point with the real detector, because the
things worth pinning here are integration facts: which face got embedded, what
the evidence says about how it was chosen, and whether the original image
survives a crop. Search is neutralised (no engines resolve), which is enough —
`face_selection` and `face` are written before the search stage runs.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from facechain.config import REPO_ROOT, settings
from facechain.runner import RunOptions, run

SAMPLES = REPO_ROOT / "samples"
ONE_FACE = SAMPLES / "satya_nadella.jpg"
OTHER_FACE = SAMPLES / "sundar_pichai.jpg"

pytestmark = pytest.mark.skipif(
    not (ONE_FACE.exists() and OTHER_FACE.exists()),
    reason="sample photographs are required for the real face stage",
)


@pytest.fixture(autouse=True)
def isolated_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "evidence_dir", tmp_path / "evidence")
    return tmp_path


def two_face_photo(tmp_path) -> tuple:
    """Two real faces of comparable size, side by side.

    A composite of the two sample photographs — a genuine two-person image, so
    the ambiguity the selection UI exists for is real rather than simulated.
    """
    a = cv2.imread(str(ONE_FACE))
    b = cv2.imread(str(OTHER_FACE))
    h = 900
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    canvas = np.hstack([a, b])
    path = tmp_path / "two_faces.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path, canvas


def opts(image, **kw) -> RunOptions:
    base = dict(image=str(image), chain_mode="skip", engines=["__none__"])
    base.update(kw)
    return RunOptions(**base)


# ---- single face: auto-selected, honestly labelled ----------------------

def test_a_single_face_photo_is_auto_selected():
    case = run(opts(ONE_FACE))
    assert case.face.detected
    assert case.face.faces_found == 1
    assert case.face_selection.mode == "auto"
    assert case.face_selection.face_index == 0
    assert case.face_selection.crop_rect is None
    assert case.face.quality.passed


def test_an_explicit_auto_mode_is_not_relabelled_as_manual():
    """Regression: the UI auto-selects, then passes the index it was given plus
    `selection_mode="auto"`. The runner inferred "manual-face" from the presence
    of an index, so the evidence claimed a human chose a face they never chose.
    """
    case = run(opts(ONE_FACE, face_index=0, selection_mode="auto"))
    assert case.face_selection.mode == "auto"


def test_an_index_with_no_declared_mode_is_recorded_as_manual():
    case = run(opts(ONE_FACE, face_index=0))
    assert case.face_selection.mode == "manual-face"


def test_the_embedding_hash_is_recorded_and_the_vector_is_not():
    case = run(opts(ONE_FACE))
    dumped = json.dumps(case.model_dump(mode="json"))
    assert case.face.embedding_sha256 and len(case.face.embedding_sha256) == 64
    assert case.face.embedding_dimension == 512
    assert "embedding\":" not in dumped.replace("embedding_sha256", "").replace(
        "embedding_dimension", "")


# ---- two faces: the pipeline refuses to guess ---------------------------

def test_two_comparable_faces_stop_the_scan_for_a_selection(tmp_path):
    path, _ = two_face_photo(tmp_path)
    case = run(opts(path))

    assert case.verdict == "FACE_SELECTION_REQUIRED"
    assert case.face.faces_found >= 2
    assert len(case.face.faces) >= 2
    assert case.face_selection.mode == "pending"
    # The reason is actionable, not a generic failure.
    assert "choose" in case.failure_reason.lower() or "select" in case.failure_reason.lower()
    # No search ran and no match was invented.
    assert case.best_match is None
    assert case.reverse_search is None


def test_choosing_a_face_lets_the_scan_continue(tmp_path):
    path, _ = two_face_photo(tmp_path)
    offered = run(opts(path))
    assert offered.verdict == "FACE_SELECTION_REQUIRED"

    chosen = run(opts(path, face_index=1, selection_mode="manual-face"))
    assert chosen.verdict != "FACE_SELECTION_REQUIRED"
    assert chosen.face_selection.face_index == 1
    assert chosen.face_selection.mode == "manual-face"
    assert chosen.face.embedding_sha256


def test_different_faces_in_one_photo_give_different_embeddings(tmp_path):
    """The selection has to actually change what gets embedded."""
    path, _ = two_face_photo(tmp_path)
    first = run(opts(path, face_index=0, selection_mode="manual-face"))
    second = run(opts(path, face_index=1, selection_mode="manual-face"))
    assert first.face.embedding_sha256 != second.face.embedding_sha256
    assert first.face.bbox != second.face.bbox


def test_an_out_of_range_face_index_is_refused_not_silently_reassigned():
    case = run(opts(ONE_FACE, face_index=7))
    assert case.verdict == "INVALID_FACE_SELECTION"
    assert "does not exist" in case.failure_reason
    assert case.face_selection is None


# ---- crops never replace the original ----------------------------------

def test_a_crop_is_applied_and_the_original_hash_is_still_recorded(tmp_path):
    path, canvas = two_face_photo(tmp_path)
    from facechain.evidence.hashing import sha256_file
    from facechain.face.encoder import read_image

    original_sha = sha256_file(path)
    working = read_image(path)
    # Crop the left half, where the first face is.
    rect = [0, 0, working.shape[1] // 2, working.shape[0]]
    case = run(opts(path, crop_rect=rect, selection_mode="manual-crop"))

    sel = case.face_selection
    assert sel.mode == "manual-crop"
    assert sel.crop_rect == rect
    assert sel.crop_sha256 and len(sel.crop_sha256) == 64
    # The crop is an *additional* artefact; the upload's own hash is unchanged.
    assert sel.original_sha256 == original_sha
    assert case.input.sha256 == original_sha
    assert sel.crop_sha256 != sel.original_sha256
    assert (sel.original_width, sel.original_height) == (working.shape[1], working.shape[0])


def test_the_crop_artefact_is_written_into_the_bundle(tmp_path):
    path, working = two_face_photo(tmp_path)
    from facechain.face.encoder import read_image
    img = read_image(path)
    case = run(opts(path, crop_rect=[0, 0, img.shape[1] // 2, img.shape[0]],
                    selection_mode="manual-crop"))

    bundle = settings.evidence_dir / case.case_id
    crop = bundle / "artifacts" / "selected_crop.png"
    assert crop.exists(), "the operator's crop must be preserved as evidence"

    from facechain.evidence.hashing import sha256_bytes
    assert sha256_bytes(crop.read_bytes()) == case.face_selection.crop_sha256
    assert (bundle / "face_selection.json").exists()
    assert (bundle / "selected_crop.sha256").exists()
    # And the untouched original is in there too.
    assert list((bundle / "artifacts").glob("input.*"))


def test_a_crop_around_one_face_resolves_the_ambiguity(tmp_path):
    """Drawing a box around the person you mean is itself the answer, so no
    further prompt is warranted."""
    path, _ = two_face_photo(tmp_path)
    from facechain.face.encoder import read_image
    img = read_image(path)

    case = run(opts(path, crop_rect=[0, 0, img.shape[1] // 2, img.shape[0]]))
    assert case.verdict != "FACE_SELECTION_REQUIRED"
    assert case.face.embedding_sha256


@pytest.mark.parametrize("rect", [[0, 0, 1, 1], [99999, 99999, 100, 100], [0, 0, -5, -5]])
def test_an_invalid_crop_stops_the_scan_with_a_reason(rect):
    case = run(opts(ONE_FACE, crop_rect=rect))
    assert case.verdict == "INVALID_CROP"
    assert "crop rejected" in case.failure_reason


# ---- quality gate is actually wired in ---------------------------------

def test_a_blurred_photo_is_rejected_by_the_quality_gate(tmp_path):
    """The gate module used to exist but was never called by the pipeline."""
    img = cv2.imread(str(ONE_FACE))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=18)
    path = tmp_path / "blurred.jpg"
    cv2.imwrite(str(path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])

    case = run(opts(path))
    assert case.verdict in ("FACE_QUALITY_INSUFFICIENT", "NO_FACE")
    if case.verdict == "FACE_QUALITY_INSUFFICIENT":
        assert "insufficient" in case.failure_reason.lower()
        assert case.face.quality is not None and not case.face.quality.passed
    # Either way, no search ran and nothing was matched.
    assert case.best_match is None


def test_a_photo_with_no_face_is_refused_with_actionable_advice(tmp_path):
    rng = np.random.default_rng(11)
    noise = rng.integers(60, 200, (500, 400, 3), dtype=np.uint8)
    path = tmp_path / "noise.jpg"
    cv2.imwrite(str(path), noise)

    case = run(opts(path))
    assert case.verdict == "NO_FACE"
    assert "clearer image" in case.failure_reason
    assert case.best_match is None


def test_the_quality_report_is_recorded_even_on_a_passing_run():
    case = run(opts(ONE_FACE))
    q = case.face.quality
    assert q is not None and q.passed
    assert q.blur_score > 0 and q.face_px > 0


def test_every_case_carries_the_threshold_snapshot_that_governed_it():
    """Even a run that never reaches search (no engines configured) must
    record which thresholds were in force — a threshold change must be
    auditable in the evidence, not just in the current config file."""
    from facechain.config import settings as global_settings

    case = run(opts(ONE_FACE))
    snap = case.threshold_snapshot
    assert snap is not None
    assert snap.face_match_threshold == global_settings.face_match_threshold
    assert snap.image_match_threshold == global_settings.image_match_threshold
    assert snap.verify_min_score == global_settings.verify_min_score
    assert snap.calibration_status == "DEFAULT"


def test_a_configured_calibration_file_is_reflected_in_the_snapshot(tmp_path, monkeypatch):
    import json
    from facechain.config import settings as global_settings

    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({"status": "CALIBRATED", "note": "60/60 pairs"}))
    monkeypatch.setattr(global_settings, "calibration_file", str(calib))

    case = run(opts(ONE_FACE))
    assert case.threshold_snapshot.calibration_status == "CALIBRATED"
    assert "60/60" in case.threshold_snapshot.calibration_note


# ---- the bundle is written on every terminal path -----------------------

@pytest.mark.parametrize("kwargs,expected", [
    ({}, "auto"),
    ({"face_index": 7}, None),
])
def test_a_bundle_is_written_even_when_the_face_stage_refuses(kwargs, expected):
    case = run(opts(ONE_FACE, **kwargs))
    bundle = settings.evidence_dir / case.case_id
    assert (bundle / "case.json").exists()
    if expected:
        assert json.loads((bundle / "face_selection.json").read_text())["mode"] == expected


# ---- what actually gets searched ---------------------------------------

def test_a_crop_becomes_the_search_query(tmp_path, monkeypatch):
    """Regression: cropping one person out of a group shot still searched the
    whole frame, so every hit matched the composite. A real run scored 0.079 on
    the cropped face where the uncropped run on the same person scored 0.96."""
    from facechain import runner as runner_mod
    from facechain.models import SearchReport

    seen: dict = {}

    def fake_search(image_path, engines=None, image_url=None, on_event=None, variants=None):
        seen["path"] = image_path
        return SearchReport(engines_attempted=list(engines or [])), None

    monkeypatch.setattr(runner_mod, "run_reverse_search", fake_search)

    path, _ = two_face_photo(tmp_path)
    from facechain.face.encoder import read_image
    img = read_image(path)
    case = run(opts(path, crop_rect=[0, 0, img.shape[1] // 2, img.shape[0]]))

    assert seen["path"].endswith("selected_crop.png"), seen["path"]
    # And the original upload is still the evidence anchor.
    assert case.input.sha256 == case.face_selection.original_sha256


def test_a_face_selection_without_a_crop_searches_the_whole_upload(tmp_path, monkeypatch):
    """Deliberate: the other people in a photo are part of the photograph, the
    full frame can match the original exactly, and it is face verification —
    not the query framing — that decides whether a hit is the right person."""
    from facechain import runner as runner_mod
    from facechain.models import SearchReport

    seen: dict = {}

    def fake_search(image_path, engines=None, image_url=None, on_event=None, variants=None):
        seen["path"] = image_path
        return SearchReport(engines_attempted=list(engines or [])), None

    monkeypatch.setattr(runner_mod, "run_reverse_search", fake_search)

    path, _ = two_face_photo(tmp_path)
    run(opts(path, face_index=1, selection_mode="manual-face"))
    assert seen["path"] == str(path)


def test_the_download_budget_guard_uses_real_bytes_not_an_estimate(tmp_path, monkeypatch):
    """Regression: the guard used to estimate `downloads * 500_000` bytes
    instead of summing what verification actually downloaded, so it could
    keep running well past the real 50MB ceiling (or stop early for small
    images). It must react to `MediaCache.total_bytes` directly."""
    from facechain import runner as runner_mod
    from facechain.models import SearchCandidate, SearchReport

    monkeypatch.setattr(runner_mod, "MAX_DOWNLOAD_BYTES", 10_000_000)  # 10MB for this test

    candidates = [
        SearchCandidate(engine="test", url=f"https://example.com/{i}", domain="example.com")
        for i in range(5)
    ]

    def fake_search(image_path, engines=None, image_url=None, on_event=None, variants=None):
        return SearchReport(engines_attempted=list(engines or []), candidates=candidates,
                            total_candidates=len(candidates)), None

    calls: list[int] = []

    def fake_verify_candidate(cand, input_hashes, input_embedding, cache=None):
        calls.append(1)
        # Each call "downloads" 6MB — the second call should push the running
        # total past the 10MB budget and stop a third call from happening.
        cache.total_bytes += 6_000_000
        from facechain.models import VerifiedCandidate
        return VerifiedCandidate(engine=cand.engine, url=cand.url, domain=cand.domain,
                                 canonical_url=cand.url)

    monkeypatch.setattr(runner_mod, "run_reverse_search", fake_search)
    monkeypatch.setattr(runner_mod, "verify_candidate", fake_verify_candidate)

    run(opts(ONE_FACE))

    # First call: 0MB -> proceeds, ends at 6MB. Second call: 6MB -> proceeds
    # (under budget when checked), ends at 12MB. Third call: 12MB >= 10MB ->
    # stops before verifying candidate 3, 4, or 5.
    assert len(calls) == 2


def test_the_case_carries_an_evidence_graph_with_independent_sources(tmp_path, monkeypatch):
    """End-to-end: two verified hits on different platforms/domains should
    show up in `case.evidence_graph` as an independent_source relationship,
    not just as two rows in the candidate list."""
    from facechain import runner as runner_mod
    from facechain.models import SearchCandidate, SearchReport, VerifiedCandidate

    candidates = [
        SearchCandidate(engine="test", url="https://linkedin.com/in/alice",
                        domain="linkedin.com", platform="LinkedIn", is_social=True),
        SearchCandidate(engine="test", url="https://github.com/alice",
                        domain="github.com", platform="GitHub", is_social=True),
    ]

    def fake_search(image_path, engines=None, image_url=None, on_event=None, variants=None):
        return SearchReport(engines_attempted=list(engines or []), candidates=candidates,
                            total_candidates=len(candidates)), None

    def fake_verify_candidate(cand, input_hashes, input_embedding, cache=None):
        # score_candidate() recomputes verified/final_score/stages from these
        # measurements — face+image similarity here are high enough to clear
        # verify_min_score (0.70) once weighted, so both genuinely verify.
        return VerifiedCandidate(
            engine=cand.engine, url=cand.url, domain=cand.domain, platform=cand.platform,
            is_social=True, canonical_url=cand.url,
            candidate_image_phash="0011" if "linkedin" in cand.url else "ff00",
            face_detected=True, face_similarity=0.95, image_similarity=0.6,
        )

    monkeypatch.setattr(runner_mod, "run_reverse_search", fake_search)
    monkeypatch.setattr(runner_mod, "verify_candidate", fake_verify_candidate)

    case = run(opts(ONE_FACE))

    assert case.evidence_graph is not None
    assert case.evidence_graph.independent_evidence_count == 2
    assert any(e.type == "independent_source" for e in case.evidence_graph.edges)
    assert any(n.type == "domain" and n.label == "linkedin.com" for n in case.evidence_graph.nodes)
    assert any(n.type == "domain" and n.label == "github.com" for n in case.evidence_graph.nodes)
