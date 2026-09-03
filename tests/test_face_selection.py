"""Face selection: auto-select, multi-face prompts, manual crops, quality gate.

The property that matters most here is that the pipeline refuses to guess. A
wrong automatic pick does not produce a weak result — it produces a confident
result about the wrong person, which is worse than asking. So the tests below
lean on: when is auto-select allowed, when must we prompt, and can a crop ever
quietly stand in for the original evidence.

Synthetic images are used so the geometry is known exactly; `tests/test_face.py`
covers the real detector against real photographs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from facechain.config import settings
from facechain.face import selection as fs
from facechain.face.detector import DetectedFace
from facechain.face.quality import QualityError


def canvas(w: int = 600, h: int = 800, value: int = 128) -> np.ndarray:
    """A mid-grey image with texture, so exposure and blur gates pass."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(1234)
    img = np.clip(img.astype(np.int16) + rng.integers(-60, 60, img.shape), 0, 255).astype(np.uint8)
    return img


def face(x1: int, y1: int, x2: int, y2: int, score: float = 0.9, dim: int = 512) -> DetectedFace:
    emb = np.zeros(dim, dtype=np.float32)
    emb[0] = 1.0
    return DetectedFace(bbox=(x1, y1, x2, y2), det_score=score, embedding=emb)


# ---- auto-selection ------------------------------------------------------

def test_one_good_face_is_auto_selected_without_prompting():
    img = canvas()
    offer = fs.offer(img, [face(100, 100, 300, 340)])
    assert offer.auto_index == 0
    assert not offer.selection_required
    assert offer.faces[0].usable


def test_two_comparable_faces_require_a_choice():
    """"Largest wins" is exactly wrong for a two-person photo."""
    img = canvas()
    offer = fs.offer(img, [face(50, 50, 250, 290), face(320, 60, 510, 295)])
    assert offer.selection_required
    assert offer.auto_index is None
    assert "comparable size" in offer.reason


def test_a_clearly_dominant_face_still_auto_selects():
    """A subject plus a distant bystander is not ambiguous."""
    img = canvas()
    big = face(50, 50, 350, 400)                        # 300x350
    small = face(500, 700, 545, 745, score=0.8)         # 45x45, far smaller
    offer = fs.offer(img, [big, small])
    assert not offer.selection_required
    assert offer.auto_index == 0


def test_low_confidence_detection_requires_confirmation():
    img = canvas()
    offer = fs.offer(img, [face(100, 100, 300, 340, score=0.55)])
    assert offer.selection_required
    assert "uncertain" in offer.reason


def test_a_face_below_the_size_floor_is_marked_unusable_and_prompts():
    img = canvas()
    tiny = settings.min_face_px - 10
    offer = fs.offer(img, [face(10, 10, 10 + tiny, 10 + tiny)])
    assert offer.selection_required
    assert not offer.faces[0].usable
    assert "not usable" in offer.reason
    assert str(tiny) in offer.faces[0].note


def test_no_faces_needs_no_selection_and_says_so():
    offer = fs.offer(canvas(), [])
    assert offer.faces == []
    assert offer.auto_index is None
    assert not offer.selection_required
    assert "no usable face" in offer.reason


def test_reject_policy_forces_a_choice_for_multiple_faces(monkeypatch):
    monkeypatch.setattr(settings, "multi_face_policy", "reject")
    img = canvas()
    offer = fs.offer(img, [face(50, 50, 350, 400), face(500, 700, 560, 760, score=0.8)])
    assert offer.selection_required
    assert "multi_face_policy=reject" in offer.reason


def test_described_faces_never_carry_an_embedding():
    """The UI gets geometry and quality; the vector stays on the server."""
    infos = fs.describe_faces([face(10, 10, 200, 220)], canvas())
    dumped = infos[0].model_dump()
    assert "embedding" not in dumped
    assert set(dumped) == {"index", "bbox", "det_score", "face_px", "area", "usable", "note"}


def test_boxes_are_clamped_to_the_image():
    img = canvas(600, 800)
    infos = fs.describe_faces([face(-50, -30, 900, 1200)], img)
    assert infos[0].bbox == [0, 0, 600, 800]


# ---- manual crops --------------------------------------------------------

def test_a_valid_crop_is_applied_exactly():
    img = canvas(600, 800)
    cropped, rect = fs.apply_crop(img, [100, 150, 200, 240])
    assert rect == (100, 150, 200, 240)
    assert cropped.shape[:2] == (240, 200)


def test_a_crop_overhanging_the_edge_is_clamped_not_rejected():
    img = canvas(600, 800)
    _, rect = fs.apply_crop(img, [500, 700, 400, 400])
    assert rect == (500, 700, 100, 100)


@pytest.mark.parametrize(
    "rect,message",
    [
        ([10, 10, 0, 100], "non-positive"),
        ([10, 10, 100, -5], "non-positive"),
        ([700, 900, 100, 100], "outside"),
        ([-500, -500, 100, 100], "outside"),
        ([0, 0, 10, 10], "at least"),
        ([1, 2, 3], "x, y, width, height"),
    ],
)
def test_invalid_crops_are_rejected_with_a_reason(rect, message):
    """Rejected rather than silently clamped to *something*.

    A degenerate rectangle means the client and server disagree about the
    coordinate space, and cropping anyway would attach a misleading rectangle
    to the evidence.
    """
    with pytest.raises(fs.CropError, match=message):
        fs.apply_crop(canvas(600, 800), rect)


def test_crop_values_must_be_numbers():
    with pytest.raises(fs.CropError, match="integers"):
        fs.normalise_crop(["a", "b", "c", "d"], canvas())


def test_a_crop_produces_its_own_hashable_artefact():
    """The crop is an *additional* artefact — the original is untouched."""
    img = canvas(600, 800)
    cropped, _ = fs.apply_crop(img, [100, 150, 200, 240])
    crop_png = fs.encode_png(cropped)
    original_png = fs.encode_png(img)
    assert crop_png and crop_png != original_png
    assert cv2.imdecode(np.frombuffer(crop_png, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (240, 200)


# ---- quality gate on the *selected* face --------------------------------

def test_gate_reports_on_the_selected_face_not_the_largest():
    """Gating one face while embedding another would let a scan run on a face
    nothing ever checked."""
    img = canvas()
    big = face(50, 50, 350, 400)
    tiny = face(500, 700, 500 + settings.min_face_px - 20, 700 + settings.min_face_px - 20)

    on_big = fs.gate_selected(img, [big, tiny], face_index=0)
    on_tiny = fs.gate_selected(img, [big, tiny], face_index=1)

    assert on_big.passed
    assert not on_tiny.passed
    assert on_tiny.error == QualityError.FACE_TOO_SMALL.value


def test_gate_records_the_total_face_count_even_when_one_is_selected():
    img = canvas()
    report = fs.gate_selected(img, [face(50, 50, 350, 400), face(400, 50, 500, 170)], face_index=0)
    assert report.face_count == 2


def test_gate_fails_closed_on_a_dark_image():
    dark = np.zeros((400, 400, 3), dtype=np.uint8)
    report = fs.gate_selected(dark, [face(50, 50, 250, 250)], face_index=0)
    assert not report.passed
    assert report.error == QualityError.LOW_EXPOSURE.value


def test_gate_fails_closed_on_a_blurred_image():
    img = cv2.GaussianBlur(canvas(400, 400), (0, 0), sigmaX=25)
    report = fs.gate_selected(img, [face(50, 50, 250, 250)], face_index=0)
    assert not report.passed
    assert report.error == QualityError.BLURRY.value


def test_gate_reports_no_face_when_nothing_was_detected():
    report = fs.gate_selected(canvas(), [], face_index=None)
    assert not report.passed
    assert report.error == QualityError.NO_FACE.value
    assert report.face_count == 0


# ---- graded quality metrics (informational, never gate on their own) ----

def test_a_good_pass_gets_good_bands_and_high_overall_quality():
    img = canvas()
    big = face(50, 50, 350, 400, score=0.95)
    report = fs.gate_selected(img, [big], face_index=0)
    assert report.passed
    assert report.bands["detection"] == "PASS"
    assert report.bands["resolution"] == "GOOD"
    assert report.overall_quality > 0.5


def test_a_marginal_pass_still_reports_graded_metrics():
    """A face just above the hard floor should pass but read as lower quality,
    not be silently indistinguishable from a great photo."""
    img = canvas()
    marginal = face(50, 50, 50 + settings.min_face_px + 2, 50 + settings.min_face_px + 2)
    report = fs.gate_selected(img, [marginal], face_index=0)
    assert report.passed
    assert report.bands["resolution"] in ("ACCEPTABLE", "POOR")
    assert report.overall_quality < 1.0


def test_rejected_reports_still_carry_graded_metrics_for_the_primary_face():
    """MULTI_FACE / FACE_TOO_SMALL rejections should still explain *why* via
    bands rather than leaving the operator with only a bare error code."""
    img = canvas()
    tiny = face(500, 700, 500 + settings.min_face_px - 20, 700 + settings.min_face_px - 20)
    report = fs.gate_selected(img, [tiny], face_index=0)
    assert not report.passed
    assert report.error == QualityError.FACE_TOO_SMALL.value
    assert report.bands  # populated, not empty
    assert report.bands["detection"] == "PASS"


def test_frontal_face_has_low_pose_angles():
    img = canvas()
    frontal = face(100, 100, 300, 340, score=0.95)
    report = fs.gate_selected(img, [frontal], face_index=0)
    # No landmarks on this synthetic face -> pose defaults to 0.
    assert report.yaw_deg == pytest.approx(0.0)
    assert report.roll_deg == pytest.approx(0.0)
    assert report.bands["pose"] == "GOOD"


