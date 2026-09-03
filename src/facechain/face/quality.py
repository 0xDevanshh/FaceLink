"""Face quality gating — reject garbage before embedding.

Every check is a separate, named failure reason so the evidence bundle records
exactly which gate a rejected image failed. Thresholds are all in config so
operators can tune them for their hardware and use-case without touching code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..config import settings

log = logging.getLogger(__name__)


class QualityError(str, Enum):
    """Structured error codes — recorded in evidence and returned as HTTP 422 from the API."""

    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    BLURRY = "BLURRY"
    INVALID_IMAGE = "INVALID_IMAGE"
    CORRUPT_FILE = "CORRUPT_FILE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    LOW_EXPOSURE = "LOW_EXPOSURE"
    HIGH_EXPOSURE = "HIGH_EXPOSURE"


@dataclass
class QualityReport:
    passed: bool
    error: QualityError | None = None
    detail: str = ""
    blur_score: float = 0.0
    face_px: int = 0
    face_count: int = 0
    # ---- graded metrics (populated whenever a primary face is available,
    # even when `passed=False` for a non-face reason like blur/exposure) ----
    det_score: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    brightness: float = 0.0
    bands: dict[str, str] = field(default_factory=dict)  # metric -> GOOD|ACCEPTABLE|POOR
    overall_quality: float = 0.0  # 0..1, informational only — never a gate


def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance — proxy for sharpness. Low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_check(gray: np.ndarray) -> tuple[bool, bool]:
    """(too_dark, too_bright) based on mean luminance."""
    mean = float(gray.mean())
    return mean < settings.quality_min_brightness, mean > settings.quality_max_brightness


def estimate_pose(landmarks: np.ndarray | None) -> tuple[float, float]:
    """Rough (yaw, roll) in degrees from 5-point landmarks.

    Not a real 3D pose solve — just enough geometry to flag a badly turned or
    tilted face for the quality band. Order is left_eye, right_eye, nose,
    mouth_left, mouth_right, which both backends' detectors emit.
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0
    le, re, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_dx, eye_dy = float(re[0] - le[0]), float(re[1] - le[1])
    roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

    eye_mid_x = (le[0] + re[0]) / 2.0
    eye_span = float(np.linalg.norm([eye_dx, eye_dy])) or 1.0
    # Nose offset from the eye midline, normalised by inter-eye distance —
    # near 0 when facing the camera, growing as the head turns.
    yaw = float((nose[0] - eye_mid_x) / eye_span) * 90.0
    return yaw, roll


def _band(value: float, good: float, acceptable: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= good:
            return "GOOD"
        if value >= acceptable:
            return "ACCEPTABLE"
        return "POOR"
    if value <= good:
        return "GOOD"
    if value <= acceptable:
        return "ACCEPTABLE"
    return "POOR"


def _score_metrics(gray: np.ndarray, face, blur: float) -> tuple[dict[str, str], float, float, float]:
    """Compute per-metric bands + a 0..1 overall score for the primary face.

    `face` is a `DetectedFace` (see `face/detector.py`) or `None`.
    Returns (bands, overall_quality, det_score, brightness).
    """
    brightness = float(gray.mean())
    x1, y1, x2, y2 = face.bbox if face is not None else (0, 0, 0, 0)
    face_px = min(x2 - x1, y2 - y1) if face is not None else 0
    det_score = float(face.det_score) if face is not None else 0.0
    yaw, roll = estimate_pose(face.landmarks if face is not None else None)
    pose_deg = max(abs(yaw), abs(roll))

    bands = {
        "detection": "PASS" if face is not None else "FAIL",
        "resolution": _band(face_px, settings.quality_good_face_px, settings.min_face_px),
        "blur": _band(blur, settings.quality_good_blur, settings.quality_blur_threshold),
        "exposure": _band(
            min(
                brightness - settings.quality_min_brightness,
                settings.quality_max_brightness - brightness,
            ),
            settings.quality_brightness_margin,
            0.0,
        ),
        "pose": _band(pose_deg, settings.quality_good_pose_deg, settings.quality_acceptable_pose_deg,
                       higher_is_better=False),
        "landmarks": _band(det_score, settings.quality_good_det_score, settings.quality_acceptable_det_score),
    }

    # Normalised 0..1 sub-scores feeding the overall figure — deliberately
    # simple linear scaling, not learned weights (see config.py comment).
    res_norm = min(1.0, face_px / max(settings.quality_good_face_px, 1))
    blur_norm = min(1.0, blur / max(settings.quality_good_blur, 1e-6))
    exposure_norm = min(1.0, max(0.0, min(
        brightness - settings.quality_min_brightness,
        settings.quality_max_brightness - brightness,
    )) / max(settings.quality_brightness_margin, 1e-6))
    pose_norm = max(0.0, 1.0 - pose_deg / max(settings.quality_acceptable_pose_deg, 1e-6))
    det_norm = min(1.0, det_score / max(settings.quality_good_det_score, 1e-6))

    overall = (res_norm + blur_norm + exposure_norm + pose_norm + det_norm) / 5.0 if face is not None else 0.0
    return bands, float(np.clip(overall, 0.0, 1.0)), det_score, brightness


def score_face_quality(image_bgr: np.ndarray, face) -> tuple[dict[str, str], float]:
    """Graded quality bands + overall score for one face, no pass/fail verdict.

    `gate()` exists to accept or reject the *query* image outright. A
    candidate's matched face is never accepted or rejected the same way — it
    is scored, so a poor-quality candidate photo can still count as a weak
    corroborating hit rather than being silently thrown out or, worse, trusted
    as much as a sharp, frontal, well-lit one. Same metrics, different use.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_variance(gray)
    bands, overall, _det_score, _brightness = _score_metrics(gray, face, blur)
    return bands, overall


def gate(
    image_bgr: np.ndarray,
    faces,  # list[DetectedFace]
) -> QualityReport:
    """Run all quality checks and return the first failure or a PASS report."""
    h, w = image_bgr.shape[:2]
    if max(h, w) > settings.max_image_edge:
        return QualityReport(
            passed=False,
            error=QualityError.IMAGE_TOO_LARGE,
            detail=f"{w}x{h} exceeds max edge {settings.max_image_edge}",
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    dark, bright = exposure_check(gray)
    if dark:
        return QualityReport(
            passed=False,
            error=QualityError.LOW_EXPOSURE,
            detail=f"mean luminance {gray.mean():.1f} < {settings.quality_min_brightness}",
        )
    if bright:
        return QualityReport(
            passed=False,
            error=QualityError.HIGH_EXPOSURE,
            detail=f"mean luminance {gray.mean():.1f} > {settings.quality_max_brightness}",
        )

    blur = laplacian_variance(gray)
    if blur < settings.quality_blur_threshold:
        return QualityReport(
            passed=False,
            error=QualityError.BLURRY,
            detail=f"Laplacian variance {blur:.1f} < threshold {settings.quality_blur_threshold}",
            blur_score=blur,
        )

    if not faces:
        return QualityReport(passed=False, error=QualityError.NO_FACE, detail="no face detected",
                             blur_score=blur, face_count=0)

    n = len(faces)
    policy = settings.multi_face_policy
    # Computed once faces exist so every downstream report — including the
    # MULTI_FACE/FACE_TOO_SMALL rejections — still carries the graded metrics
    # for the primary face, per the "explain why, not just pass/fail" goal.
    primary = max(faces, key=lambda f: f.area)
    bands, overall, det_score, brightness = _score_metrics(gray, primary, blur)
    yaw, roll = estimate_pose(primary.landmarks)

    if n > 1 and policy == "reject":
        return QualityReport(
            passed=False,
            error=QualityError.MULTI_FACE,
            detail=f"{n} faces detected; multi_face_policy=reject",
            blur_score=blur,
            face_count=n,
            det_score=det_score,
            yaw_deg=yaw,
            roll_deg=roll,
            brightness=brightness,
            bands=bands,
            overall_quality=overall,
        )

    x1, y1, x2, y2 = primary.bbox
    face_px = min(x2 - x1, y2 - y1)
    if face_px < settings.min_face_px:
        return QualityReport(
            passed=False,
            error=QualityError.FACE_TOO_SMALL,
            detail=f"smallest face dimension {face_px}px < min {settings.min_face_px}px",
            blur_score=blur,
            face_count=n,
            face_px=face_px,
            det_score=det_score,
            yaw_deg=yaw,
            roll_deg=roll,
            brightness=brightness,
            bands=bands,
            overall_quality=overall,
        )

    return QualityReport(
        passed=True,
        blur_score=blur,
        face_count=n,
        face_px=face_px,
        det_score=det_score,
        yaw_deg=yaw,
        roll_deg=roll,
        brightness=brightness,
        bands=bands,
        overall_quality=overall,
    )
