"""Face quality gating — reject garbage before embedding.

Every check is a separate, named failure reason so the evidence bundle records
exactly which gate a rejected image failed. Thresholds are all in config so
operators can tune them for their hardware and use-case without touching code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance — proxy for sharpness. Low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_check(gray: np.ndarray) -> tuple[bool, bool]:
    """(too_dark, too_bright) based on mean luminance."""
    mean = float(gray.mean())
    return mean < settings.quality_min_brightness, mean > settings.quality_max_brightness


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

    if n > 1 and policy == "reject":
        return QualityReport(
            passed=False,
            error=QualityError.MULTI_FACE,
            detail=f"{n} faces detected; multi_face_policy=reject",
            blur_score=blur,
            face_count=n,
        )

    # Check primary (largest) face size.
    primary = max(faces, key=lambda f: f.area)
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
        )

    return QualityReport(
        passed=True,
        blur_score=blur,
        face_count=n,
        face_px=face_px,
    )
