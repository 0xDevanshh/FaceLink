"""Choosing *which* face in an uploaded photo the scan is about.

A photo is not always a portrait. Group shots, crowd shots, badly framed
snapshots and images where the detector is unsure all need an operator to say
which face they mean, and the answer has to be recorded as evidence rather than
inferred later — otherwise nobody reading the bundle can tell whether the match
was about the subject or about a bystander.

Two rules hold this module together.

1. **Auto-select only when it is unambiguous.** One clearly dominant, usable
   face is selected without asking. Anything else — several faces, a marginal
   detection, a face too small to measure — returns a request for selection
   instead of quietly guessing.

2. **A crop never replaces the original.** Cropping produces an additional
   artefact with its own hash. The original bytes and the original hash stay in
   the bundle, alongside the crop rectangle, so the whole selection is
   reproducible from the evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings
from ..models import DetectedFaceInfo, FaceQuality
from .detector import DetectedFace, load_backend
from .quality import QualityError, gate

log = logging.getLogger(__name__)

# A second face this fraction of the primary's area or larger makes the choice
# ambiguous: two comparably sized faces in one frame is exactly the case where
# "largest wins" picks the wrong person.
AMBIGUITY_AREA_RATIO = 0.55
# Below this detector confidence we ask rather than assume, even for a lone face.
AUTO_SELECT_MIN_DET_SCORE = 0.60


class CropError(ValueError):
    """A crop rectangle that cannot be applied to the image it was given for."""


@dataclass
class FaceOffer:
    """What the API hands the UI so a human can choose."""

    faces: list[DetectedFaceInfo]
    auto_index: int | None          # index to use without asking, if any
    selection_required: bool
    reason: str = ""                # why selection is required, in plain language


def describe_faces(faces: list[DetectedFace], image_bgr: np.ndarray) -> list[DetectedFaceInfo]:
    """Summarise detections for the UI — geometry and quality, never embeddings."""
    h, w = image_bgr.shape[:2]
    out: list[DetectedFaceInfo] = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_px = min(x2 - x1, y2 - y1)
        too_small = face_px < settings.min_face_px
        low_conf = f.det_score < settings.face_det_threshold
        notes = []
        if too_small:
            notes.append(f"{face_px}px is below the {settings.min_face_px}px minimum")
        if low_conf:
            notes.append(f"detector confidence {f.det_score:.2f} is low")
        out.append(
            DetectedFaceInfo(
                index=i,
                bbox=[x1, y1, x2, y2],
                det_score=round(float(f.det_score), 4),
                face_px=face_px,
                area=int((x2 - x1) * (y2 - y1)),
                usable=not (too_small or low_conf),
                note="; ".join(notes),
            )
        )
    return out


def offer(image_bgr: np.ndarray, faces: list[DetectedFace]) -> FaceOffer:
    """Decide whether we can pick a face ourselves, or must ask.

    Erring towards asking is intentional: a wrong automatic pick produces a
    confidently wrong scan of the wrong person, while an unnecessary prompt
    costs one click.
    """
    infos = describe_faces(faces, image_bgr)
    if not infos:
        return FaceOffer(faces=[], auto_index=None, selection_required=False,
                         reason="no usable face detected")

    ranked = sorted(infos, key=lambda f: (f.area, f.det_score), reverse=True)
    primary = ranked[0]

    if not primary.usable:
        return FaceOffer(
            faces=infos, auto_index=None, selection_required=True,
            reason=(f"the largest detected face is not usable ({primary.note}). "
                    "Select a face or draw a tighter crop around one."),
        )

    if len(ranked) > 1:
        runner_up = ranked[1]
        if primary.area > 0 and runner_up.area / primary.area >= AMBIGUITY_AREA_RATIO:
            return FaceOffer(
                faces=infos, auto_index=None, selection_required=True,
                reason=(f"{len(infos)} faces of comparable size were detected — "
                        "choose which one the scan is about."),
            )

    if primary.det_score < AUTO_SELECT_MIN_DET_SCORE:
        return FaceOffer(
            faces=infos, auto_index=None, selection_required=True,
            reason=(f"detection is uncertain (confidence {primary.det_score:.2f}) — "
                    "confirm the face or draw a crop."),
        )

    if len(infos) > 1 and settings.multi_face_policy == "reject":
        return FaceOffer(
            faces=infos, auto_index=None, selection_required=True,
            reason=f"{len(infos)} faces detected and multi_face_policy=reject — choose one.",
        )

    return FaceOffer(faces=infos, auto_index=primary.index, selection_required=False)


def normalise_crop(rect: list[int] | tuple[int, ...], image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Validate an operator-supplied `[x, y, w, h]` against the image.

    Rejects rather than silently clamps a rectangle that is degenerate or
    entirely outside the image: those indicate the client and the server
    disagree about the image's coordinate space, and quietly cropping
    *something* would attach a misleading rectangle to the evidence.
    """
    if len(rect) != 4:
        raise CropError("crop must be [x, y, width, height]")
    try:
        x, y, cw, ch = (int(v) for v in rect)
    except (TypeError, ValueError) as exc:
        raise CropError("crop values must be integers") from exc

    h, w = image_bgr.shape[:2]
    if cw <= 0 or ch <= 0:
        raise CropError(f"crop has non-positive size ({cw}x{ch})")
    if x >= w or y >= h or x + cw <= 0 or y + ch <= 0:
        raise CropError(f"crop {x},{y} {cw}x{ch} lies entirely outside the {w}x{h} image")

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + cw), min(h, y + ch)
    if x1 - x0 < settings.min_face_px or y1 - y0 < settings.min_face_px:
        raise CropError(
            f"crop is {x1 - x0}x{y1 - y0} after clamping to the image; "
            f"needs at least {settings.min_face_px}px on each side"
        )
    return x0, y0, x1 - x0, y1 - y0


def apply_crop(image_bgr: np.ndarray, rect: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop and return `(cropped_image, applied_rect)`."""
    x, y, cw, ch = normalise_crop(rect, image_bgr)
    return image_bgr[y:y + ch, x:x + cw].copy(), (x, y, cw, ch)


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise CropError("could not encode the selected region as PNG")
    return buf.tobytes()


def gate_selected(image_bgr: np.ndarray, faces: list[DetectedFace], face_index: int | None) -> FaceQuality:
    """Run the quality gate against the *selected* face, not the largest one.

    The gate reports on whichever face the scan will actually use. Gating the
    largest face while embedding a different one would let a scan run on a face
    nothing ever checked.
    """
    subject = faces
    if face_index is not None and 0 <= face_index < len(faces):
        subject = [faces[face_index]]
    report = gate(image_bgr, subject)
    return FaceQuality(
        passed=report.passed,
        error=report.error.value if report.error else None,
        detail=report.detail,
        blur_score=report.blur_score,
        face_px=report.face_px,
        face_count=len(faces),
        det_score=report.det_score,
        yaw_deg=report.yaw_deg,
        roll_deg=report.roll_deg,
        brightness=report.brightness,
        bands=report.bands,
        overall_quality=report.overall_quality,
    )


def detect_faces(image_bgr: np.ndarray, backend_name: str | None = None) -> list[DetectedFace]:
    return load_backend(backend_name).detect(image_bgr)


__all__ = [
    "AMBIGUITY_AREA_RATIO",
    "AUTO_SELECT_MIN_DET_SCORE",
    "CropError",
    "FaceOffer",
    "QualityError",
    "apply_crop",
    "describe_faces",
    "detect_faces",
    "encode_png",
    "gate_selected",
    "normalise_crop",
    "offer",
]
