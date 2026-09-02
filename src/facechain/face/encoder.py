"""Turn an image into (face record, embedding) — the pipeline's face stage."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..evidence.hashing import embedding_hash
from ..models import FaceRecord
from .detector import DetectedFace, load_backend, select_primary

log = logging.getLogger(__name__)

MAX_EDGE = 2000  # downscale huge photos; detection quality is unaffected


class NoFaceError(RuntimeError):
    pass


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, EXIF-oriented, downscaled to a sane maximum."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not a decodable image: {path}")
    return _downscale(img)


def decode_image(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return _downscale(img) if img is not None else None


def _downscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    edge = max(h, w)
    if edge <= MAX_EDGE:
        return img
    scale = MAX_EDGE / edge
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_face(image_bgr: np.ndarray, backend_name: str | None = None):
    """Detect faces and embed the primary one.

    Returns `(FaceRecord, embedding | None, all_faces)`.
    """
    backend = load_backend(backend_name)
    faces = backend.detect(image_bgr)
    primary = select_primary(faces)

    if primary is None:
        return (
            FaceRecord(
                detected=False,
                backend=backend.name,
                model=backend.model_name,
                faces_found=0,
            ),
            None,
            faces,
        )

    record = FaceRecord(
        detected=True,
        backend=backend.name,
        model=backend.model_name,
        faces_found=len(faces),
        bbox=[int(v) for v in primary.bbox],
        det_score=round(float(primary.det_score), 4),
        embedding_dimension=int(primary.embedding.size),
        embedding_sha256=embedding_hash(primary.embedding),
    )
    return record, primary.embedding, faces


def encode_detected(
    backend, faces: list[DetectedFace], face_index: int | None = None
) -> tuple[FaceRecord, np.ndarray | None, DetectedFace | None]:
    """Build the `FaceRecord` for an already-detected face set.

    Split out from `encode_face` so a caller that has run detection once — to
    offer the faces to an operator — does not have to run it again to embed the
    one that was chosen. Detection is the expensive half, and running it twice
    could in principle return a different face set for the same image.

    `face_index` selects a specific detection; `None` keeps the historical
    "largest face wins" behaviour. An out-of-range index is an error rather
    than a silent fallback: embedding a different face from the one the operator
    picked would make the evidence describe the wrong person.
    """
    if face_index is None:
        primary = select_primary(faces)
    else:
        if not 0 <= face_index < len(faces):
            raise IndexError(f"face_index {face_index} out of range (0..{len(faces) - 1})")
        primary = faces[face_index]

    if primary is None:
        return (
            FaceRecord(detected=False, backend=backend.name, model=backend.model_name,
                       faces_found=0),
            None,
            None,
        )

    record = FaceRecord(
        detected=True,
        backend=backend.name,
        model=backend.model_name,
        faces_found=len(faces),
        bbox=[int(v) for v in primary.bbox],
        det_score=round(float(primary.det_score), 4),
        embedding_dimension=int(primary.embedding.size),
        embedding_sha256=embedding_hash(primary.embedding),
    )
    return record, primary.embedding, primary


def crop_face(image_bgr: np.ndarray, face: DetectedFace, margin: float = 0.25) -> np.ndarray:
    """Crop with margin — used to save a visual artefact into the evidence bundle."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return image_bgr[max(0, y1 - my) : min(h, y2 + my), max(0, x1 - mx) : min(w, x2 + mx)]
