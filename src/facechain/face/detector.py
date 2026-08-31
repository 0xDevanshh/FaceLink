"""Face detection + embedding backends.

Primary backend is InsightFace (SCRFD detector + ArcFace w600k recogniser,
the `buffalo_l` bundle). A lightweight OpenCV backend (YuNet + SFace, ~11 MB
of ONNX) is kept as a fallback so the pipeline still runs on machines where
the InsightFace model zoo cannot be fetched or onnxruntime is unavailable.

Both backends expose the same surface, and both return L2-normalised
embeddings so cosine similarity is just a dot product.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT, settings

log = logging.getLogger(__name__)

MODEL_CACHE = Path(os.environ.get("FACECHAIN_MODEL_DIR", REPO_ROOT / ".models"))

OPENCV_MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray  # L2-normalised
    landmarks: np.ndarray | None = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def _download(name: str, url: str) -> Path:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    dest = MODEL_CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log.info("downloading model %s", name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": settings.user_agent})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest


def _l2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class FaceBackend:
    name = "abstract"
    model_name = "abstract"

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:  # pragma: no cover
        raise NotImplementedError


class InsightFaceBackend(FaceBackend):
    """SCRFD detection + ArcFace (glint360k/w600k) 512-D embeddings."""

    name = "insightface"

    def __init__(self) -> None:
        # INSIGHTFACE_HOME is read at import time, so it must be set first.
        os.environ.setdefault("INSIGHTFACE_HOME", str(MODEL_CACHE / "insightface"))
        from insightface.app import FaceAnalysis  # imported lazily: heavy

        self.model_name = f"{settings.insightface_model}/SCRFD+ArcFace"
        self._app = FaceAnalysis(
            name=settings.insightface_model,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        det = settings.face_det_size
        self._app.prepare(ctx_id=-1, det_size=(det, det), det_thresh=settings.face_det_threshold)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self._app.get(image_bgr)
        out: list[DetectedFace] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = _l2(f.embedding)
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            out.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                    embedding=_l2(emb),
                    landmarks=getattr(f, "kps", None),
                )
            )
        return out


class OpenCvBackend(FaceBackend):
    """YuNet detector + SFace recogniser (128-D). Fallback backend."""

    name = "opencv"
    model_name = "YuNet+SFace"

    def __init__(self) -> None:
        det_path = _download(*OPENCV_MODELS["yunet"])
        rec_path = _download(*OPENCV_MODELS["sface"])
        self._det = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), settings.face_det_threshold, 0.3, 5000
        )
        self._rec = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        h, w = image_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, raw = self._det.detect(image_bgr)
        if raw is None:
            return []
        out: list[DetectedFace] = []
        for row in raw:
            x, y, bw, bh = (int(v) for v in row[:4])
            # alignCrop wants the raw detection row (bbox + 5 landmarks).
            aligned = self._rec.alignCrop(image_bgr, row)
            emb = self._rec.feature(aligned)
            out.append(
                DetectedFace(
                    bbox=(x, y, x + bw, y + bh),
                    det_score=float(row[-1]),
                    embedding=_l2(emb),
                    landmarks=np.array(row[4:14], dtype=np.float32).reshape(5, 2),
                )
            )
        return out


_CACHED: FaceBackend | None = None


def load_backend(force: str | None = None) -> FaceBackend:
    """Resolve and cache the face backend according to config/`force`."""
    global _CACHED
    choice = force or settings.face_backend
    if _CACHED is not None and (choice in ("auto", _CACHED.name)):
        return _CACHED

    errors: dict[str, str] = {}
    order = {
        "auto": ["insightface", "opencv"],
        "insightface": ["insightface"],
        "opencv": ["opencv"],
    }[choice]

    for name in order:
        try:
            _CACHED = InsightFaceBackend() if name == "insightface" else OpenCvBackend()
            log.info("face backend: %s (%s)", _CACHED.name, _CACHED.model_name)
            return _CACHED
        except Exception as exc:  # noqa: BLE001 - we genuinely want to fall through
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("face backend %s unavailable: %s", name, errors[name])

    raise RuntimeError(f"no face backend available: {errors}")


def select_primary(faces: list[DetectedFace]) -> DetectedFace | None:
    """Largest face wins — the subject of a photo, not a bystander."""
    return max(faces, key=lambda f: (f.area, f.det_score)) if faces else None
