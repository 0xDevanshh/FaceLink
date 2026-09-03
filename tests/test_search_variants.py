"""Search-variant generation: budgets, dedup, and content-based skipping.

Offline and synthetic — the property under test is the *policy* (how many
variants per depth, when a crop is skipped as redundant), not face detection
itself (covered against real photos in `tests/test_face.py`).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from facechain.face.detector import DetectedFace
from facechain.search.variants import (
    VARIANT_BUDGETS,
    cleanup_variants,
    generate_variants,
)


def textured_canvas(w: int = 800, h: int = 600) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def face(x1: int, y1: int, x2: int, y2: int) -> DetectedFace:
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    return DetectedFace(bbox=(x1, y1, x2, y2), det_score=0.9, embedding=emb)


def write_png(img: np.ndarray, tmp_path) -> str:
    path = str(tmp_path / "original.png")
    cv2.imwrite(path, img)
    return path


def test_fast_depth_never_generates_extra_variants(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "fast")
    assert len(variants) == 1
    assert variants[0].variant_type == "original"
    assert variants[0].image_path == path


def test_standard_depth_adds_one_variant_when_a_face_is_known(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "standard")
    assert 1 <= len(variants) <= VARIANT_BUDGETS["standard"]
    assert variants[0].variant_type == "original"


def test_deep_depth_allows_up_to_the_deep_budget(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "deep")
    assert len(variants) <= VARIANT_BUDGETS["deep"]


def test_no_face_means_only_the_original_regardless_of_depth(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, None, "deep")
    assert len(variants) == 1


def test_a_face_that_already_fills_the_frame_does_not_earn_a_redundant_crop(tmp_path):
    """A tight/loose crop of a face that is nearly the whole image is close to
    pixel-identical to the original — budget should not be spent on it."""
    img = textured_canvas(300, 300)
    path = write_png(img, tmp_path)
    # The face covers almost the entire canvas.
    variants = generate_variants(img, path, face(5, 5, 295, 295), "deep")
    assert len(variants) == 1


def test_variant_ids_are_unique_and_types_are_named(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "deep")
    ids = [v.variant_id for v in variants]
    assert len(ids) == len(set(ids))
    assert variants[0].variant_type == "original"
    for v in variants[1:]:
        assert v.variant_type in ("tight_crop", "loose_crop")


def test_each_variant_records_its_own_sha256_and_dimensions(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "deep")
    for v in variants:
        assert v.sha256 and len(v.sha256) == 64
        assert v.width > 0 and v.height > 0


def test_cleanup_removes_generated_files_but_keeps_the_original(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, face(50, 50, 250, 250), "deep")
    generated_paths = [v.image_path for v in variants if v.image_path != path]
    assert generated_paths, "test setup expects at least one generated crop file"
    for p in generated_paths:
        assert os.path.exists(p)

    cleanup_variants(variants, keep_path=path)

    assert os.path.exists(path)  # the caller-owned original survives
    for p in generated_paths:
        assert not os.path.exists(p)


def test_cleanup_is_a_no_op_for_a_variant_list_with_only_the_original(tmp_path):
    img = textured_canvas()
    path = write_png(img, tmp_path)
    variants = generate_variants(img, path, None, "fast")
    cleanup_variants(variants, keep_path=path)
    assert os.path.exists(path)
