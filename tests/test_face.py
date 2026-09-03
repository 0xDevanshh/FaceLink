"""Face + image-similarity tests.

The perceptual-hash tests are fast and synthetic. The face-model tests need the
InsightFace weights (~275 MB, downloaded on first use) and a real photo in
`samples/`, so they skip automatically when those are absent:

    python -m pytest tests/test_face.py                 # fast subset
    python -m pytest tests/test_face.py -m "" -q        # everything
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from facechain.face.similarity import best_cosine, best_match_index, cosine
from facechain.verification.image_similarity import compare, perceptual_hashes

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((REPO_ROOT / "samples").glob("*.jpg"))


# ---- cosine similarity ---------------------------------------------------

def test_cosine_of_identical_vectors_is_one():
    v = np.random.RandomState(0).randn(512).astype(np.float32)
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_of_opposite_vectors_is_minus_one():
    v = np.random.RandomState(1).randn(512).astype(np.float32)
    assert cosine(v, -v) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_is_scale_invariant():
    """Defensive re-normalisation: an unnormalised vector must not inflate."""
    a = np.random.RandomState(2).randn(512).astype(np.float32)
    b = np.random.RandomState(3).randn(512).astype(np.float32)
    assert cosine(a, b) == pytest.approx(cosine(a * 17.0, b * 0.03), abs=1e-5)


def test_cosine_handles_degenerate_input():
    assert cosine(np.zeros(512), np.ones(512)) == 0.0
    assert cosine(np.ones(4), np.ones(8)) == 0.0  # dimension mismatch
    assert cosine(np.array([]), np.array([])) == 0.0


def test_cosine_is_clamped():
    v = np.ones(512, dtype=np.float32)
    assert -1.0 <= cosine(v, v) <= 1.0


def test_best_cosine_picks_the_best_face_in_a_group_photo():
    ref = np.random.RandomState(4).randn(512).astype(np.float32)
    others = [np.random.RandomState(i).randn(512).astype(np.float32) for i in (5, 6)]
    assert best_cosine(ref, others + [ref]) == pytest.approx(1.0, abs=1e-5)


def test_best_cosine_of_no_faces_is_zero():
    assert best_cosine(np.ones(512), []) == 0.0


def test_best_match_index_reports_which_face_won():
    ref = np.random.RandomState(4).randn(512).astype(np.float32)
    others = [np.random.RandomState(i).randn(512).astype(np.float32) for i in (5, 6)]
    candidates = others + [ref]  # the matching face is index 2
    score, idx = best_match_index(ref, candidates)
    assert idx == 2
    assert score == pytest.approx(1.0, abs=1e-5)


def test_best_match_index_of_no_faces_is_minus_one():
    score, idx = best_match_index(np.ones(512), [])
    assert score == 0.0
    assert idx == -1


# ---- perceptual hashing --------------------------------------------------

def synth(seed: int, size=(256, 256)) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (*size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (9, 9), 0)  # low-freq structure for pHash


def test_identical_images_compare_as_one():
    img = synth(10)
    h = perceptual_hashes(img)
    assert compare(h, h) == pytest.approx(1.0)


def test_hashes_survive_recompression_and_resize():
    """A social repost is re-encoded and resized; pHash must survive that."""
    img = synth(11, (512, 512))
    small = cv2.resize(img, (170, 170))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
    assert ok
    assert compare(perceptual_hashes(img), perceptual_hashes(buf.tobytes())) > 0.85


def test_unrelated_images_score_low():
    assert compare(perceptual_hashes(synth(12)), perceptual_hashes(synth(99))) < 0.8


def test_hashes_are_stable_across_bytes_and_array_input():
    img = synth(13)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    assert perceptual_hashes(img)["phash"] == perceptual_hashes(buf.tobytes())["phash"]


def test_similarity_is_bounded():
    a, b = perceptual_hashes(synth(14)), perceptual_hashes(synth(15))
    assert 0.0 <= compare(a, b) <= 1.0


# ---- real model (slow; needs weights + a sample photo) -------------------

requires_sample = pytest.mark.skipif(not SAMPLES, reason="no samples/*.jpg — run scripts/fetch_sample.py")


@requires_sample
def test_detects_and_encodes_a_real_face():
    from facechain.face.encoder import encode_face, read_image

    record, embedding, faces = encode_face(read_image(SAMPLES[0]))
    assert record.detected and record.faces_found >= 1
    assert record.embedding_dimension in (128, 512)
    assert record.embedding_sha256 and len(record.embedding_sha256) == 64
    assert embedding is not None
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)
    x1, y1, x2, y2 = record.bbox
    assert x2 > x1 and y2 > y1


@requires_sample
def test_same_photo_recompressed_still_matches_the_same_face():
    """End-to-end threshold sanity: the demo's core claim must hold."""
    from facechain.config import settings
    from facechain.face.encoder import decode_image, encode_face, read_image

    img = read_image(SAMPLES[0])
    _, ref, _ = encode_face(img)

    degraded = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    ok, buf = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, 55])
    assert ok
    _, other, _ = encode_face(decode_image(buf.tobytes()))

    assert cosine(ref, other) > settings.face_match_threshold
    assert cosine(ref, other) > 0.8  # in practice ~0.96


@pytest.mark.skipif(len(SAMPLES) < 2, reason="needs two different people in samples/")
def test_different_people_score_far_below_threshold():
    from facechain.config import settings
    from facechain.face.encoder import encode_face, read_image

    _, a, _ = encode_face(read_image(SAMPLES[0]))
    _, b, _ = encode_face(read_image(SAMPLES[1]))
    assert cosine(a, b) < settings.face_match_threshold


@requires_sample
def test_no_face_in_a_blank_image():
    from facechain.face.encoder import encode_face

    record, embedding, faces = encode_face(np.full((640, 640, 3), 220, dtype=np.uint8))
    assert not record.detected and embedding is None and faces == []


@pytest.mark.skipif(len(SAMPLES) < 2, reason="needs two different people in samples/")
def test_candidate_verification_finds_the_matching_face_in_a_group_photo():
    """A candidate image showing two people must report *which* face matched,
    not just that a face matched somewhere in the frame."""
    from facechain.config import settings
    from facechain.face.detector import load_backend
    from facechain.face.encoder import encode_face, read_image

    img_a, img_b = read_image(SAMPLES[0]), read_image(SAMPLES[1])
    _, ref_embedding, _ = encode_face(img_a)

    # Build a synthetic "group photo": person B on the left, person A on the
    # right, so the correct match is not the first face the detector reports.
    h = 480
    a_resized = cv2.resize(img_a, (int(img_a.shape[1] * h / img_a.shape[0]), h))
    b_resized = cv2.resize(img_b, (int(img_b.shape[1] * h / img_b.shape[0]), h))
    group = np.hstack([b_resized, a_resized])
    ok, buf = cv2.imencode(".jpg", group, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    group_bytes = buf.tobytes()

    faces_in_group = load_backend().detect(group)
    assert len(faces_in_group) >= 2, "test setup needs both faces detected in the composite"

    # Exercise the same face-matching path `verify_candidate` uses internally
    # (see `verification/candidate.py`) without a network fetch.
    from facechain.face.encoder import decode_image
    from facechain.face.similarity import best_match_index

    decoded = decode_image(group_bytes)
    faces = load_backend().detect(decoded)
    sim, idx = best_match_index(ref_embedding, [f.embedding for f in faces])
    assert idx != -1
    assert sim > settings.face_match_threshold
    # The matched face's own bbox should sit in the right half of the frame,
    # where person A (the reference) was placed.
    x1, _, x2, _ = faces[idx].bbox
    assert (x1 + x2) / 2 > group.shape[1] / 2


@requires_sample
def test_verify_candidate_scores_the_matched_candidate_faces_quality(monkeypatch):
    """`VerifiedCandidate.candidate_face_quality` must reflect real graded
    quality (resolution/blur/exposure/pose/detection) of the *matched* face,
    not just a bare detector-confidence number — a sharp, well-lit sample
    photo should score meaningfully above zero with sensible bands."""
    import httpx

    from facechain.face.encoder import encode_face, read_image
    from facechain.models import SearchCandidate
    from facechain.verification import candidate as candmod

    img = read_image(SAMPLES[0])
    _, ref_embedding, _ = encode_face(img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    data = buf.tobytes()

    def fake_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=data)

    def patched_client() -> httpx.Client:
        return httpx.Client(follow_redirects=False, transport=httpx.MockTransport(fake_transport))

    monkeypatch.setattr(candmod, "_client", patched_client)
    monkeypatch.setattr(candmod, "safe_url_or_none", lambda u: u)

    cand = SearchCandidate(engine="test", url="https://example.com/photo.jpg", domain="example.com")
    vc = candmod.verify_candidate(cand, perceptual_hashes(data), ref_embedding)

    assert vc.face_detected
    assert vc.candidate_face_index != -1
    assert 0.0 < vc.candidate_face_quality <= 1.0
    assert vc.candidate_face_bands.get("detection") == "PASS"
