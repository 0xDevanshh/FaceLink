"""FastAPI server tests — no real pipeline run, all mocked.

Tests: upload validation, magic-byte check, CORS headers, health endpoint,
path-traversal protection on case IDs, evidence verify endpoint.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def client():
    from server import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _jpeg_bytes(size: int = 100) -> bytes:
    """Minimal valid JPEG magic bytes followed by padding."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * max(size, 100)


def _png_bytes() -> bytes:
    """Real 1x1 PNG so magic check passes."""
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (4, 4), (128, 64, 32)).save(buf, "PNG")
    return buf.getvalue()


# ---- health endpoint -----------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "engines_configured" in body
    # Secrets must not appear in health response.
    raw = json.dumps(body)
    assert "PRIVATE_KEY" not in raw
    assert "serpapi_key" not in raw


# ---- upload validation ---------------------------------------------------

def test_upload_rejects_too_large(client):
    # 11 MB > 10 MB limit.
    big = _jpeg_bytes(11 * 1024 * 1024)
    r = client.post("/api/v1/scan", files={"image": ("x.jpg", big, "image/jpeg")},
                    data={"no_chain": "true"})
    assert r.status_code == 413


def test_upload_rejects_wrong_mime(client):
    r = client.post("/api/v1/scan",
                    files={"image": ("x.txt", b"hello world", "text/plain")},
                    data={"no_chain": "true"})
    assert r.status_code == 415


def test_upload_rejects_bad_magic(client):
    # JPEG MIME but ZIP magic bytes.
    r = client.post("/api/v1/scan",
                    files={"image": ("x.jpg", b"PK\x03\x04" + b"\x00" * 100, "image/jpeg")},
                    data={"no_chain": "true"})
    assert r.status_code == 422


def test_upload_accepts_valid_jpeg(client):
    """Valid JPEG magic bytes should pass validation and reach the pipeline."""
    with patch("server._run_pipeline_background"):
        r = client.post("/api/v1/scan",
                        files={"image": ("face.jpg", _jpeg_bytes(500), "image/jpeg")},
                        data={"no_chain": "true"})
    # Should be accepted (202 or 200) — pipeline runs async.
    assert r.status_code == 200
    body = r.json()
    assert "case_id" in body
    assert "events_url" in body


def test_upload_accepts_valid_png(client):
    with patch("server._run_pipeline_background"):
        r = client.post("/api/v1/scan",
                        files={"image": ("face.png", _png_bytes(), "image/png")},
                        data={"no_chain": "true"})
    assert r.status_code == 200


# ---- path traversal protection -------------------------------------------

def test_status_rejects_traversal_case_id(client):
    r = client.get("/api/v1/scan/../../etc/passwd/status")
    assert r.status_code in (400, 404, 422)


def test_result_rejects_traversal_case_id(client):
    r = client.get("/api/v1/scan/../../../secret/result")
    assert r.status_code in (400, 404, 422)


def test_events_rejects_traversal_case_id(client):
    r = client.get("/api/v1/scan/../secret/events")
    assert r.status_code in (400, 404, 422)


# ---- unknown case 404 ----------------------------------------------------

def test_status_unknown_case_returns_404(client):
    r = client.get("/api/v1/scan/case_99991231_999999/status")
    assert r.status_code == 404


def test_result_unknown_case_returns_404(client):
    r = client.get("/api/v1/scan/case_99991231_999999/result")
    assert r.status_code == 404


# ---- CORS ----------------------------------------------------------------

def test_cors_allowed_origin(client):
    r = client.options("/api/v1/health",
                       headers={"Origin": "http://localhost:5173",
                                "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_wildcard_is_not_set(client):
    """CORS must never be wildcard — our config uses an explicit allowlist."""
    r = client.get("/api/v1/health", headers={"Origin": "http://evil.example.com"})
    # Either the header is absent or it's not *.
    acao = r.headers.get("access-control-allow-origin", "")
    assert acao != "*"


# ---- evidence verify endpoint -------------------------------------------

def _make_evidence_zip(tmp_path: Path) -> bytes:
    """Build a minimal valid evidence ZIP for the /verify endpoint."""
    from facechain.evidence.hashing import sha256_file, sha256_canonical, sha256_text
    from facechain.evidence.writer import EvidenceWriter, new_case_id
    from facechain.models import Case, FaceRecord, InputImage, Stage, VerifiedCandidate

    img = tmp_path / "input.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"x" * 500)
    h = sha256_file(img)

    case = Case(
        case_id=new_case_id(),
        created_at="2026-09-01T00:00:00+00:00",
        observed_at=1788202747,
        input=InputImage(path=str(img), filename="input.jpg", bytes_len=img.stat().st_size,
                         width=640, height=480, sha256=h, phash="aabbccdd11223344"),
        face=FaceRecord(detected=True, backend="insightface",
                        model="buffalo_l/SCRFD+ArcFace", faces_found=1,
                        bbox=[10, 20, 110, 140], det_score=0.9,
                        embedding_dimension=512, embedding_sha256="bb" * 32),
        best_match=VerifiedCandidate(
            engine="yandex", url="https://instagram.com/p/ABC/", domain="instagram.com",
            platform="Instagram", is_social=True, fetched=True,
            candidate_image_sha256="dd" * 32, candidate_image_phash="fafc13e1a083a6d8",
            candidate_image_url="https://cdn.example/a.jpg", candidate_image_source="og:image",
            image_similarity=0.95, face_similarity=0.97, metadata_consistency=0.7,
            match_type="exact-image", final_score=0.87, verified=True,
            stages=[Stage.SEARCH_FOUND, Stage.SOCIAL_MATCH, Stage.IMAGE_MATCH,
                    Stage.FACE_MATCH, Stage.VERIFIED],
        ),
        verdict="VERIFIED",
    )
    case.verification = [case.best_match]

    writer = EvidenceWriter(case.case_id, root=tmp_path / "evidence")
    payload = writer.build_payload(case)
    _, ev_hash = writer.write_payload(payload)
    case.evidence_sha256 = ev_hash
    writer.write_bundle(case, img)
    bundle_dir = writer.dir

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fp in bundle_dir.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(bundle_dir.parent))
    return buf.getvalue()


def test_verify_endpoint_passes_on_valid_evidence(client, tmp_path):
    zdata = _make_evidence_zip(tmp_path)
    r = client.post("/api/v1/verify",
                    files={"evidence_zip": ("evidence.zip", zdata, "application/zip")})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] == "PASS"
    assert all(c["passed"] for c in body["checks"])


def test_verify_endpoint_fails_on_tampered_evidence(client, tmp_path):
    zdata = _make_evidence_zip(tmp_path)
    # Tamper with the payload inside the ZIP.
    buf = io.BytesIO(zdata)
    out = io.BytesIO()
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if "attested_payload.json" in item.filename:
                payload = json.loads(data)
                payload["matched_url"] = "https://instagram.com/p/TAMPERED/"
                data = json.dumps(payload).encode()
            zout.writestr(item, data)
    out.seek(0)
    r = client.post("/api/v1/verify",
                    files={"evidence_zip": ("evidence.zip", out.read(), "application/zip")})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] == "FAIL"


def test_verify_endpoint_rejects_zip_traversal(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../etc/passwd", "root:x:0:0:root:/root:/bin/bash")
    buf.seek(0)
    r = client.post("/api/v1/verify",
                    files={"evidence_zip": ("evil.zip", buf.read(), "application/zip")})
    assert r.status_code == 422
