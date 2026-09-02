"""The face-selection endpoints, chain status, and SSE termination.

Two guarantees are pinned here.

  * **The stream always ends.** Success, no-match, insufficient evidence or an
    unrecoverable error — all four are fine, and all four must arrive as a
    terminal event. A client left on "Scanning…" forever is a product failure,
    not a slow scan.

  * **No secrets leak.** Every response a browser can see is checked for the
    attester key and the API keys, including the error paths.

The pipeline itself is replaced with a stub: these tests are about the transport
and the job lifecycle, and the real pipeline is exercised end to end separately.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def client():
    from server import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def photo_bytes(size=(400, 500)) -> bytes:
    """A real decodable JPEG (content, not just magic bytes)."""
    import numpy as np
    rng = np.random.default_rng(3)
    arr = rng.integers(60, 200, (*size[::-1], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


SECRETS = ("PRIVATE_KEY", "private_key", "serpapi_key", "SERPAPI_KEY", "mnemonic")


def assert_no_secrets(text: str) -> None:
    from facechain.config import settings

    for name in SECRETS:
        assert name not in text, f"{name} appeared in a client-visible response"
    for value in (settings.private_key, settings.serpapi_key):
        if value:
            assert value not in text
            assert value.lstrip("0x") not in text


# ---- POST /api/v1/faces -------------------------------------------------

def test_faces_endpoint_validates_the_upload_like_scan_does(client):
    r = client.post("/api/v1/faces", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415

    r = client.post("/api/v1/faces",
                    files={"image": ("x.jpg", b"PK\x03\x04" + b"\x00" * 200, "image/jpeg")})
    assert r.status_code == 422


def test_faces_endpoint_rejects_an_oversized_upload(client):
    big = bytearray(b"\xff\xd8\xff\xe0") + bytearray(11 * 1024 * 1024)
    r = client.post("/api/v1/faces", files={"image": ("big.jpg", bytes(big), "image/jpeg")})
    assert r.status_code == 413


def test_faces_endpoint_returns_geometry_and_a_reusable_upload_id(client):
    r = client.post("/api/v1/faces",
                    files={"image": ("noise.jpg", photo_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["upload_id"].startswith("upl_")
    assert len(body["sha256"]) == 64
    # Coordinates are meaningless without the space they live in.
    assert body["image_width"] > 0 and body["image_height"] > 0
    assert isinstance(body["faces"], list)
    assert "selection_required" in body and "auto_index" in body
    assert "quality" in body
    assert_no_secrets(r.text)


def test_faces_endpoint_never_returns_an_embedding(client):
    """The UI needs boxes and quality. The 512-D vector stays server-side."""
    r = client.post("/api/v1/faces",
                    files={"image": ("noise.jpg", photo_bytes(), "image/jpeg")})
    raw = r.text
    assert "embedding" not in raw


def test_a_noise_image_reports_no_faces_rather_than_inventing_one(client):
    r = client.post("/api/v1/faces",
                    files={"image": ("noise.jpg", photo_bytes(), "image/jpeg")})
    body = r.json()
    assert body["faces"] == []
    assert body["auto_index"] is None


# ---- scan: face selection parameters -----------------------------------

def test_scan_requires_either_a_file_or_an_upload_id(client):
    r = client.post("/api/v1/scan", data={"no_chain": "true"})
    assert r.status_code == 422
    assert "upload_id" in r.text


def test_scan_reports_a_wellformed_but_expired_upload_id_as_missing(client):
    """A handle that could have existed but does not: 404, with advice."""
    r = client.post("/api/v1/scan",
                    data={"no_chain": "true", "upload_id": "upl_" + "ab" * 8})
    assert r.status_code == 404
    assert "upload the image again" in r.text


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "upl_../../etc", "upl_zzzz", "case_20260101_000000",
     "upl_", "upl_ab/cd", "", "upl_" + "a" * 200],
)
def test_scan_rejects_a_malformed_upload_id(client, bad):
    """Upload ids are interpolated into a filename, so anything that is not the
    server's own `upl_<hex>` shape is refused before it reaches the filesystem."""
    r = client.post("/api/v1/scan", data={"no_chain": "true", "upload_id": bad})
    # An empty value means "no upload_id given", which is the missing-image case.
    assert r.status_code in (400, 422), r.text


@pytest.mark.parametrize("crop", ["1,2,3", "a,b,c,d", "1,2,3,4,5", "not-a-crop"])
def test_scan_rejects_a_malformed_crop(client, crop):
    r = client.post("/api/v1/scan",
                    files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                    data={"no_chain": "true", "crop": crop})
    assert r.status_code == 422


@pytest.mark.parametrize("index", ["-1", "abc", "1.5.2"])
def test_scan_rejects_a_malformed_face_index(client, index):
    r = client.post("/api/v1/scan",
                    files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                    data={"no_chain": "true", "face_index": index})
    assert r.status_code == 422


def test_scan_passes_the_selection_through_to_the_pipeline(client):
    """The crop and face index the operator chose must reach RunOptions intact —
    a dropped crop would silently scan the whole photo instead."""
    from facechain.models import Case

    captured = {}

    def fake_run(opts, reporter=None):
        captured["opts"] = opts
        if reporter:
            reporter("done", "ok", "stubbed")
        return Case(case_id=opts.case_id or "c", created_at="now", observed_at=1,
                    verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true", "face_index": "2",
                              "crop": "10,20,300,400", "selection_mode": "manual-crop"})
        assert r.status_code == 200
        client.get(f"/api/v1/scan/{r.json()['case_id']}/result")

    opts = captured["opts"]
    assert opts.face_index == 2
    assert opts.crop_rect == [10, 20, 300, 400]
    assert opts.selection_mode == "manual-crop"


# ---- GET /api/v1/chain/status ------------------------------------------

def test_chain_status_reports_configuration_without_exposing_it(client):
    r = client.get("/api/v1/chain/status")
    assert r.status_code == 200
    body = r.json()

    for key in ("network", "chain_id", "signer_configured", "rpc_reachable",
                "schema_registered", "ready"):
        assert key in body
    # Whether a key exists, never the key itself.
    assert isinstance(body["signer_configured"], bool)
    assert_no_secrets(r.text)


def test_health_reports_chain_configuration_as_a_boolean(client):
    r = client.get("/api/v1/health")
    body = r.json()
    assert isinstance(body["chain_configured"], bool)
    assert body["priority_platforms"][:4] == ["LinkedIn", "Instagram", "X/Twitter", "GitHub"]
    assert_no_secrets(r.text)


# ---- SSE ---------------------------------------------------------------

def _events_from(client, case_id: str) -> list[dict]:
    out: list[dict] = []
    with client.stream("GET", f"/api/v1/scan/{case_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            evt = json.loads(line[5:].strip())
            if evt["stage"] == "ping":
                continue
            out.append(evt)
            if evt["stage"] in ("done", "error"):
                break
    return out


def test_sse_emits_stage_events_in_pipeline_order_and_terminates(client):
    from facechain.models import Case

    def fake_run(opts, reporter=None):
        for stage, status, detail in [
            ("input", "ok", "400x500"),
            ("face", "ok", "1 face"),
            ("search:yandex", "ok", "COMPLETED: 12 candidates"),
            ("search:bing", "fail", "CHALLENGED: refused"),
            ("verify:candidate", "ok", "instagram.com score 0.88"),
            ("evidence", "ok", "evidenceHash sha256:abc"),
        ]:
            reporter(stage, status, detail)
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="VERIFIED_OFFCHAIN")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true"})
        events = _events_from(client, r.json()["case_id"])

    stages = [e["stage"] for e in events]
    assert stages[-1] == "done"
    assert stages.index("input") < stages.index("face") < stages.index("search:yandex")
    assert stages.index("evidence") < stages.index("done")


def test_a_challenged_provider_does_not_end_the_stream(client):
    from facechain.models import Case

    def fake_run(opts, reporter=None):
        reporter("search:bing", "fail", "CHALLENGED: engine refused")
        reporter("search:yandex", "ok", "COMPLETED: 30 candidates")
        reporter("evidence", "ok", "evidenceHash sha256:abc")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="VERIFIED_OFFCHAIN")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true"})
        events = _events_from(client, r.json()["case_id"])

    stages = [e["stage"] for e in events]
    # The provider failure is visible, the successful one still ran, and the
    # stream reached a terminal event regardless.
    assert "search:bing" in stages
    assert "search:yandex" in stages
    assert stages[-1] == "done"


def test_an_unhandled_pipeline_error_still_produces_a_terminal_event(client):
    def boom(opts, reporter=None):
        raise RuntimeError("something deep broke")

    with patch("server.run", side_effect=boom):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true"})
        case_id = r.json()["case_id"]
        events = _events_from(client, case_id)

    assert events[-1]["stage"] == "error"
    # And the message is a type name, not an internal traceback.
    assert "something deep broke" not in events[-1]["detail"]
    assert client.get(f"/api/v1/scan/{case_id}/status").json()["status"] == "failed"


def test_a_chain_failure_does_not_lose_the_evidence(client):
    """Attestation is the last stage. When it fails, local verification and the
    evidence hash it produced must both survive into the result."""
    from facechain.models import Case, ChainRecord

    def fake_run(opts, reporter=None):
        reporter("evidence", "ok", "evidenceHash sha256:deadbeef")
        reporter("chain", "fail", "RPC unreachable")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1,
                    verdict="VERIFIED_OFFCHAIN", evidence_sha256="ab" * 32,
                    blockchain=ChainRecord(mode="failed", note="RPC unreachable"))

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "false", "chain_mode": "onchain"})
        case_id = r.json()["case_id"]
        events = _events_from(client, case_id)
        result = client.get(f"/api/v1/scan/{case_id}/result").json()

    assert events[-1]["stage"] == "done"
    assert result["verdict"] == "VERIFIED_OFFCHAIN"
    assert result["evidence_sha256"] == "ab" * 32
    assert result["blockchain"]["mode"] == "failed"


def test_events_replay_for_a_late_subscriber(client):
    """A client that connects after the scan finished still sees the whole
    history and a terminal event, rather than hanging on an empty stream."""
    from facechain.models import Case

    def fake_run(opts, reporter=None):
        reporter("input", "ok", "done fast")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true"})
        case_id = r.json()["case_id"]
        client.get(f"/api/v1/scan/{case_id}/result")   # ensure it has finished
        events = _events_from(client, case_id)

    assert [e["stage"] for e in events][0] == "input"
    assert events[-1]["stage"] == "done"


def test_no_secrets_appear_in_any_scan_response(client):
    from facechain.models import Case

    def fake_run(opts, reporter=None):
        reporter("chain", "fail", "boom")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "true"})
        case_id = r.json()["case_id"]
        assert_no_secrets(r.text)
        for path in ("status", "result"):
            assert_no_secrets(client.get(f"/api/v1/scan/{case_id}/{path}").text)


# ---- chain mode reaches the pipeline -----------------------------------

def test_turning_off_no_chain_actually_requests_an_attestation(client):
    """Regression: `chain_mode` defaulted to "skip" on the server while the UI
    only ever sent `no_chain`. Unticking "skip attestation" in the browser
    therefore still produced a skipped attestation, and the on-chain path was
    unreachable from the UI entirely."""
    from facechain.models import Case

    captured = {}

    def fake_run(opts, reporter=None):
        captured["chain_mode"] = opts.chain_mode
        if reporter:
            reporter("done", "ok", "stubbed")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "false"})
        client.get(f"/api/v1/scan/{r.json()['case_id']}/result")

    assert captured["chain_mode"] == "onchain"


def test_the_default_is_still_to_skip_attestation(client):
    """Spending gas must never be the default for a bare API call."""
    from facechain.models import Case

    captured = {}

    def fake_run(opts, reporter=None):
        captured["chain_mode"] = opts.chain_mode
        if reporter:
            reporter("done", "ok", "stubbed")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")})
        client.get(f"/api/v1/scan/{r.json()['case_id']}/result")

    assert captured["chain_mode"] == "skip"


def test_an_explicit_simulate_mode_is_respected(client):
    from facechain.models import Case

    captured = {}

    def fake_run(opts, reporter=None):
        captured["chain_mode"] = opts.chain_mode
        if reporter:
            reporter("done", "ok", "stubbed")
        return Case(case_id=opts.case_id, created_at="now", observed_at=1, verdict="UNVERIFIED")

    with patch("server.run", side_effect=fake_run):
        r = client.post("/api/v1/scan",
                        files={"image": ("x.jpg", photo_bytes(), "image/jpeg")},
                        data={"no_chain": "false", "chain_mode": "simulate"})
        client.get(f"/api/v1/scan/{r.json()['case_id']}/result")

    assert captured["chain_mode"] == "simulate"
