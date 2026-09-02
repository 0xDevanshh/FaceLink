#!/usr/bin/env python3
"""FaceChain — FastAPI backend server.

    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Wraps the existing pipeline programmatically — CLI and UI share identical
stage logic and evidence generation. The server never subprocesses pipeline.py.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from facechain.config import PLATFORM_PRIORITY, settings
from facechain.runner import PipelineError, RunOptions, run
from facechain.security.paths import PathTraversalError, safe_case_id, safe_upload_id
from facechain.security.scrubber import install as install_scrubber, scrub

# ---- startup -------------------------------------------------------------
install_scrubber()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---- constants -----------------------------------------------------------
MAX_UPLOAD_BYTES = settings.api_upload_max_mb * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/avif"}
# Magic bytes for supported image formats.
MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",  # partial — also need to check offset 8
    b"GIF8": "image/gif",
    b"GIF9": "image/gif",
    b"BM": "image/bmp",
    b"\x00\x00\x00": "image/avif",  # ftyp box — partial check only
}
_SEMAPHORE: asyncio.Semaphore | None = None


# ---- job registry --------------------------------------------------------
class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job:
    def __init__(self, case_id: str, image_path: Path, opts: RunOptions) -> None:
        self.case_id = case_id
        self.image_path = image_path
        self.opts = opts
        self.status = JobStatus.QUEUED
        self.events: list[dict[str, str]] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self._waiters: list[asyncio.Queue] = []

    def push(self, stage: str, status: str, detail: str) -> None:
        evt = {"stage": scrub(stage), "status": status, "detail": scrub(detail),
               "ts": datetime.now(timezone.utc).isoformat()}
        self.events.append(evt)
        for q in self._waiters:
            q.put_nowait(evt)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._waiters.append(q)
        # Replay existing events.
        for evt in self.events:
            q.put_nowait(evt)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._waiters.remove(q)
        except ValueError:
            pass


_jobs: dict[str, Job] = {}
_upload_dir = Path(os.environ.get("FACECHAIN_UPLOAD_DIR", "/tmp/facechain_uploads"))

# Staged uploads awaiting a face choice: upload_id -> (path, created_at).
# Kept separate from jobs because a staged upload is not yet a scan, and it must
# expire on its own if the operator abandons the selection step.
_uploads: dict[str, tuple[Path, float]] = {}
UPLOAD_TTL_S = 1800

# A scan that has not reached a terminal state by this deadline is reported as
# timed out. Sized above the search stage's own total budget so a slow-but-alive
# search is never cut short by the watchdog.
SCAN_DEADLINE_S = max(600, settings.search_total_timeout_s + 300)


def _cleanup_jobs() -> None:
    now = time.time()
    stale = [k for k, j in _jobs.items() if now - j.created_at > settings.api_job_ttl_s]
    for k in stale:
        try:
            _jobs[k].image_path.unlink(missing_ok=True)
        except Exception:
            pass
        del _jobs[k]
    for uid in [u for u, (_, ts) in _uploads.items() if now - ts > UPLOAD_TTL_S]:
        path, _ = _uploads.pop(uid)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    global _SEMAPHORE
    _upload_dir.mkdir(parents=True, exist_ok=True)
    _SEMAPHORE = asyncio.Semaphore(settings.api_max_concurrent_scans)
    yield
    # Cleanup on shutdown.
    for job in _jobs.values():
        try:
            job.image_path.unlink(missing_ok=True)
        except Exception:
            pass
    for path, _ in _uploads.values():
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# ---- app -----------------------------------------------------------------
app = FastAPI(
    title="FaceChain API",
    version="1.0.0",
    description="Face ID + Blockchain Verification pipeline API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# ---- helpers -------------------------------------------------------------

def _check_magic(data: bytes) -> bool:
    """Validate image magic bytes (not just MIME header)."""
    if len(data) < 4:
        return False
    # JPEG: FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return True
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data[:4] == b"\x89PNG":
        return True
    # WebP: RIFF????WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    # GIF87a / GIF89a
    if data[:4] in (b"GIF8", b"GIF9"):
        return True
    # BMP: BM
    if data[:2] == b"BM":
        return True
    return False


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large: max {settings.api_upload_max_mb}MB")
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported content type: {mime}")
    if not _check_magic(data):
        raise HTTPException(422, "File magic bytes do not match a supported image format")


def _parse_face_index(raw: str) -> int | None:
    if not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise HTTPException(422, "face_index must be an integer") from None
    if value < 0:
        raise HTTPException(422, "face_index must be >= 0")
    return value


def _parse_crop(raw: str) -> list[int] | None:
    """Parse `"x,y,width,height"`.

    Rejected here rather than clamped: a malformed rectangle means the client
    and the server disagree about the image's coordinate space, and cropping
    *something* would attach a misleading rectangle to the evidence.
    """
    text = raw.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise HTTPException(422, "crop must be 'x,y,width,height'")
    try:
        return [int(float(p)) for p in parts]
    except ValueError:
        raise HTTPException(422, "crop values must be numbers") from None


def _run_pipeline_sync(job: Job) -> None:
    """Run the pipeline in a thread (called via run_in_executor)."""
    def reporter(stage: str, status: str, detail: str) -> None:
        job.push(stage, status, detail)

    try:
        case = run(job.opts, reporter)
        job.result = case.model_dump(mode="json")
        job.status = JobStatus.DONE
        job.push("done", "ok", f"verdict={case.verdict}")
    except PipelineError as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.push("error", "fail", str(exc))
    except Exception as exc:
        log.exception("unexpected pipeline error for case %s", job.case_id)
        job.status = JobStatus.FAILED
        job.error = f"internal error: {type(exc).__name__}"
        job.push("error", "fail", job.error)


# ---- routes --------------------------------------------------------------

@app.get("/api/v1/health")
async def health() -> dict:
    from facechain import PIPELINE_VERSION
    return {
        "status": "ok",
        "version": PIPELINE_VERSION,
        "engines_configured": {
            "serpapi": bool(settings.serpapi_key),
            "facecheck": bool(settings.facecheck_api_key),
            "search4faces": bool(settings.search4faces_api_key),
            "upload_host": settings.allow_upload_host,
        },
        "face_backend": settings.face_backend,
        "chain_mode_default": "skip",
        # Whether the attestation path is configured at all. Booleans only —
        # never the key itself.
        "chain_configured": bool(settings.private_key),
        "network": settings.network,
        "chain_id": settings.chain_id,
        "priority_platforms": list(PLATFORM_PRIORITY),
        "engine_timeout_s": settings.engine_timeout_s,
        "search_total_timeout_s": settings.search_total_timeout_s,
    }


@app.get("/api/v1/chain/status")
async def chain_status() -> dict:
    """Public readiness of the attestation path — no secrets, ever.

    Only public chain data is exposed: the network, the EAS addresses, whether
    the schema is registered, the attester's *address* and its balance. The
    private key never appears here, and the endpoint reports configuration state
    rather than values, so a missing key reads as `signer_configured: false`.
    """
    from facechain.chain.schema import schema_uid as predict_schema_uid

    out: dict[str, Any] = {
        "network": settings.network,
        "network_name": settings.chain_name,
        "chain_id": settings.chain_id,
        "eas_contract": settings.eas_contract,
        "signer_configured": bool(settings.private_key),
        "rpc_reachable": False,
        "schema_registered": False,
        "schema_uid": None,
        "attester": None,
        "balance_eth": None,
        "ready": False,
        "note": "",
    }
    if not out["signer_configured"]:
        out["note"] = "PRIVATE_KEY is not configured; attestation is unavailable"
        return out

    def probe() -> None:
        from facechain.chain.eas import EasClient

        client = EasClient()
        out["rpc_reachable"] = True
        uid = settings.eas_schema_uid or predict_schema_uid()
        out["schema_uid"] = uid
        out["schema_registered"] = client._schema_exists(uid)
        out["attester"] = client.address
        out["balance_eth"] = round(client.balance_eth(), 6)
        out["ready"] = bool(out["schema_registered"] and out["balance_eth"] > 0)

    try:
        # A slow public RPC must not block the event loop or hang the UI.
        await asyncio.wait_for(asyncio.to_thread(probe), timeout=20)
    except asyncio.TimeoutError:
        out["note"] = "RPC probe timed out"
    except Exception as exc:  # noqa: BLE001
        out["note"] = scrub(f"{type(exc).__name__}: {str(exc)[:200]}")
    return out


@app.post("/api/v1/faces")
async def detect_faces_endpoint(image: UploadFile = File(...)) -> dict:
    """Detect faces in an upload and say whether we can pick one safely.

    Staging the upload here and returning an `upload_id` means the operator's
    photo crosses the wire once, not once per selection attempt, and the bytes
    the scan runs on are byte-identical to the bytes the boxes were computed
    from — so a box the operator clicked cannot refer to a different image.

    Coordinates are in the pipeline's working image space (EXIF-oriented and
    downscaled), and `image_width`/`image_height` are returned so a client can
    map them onto whatever it is displaying.
    """
    data = await image.read()
    _validate_upload(image, data)

    _cleanup_jobs()
    upload_id = f"upl_{uuid.uuid4().hex[:16]}"
    suffix = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"):
        suffix = ".jpg"
    path = _upload_dir / f"{upload_id}{suffix}"
    path.write_bytes(data)

    def analyse() -> dict:
        from facechain.evidence.hashing import sha256_bytes
        from facechain.face import selection as face_selection
        from facechain.face.encoder import read_image

        img = read_image(path)
        faces = face_selection.detect_faces(img)
        offer = face_selection.offer(img, faces)
        quality = face_selection.gate_selected(
            img, faces, offer.auto_index if offer.auto_index is not None else None
        )
        return {
            "upload_id": upload_id,
            "sha256": sha256_bytes(data),
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),
            "faces": [f.model_dump(mode="json") for f in offer.faces],
            "auto_index": offer.auto_index,
            "selection_required": offer.selection_required,
            "reason": offer.reason,
            "quality": quality.rounded().model_dump(mode="json"),
        }

    try:
        return await asyncio.to_thread(analyse)
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        log.exception("face detection failed")
        raise HTTPException(422, f"could not analyse image: {type(exc).__name__}") from exc
    finally:
        if path.exists():
            _uploads[upload_id] = (path, time.time())


@app.post("/api/v1/scan")
async def start_scan(
    background_tasks: BackgroundTasks,
    image: UploadFile | None = File(default=None),
    upload_id: str = Form(default=""),
    engines: str = Form(default=""),
    # What to do when attestation is *not* skipped. `no_chain` below is the
    # gate, and it defaults to true, so the default behaviour is unchanged —
    # but a caller that turns the gate off now gets a real transaction instead
    # of silently falling back to "skip", which made on-chain attestation
    # unreachable from the UI entirely.
    chain_mode: str = Form(default="onchain"),
    max_verify: int = Form(default=0),
    user_declaration: str = Form(default="false"),
    no_chain: str = Form(default="true"),
    face_index: str = Form(default=""),
    crop: str = Form(default=""),
    selection_mode: str = Form(default=""),
    scan_depth: str = Form(default="standard"),
) -> dict:
    """Start a scan from a fresh upload or from an upload already staged by
    `POST /api/v1/faces`.

    `face_index` and `crop` are optional. When neither is given and the image is
    ambiguous, the pipeline stops with `FACE_SELECTION_REQUIRED` rather than
    guessing which person the scan is about.
    """
    _cleanup_jobs()

    _no_chain = no_chain.lower() in ("true", "1", "yes")
    _user_decl = user_declaration.lower() in ("true", "1", "yes")

    if len(_jobs) >= settings.api_max_concurrent_scans * 4:
        raise HTTPException(429, "Too many pending jobs; try again shortly")

    case_id = f"case_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    if upload_id:
        try:
            safe_upload_id(upload_id)
        except PathTraversalError:
            raise HTTPException(400, "invalid upload_id") from None
        staged = _uploads.get(upload_id)
        if staged is None:
            raise HTTPException(404, "upload_id not found or expired; upload the image again")
        source, _ = staged
        img_path = _upload_dir / f"{case_id}{source.suffix}"
        img_path.write_bytes(source.read_bytes())
    elif image is not None:
        data = await image.read()
        _validate_upload(image, data)
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        img_path = _upload_dir / f"{case_id}{suffix}"
        img_path.write_bytes(data)
    else:
        raise HTTPException(422, "provide either an image file or an upload_id")

    engine_list = [e.strip() for e in engines.split(",") if e.strip()] or None
    mode = "skip" if _no_chain else chain_mode

    opts = RunOptions(
        image=str(img_path),
        engines=engine_list,
        chain_mode=mode,
        max_verify=max_verify or None,
        case_id=case_id,
        face_index=_parse_face_index(face_index),
        crop_rect=_parse_crop(crop),
        selection_mode=selection_mode or None,
        scan_depth=scan_depth if scan_depth in ("fast", "standard", "deep") else "standard",
    )
    job = Job(case_id=case_id, image_path=img_path, opts=opts)
    if _user_decl:
        job.push("declaration", "info", "user_declaration=true recorded in job")
    _jobs[case_id] = job

    background_tasks.add_task(_run_pipeline_background, case_id)

    return {
        "case_id": case_id,
        "status_url": f"/api/v1/scan/{case_id}/status",
        "events_url": f"/api/v1/scan/{case_id}/events",
        "result_url": f"/api/v1/scan/{case_id}/result",
    }


async def _run_pipeline_background(case_id: str) -> None:
    job = _jobs.get(case_id)
    if not job:
        return
    try:
        assert _SEMAPHORE is not None
        async with _SEMAPHORE:
            job.status = JobStatus.RUNNING
            await asyncio.to_thread(_run_pipeline_sync, job)
    except Exception as exc:  # noqa: BLE001
        # A failure in the scheduling layer itself (never mind the pipeline) must
        # still terminate the job, or the SSE stream has nothing to close on.
        log.exception("scan scheduling failed for %s", case_id)
        job.status = JobStatus.FAILED
        job.error = f"scheduling error: {type(exc).__name__}"
        job.push("error", "fail", job.error)


@app.get("/api/v1/scan/{case_id}/status")
async def scan_status(case_id: str) -> dict:
    try:
        safe_case_id(case_id)
    except PathTraversalError:
        raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job:
        raise HTTPException(404, "case not found")
    return {
        "case_id": case_id,
        "status": job.status,
        "event_count": len(job.events),
        "error": job.error,
    }


@app.get("/api/v1/scan/{case_id}/events")
async def scan_events(request: Request, case_id: str) -> EventSourceResponse:
    try:
        safe_case_id(case_id)
    except PathTraversalError:
        raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job:
        raise HTTPException(404, "case not found")

    async def generator():
        # Every stream reaches a terminal event. A scan can end in success, in
        # no-match, in insufficient evidence or in an unrecoverable error — but
        # a client must never be left on "Scanning…" forever, so the deadline
        # below turns even a wedged pipeline into a reported outcome.
        q = job.subscribe()
        deadline = job.created_at + SCAN_DEADLINE_S
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield {"data": json.dumps(evt)}
                    if evt.get("stage") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    if job.status in (JobStatus.DONE, JobStatus.FAILED):
                        # The worker finished between our last read and now;
                        # drain anything it queued, then close.
                        while not q.empty():
                            yield {"data": json.dumps(q.get_nowait())}
                        if not any(e["stage"] in ("done", "error") for e in job.events):
                            yield {"data": json.dumps({
                                "stage": "done", "status": "ok",
                                "detail": f"status={job.status}",
                                "ts": datetime.now(timezone.utc).isoformat()})}
                        break
                    if time.time() > deadline:
                        job.status = JobStatus.FAILED
                        job.error = f"scan exceeded the {SCAN_DEADLINE_S}s deadline"
                        job.push("error", "fail", job.error)
                        yield {"data": json.dumps(job.events[-1])}
                        break
                    yield {"data": json.dumps({"stage": "ping", "status": "ok", "detail": ""})}
        finally:
            job.unsubscribe(q)

    return EventSourceResponse(generator())


@app.get("/api/v1/scan/{case_id}/result")
async def scan_result(case_id: str) -> dict:
    try:
        safe_case_id(case_id)
    except PathTraversalError:
        raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job:
        raise HTTPException(404, "case not found")
    if job.status == JobStatus.QUEUED or job.status == JobStatus.RUNNING:
        raise HTTPException(202, "scan still in progress")
    if job.status == JobStatus.FAILED:
        raise HTTPException(500, job.error or "pipeline failed")
    return job.result or {}


@app.get("/api/v1/scan/{case_id}/evidence")
async def scan_evidence(case_id: str) -> Response:
    try:
        safe_case_id(case_id)
    except PathTraversalError:
        raise HTTPException(400, "invalid case_id")

    evidence_path = Path(settings.evidence_dir) / case_id
    if not evidence_path.exists():
        raise HTTPException(404, "evidence not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(evidence_path.rglob("*")):
            if fp.is_file():
                arcname = fp.relative_to(evidence_path.parent)
                zf.write(fp, arcname)
        # Add a checksums file.
        from facechain.evidence.hashing import sha256_file
        lines = []
        for fp in sorted(evidence_path.rglob("*")):
            if fp.is_file():
                lines.append(f"{sha256_file(fp)}  {fp.relative_to(evidence_path.parent)}")
        zf.writestr(f"{case_id}/CHECKSUMS.sha256", "\n".join(lines))

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{case_id}.zip"'},
    )


@app.post("/api/v1/verify")
async def verify_evidence_upload(evidence_zip: UploadFile = File(...)) -> dict:
    """Accept an uploaded evidence ZIP, run the independent verifier, return per-field report."""
    data = await evidence_zip.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Evidence ZIP too large (max 50MB)")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Sanitize extraction paths.
            for member in zf.namelist():
                if ".." in member or member.startswith("/"):
                    raise HTTPException(422, f"Unsafe path in ZIP: {member}")
                zf.extract(member, tmp_path)

        # Find the case directory.
        case_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("case_")]
        if not case_dirs:
            raise HTTPException(422, "No case_* directory found in ZIP")
        case_dir = case_dirs[0]

        from facechain.evidence.writer import verify_bundle_integrity
        from facechain.evidence.hashing import sha256_canonical, sha256_text, sha256_file
        import json as _json

        passed, problems = verify_bundle_integrity(case_dir)
        checks: list[dict] = []

        case_path = case_dir / "case.json"
        if case_path.exists():
            case_data = _json.loads(case_path.read_text())
            payload_path = case_dir / "attested_payload.json"
            if payload_path.exists():
                payload = _json.loads(payload_path.read_text())
                ev_hash = sha256_canonical(payload)
                checks.append({
                    "check": "evidenceHash recomputed",
                    "passed": ev_hash == case_data.get("evidence_sha256"),
                    "detail": f"sha256:{ev_hash[:24]}…",
                })
                url_hash_ok = sha256_text(payload.get("matched_url", "")) == payload.get("matched_url_sha256", "")
                checks.append({"check": "matched_url hash consistent", "passed": url_hash_ok})

        for p in problems:
            checks.append({"check": "bundle integrity", "passed": False, "detail": p})
        if not problems:
            checks.append({"check": "bundle integrity", "passed": True})

        return {
            "overall": "PASS" if passed and all(c["passed"] for c in checks) else "FAIL",
            "checks": checks,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.api_host, port=settings.api_port, reload=True)
