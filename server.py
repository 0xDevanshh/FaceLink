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

from facechain.config import settings
from facechain.runner import PipelineError, RunOptions, run
from facechain.security.paths import PathTraversalError, safe_case_id
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


def _cleanup_jobs() -> None:
    now = time.time()
    stale = [k for k, j in _jobs.items() if now - j.created_at > settings.api_job_ttl_s]
    for k in stale:
        try:
            _jobs[k].image_path.unlink(missing_ok=True)
        except Exception:
            pass
        del _jobs[k]


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
    }


@app.post("/api/v1/scan")
async def start_scan(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    engines: str = Form(default=""),
    chain_mode: str = Form(default="skip"),
    max_verify: int = Form(default=0),
    user_declaration: str = Form(default="false"),
    no_chain: str = Form(default="true"),
) -> dict:
    _cleanup_jobs()

    _no_chain = no_chain.lower() in ("true", "1", "yes")
    _user_decl = user_declaration.lower() in ("true", "1", "yes")

    if len(_jobs) >= settings.api_max_concurrent_scans * 4:
        raise HTTPException(429, "Too many pending jobs; try again shortly")

    data = await image.read()
    _validate_upload(image, data)

    case_id = f"case_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    img_path = _upload_dir / f"{case_id}{Path(image.filename or 'upload.jpg').suffix or '.jpg'}"
    img_path.write_bytes(data)

    engine_list = [e.strip() for e in engines.split(",") if e.strip()] or None
    mode = "skip" if _no_chain else chain_mode

    opts = RunOptions(
        image=str(img_path),
        engines=engine_list,
        chain_mode=mode,
        max_verify=max_verify or None,
        case_id=case_id,
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
    assert _SEMAPHORE is not None
    async with _SEMAPHORE:
        job.status = JobStatus.RUNNING
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_pipeline_sync, job)


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
        q = job.subscribe()
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
