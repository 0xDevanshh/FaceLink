#!/usr/bin/env python3
"""FaceChain — FastAPI backend server.

    uvicorn server:app --reload --host 0.0.0.0 --port 8000

The pipeline runs in a ThreadPoolExecutor. Events are appended to a plain
list by the worker thread (GIL protects list.append). The SSE generator
tail-follows that list by index — no asyncio.Queue, no call_soon_threadsafe,
no thread-safety bugs.
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
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse

from facechain.config import settings
from facechain.runner import PipelineError, RunOptions, run
from facechain.security.paths import PathTraversalError, safe_case_id
from facechain.security.scrubber import install as install_scrubber, scrub

install_scrubber()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = settings.api_upload_max_mb * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/avif"}

# Windows-safe upload dir (no /tmp)
_UPLOAD_DIR = Path(os.environ.get(
    "FACECHAIN_UPLOAD_DIR",
    str(Path(__file__).parent / "_uploads"),
))

_SEMAPHORE: asyncio.Semaphore | None = None


# ── job ───────────────────────────────────────────────────────────────────────
class JobStatus:
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class Job:
    """
    Events are appended to `self.events` by the worker thread.
    Python's GIL makes list.append() atomic enough for our read/write pattern
    (one writer thread, many reader coroutines that only read by index).
    SSE generators tail-follow by keeping a local cursor into this list.
    """
    def __init__(self, case_id: str, image_path: Path, opts: RunOptions) -> None:
        self.case_id    = case_id
        self.image_path = image_path
        self.opts       = opts
        self.status     = JobStatus.QUEUED
        self.events: list[dict]    = []   # append-only, GIL-safe
        self.result: dict | None   = None
        self.error:  str  | None   = None
        self.created_at            = time.monotonic()

    # Called from the worker thread — just append.
    def push(self, stage: str, status: str, detail: str) -> None:
        self.events.append({
            "stage":  scrub(stage),
            "status": status,
            "detail": scrub(detail),
            "ts":     datetime.now(timezone.utc).isoformat(),
        })


_jobs: dict[str, Job] = {}


def _cleanup_jobs() -> None:
    cutoff = time.monotonic() - settings.api_job_ttl_s
    stale  = [k for k, j in _jobs.items() if j.created_at < cutoff]
    for k in stale:
        try: _jobs[k].image_path.unlink(missing_ok=True)
        except Exception: pass  # noqa: BLE001
        del _jobs[k]


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _SEMAPHORE
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _SEMAPHORE = asyncio.Semaphore(settings.api_max_concurrent_scans)
    yield
    for j in _jobs.values():
        try: j.image_path.unlink(missing_ok=True)
        except Exception: pass  # noqa: BLE001


# ── app ───────────────────────────────────────────────────────────────────────
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


# ── upload validation ─────────────────────────────────────────────────────────
def _check_magic(data: bytes) -> bool:
    if len(data) < 4:                                                return False
    if data[:3] == b"\xff\xd8\xff":                                  return True  # JPEG
    if data[:4] == b"\x89PNG":                                       return True  # PNG
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP": return True  # WebP
    if data[:4] in (b"GIF8", b"GIF9"):                               return True  # GIF
    if data[:2] == b"BM":                                            return True  # BMP
    return False


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large: max {settings.api_upload_max_mb} MB")
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported content type: {mime!r}")
    if not _check_magic(data):
        raise HTTPException(422, "File magic bytes do not match a supported image format")


# ── pipeline worker ───────────────────────────────────────────────────────────
def _run_pipeline_sync(job: Job) -> None:
    """Runs in a thread. Calls job.push() which is GIL-safe."""
    try:
        case = run(job.opts, job.push)
        job.result = case.model_dump(mode="json")
        job.status = JobStatus.DONE
        job.push("done", "ok", f"verdict={case.verdict}")
    except PipelineError as exc:
        job.status = JobStatus.FAILED
        job.error  = str(exc)
        job.push("error", "fail", str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline error for %s", job.case_id)
        job.status = JobStatus.FAILED
        job.error  = f"{type(exc).__name__}: {exc}"
        job.push("error", "fail", job.error)


async def _launch(case_id: str) -> None:
    job = _jobs.get(case_id)
    if not job:
        return
    assert _SEMAPHORE
    async with _SEMAPHORE:
        job.status = JobStatus.RUNNING
        await asyncio.get_event_loop().run_in_executor(None, _run_pipeline_sync, job)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health() -> dict:
    from facechain import PIPELINE_VERSION
    return {
        "status":  "ok",
        "version": PIPELINE_VERSION,
        "engines_configured": {
            "serpapi":      bool(settings.serpapi_key),
            "facecheck":    bool(settings.facecheck_api_key),
            "search4faces": bool(settings.search4faces_api_key),
            "upload_host":  settings.allow_upload_host,
        },
        "face_backend":      settings.face_backend,
        "chain_mode_default": "skip",
    }


@app.post("/api/v1/scan")
async def start_scan(
    background_tasks: BackgroundTasks,
    image:            UploadFile = File(...),
    engines:          str = Form(default=""),
    chain_mode:       str = Form(default="skip"),
    max_verify:       int = Form(default=0),
    user_declaration: str = Form(default="false"),
    no_chain:         str = Form(default="true"),
) -> dict:
    _cleanup_jobs()

    _no_chain  = no_chain.lower() in ("true", "1", "yes")
    _user_decl = user_declaration.lower() in ("true", "1", "yes")

    if len(_jobs) >= settings.api_max_concurrent_scans * 4:
        raise HTTPException(429, "Too many pending jobs; try again shortly")

    data = await image.read()
    _validate_upload(image, data)

    ts      = datetime.now(timezone.utc)
    case_id = f"case_{ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    suffix  = Path(image.filename or "upload.jpg").suffix or ".jpg"
    img_path = _UPLOAD_DIR / f"{case_id}{suffix}"
    img_path.write_bytes(data)

    engine_list = [e.strip() for e in engines.split(",") if e.strip()] or None
    mode        = "skip" if _no_chain else chain_mode

    opts = RunOptions(
        image=str(img_path),
        engines=engine_list,
        chain_mode=mode,
        max_verify=max_verify or None,
        case_id=case_id,
    )
    job = Job(case_id=case_id, image_path=img_path, opts=opts)
    if _user_decl:
        job.push("declaration", "info", "user_declaration=true")
    _jobs[case_id] = job

    background_tasks.add_task(_launch, case_id)

    return {
        "case_id":    case_id,
        "status_url": f"/api/v1/scan/{case_id}/status",
        "events_url": f"/api/v1/scan/{case_id}/events",
        "result_url": f"/api/v1/scan/{case_id}/result",
    }


@app.get("/api/v1/scan/{case_id}/status")
async def scan_status(case_id: str) -> dict:
    try:    safe_case_id(case_id)
    except PathTraversalError: raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job: raise HTTPException(404, "case not found")
    return {
        "case_id":     case_id,
        "status":      job.status,
        "event_count": len(job.events),
        "error":       job.error,
    }


@app.get("/api/v1/scan/{case_id}/events")
async def scan_events(request: Request, case_id: str) -> EventSourceResponse:
    """
    Tail-follow job.events by index.  No queues, no thread-safety issues.
    Works even if the pipeline finishes before the SSE client connects
    (it will stream the whole backlog immediately then close).
    """
    try:    safe_case_id(case_id)
    except PathTraversalError: raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job: raise HTTPException(404, "case not found")

    async def _gen():
        cursor = 0
        while True:
            # Drain everything buffered so far
            snapshot = job.events          # reference; list grows, never shrinks
            while cursor < len(snapshot):
                evt = snapshot[cursor]
                cursor += 1
                yield {"data": json.dumps(evt)}
                if evt.get("stage") in ("done", "error"):
                    return

            # Are we done?
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                # Drain one last time (race: status set just after snapshot)
                snapshot = job.events
                while cursor < len(snapshot):
                    evt = snapshot[cursor]
                    cursor += 1
                    yield {"data": json.dumps(evt)}
                return

            # Client disconnected?
            if await request.is_disconnected():
                return

            # Nothing new yet — yield a keepalive and wait
            yield {"data": json.dumps({"stage": "ping", "status": "ok", "detail": ""})}
            await asyncio.sleep(0.4)   # poll every 400 ms — fast enough, not spammy

    return EventSourceResponse(_gen())


@app.get("/api/v1/scan/{case_id}/result")
async def scan_result(case_id: str) -> dict:
    try:    safe_case_id(case_id)
    except PathTraversalError: raise HTTPException(400, "invalid case_id")
    job = _jobs.get(case_id)
    if not job: raise HTTPException(404, "case not found")
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(202, "scan still in progress")
    if job.status == JobStatus.FAILED:
        raise HTTPException(500, job.error or "pipeline failed")
    return job.result or {}


@app.get("/api/v1/scan/{case_id}/evidence")
async def scan_evidence(case_id: str) -> Response:
    try:    safe_case_id(case_id)
    except PathTraversalError: raise HTTPException(400, "invalid case_id")
    ep = Path(settings.evidence_dir) / case_id
    if not ep.exists(): raise HTTPException(404, "evidence not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        from facechain.evidence.hashing import sha256_file
        lines: list[str] = []
        for fp in sorted(ep.rglob("*")):
            if fp.is_file():
                arc = fp.relative_to(ep.parent)
                zf.write(fp, arc)
                lines.append(f"{sha256_file(fp)}  {arc}")
        zf.writestr(f"{case_id}/CHECKSUMS.sha256", "\n".join(lines))
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{case_id}.zip"'},
    )


@app.post("/api/v1/verify")
async def verify_evidence_upload(evidence_zip: UploadFile = File(...)) -> dict:
    data = await evidence_zip.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Evidence ZIP too large (max 50 MB)")

    import tempfile
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for m in zf.namelist():
                if ".." in m or m.startswith("/"):
                    raise HTTPException(422, f"Unsafe path in ZIP: {m}")
                zf.extract(m, tmp_path)

        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("case_")]
        if not dirs: raise HTTPException(422, "No case_* directory in ZIP")
        case_dir = dirs[0]

        from facechain.evidence.writer  import verify_bundle_integrity
        from facechain.evidence.hashing import sha256_canonical, sha256_text

        passed, problems = verify_bundle_integrity(case_dir)
        checks: list[dict] = []

        cp = case_dir / "case.json"
        if cp.exists():
            cd = _json.loads(cp.read_text())
            pp = case_dir / "attested_payload.json"
            if pp.exists():
                payload  = _json.loads(pp.read_text())
                ev_hash  = sha256_canonical(payload)
                checks.append({
                    "check":  "evidenceHash recomputed",
                    "passed": ev_hash == cd.get("evidence_sha256"),
                    "detail": f"sha256:{ev_hash[:24]}…",
                })
                checks.append({
                    "check":  "matched_url hash consistent",
                    "passed": sha256_text(payload.get("matched_url", ""))
                              == payload.get("matched_url_sha256", ""),
                })

        for p in problems:
            checks.append({"check": "bundle integrity", "passed": False, "detail": p})
        if not problems:
            checks.append({"check": "bundle integrity", "passed": True})

        return {
            "overall": "PASS" if passed and all(c["passed"] for c in checks) else "FAIL",
            "checks":  checks,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.api_host, port=settings.api_port, reload=True)
