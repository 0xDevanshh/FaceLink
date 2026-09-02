"""Fan out one image across several real reverse-image engines and merge hits.

No single engine is trusted or required: Lens may find nothing while Yandex
finds an Instagram post, and any engine can be CAPTCHA'd on any given run.
Every engine's outcome — its lifecycle state, how long it took, which query
mode it used — is recorded in the evidence bundle so the search stage is
auditable rather than magical.

Three properties this module is responsible for:

1. **Isolation.** One provider's failure, challenge or hang can never stop the
   others or abort the scan. Every adapter call is wrapped, and every provider
   ends in exactly one terminal `ProviderStatus`.

2. **Bounded time.** Each provider has a hard wall-clock budget
   (`engine_timeout_s`) and the stage as a whole has another
   (`search_total_timeout_s`). Playwright's own timeouts bound individual
   operations, but not the gaps between them — so a provider that wedges is
   abandoned and reported as TIMEOUT rather than being waited on.

3. **Real concurrency, bounded.** Providers run in a pool of at most
   `search_concurrency` threads. Each browser provider owns its own Chromium
   because Playwright's sync API is thread-affine, which also means one
   provider's crashed browser cannot take another's down with it.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from ..config import OTHER_WEB_PRIORITY, PLATFORM_PRIORITY, settings
from ..models import ProviderReport, ProviderStatus, SearchCandidate, SearchReport
from .base import EngineResult, looks_like_post
from .bing import BingVisualAdapter
from .browser import BrowserSession
from .google_lens import GoogleLensAdapter
from .serpapi import SerpApiAdapter
from .tineye import TinEyeAdapter
from .uploader import UploadError, publish_temporarily
from .yandex import YandexAdapter

log = logging.getLogger(__name__)

BROWSER_ADAPTERS = {
    "google_lens": GoogleLensAdapter,
    "yandex": YandexAdapter,
    "bing": BingVisualAdapter,
    "tineye": TinEyeAdapter,
}
API_ADAPTERS = {
    "serpapi_google_lens": lambda: SerpApiAdapter("google_lens"),
    "serpapi_yandex": lambda: SerpApiAdapter("yandex_images"),
}

# Error-text fingerprints, checked in order. Only used when an adapter did not
# already report a precise status of its own.
_ERROR_STATUS_HINTS: tuple[tuple[tuple[str, ...], ProviderStatus], ...] = (
    (("captcha", "bot challenge", "are you a robot", "unusual traffic",
      "verify you are human", "human confirmation", "access denied"),
     ProviderStatus.CHALLENGED),
    (("too many requests", "rate limit", "429"), ProviderStatus.RATE_LIMITED),
    (("timeout", "timed out"), ProviderStatus.TIMEOUT),
    (("no usable page links", "hasn't returned any results",
      "no outbound result links"), ProviderStatus.NO_RESULTS),
    (("not set", "needs a public image url", "not configured"),
     ProviderStatus.NOT_CONFIGURED),
)


def classify_error(error: str) -> ProviderStatus:
    """Best-effort status for an adapter that only gave us prose.

    Defaults to FAILED rather than NO_RESULTS: an unrecognised error means we do
    not know that the search ran, and claiming "searched, found nothing" would
    overstate what happened.
    """
    lowered = (error or "").lower()
    for needles, status in _ERROR_STATUS_HINTS:
        if any(n in lowered for n in needles):
            return status
    return ProviderStatus.FAILED


def _status_for(res: EngineResult) -> ProviderStatus:
    if res.status is not None:
        return res.status
    if res.ok and res.candidates:
        return ProviderStatus.COMPLETED
    if res.ok:
        return ProviderStatus.NO_RESULTS
    return classify_error(res.error)


def _run_browser_engine(name: str, image_path: str, public_url: str | None) -> EngineResult:
    """Drive one browser engine in its own Chromium, in this thread.

    A per-provider browser is what makes provider isolation and provider
    concurrency possible at once: the sync Playwright API may only be used from
    the thread that created it.
    """
    with BrowserSession() as session:
        adapter = BROWSER_ADAPTERS[name](session)
        res = adapter.search(image_path, public_url)
        # An engine's by-URL path can fail while its upload path works. Only
        # worth retrying when the failure was not a refusal to serve us at all.
        if not res.ok and public_url and _status_for(res) not in (
            ProviderStatus.CHALLENGED, ProviderStatus.RATE_LIMITED
        ):
            log.info("%s by-url failed (%s); retrying via upload", name, res.error)
            res = adapter.search(image_path, None)
        return res


def _run_api_engine(name: str, image_path: str, public_url: str | None) -> EngineResult:
    return API_ADAPTERS[name]().search(image_path, public_url)


def run_reverse_search(
    image_path: str,
    engines: list[str] | None = None,
    image_url: str | None = None,
    on_event: Callable[[str, str, str], None] | None = None,
) -> tuple[SearchReport, str | None]:
    """Search `image_path` on every requested engine.

    `on_event(engine, status, detail)` is called for live feedback, where status
    is one of "start" | "ok" | "fail". The detail string always names the
    provider's terminal state, so a UI can show CHALLENGED and TIMEOUT as the
    distinct outcomes they are.

    Returns the merged report and the public image URL actually used (if any).
    """
    engines = engines or settings.engine_list
    report = SearchReport(engines_attempted=list(engines))
    emit = on_event or (lambda *_: None)

    # If a public URL is available (given, or opted-in temp upload), engines can
    # use their much more reliable by-URL endpoints.
    public_url = image_url
    if public_url is None and settings.allow_upload_host:
        try:
            public_url = publish_temporarily(image_path)
            emit("host", "ok", public_url)
        except UploadError as exc:
            log.warning("temp hosting failed, falling back to upload flows: %s", exc)
            emit("host", "fail", str(exc))

    # ---- plan ------------------------------------------------------------
    planned: list[tuple[str, Callable[[], EngineResult]]] = []
    for name in engines:
        if name in BROWSER_ADAPTERS:
            planned.append((name, lambda n=name: _run_browser_engine(n, image_path, public_url)))
        elif name in API_ADAPTERS:
            planned.append((name, lambda n=name: _run_api_engine(n, image_path, public_url)))
        else:
            report.providers.append(ProviderReport(
                engine=name, status=ProviderStatus.FAILED, error="unknown engine"))
            emit(name, "fail", "FAILED: unknown engine")

    results: list[EngineResult] = []

    if planned:
        stage_deadline = time.monotonic() + settings.search_total_timeout_s
        workers = max(1, min(settings.search_concurrency, len(planned)))
        started_at: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], EngineResult]) -> EngineResult:
            # Start time is taken inside the task, so a provider that waited for
            # a free worker is not charged for the queueing.
            started_at[name] = time.monotonic()
            emit(name, "start", "")
            return fn()

        # NOT a `with` block, deliberately. ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which joins every running worker — so a wedged
        # provider would be waited on anyway and the deadline below would do
        # nothing. Owning the shutdown explicitly is what makes abandonment real.
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="search")
        try:
            futures = {}
            for name, fn in planned:
                started_at[name] = time.monotonic()
                futures[pool.submit(timed, name, fn)] = name

            pending = set(futures)
            while pending:
                remaining = stage_deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                for fut in done:
                    name = futures[fut]
                    elapsed = time.monotonic() - started_at[name]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001 - isolation is the point
                        log.warning("provider %s raised: %s", name, exc)
                        res = EngineResult(name, ok=False,
                                           error=f"{type(exc).__name__}: {str(exc)[:200]}")
                    status = _status_for(res)
                    # A provider that used its whole budget and still failed is
                    # a timeout, whatever the adapter called it.
                    if not res.ok and elapsed >= settings.engine_timeout_s:
                        status = ProviderStatus.TIMEOUT
                    results.append(res)
                    report.providers.append(ProviderReport(
                        engine=res.engine or name,
                        status=status,
                        candidates=len(res.candidates),
                        duration_s=elapsed,
                        query_mode=res.query_mode,
                        error="" if status.produced_results else res.error[:300],
                    ))
                    detail = (f"{status.value}: {len(res.candidates)} candidates"
                              if status.produced_results
                              else f"{status.value}: {res.error[:160]}")
                    emit(name, "ok" if status.produced_results else "fail", detail)

            # Anything still pending blew the stage budget. Those workers are
            # abandoned rather than joined: Playwright's own per-operation
            # timeouts guarantee they eventually unwind and close their browser,
            # so the scan is never held hostage by a wedged provider.
            for fut in pending:
                name = futures[fut]
                fut.cancel()
                report.timed_out = True
                report.providers.append(ProviderReport(
                    engine=name,
                    status=ProviderStatus.TIMEOUT,
                    duration_s=time.monotonic() - started_at[name],
                    error=f"exceeded the {settings.search_total_timeout_s}s search budget",
                ))
                emit(name, "fail", f"{ProviderStatus.TIMEOUT.value}: search budget exhausted")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # ---- merge -----------------------------------------------------------
    merged: dict[str, SearchCandidate] = {}
    for res in results:
        report.query_mode[res.engine] = res.query_mode
        provider = report.provider(res.engine)
        if provider is not None and provider.status.produced_results:
            report.engines_succeeded.append(res.engine)
        elif provider is not None and provider.error:
            report.engine_errors[res.engine] = provider.error
        for cand in res.candidates:
            key = cand.url.split("#")[0].rstrip("/")
            existing = merged.get(key)
            if existing is None:
                merged[key] = cand
            elif res.engine not in existing.engine:
                # Corroboration across engines is a positive signal; keep both names.
                existing.engine = f"{existing.engine}+{res.engine}"

    # Priority platforms first, a specific post ahead of a bare profile, then
    # the wider web. Ordering only — verification decides what is true.
    candidates = sorted(
        merged.values(),
        key=lambda c: (c.platform_priority, not looks_like_post(c.url), c.domain),
    )
    report.candidates = candidates
    report.total_candidates = len(candidates)
    report.social_candidates = sum(1 for c in candidates if c.is_social)
    report.platform_counts = count_by_platform(candidates)
    return report, public_url


def count_by_platform(candidates: list[SearchCandidate]) -> dict[str, int]:
    """Candidates per platform, with an explicit 0 for every priority platform.

    The zeros matter: "we looked for LinkedIn and found none" and "we never
    looked" must not render identically.
    """
    counts: dict[str, int] = {name: 0 for name in PLATFORM_PRIORITY}
    counts["Other Web"] = 0
    for cand in candidates:
        if cand.platform and cand.platform_priority < OTHER_WEB_PRIORITY:
            counts[cand.platform] = counts.get(cand.platform, 0) + 1
        else:
            counts["Other Web"] += 1
    return counts
