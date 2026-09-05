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
from ..models import ProviderReport, ProviderStatus, SearchCandidate, SearchReport, SearchVariantReport
from .base import EngineResult, looks_like_post
from .bing import BingVisualAdapter
from .browser import BrowserSession
from .google_lens import GoogleLensAdapter
from .serpapi import SerpApiAdapter
from .tineye import TinEyeAdapter
from .uploader import UploadError, publish_image, publish_temporarily
from .variants import SearchVariant
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
    # SerpAPI's actual engine id is "bing_reverse_image" (confirmed against
    # SerpAPI's own docs) — not "bing", which silently returns nothing.
    "serpapi_bing": lambda: SerpApiAdapter("bing_reverse_image"),
}

# Browser challenges are terminal for that browser session. When the same
# provider has a configured first-party API, use it once as a legitimate
# fallback instead of retrying the blocked UI.
BROWSER_API_FALLBACKS = {
    "google_lens": "serpapi_google_lens",
    "bing": "serpapi_bing",
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

    The Windows ProactorEventLoop policy is set at server.py startup so
    subprocess spawning works from any thread in the process.
    """
    with BrowserSession() as session:
        adapter = BROWSER_ADAPTERS[name](session)
        res = adapter.search(image_path, public_url)
        if not res.ok and public_url and _status_for(res) not in (
            ProviderStatus.CHALLENGED, ProviderStatus.RATE_LIMITED
        ):
            log.info("%s by-url failed (%s); retrying via upload", name, res.error)
            res = adapter.search(image_path, None)
        return res


def _run_api_engine(
    name: str, image_path: str, public_url: str | None, upload_failure: str | None = None,
) -> EngineResult:
    adapter = API_ADAPTERS[name]()
    if public_url is None and upload_failure and adapter.requires_public_url:
        # This adapter has no way to run without a public URL, and one was
        # genuinely attempted and failed — not simply never requested. Report
        # the real cause rather than calling `.search()`, which can only ever
        # produce its generic "needs a public image URL" message here and
        # would misleadingly suggest the fix is to enable a flag that is
        # already enabled.
        return EngineResult(
            name, ok=False, query_mode="api", status=ProviderStatus.FAILED,
            error=f"temporary image publication failed: {upload_failure}",
        )
    res = adapter.search(image_path, public_url)
    # `SerpApiAdapter.name` is derived from its own `serp_engine` id (e.g.
    # "serpapi_yandex_images"), which does not always match the registry key
    # a caller actually requested (e.g. "serpapi_yandex"). Normalising here
    # keeps `report.providers`, the live progress events, and
    # `engines_attempted` all referring to the same name for the same engine.
    res.engine = name
    return res


def _run_challenge_fallback(
    browser_name: str, image_path: str, public_url: str | None,
) -> EngineResult | None:
    fallback_name = BROWSER_API_FALLBACKS.get(browser_name)
    if not fallback_name or fallback_name not in API_ADAPTERS:
        return None
    if not settings.serpapi_key:
        return None
    log.info("%s challenged; using configured %s fallback once", browser_name, fallback_name)
    try:
        return _run_api_engine(fallback_name, image_path, public_url)
    except Exception as exc:  # noqa: BLE001 - fallback must remain isolated
        return EngineResult(
            fallback_name, ok=False, query_mode="api",
            error=f"fallback error: {type(exc).__name__}: {str(exc)[:200]}",
        )


def _requires_public_url(name: str) -> bool:
    if name in BROWSER_ADAPTERS:
        return bool(getattr(BROWSER_ADAPTERS[name], "requires_public_url", False))
    if name in API_ADAPTERS:
        return bool(getattr(API_ADAPTERS[name](), "requires_public_url", False))
    return False


def _has_reliable_upload_alternative(name: str) -> bool:
    try:
        if name in BROWSER_ADAPTERS:
            return bool(getattr(BROWSER_ADAPTERS[name], "has_reliable_upload_alternative", False))
        if name in API_ADAPTERS:
            return bool(getattr(API_ADAPTERS[name](), "has_reliable_upload_alternative", False))
    except Exception:  # noqa: BLE001 — a broken adapter stub must not abort the check
        pass
    return False


def _run_engine_pass(
    image_path: str,
    engines: list[str],
    public_url: str | None,
    stage_deadline: float,
    emit: Callable[[str, str, str], None],
    emit_prefix: str = "",
    upload_failure: str | None = None,
) -> tuple[list[EngineResult], list[ProviderReport], bool]:
    """Fan `image_path` out across `engines` once, bounded by `stage_deadline`.

    Shared by the primary (original-image) pass and every extra search-variant
    pass, so a variant gets the exact same isolation/timeout/concurrency
    guarantees as the main search rather than a cheaper approximation of them.

    `upload_failure`, when set, is the reason a central public-URL publish was
    attempted and failed this run — passed through so an adapter that
    genuinely cannot run without one reports that real cause instead of a
    generic "not configured" message.
    """

    def tag(engine: str) -> str:
        return f"{emit_prefix}{engine}" if emit_prefix else engine

    planned: list[tuple[str, Callable[[], EngineResult]]] = []
    providers: list[ProviderReport] = []
    for name in engines:
        if name in BROWSER_ADAPTERS:
            planned.append((name, lambda n=name: _run_browser_engine(n, image_path, public_url)))
        elif name in API_ADAPTERS:
            planned.append((name, lambda n=name: _run_api_engine(
                n, image_path, public_url, upload_failure)))
        else:
            providers.append(ProviderReport(
                engine=name, status=ProviderStatus.FAILED, error="unknown engine"))
            emit(tag(name), "fail", "FAILED: unknown engine")

    results: list[EngineResult] = []
    completed_by_name: dict[str, EngineResult] = {}
    timed_out = False
    if not planned:
        return results, providers, timed_out

    remaining_budget = stage_deadline - time.monotonic()
    if remaining_budget <= 0:
        for name, _ in planned:
            providers.append(ProviderReport(
                engine=name, status=ProviderStatus.TIMEOUT,
                error="search budget exhausted before this pass could start"))
            emit(tag(name), "fail", f"{ProviderStatus.TIMEOUT.value}: search budget exhausted")
        return results, providers, True

    workers = max(1, min(settings.search_concurrency, len(planned)))
    started_at: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], EngineResult]) -> EngineResult:
        started_at[name] = time.monotonic()
        emit(tag(name), "start", "")
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
                completed_by_name[name] = res
                providers.append(ProviderReport(
                    engine=res.engine or name,
                    status=status,
                    candidates=len(res.candidates),
                    duration_s=elapsed,
                    query_mode=res.query_mode,
                    error="" if status.produced_results else res.error[:300],
                    public_url_available=bool(public_url),
                ))
                detail = (f"{status.value}: {len(res.candidates)} candidates"
                          if status.produced_results
                          else f"{status.value}: {res.error[:160]}")
                emit(tag(name), "ok" if status.produced_results else "fail", detail)

        # Anything still pending blew the stage budget. Those workers are
        # abandoned rather than joined: Playwright's own per-operation
        # timeouts guarantee they eventually unwind and close their browser,
        # so the scan is never held hostage by a wedged provider.
        for fut in pending:
            name = futures[fut]
            fut.cancel()
            timed_out = True
            providers.append(ProviderReport(
                engine=name,
                status=ProviderStatus.TIMEOUT,
                duration_s=time.monotonic() - started_at[name],
                error="exceeded the search budget",
            ))
            emit(tag(name), "fail", f"{ProviderStatus.TIMEOUT.value}: search budget exhausted")

        # A challenge is a provider-specific terminal state, not a reason to
        # spend another browser request. A configured API fallback gets one
        # bounded attempt and is recorded under its own provider name.
        for name, res in list(completed_by_name.items()):
            if _status_for(res) != ProviderStatus.CHALLENGED:
                continue
            fallback_name = BROWSER_API_FALLBACKS.get(name)
            if not fallback_name or fallback_name in completed_by_name:
                continue
            if time.monotonic() >= stage_deadline:
                continue
            fallback_started = time.monotonic()
            fallback = _run_challenge_fallback(name, image_path, public_url)
            if fallback is None:
                continue
            fallback_status = _status_for(fallback)
            fallback_elapsed = time.monotonic() - fallback_started
            results.append(fallback)
            providers.append(ProviderReport(
                engine=fallback_name,
                status=fallback_status,
                candidates=len(fallback.candidates),
                duration_s=fallback_elapsed,
                query_mode=fallback.query_mode,
                error="" if fallback_status.produced_results else fallback.error[:300],
                public_url_available=bool(public_url),
                fallback_used=name,
            ))
            emit(tag(fallback_name),
                 "ok" if fallback_status.produced_results else "fail",
                 (f"{fallback_status.value}: {len(fallback.candidates)} candidates"
                  if fallback_status.produced_results
                  else f"{fallback_status.value}: {fallback.error[:160]}"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results, providers, timed_out


def _merge_candidates(
    merged: dict[str, SearchCandidate], results: list[EngineResult], variant_id: str = ""
) -> None:
    for res in results:
        for cand in res.candidates:
            key = cand.url.split("#")[0].rstrip("/")
            existing = merged.get(key)
            if existing is None:
                if variant_id:
                    cand.found_via_variant = variant_id
                merged[key] = cand
            elif res.engine not in existing.engine:
                # Corroboration across engines is a positive signal; keep both names.
                existing.engine = f"{existing.engine}+{res.engine}"


def run_reverse_search(
    image_path: str,
    engines: list[str] | None = None,
    image_url: str | None = None,
    on_event: Callable[[str, str, str], None] | None = None,
    variants: list[SearchVariant] | None = None,
) -> tuple[SearchReport, str | None]:
    """Search `image_path` on every requested engine.

    `on_event(engine, status, detail)` is called for live feedback, where status
    is one of "start" | "ok" | "fail". The detail string always names the
    provider's terminal state, so a UI can show CHALLENGED and TIMEOUT as the
    distinct outcomes they are.

    `variants` (see `search/variants.py`) optionally adds extra search passes
    over crops of the same photo, on top of the pass over `image_path` this
    function has always done. Each variant gets a share of the overall search
    budget; its own engine-level events do not overwrite the primary pass's
    `report.providers`, since that remains the historical single-search view
    every existing consumer (CLI, UI, evidence) already expects. Extra
    variants are attributed instead through `report.variants` and
    `SearchCandidate.found_via_variant`.

    Returns the merged report and the public image URL actually used (if any).
    """
    engines = engines or settings.engine_list
    report = SearchReport(engines_attempted=list(engines))
    emit = on_event or (lambda *_: None)

    # If a public URL is available (given, or opted-in temp upload), engines can
    # use their much more reliable by-URL endpoints.
    public_url = image_url
    upload_failure: str | None = None
    if public_url is None:
        needs_url = any(not _has_reliable_upload_alternative(name) for name in engines)
        if needs_url:
            try:
                # Local server takes priority and validates that the route is
                # reachable. A third-party host is only used when explicitly
                # enabled and local publication cannot serve the image.
                if (settings.local_image_base_url or "").strip():
                    from .uploader import publish_image as _publish_image
                    public_url = _publish_image(image_path, validate=True)
                else:
                    public_url = publish_temporarily(image_path)
                emit("host", "ok", public_url)
            except UploadError as exc:
                upload_failure = str(exc)
                log.warning("image publication failed, falling back to direct upload flows: %s", exc)
                emit("host", "fail", str(exc))
        else:
            emit("host", "skip",
                 "no selected engine needs central hosting (each has its own reliable path)")

    extra_variants = [v for v in (variants or []) if v.image_path != image_path]
    n_passes = 1 + len(extra_variants)
    total_budget = settings.search_total_timeout_s
    # Each pass gets an even share of the overall budget rather than the full
    # budget each — otherwise the second variant could never run at all once
    # the first pass had already used its wall-clock allowance.
    per_pass_budget = total_budget / n_passes if n_passes > 1 else total_budget
    now = time.monotonic()

    # ---- primary pass: the image already chosen upstream (crop or upload) ---
    primary_deadline = now + per_pass_budget
    results, providers, timed_out = _run_engine_pass(
        image_path, engines, public_url, primary_deadline, emit,
        upload_failure=upload_failure,
    )
    report.providers = providers
    report.timed_out = timed_out

    merged: dict[str, SearchCandidate] = {}
    _merge_candidates(merged, results)

    if extra_variants:
        report.variants.append(SearchVariantReport(
            variant_id="v0-original", variant_type="original",
            sha256="", candidates_found=len(merged),
        ))

    # ---- extra variant passes -----------------------------------------------
    overall_deadline = now + total_budget
    for variant in extra_variants:
        pass_deadline = min(time.monotonic() + per_pass_budget, overall_deadline)
        if time.monotonic() >= overall_deadline:
            report.variants.append(SearchVariantReport(
                variant_id=variant.variant_id, variant_type=variant.variant_type,
                sha256=variant.sha256, width=variant.width, height=variant.height,
                skipped=True, skip_reason="overall search budget exhausted",
            ))
            continue

        emit(f"search:variant:{variant.variant_type}", "start", variant.variant_id)
        v_results, _v_providers, v_timed_out = _run_engine_pass(
            variant.image_path, engines, public_url, pass_deadline, emit,
            emit_prefix=f"variant:{variant.variant_type}:",
            upload_failure=upload_failure,
        )
        report.timed_out = report.timed_out or v_timed_out
        before = len(merged)
        _merge_candidates(merged, v_results, variant_id=variant.variant_id)
        new_hits = len(merged) - before
        report.variants.append(SearchVariantReport(
            variant_id=variant.variant_id, variant_type=variant.variant_type,
            sha256=variant.sha256, width=variant.width, height=variant.height,
            candidates_found=new_hits,
        ))
        emit(f"search:variant:{variant.variant_type}", "ok" if not v_timed_out else "fail",
             f"{new_hits} new candidate(s)")

    for res in results:
        report.query_mode[res.engine] = res.query_mode
        provider = report.provider(res.engine)
        if provider is not None and provider.status.produced_results:
            report.engines_succeeded.append(res.engine)
        elif provider is not None and provider.error:
            report.engine_errors[res.engine] = provider.error

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
