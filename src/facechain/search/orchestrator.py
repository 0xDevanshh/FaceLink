"""Fan out one image across several real reverse-image engines and merge hits.

No single engine is trusted or required: Lens may find nothing while Yandex
finds an Instagram post, and any engine can be CAPTCHA'd on any given run.
Every engine's outcome (success, error, which query mode it used) is recorded
in the evidence bundle so the search stage is auditable rather than magical.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..config import settings
from ..models import SearchCandidate, SearchReport
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


def run_reverse_search(
    image_path: str,
    engines: list[str] | None = None,
    image_url: str | None = None,
    on_event: Callable[[str, str, str], None] | None = None,
) -> tuple[SearchReport, str | None]:
    """Search `image_path` on every requested engine.

    `on_event(engine, status, detail)` is called for live CLI feedback, where
    status is one of "start" | "ok" | "fail".

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

    results: list[EngineResult] = []
    browser_engines = [e for e in engines if e in BROWSER_ADAPTERS]
    api_engines = [e for e in engines if e in API_ADAPTERS]

    for unknown in set(engines) - set(browser_engines) - set(api_engines):
        report.engine_errors[unknown] = "unknown engine"
        emit(unknown, "fail", "unknown engine")

    if browser_engines:
        with BrowserSession() as session:
            for name in browser_engines:
                emit(name, "start", "")
                adapter = BROWSER_ADAPTERS[name](session)
                res = adapter.search(image_path, public_url)
                # An engine's by-URL path can fail while its upload path works.
                if not res.ok and public_url:
                    log.info("%s by-url failed (%s); retrying via upload", name, res.error)
                    res = adapter.search(image_path, None)
                results.append(res)
                emit(name, "ok" if res.ok else "fail",
                     f"{len(res.candidates)} candidates" if res.ok else res.error)

    for name in api_engines:
        emit(name, "start", "")
        res = API_ADAPTERS[name]().search(image_path, public_url)
        results.append(res)
        emit(name, "ok" if res.ok else "fail",
             f"{len(res.candidates)} candidates" if res.ok else res.error)

    # ---- merge -----------------------------------------------------------
    merged: dict[str, SearchCandidate] = {}
    for res in results:
        report.query_mode[res.engine] = res.query_mode
        if res.ok:
            report.engines_succeeded.append(res.engine)
        else:
            report.engine_errors[res.engine] = res.error
        for cand in res.candidates:
            key = cand.url.split("#")[0].rstrip("/")
            existing = merged.get(key)
            if existing is None:
                merged[key] = cand
            elif res.engine not in existing.engine:
                # Corroboration across engines is a positive signal; keep both names.
                existing.engine = f"{existing.engine}+{res.engine}"

    candidates = sorted(
        merged.values(),
        key=lambda c: (not c.is_social, not looks_like_post(c.url), c.domain),
    )
    report.candidates = candidates
    report.total_candidates = len(candidates)
    report.social_candidates = sum(1 for c in candidates if c.is_social)
    return report, public_url
