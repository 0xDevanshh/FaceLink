"""Playwright browser plumbing shared by every browser-driven adapter."""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Iterator

from ..config import settings

log = logging.getLogger(__name__)

# Minimal, honest hardening: a real UA and locale, plus removing the
# `navigator.webdriver` flag. We are not defeating CAPTCHAs — when an engine
# challenges us, the adapter reports it and the orchestrator moves on.
INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
"""


class BrowserSession:
    """One Chromium instance reused across all engines in a run."""

    def __init__(self, headless: bool | None = None) -> None:
        self.headless = settings.headless if headless is None else headless
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self) -> "BrowserSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = self._browser.new_context(
            user_agent=settings.user_agent,
            locale="en-US",
            timezone_id="Asia/Kolkata",
            viewport={"width": 1440, "height": 900},
            accept_downloads=False,
        )
        self._context.add_init_script(INIT_SCRIPT)
        self._context.set_default_timeout(settings.search_timeout_s * 1000)
        return self

    def __exit__(self, *exc) -> None:
        for closer in (self._context, self._browser):
            with contextlib.suppress(Exception):
                closer.close()
        with contextlib.suppress(Exception):
            self._pw.stop()

    @contextlib.contextmanager
    def page(self) -> Iterator:
        page = self._context.new_page()
        # Block heavy media: results arrive faster and we never need the pixels.
        def _route(route):
            if route.request.resource_type in ("media", "font"):
                return route.abort()
            return route.continue_()

        with contextlib.suppress(Exception):
            page.route("**/*", _route)
        try:
            yield page
        finally:
            with contextlib.suppress(Exception):
                page.close()


def settle(page, extra_ms: int = 1500) -> None:
    """Wait for the network to go quiet, then a beat for lazy render."""
    with contextlib.suppress(Exception):
        page.wait_for_load_state("networkidle", timeout=settings.search_timeout_s * 1000)
    with contextlib.suppress(Exception):
        page.wait_for_timeout(extra_ms)


def scroll_through(page, rounds: int = 3, pause_ms: int = 900) -> None:
    """Trigger lazy-loaded result batches."""
    for _ in range(rounds):
        with contextlib.suppress(Exception):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(pause_ms)


def attach_file(page, path: str, selectors: tuple[str, ...], triggers: tuple[str, ...] = ()) -> bool:
    """Put `path` into the engine's file input.

    Tries the known input selectors first; if none are attached yet, clicks the
    engine's camera/upload trigger to materialise the input and retries. Also
    handles engines that only open a native file chooser (via filechooser).
    """
    def _try_inputs() -> bool:
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    el.set_input_files(path)
                    log.debug("attached file via %s", sel)
                    return True
            except Exception as exc:  # noqa: BLE001
                log.debug("selector %s rejected file: %s", sel, exc)
        return False

    if _try_inputs():
        return True

    for trig in triggers:
        try:
            el = page.query_selector(trig)
            if not el:
                continue
            # Some engines open a native chooser rather than exposing an input.
            try:
                with page.expect_file_chooser(timeout=4000) as fc:
                    el.click()
                fc.value.set_files(path)
                log.debug("attached file via file chooser from %s", trig)
                return True
            except Exception:  # noqa: BLE001
                page.wait_for_timeout(800)
                if _try_inputs():
                    return True
        except Exception as exc:  # noqa: BLE001
            log.debug("trigger %s failed: %s", trig, exc)

    return _try_inputs()


def click_text(page, pattern: str, timeout_ms: int = 4000) -> bool:
    """Click the first visible element whose text matches a regex (case-insensitive)."""
    for role in ("button", "link", "tab"):
        try:
            loc = page.get_by_role(role, name=re.compile(pattern, re.I)).first
            loc.click(timeout=timeout_ms)
            return True
        except Exception:  # noqa: BLE001
            continue
    try:
        page.locator(f"text=/{pattern}/i").first.click(timeout=timeout_ms)
        return True
    except Exception:  # noqa: BLE001
        return False


# Interstitials whose URL alone proves we were challenged.
BLOCK_URL_MARKERS = ("/sorry/", "/showcaptcha", "captcha", "checkcaptcha", "/blocked")


def detect_block(page) -> str:
    """Return a human reason if the engine served a challenge instead of results."""
    try:
        url = (page.url or "").lower()
    except Exception:  # noqa: BLE001
        url = ""
    for marker in BLOCK_URL_MARKERS:
        if marker in url:
            return f"bot challenge interstitial ({marker})"

    try:
        body = (page.inner_text("body") or "").lower()[:4000]
    except Exception:  # noqa: BLE001
        return ""
    signals = {
        "captcha": "CAPTCHA / bot challenge",
        "unusual traffic": "engine flagged unusual traffic",
        "are you a robot": "robot check",
        "confirm that you": "human confirmation required",
        "verify you are human": "human verification required",
        "access denied": "access denied",
        "too many requests": "rate limited",
        # Bing's soft refusal for automated visual search. It looks like a bad
        # image ("try a different image") but it fires for every image and every
        # size, so it is a refusal to serve, not a complaint about the input.
        # Reporting it as CHALLENGED is the accurate description; it is not
        # something to work around.
        "unable to process this search": "engine refused to process the image "
                                        "(automated-request block)",
    }
    for needle, reason in signals.items():
        if needle in body:
            return reason
    return ""
