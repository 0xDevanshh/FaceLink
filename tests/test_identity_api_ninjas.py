"""API Ninjas celebrity enrichment — offline, `httpx.get` monkeypatched.

Never makes a real request: every test stubs `httpx.get` directly. Covers
the mission's exact test list: success, schema mapping, multiple results,
empty response, timeout, 401/403, 429, malformed JSON, missing key, cache
hit, and that metadata can never override the already-resolved identity.
"""

from __future__ import annotations

import httpx
import pytest

from facechain.config import settings
from facechain.identity import api_ninjas
from facechain.identity.api_ninjas import _normalise_name, _select_best_match, fetch_celebrity_info


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    api_ninjas.reset_cache()
    monkeypatch.setattr(settings, "api_ninjas_api_key", "test-key")
    yield
    api_ninjas.reset_cache()


def _fake_get(payload=None, status_code=200, raise_exc=None):
    def _get(url, params=None, headers=None, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(status_code, json=payload, request=httpx.Request("GET", url))
    return _get


# ---- missing key ------------------------------------------------------

def test_missing_api_key_is_a_clean_no_op(monkeypatch):
    monkeypatch.setattr(settings, "api_ninjas_api_key", "")
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.available is False
    assert result.source == "api_ninjas"


def test_empty_resolved_name_is_a_clean_no_op():
    assert fetch_celebrity_info("").available is False
    assert fetch_celebrity_info(None).available is False  # type: ignore[arg-type]


# ---- successful response + exact schema mapping ------------------------

def test_successful_response_maps_every_field(monkeypatch):
    payload = [{
        "name": "sundar pichai", "net_worth": 1500000000, "gender": "male",
        "nationality": "in", "occupation": ["businessperson", "executive"],
        "height": 1.73, "birthday": "1972-07-12", "age": 54, "is_alive": True,
    }]
    monkeypatch.setattr(httpx, "get", _fake_get(payload))
    result = fetch_celebrity_info("Sundar Pichai")

    assert result.available is True
    assert result.name == "Sundar Pichai"  # the RESOLVED name, not API Ninjas' own casing
    assert result.nationality == "in"
    assert result.occupations == ["businessperson", "executive"]
    assert result.birthday == "1972-07-12"
    assert result.age == 54
    assert result.gender == "male"
    assert result.height == 1.73
    assert result.net_worth == 1500000000
    assert result.is_alive is True
    assert result.source == "api_ninjas"


def test_missing_and_null_fields_do_not_crash(monkeypatch):
    payload = [{"name": "sundar pichai", "occupation": None, "age": None}]
    monkeypatch.setattr(httpx, "get", _fake_get(payload))
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.available is True
    assert result.occupations == []
    assert result.age is None
    assert result.nationality is None


# ---- multiple results: never blindly take the first --------------------

def test_multiple_results_selects_the_best_name_match(monkeypatch):
    payload = [
        {"name": "someone else entirely", "age": 30},
        {"name": "sundar pichai", "age": 54},
    ]
    monkeypatch.setattr(httpx, "get", _fake_get(payload))
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.available is True
    assert result.age == 54


def test_no_plausible_match_among_multiple_results_is_unavailable(monkeypatch):
    payload = [{"name": "a totally different person"}, {"name": "another unrelated name"}]
    monkeypatch.setattr(httpx, "get", _fake_get(payload))
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.available is False


# ---- empty / malformed responses ---------------------------------------

def test_empty_array_response_is_unavailable(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get([]))
    assert fetch_celebrity_info("Sundar Pichai").available is False


def test_malformed_json_does_not_crash(monkeypatch):
    def _get(url, params=None, headers=None, timeout=None):
        return httpx.Response(200, content=b"not json at all", request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", _get)
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.available is False


def test_response_that_is_not_a_list_is_unavailable(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"error": "unexpected shape"}))
    assert fetch_celebrity_info("Sundar Pichai").available is False


# ---- HTTP error handling -------------------------------------------------

@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_unavailable_not_a_crash(monkeypatch, status):
    monkeypatch.setattr(httpx, "get", _fake_get([{"name": "sundar pichai"}], status_code=status))
    assert fetch_celebrity_info("Sundar Pichai").available is False


def test_rate_limit_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get([{"name": "sundar pichai"}], status_code=429))
    assert fetch_celebrity_info("Sundar Pichai").available is False


def test_server_error_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get([{"name": "sundar pichai"}], status_code=500))
    assert fetch_celebrity_info("Sundar Pichai").available is False


def test_timeout_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(raise_exc=httpx.TimeoutException("timed out")))
    assert fetch_celebrity_info("Sundar Pichai").available is False


def test_connection_error_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(raise_exc=httpx.ConnectError("refused")))
    assert fetch_celebrity_info("Sundar Pichai").available is False


# ---- caching -------------------------------------------------------------

def test_a_second_lookup_for_the_same_name_is_served_from_cache(monkeypatch):
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(params["name"])
        return httpx.Response(200, json=[{"name": "sundar pichai", "age": 54}], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _get)
    first = fetch_celebrity_info("Sundar Pichai")
    second = fetch_celebrity_info("Sundar Pichai")
    assert len(calls) == 1  # only the first call hit the network
    assert first == second


def test_cache_key_normalises_case_and_whitespace(monkeypatch):
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return httpx.Response(200, json=[{"name": "sundar pichai"}], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _get)
    fetch_celebrity_info("Sundar Pichai")
    fetch_celebrity_info("  SUNDAR   PICHAI  ")
    assert len(calls) == 1


def test_negative_results_are_also_cached(monkeypatch):
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return httpx.Response(200, json=[], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _get)
    fetch_celebrity_info("Nobody Notable")
    fetch_celebrity_info("Nobody Notable")
    assert len(calls) == 1


def test_expired_cache_entry_is_refetched(monkeypatch):
    monkeypatch.setattr(settings, "api_ninjas_cache_ttl_s", 0.0)
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append(1)
        return httpx.Response(200, json=[{"name": "sundar pichai"}], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _get)
    fetch_celebrity_info("Sundar Pichai")
    fetch_celebrity_info("Sundar Pichai")
    assert len(calls) == 2


# ---- name normalisation / matching helpers --------------------------------

def test_normalise_name_strips_case_whitespace_and_punctuation():
    assert _normalise_name("  Sundar   Pichai! ") == "sundar pichai"
    assert _normalise_name("J.K. Rowling") == "jk rowling"


def test_select_best_match_prefers_exact_normalised_match():
    results = [{"name": "j k rowling"}, {"name": "J.K. Rowling"}]
    best = _select_best_match(results, "J.K. Rowling")
    assert best["name"] == "J.K. Rowling"


# ---- metadata can never override the resolved identity --------------------

def test_celebrity_metadata_never_changes_the_resolved_name(monkeypatch):
    """API Ninjas' own (possibly differently-cased/spelled) name must never
    replace the name the face-evidence identity layer already resolved."""
    payload = [{"name": "SUNDAR PICHAI (Google CEO)", "age": 54}]
    monkeypatch.setattr(httpx, "get", _fake_get(payload))
    result = fetch_celebrity_info("Sundar Pichai")
    assert result.name == "Sundar Pichai"


def test_a_failed_lookup_cannot_affect_the_caller_supplied_name(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(raise_exc=httpx.ConnectError("boom")))
    result = fetch_celebrity_info("Sundar Pichai")
    # available=False and everything else empty — nothing here could ever
    # be mistaken for a name/identity assertion.
    assert result.available is False
    assert result.name is None
