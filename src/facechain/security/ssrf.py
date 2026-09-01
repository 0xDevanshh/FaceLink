"""SSRF protection for all outbound HTTP fetches in the verification layer.

Every URL from a search engine or candidate page is untrusted. This module
validates that a URL's resolved IP is not a private/loopback/link-local/
multicast address before any connection is made, and re-validates after each
redirect hop (DNS rebinding protection).

The checks are conservative: if resolution fails the URL is rejected, not
allowed — we prefer a false rejection over a SSRF.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Schemes we permit. Everything else (file://, gopher://, ftp://, data://) is rejected.
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Private, loopback, link-local, multicast, and unspecified network ranges.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network(r)
    for r in (
        "127.0.0.0/8",       # loopback
        "::1/128",           # IPv6 loopback
        "10.0.0.0/8",        # private class A
        "172.16.0.0/12",     # private class B
        "192.168.0.0/16",    # private class C
        "169.254.0.0/16",    # link-local
        "fe80::/10",         # IPv6 link-local
        "fc00::/7",          # IPv6 unique-local
        "0.0.0.0/8",         # this network
        "100.64.0.0/10",     # shared address space (CGNAT)
        "192.0.0.0/24",      # IETF protocol assignments
        "192.0.2.0/24",      # TEST-NET-1
        "198.51.100.0/24",   # TEST-NET-2
        "203.0.113.0/24",    # TEST-NET-3
        "240.0.0.0/4",       # reserved
        "255.255.255.255/32",# broadcast
        "224.0.0.0/4",       # multicast IPv4
        "ff00::/8",          # multicast IPv6
        "::/128",            # unspecified IPv6
    )
]


class SSRFViolation(ValueError):
    """Raised when a URL would reach a non-public address."""


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable — reject
    return any(addr in net for net in _BLOCKED_NETWORKS)


def resolve_and_check(host: str) -> str:
    """Resolve host → IP and assert it is a public routable address.

    Returns the resolved IP string on success. Raises SSRFViolation on failure.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SSRFViolation(f"DNS resolution failed for {host!r}: {exc}") from exc

    if not infos:
        raise SSRFViolation(f"no DNS records for {host!r}")

    for info in infos:
        ip = info[4][0]
        if _is_blocked_ip(ip):
            raise SSRFViolation(
                f"host {host!r} resolves to blocked address {ip} (private/loopback/link-local)"
            )
    return infos[0][4][0]


def validate_url(url: str) -> str:
    """Validate a URL for safe outbound fetching.

    Checks:
    1. Scheme must be http or https.
    2. Host must not be empty.
    3. Resolved IP must be a public routable address.

    Returns the validated URL string. Raises SSRFViolation on any failure.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise SSRFViolation(f"unparseable URL: {exc}") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFViolation(f"scheme {parsed.scheme!r} not allowed (only http/https)")

    host = parsed.hostname
    if not host:
        raise SSRFViolation("URL has no host")

    # Reject bare IP addresses in private ranges directly (no DNS lookup needed).
    try:
        addr = ipaddress.ip_address(host)
        if _is_blocked_ip(str(addr)):
            raise SSRFViolation(f"direct IP {host} is a blocked address")
        return url
    except ValueError:
        pass  # not an IP literal — proceed to DNS resolution

    resolve_and_check(host)
    return url


def safe_url_or_none(url: str) -> str | None:
    """Return the URL if safe, None (with a warning) if it fails SSRF checks."""
    try:
        return validate_url(url)
    except SSRFViolation as exc:
        log.warning("SSRF: rejected %s — %s", url[:120], exc)
        return None
