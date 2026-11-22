"""Which OAuth redirect URIs EKIP will ever send an authorization code to.

Dynamic client registration is unauthenticated, so the redirect URI a client
registers is attacker-controlled input. A code (and, through the sign-in page,
a fresh EKIP session) is delivered to that URI, so registration only accepts:

* `https` URIs whose host is explicitly allow-listed
  (`Settings.mcp_oauth_allowed_redirect_hosts`, default Claude's own hosts);
* `http` URIs only on a loopback host (RFC 8252 section 7.3 -- native apps and
  `mcp-remote` listen on `http://localhost:<port>/...`), any port.

Everything else is refused: other schemes (`javascript:`, `data:`, custom
schemes), plain `http` on a non-loopback host, other `https` hosts, URIs with
embedded credentials, and URIs with a fragment (RFC 6749 section 3.1.2).
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def redirect_uri_violation(uri: str, allowed_https_hosts: Iterable[str]) -> str | None:
    """Why `uri` is not acceptable, or `None` if it is."""
    try:
        parts = urlsplit(str(uri))
        host = (parts.hostname or "").lower()
        parts.port  # noqa: B018 -- raises ValueError on a malformed port
    except ValueError:
        return "is not a valid URL"
    if parts.fragment:
        return "must not contain a fragment"
    if parts.username is not None or parts.password is not None:
        return "must not contain credentials"
    if not host:
        return "must include a host"
    if parts.scheme == "https":
        if host in {h.strip().lower() for h in allowed_https_hosts}:
            return None
        return f"host '{host}' is not on the allowed redirect host list"
    if parts.scheme == "http":
        if host in _LOOPBACK_HOSTS:
            return None
        return "http is only allowed for loopback hosts (localhost, 127.0.0.1, [::1])"
    return f"scheme '{parts.scheme}' is not allowed"


def first_redirect_uri_violation(
    uris: Iterable[object], allowed_https_hosts: Iterable[str]
) -> str | None:
    allowed = list(allowed_https_hosts)
    for uri in uris:
        problem = redirect_uri_violation(str(uri), allowed)
        if problem:
            return f"redirect_uri '{uri}' {problem}"
    return None
