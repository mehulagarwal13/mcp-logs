"""Constants shared by the MCP OAuth bridge and the MCP identity boundary."""

from __future__ import annotations

#: The single OAuth scope EKIP's MCP server issues and requires. It is
#: advertised in authorization-server metadata, granted by default at
#: `/authorize`, returned in every token response, carried in the access-token
#: JWT, and required by both the transport (`AuthSettings.required_scopes`) and
#: the per-call identity check (`app.mcp.auth.resolve_mcp_identity`).
MCP_SCOPE = "ekip"


def mcp_resource_identifier() -> str:
    """The OAuth resource identifier (`aud`) and issuer (`iss`) of MCP-issued
    access tokens: this server's public base URL, exactly as advertised in the
    authorization-server and protected-resource metadata (no trailing slash)."""
    from app.shared.config.settings import get_settings

    return get_settings().mcp_public_base_url.rstrip("/")


def accepted_resource_indicators() -> frozenset[str]:
    """RFC 8707 `resource` values a client may send at `/authorize`: the
    identifier itself, or the MCP endpoint under it (what a client derives from
    the connector URL), with or without a trailing slash."""
    base = mcp_resource_identifier()
    return frozenset({base, f"{base}/", f"{base}/mcp", f"{base}/mcp/"})
