"""Data minimization for `mcp_requests.request_summary`.

Every MCP tool hands `run_mcp_tool` a `request_summary` that is, today, mostly
the raw tool arguments: a user's question, an incident description, an invitee's
email address, an SSO issuer URL. That is request *content*, and persisting it
verbatim turns an observability table into a store of sensitive text with no
retention and no erasure path. The table only needs to answer "which tool, who,
how big, how long, did it work" -- so this module reduces a summary to exactly
that before it is written, in one place (`service.record_mcp_request`), so no
tool handler can bypass it by accident.

Policy (default-deny):

* Keys in `_KEPT_KEYS` are identifiers or small enumerations/booleans/dates --
  incident/document ids, a collection or role name, a rule type -- and are kept
  (strings truncated to `_MAX_KEPT_LENGTH`).
* `email` becomes `email_domain`; `issuer_url` becomes `issuer_host`.
* Every other string becomes `<key>_chars` (its length); every other
  collection becomes `<key>_items` (its size); other booleans/numbers/None are
  kept as they are.
* Nothing that was not explicitly allowed can survive as text.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

_MAX_KEPT_LENGTH = 128

#: Identifiers, enumerations and dates. Deliberately not free text.
_KEPT_KEYS = frozenset(
    {
        "incident_id",
        "document_id",
        "source_incident_id",
        "collection",
        "repository",
        "since",
        "rule_type",
        "grants_role",
        "provider",
        "is_default",
    }
)


def _email_domain(value: str) -> str | None:
    _, _, domain = value.rpartition("@")
    domain = domain.strip().lower()
    return domain[:_MAX_KEPT_LENGTH] if "@" in value and domain else None


def _url_host(value: str) -> str | None:
    try:
        host = urlsplit(value).hostname
    except ValueError:
        return None
    return host.lower()[:_MAX_KEPT_LENGTH] if host else None


def minimize_request_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """The persistable form of a tool's request summary (see module docstring)."""
    if summary is None:
        return None

    minimized: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "email" and isinstance(value, str):
            domain = _email_domain(value)
            if domain is not None:
                minimized["email_domain"] = domain
            minimized["email_chars"] = len(value)
        elif key == "issuer_url" and isinstance(value, str):
            host = _url_host(value)
            if host is not None:
                minimized["issuer_host"] = host
        elif key in _KEPT_KEYS and (value is None or isinstance(value, (bool, int, float))):
            minimized[key] = value
        elif key in _KEPT_KEYS and isinstance(value, str):
            minimized[key] = value[:_MAX_KEPT_LENGTH]
        elif isinstance(value, str):
            minimized[f"{key}_chars"] = len(value)
        elif isinstance(value, (list, tuple, set, frozenset, dict)):
            minimized[f"{key}_items"] = len(value)
        elif value is None or isinstance(value, (bool, int, float)):
            minimized[key] = value
        # anything else (arbitrary objects) is dropped
    return minimized
