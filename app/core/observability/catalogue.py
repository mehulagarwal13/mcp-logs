"""Derive a tool's parameter list from its JSON Schema (what MCP `tools/list`
reports as `inputSchema`). Pure function, no I/O."""

from __future__ import annotations

import json
from typing import Any

from app.core.observability.schemas import McpToolParameter

_TYPE_MAP = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}


def _json_type(schema: dict[str, Any]) -> str:
    """The first non-null JSON type of a property schema (`T | None` is an
    `anyOf` with a `null` branch)."""
    declared = schema.get("type")
    candidates: list[Any] = [declared] if isinstance(declared, str) else list(declared or [])
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        if isinstance(branch, dict):
            branch_type = branch.get("type")
            candidates.extend([branch_type] if isinstance(branch_type, str) else branch_type or [])
    for candidate in candidates:
        if candidate in _TYPE_MAP:
            return _TYPE_MAP[candidate]
    return "string"


def _render_default(prop: dict[str, Any]) -> str | None:
    """The default as display text ('10', 'true', 'code'); None if there is none."""
    default = prop.get("default")
    if "default" not in prop or default is None:
        return None
    return default if isinstance(default, str) else json.dumps(default)


def parameters_from_schema(schema: dict[str, Any]) -> list[McpToolParameter]:
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}
    parameters = []
    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        parameters.append(
            McpToolParameter(
                name=name,
                type=_json_type(prop),  # type: ignore[arg-type]
                required=name in required,
                description=str(prop.get("description") or ""),
                default=_render_default(prop),
            )
        )
    return parameters
