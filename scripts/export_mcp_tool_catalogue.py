"""Regenerate the frontend's mock-mode MCP tool catalogue from the REAL registration.

The MCP server's `@mcp_server.tool()` registrations are the source of truth. Mock
mode (`VITE_USE_MOCK_DATA=true`) must show the same tools, so its catalogue is not
hand-written: it is this generated snapshot, and `tests/mcp/test_tool_catalogue.py`
fails if the snapshot drifts from the live registration.

    python scripts/export_mcp_tool_catalogue.py            # rewrite the snapshot
    python scripts/export_mcp_tool_catalogue.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.core.observability.catalogue import parameters_from_schema
from app.mcp.catalogue import describe_registered_tools
from app.mcp.servers import main as mcp_assembly  # registers every tool on import

SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "frontend/src/mocks/data/mcpToolCatalogue.generated.json"
)


async def build_snapshot() -> list[dict]:
    """The catalogue in the shape the frontend receives from the API (camelCase)."""
    specs = await describe_registered_tools(mcp_assembly.mcp_server)
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description,
                    "default": p.default,
                }
                for p in parameters_from_schema(spec.input_schema)
            ],
        }
        for spec in specs
    ]


def render(snapshot: list[dict]) -> str:
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the snapshot is stale")
    args = parser.parse_args()
    expected = render(asyncio.run(build_snapshot()))
    if args.check:
        current = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
        if current.replace("\r\n", "\n") != expected:
            print(f"{SNAPSHOT} is stale; run scripts/export_mcp_tool_catalogue.py", file=sys.stderr)
            return 1
        print("mock MCP tool catalogue is up to date")
        return 0
    SNAPSHOT.write_text(expected, encoding="utf-8")
    print(f"wrote {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
