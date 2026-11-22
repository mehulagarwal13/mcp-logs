"""Retention: delete `mcp_requests` rows older than `MCP_REQUEST_RETENTION_DAYS`
(default 90) across ALL tenants, including calls that failed before any identity
was resolved (which no tenant can see or delete).

Operator-run; nothing calls this automatically. Dry run unless `--apply` -- a dry
run only prints the cutoff, because the application role cannot count other
tenants' rows (that is the point of row-level security). Deletion goes through
the `purge_mcp_requests` SECURITY DEFINER function.

    python scripts/purge_mcp_requests.py
    python scripts/purge_mcp_requests.py --apply
    python scripts/purge_mcp_requests.py --apply --retention-days 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta

from app.core.observability import service as observability_service
from app.database.session import session_scope
from app.shared.config.settings import get_settings


async def _run(apply: bool, retention_days: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    if not apply:
        print(
            f"Dry run: would delete mcp_requests rows older than {cutoff.isoformat()} "
            f"({retention_days} days). Pass --apply to delete."
        )
        return 0
    async with session_scope() as session:
        deleted = await observability_service.purge_expired_mcp_requests(
            session, retention_days=retention_days
        )
    print(f"Deleted {deleted} mcp_requests rows older than {cutoff.isoformat()}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="actually delete (default is a dry run)"
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=get_settings().mcp_request_retention_days,
        help="keep this many days (default: MCP_REQUEST_RETENTION_DAYS)",
    )
    args = parser.parse_args()
    if args.retention_days < 1:
        parser.error("--retention-days must be >= 1")
    return asyncio.run(_run(args.apply, args.retention_days))


if __name__ == "__main__":
    sys.exit(main())
