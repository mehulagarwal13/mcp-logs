"""Backfill `documents.superseded_at` and purge the stale chunks of superseded
document versions.

Run this AFTER the `c3a9d5e7f1b2` migration has been applied and the new
ingestion code is deployed. It is NOT run by any migration, worker or CI job.

Two phases (see `app/ingestion/lifecycle.py` for the full contract):

  mark   set `superseded_at` on every non-latest version of each connector
         document. Metadata only; nothing is deleted. Retrieval and the
         `/knowledge` browse list stop returning those rows immediately.
  purge  delete the chunks/embeddings of superseded documents from all four
         chunk tables. `documents` / `document_metadata` rows are NEVER deleted,
         so old document ids (and citations naming them) keep resolving.

Safe by default
---------------
* DRY RUN unless `--apply` is given: it only reports what it would do.
* You must name the scope explicitly: `--org <uuid>` (repeatable) or `--all-orgs`.
* `--apply` additionally requires `--yes`, and prints the target database
  host/name first so a wrong `DATABASE_URL` is obvious before anything runs.
* Tenant-scoped, short transactions (one per batch), idempotent and resumable:
  just re-run after an interruption.
* Soft-deleted documents' chunks are left alone unless `--include-soft-deleted`.

Usage::

    python scripts/backfill_superseded_documents.py --all-orgs                  # report only
    python scripts/backfill_superseded_documents.py --org <uuid>                # report only
    python scripts/backfill_superseded_documents.py --org <uuid> --apply --yes --phase mark
    python scripts/backfill_superseded_documents.py --org <uuid> --apply --yes --phase purge
    python scripts/backfill_superseded_documents.py --all-orgs --apply --yes    # both phases

Recommended order: dry run -> `--phase mark` on one organization -> verify ->
remaining organizations -> `--phase purge` in a quiet window. Plain `DELETE`
makes space reusable by PostgreSQL; it does not shrink files on disk
(`VACUUM FULL` / `pg_repack` would, and take locks -- not done here).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from sqlalchemy.engine import make_url

from app.database.session import session_scope
from app.ingestion import lifecycle
from app.shared.config.settings import get_settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--org", type=uuid.UUID, action="append", help="organization id (repeatable)"
    )
    scope.add_argument("--all-orgs", action="store_true", help="every organization")
    parser.add_argument("--apply", action="store_true", help="perform writes (default: dry run)")
    parser.add_argument("--yes", action="store_true", help="required together with --apply")
    parser.add_argument("--phase", choices=["mark", "purge", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between batches")
    parser.add_argument(
        "--include-soft-deleted",
        action="store_true",
        help="also purge chunks of soft-deleted documents (default: superseded only)",
    )
    return parser.parse_args(argv)


def _describe_target() -> str:
    url = make_url(str(get_settings().database_url))
    return f"host={url.host} port={url.port} database={url.database} user={url.username}"


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.apply and not args.yes:
        print("--apply requires --yes (this will modify the database).", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    print(f"Target: {_describe_target()}")
    print("Mode:   " + ("APPLY" if args.apply else "DRY RUN (no writes)"))

    if args.all_orgs:
        async with session_scope() as session:
            organization_ids = await lifecycle.list_organization_ids(session)
    else:
        organization_ids = list(args.org)
    print(f"Organizations in scope: {len(organization_ids)}")

    report = await lifecycle.run_backfill(
        session_scope,
        organization_ids=organization_ids,
        apply=args.apply,
        phase=args.phase,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
        include_soft_deleted=args.include_soft_deleted,
    )

    total_to_mark = sum(o.versions_to_mark for o in report.organizations)
    total_chunks = sum(sum(o.chunks_to_purge.values()) for o in report.organizations)
    if not args.apply:
        print(
            f"\nDRY RUN summary: {total_to_mark} version(s) would be marked superseded; "
            f"{total_chunks} chunk row(s) currently belong to already-superseded documents "
            "(more become purgeable once marking runs). Re-run with --apply --yes to execute."
        )
        return 0

    leftovers = [
        o.organization_id
        for o in report.organizations
        if o.keys_still_with_multiple_current
    ]
    print(
        f"\nDone. marked={sum(o.versions_marked for o in report.organizations)} "
        f"purged={sum(sum(o.chunks_purged.values()) for o in report.organizations)}"
    )
    if leftovers and args.phase in {"mark", "all"}:
        print(f"WARNING: organizations still showing documents with >1 current row: {leftovers}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
