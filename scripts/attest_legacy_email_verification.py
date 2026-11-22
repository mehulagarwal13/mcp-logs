"""Operator tool: explicitly attest that specific LEGACY users' email addresses
are theirs, without making them click a verification link.

Context: the email-verification migration (`d5f2a8c1e9b4`) deliberately leaves
every pre-existing user UNVERIFIED -- nothing that happened before it is
defensible evidence of mailbox ownership. Those users can re-verify themselves
(login -> "resend verification email"). This script is for the cases where an
operator has their OWN evidence (e.g. the addresses all belong to your company
domain and you control it) and wants to skip the round trip. It is never run
by a migration, worker or CI job, and never automatic.

Safe by default
---------------
* DRY RUN unless `--apply` is given: lists who would be attested.
* You must name the scope: `--email <addr>` (repeatable) and/or
  `--domain <example.com>` (repeatable). There is no "everyone" option.
* `--apply` additionally requires `--yes` and `--reason "<why you trust this>"`,
  and prints the target database first. The reason is printed and logged.
* Only touches `users.email_verified_at/email_verified_via` of rows that are
  currently unverified. Idempotent. Never changes passwords, roles, orgs.
* Attested accounts' pre-existing refresh tokens are left valid: you are
  vouching for the account holder. (A link-verified account revokes them.)
  Run `POST /users/{id}/logout-all` afterwards if you doubt the sessions.

Usage::

    python scripts/attest_legacy_email_verification.py --domain example.com
    python scripts/attest_legacy_email_verification.py --domain example.com \
        --apply --yes --reason "all example.com mailboxes are company-managed"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import make_url

from app.database.models.core_models import User
from app.database.session import session_scope
from app.shared.config.settings import get_settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--email", action="append", default=[], help="exact address (repeatable)")
    parser.add_argument("--domain", action="append", default=[], help="email domain (repeatable)")
    parser.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    parser.add_argument("--yes", action="store_true", help="required together with --apply")
    parser.add_argument("--reason", help="why you trust these addresses (required with --apply)")
    args = parser.parse_args(argv)
    if not args.email and not args.domain:
        parser.error("name a scope: --email and/or --domain")
    if args.apply and not (args.yes and args.reason):
        parser.error("--apply requires --yes and --reason")
    return args


def _scope_filter(emails: list[str], domains: list[str]):
    clauses = [func.lower(User.email) == e.strip().lower() for e in emails]
    clauses += [func.lower(User.email).like(f"%@{d.strip().lower().lstrip('@')}") for d in domains]
    return or_(*clauses)


async def _run(args: argparse.Namespace) -> int:
    url = make_url(str(get_settings().database_url))
    print(f"Target database: host={url.host} name={url.database}")
    print("Mode: APPLY" if args.apply else "Mode: DRY RUN (nothing will be written)")

    async with session_scope() as session:
        stmt = select(User.id, User.email).where(
            User.email_verified_at.is_(None), _scope_filter(args.email, args.domain)
        )
        rows = (await session.execute(stmt)).all()
        print(f"{len(rows)} unverified user(s) in scope:")
        for _, email in rows:
            print(f"  - {email}")
        if not args.apply or not rows:
            return 0

        now = datetime.now(timezone.utc)
        result = await session.execute(
            update(User)
            .where(User.id.in_([r[0] for r in rows]), User.email_verified_at.is_(None))
            .values(email_verified_at=now, email_verified_via="operator_attestation")
        )
        print(f"Attested {result.rowcount} user(s). Reason: {args.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args())))
