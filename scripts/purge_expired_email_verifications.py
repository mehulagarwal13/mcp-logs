"""Housekeeping: delete abandoned email-verification challenges (pending
signups / invitation acceptances / legacy re-verification links) that expired
or were consumed more than `--retention-days` ago.

Never touches `users`, `organizations`, `invitations` or anything else, and an
expired challenge is already inert (consumption requires `expires_at > now()`),
so skipping this only costs a little disk. Dry run unless `--apply`.

    python scripts/purge_expired_email_verifications.py
    python scripts/purge_expired_email_verifications.py --apply --retention-days 7
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, select

from app.core.auth import email_verification
from app.database.models.auth_models import EmailVerification
from app.database.session import session_scope


async def _run(apply: bool, retention_days: int) -> int:
    retention = timedelta(days=retention_days)
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        count = (
            await session.execute(
                select(func.count()).select_from(EmailVerification).where(
                    or_(
                        EmailVerification.expires_at < now - retention,
                        EmailVerification.consumed_at < now - retention,
                    )
                )
            )
        ).scalar_one()
        print(f"{count} stale challenge(s) older than {retention_days} day(s).")
        if apply and count:
            deleted = await email_verification.purge_stale(session, now=now, retention=retention)
            print(f"Deleted {deleted}.")
        elif not apply:
            print("Dry run: nothing deleted (use --apply).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retention-days", type=int, default=7)
    ns = parser.parse_args()
    sys.exit(asyncio.run(_run(ns.apply, ns.retention_days)))
