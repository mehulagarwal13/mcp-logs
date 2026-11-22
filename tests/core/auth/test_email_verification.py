"""Unit tests for `core.auth.email_verification` and `app.shared.email`.

Everything that needs a real database (single-use atomicity, RLS, concurrency,
rollback) lives in `test_email_verification_postgres.py`. These cover the pure
logic and the delivery failure semantics with fakes.
"""

from __future__ import annotations

import logging
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.core.auth import email_verification as ev
from app.core.exceptions import ValidationError
from app.shared.email import EmailDeliveryError, EmailMessage, EmailNotConfiguredError
from app.shared.email.smtp import SmtpEmailSender
from app.shared.email.templates import render_verification_email, verification_link


def _delivery(token="raw-secret-token-value"):
    return ev.PendingDelivery(uuid.uuid4(), "registration", "person@example.com", token)


# --------------------------------------------------------------- email address


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Person@Example.COM ", "person@example.com"), ("a.b+tag@sub.example.org", "a.b+tag@sub.example.org")],
)
def test_normalize_email_strips_and_lowercases(raw, expected) -> None:
    assert ev.normalize_email(raw) == expected


def test_normalize_email_strips_surrounding_whitespace_including_newlines() -> None:
    assert ev.normalize_email("a@b.co\n") == "a@b.co"  # trailing whitespace is trimmed, not an injection vector


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "no-at-sign", "two@@example.com", "a b@example.com", "x@nodot", "a\n@b.co", "a@b.co\nBcc: x@y.z", "@example.com", "a" * 250 + "@x.io"],
)
def test_normalize_email_rejects_malformed_addresses(bad) -> None:
    with pytest.raises(ValidationError) as exc:
        ev.normalize_email(bad)
    assert exc.value.error_code == "auth.invalid_email"


# --------------------------------------------------------------- rate limits


def _row(*, last_sent_at=None, window=None, count=0):
    return SimpleNamespace(
        last_sent_at=last_sent_at, send_window_started_at=window, send_count=count
    )


def test_may_send_enforces_the_cooldown() -> None:
    now = datetime.now(timezone.utc)
    assert ev._may_send(_row(), now) is True
    assert ev._may_send(_row(last_sent_at=now - timedelta(seconds=10), window=now, count=1), now) is False
    assert ev._may_send(_row(last_sent_at=now - timedelta(seconds=3600), window=now - timedelta(seconds=3600), count=1), now) is True


def test_may_send_enforces_the_daily_cap_and_resets_after_the_window() -> None:
    from app.shared.config.settings import get_settings

    cap = get_settings().email_verification_max_sends_per_day
    now = datetime.now(timezone.utc)
    long_ago = now - timedelta(hours=2)
    assert ev._may_send(_row(last_sent_at=long_ago, window=long_ago, count=cap), now) is False
    # a 25-hour-old window no longer counts
    old = now - timedelta(hours=25)
    assert ev._may_send(_row(last_sent_at=old, window=old, count=cap), now) is True


def test_rotate_replaces_the_token_hash_extends_expiry_and_counts_the_send() -> None:
    now = datetime.now(timezone.utc)
    row = SimpleNamespace(
        token_hash="old-hash",
        expires_at=now,
        last_sent_at=None,
        send_window_started_at=None,
        send_count=0,
        display_name="old",
    )

    ev._rotate(row, raw_token="new-raw", now=now, payload={"display_name": "new"})

    from app.shared.security import hash_opaque_token

    assert row.token_hash == hash_opaque_token("new-raw") != "old-hash"
    assert "new-raw" not in str(vars(row))  # only the hash is kept on the row
    assert row.expires_at > now and row.send_count == 1 and row.last_sent_at == now
    assert row.display_name == "new"


# --------------------------------------------------------------- PendingDelivery / messages


def test_pending_delivery_repr_never_contains_the_token() -> None:
    delivery = _delivery("super-secret-token")
    assert "super-secret-token" not in repr(delivery)
    assert "super-secret-token" not in str(delivery)


def test_email_message_repr_redacts_recipient_and_body() -> None:
    msg = EmailMessage(to="victim@example.com", subject="s", text="LINK-WITH-TOKEN", html="<a>x</a>")
    assert "victim@example.com" not in repr(msg) and "LINK-WITH-TOKEN" not in repr(msg)


def test_verification_link_puts_the_token_in_the_fragment_not_the_query() -> None:
    link = verification_link("https://app.example.com/", "tok123")
    assert link == "https://app.example.com/verify-email#token=tok123"
    assert "?" not in link  # fragments are never sent to servers or in Referer


@pytest.mark.parametrize("kind", ["registration", "invitation", "existing_user"])
def test_rendered_mail_contains_the_link_and_a_safety_notice(kind) -> None:
    link = "https://app.example.com/verify-email#token=abc&x=<y>"
    msg = render_verification_email(to="a@b.co", link=link, ttl_minutes=1440, kind=kind)
    assert link in msg.text
    assert "ignore this email" in msg.text
    assert "&amp;" in msg.html and "<y>" not in msg.html  # html-escaped, no injection through the link
    assert "24 hour" in msg.text


# --------------------------------------------------------------- delivery semantics


class _FakeSender:
    def __init__(self, failures=0, exc=None):
        self.failures = failures
        self.exc = exc or EmailDeliveryError("SMTP delivery failed (SMTPServerDisconnected).")
        self.sent: list[EmailMessage] = []
        self.attempts = 0

    async def send(self, message):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.exc
        self.sent.append(message)


async def _no_sleep(_):
    return None


@pytest.mark.asyncio
async def test_delivery_sends_a_message_with_the_link_to_the_right_address(monkeypatch) -> None:
    monkeypatch.setattr(ev, "get_settings", lambda: SimpleNamespace(
        frontend_public_base_url="https://app.example.com", email_verification_token_ttl_minutes=1440))
    sender = _FakeSender()

    ok = await ev.deliver_verification_email(_delivery("tok-1"), sender=sender, sleep=_no_sleep)

    assert ok is True
    assert sender.sent[0].to == "person@example.com"
    assert "https://app.example.com/verify-email#token=tok-1" in sender.sent[0].text


@pytest.mark.asyncio
async def test_delivery_retries_transient_provider_failures_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(ev, "get_settings", lambda: SimpleNamespace(
        frontend_public_base_url="https://app.example.com", email_verification_token_ttl_minutes=1440))
    sender = _FakeSender(failures=2)
    sleeps: list[float] = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    ok = await ev.deliver_verification_email(_delivery(), sender=sender, sleep=record_sleep)

    assert ok is True and sender.attempts == 3
    assert sleeps == [2, 4]  # exponential backoff


@pytest.mark.asyncio
async def test_delivery_failure_is_reported_as_failure_never_as_success(monkeypatch, caplog) -> None:
    """Provider outage: after bounded retries the function returns False and logs an
    error -- it never claims delivery, and never logs the token, link or address."""
    monkeypatch.setattr(ev, "get_settings", lambda: SimpleNamespace(
        frontend_public_base_url="https://app.example.com", email_verification_token_ttl_minutes=1440))
    sender = _FakeSender(failures=99)

    with caplog.at_level(logging.DEBUG):
        ok = await ev.deliver_verification_email(
            _delivery("TOKEN-MUST-NOT-LEAK"), sender=sender, sleep=_no_sleep
        )

    assert ok is False and sender.attempts == 3 and sender.sent == []
    logged = caplog.text
    assert "TOKEN-MUST-NOT-LEAK" not in logged
    assert "person@example.com" not in logged
    assert "verify-email" not in logged


@pytest.mark.asyncio
async def test_unconfigured_provider_is_not_retried(monkeypatch) -> None:
    monkeypatch.setattr(ev, "get_settings", lambda: SimpleNamespace(
        frontend_public_base_url="https://app.example.com", email_verification_token_ttl_minutes=1440))
    sender = _FakeSender(failures=99, exc=EmailNotConfiguredError("no provider"))

    ok = await ev.deliver_verification_email(_delivery(), sender=sender, sleep=_no_sleep)

    assert ok is False and sender.attempts == 1


@pytest.mark.asyncio
async def test_duplicate_delivery_is_safe_both_mails_carry_the_same_single_use_token(monkeypatch) -> None:
    monkeypatch.setattr(ev, "get_settings", lambda: SimpleNamespace(
        frontend_public_base_url="https://app.example.com", email_verification_token_ttl_minutes=1440))
    sender = _FakeSender()
    delivery = _delivery("same-token")

    await ev.deliver_verification_email(delivery, sender=sender, sleep=_no_sleep)
    await ev.deliver_verification_email(delivery, sender=sender, sleep=_no_sleep)

    assert len(sender.sent) == 2 and sender.sent[0].text == sender.sent[1].text
    # (the second *use* of that one token is rejected as already_used -- proven against
    # real Postgres in test_reused_token_is_rejected_and_creates_nothing_more)


# --------------------------------------------------------------- SMTP sender


def _smtp(**over):
    kwargs = dict(
        host="smtp.example.com", port=587, username="u", password="p",
        security="starttls", from_address="no-reply@example.com", from_name="EKIP", timeout=5.0,
    )
    kwargs.update(over)
    return SmtpEmailSender(**kwargs)


def test_smtp_refuses_plaintext_to_a_non_loopback_host() -> None:
    with pytest.raises(EmailNotConfiguredError):
        _smtp(security="none")


def test_smtp_allows_plaintext_only_for_loopback() -> None:
    _smtp(host="127.0.0.1", security="none")  # local Mailpit: fine


@pytest.mark.parametrize("missing", [{"host": ""}, {"from_address": ""}])
def test_smtp_requires_host_and_from_address(missing) -> None:
    with pytest.raises(EmailNotConfiguredError):
        _smtp(**missing)


@pytest.mark.asyncio
async def test_smtp_translates_provider_errors_without_leaking_content(monkeypatch) -> None:
    sender = _smtp()

    def boom(self, message):
        raise smtplib.SMTPAuthenticationError(535, b"auth failed for person@example.com token=abc")

    monkeypatch.setattr(SmtpEmailSender, "_send_blocking", boom)

    with pytest.raises(EmailDeliveryError) as exc:
        await sender.send(EmailMessage(to="person@example.com", subject="s", text="body-with-link"))

    assert "person@example.com" not in str(exc.value) and "abc" not in str(exc.value)
    assert "SMTPAuthenticationError" in str(exc.value)


def test_smtp_builds_a_multipart_message_with_a_proper_from_header() -> None:
    mime = _smtp()._build(EmailMessage(to="a@b.co", subject="Hello", text="plain", html="<b>h</b>"))
    assert mime["To"] == "a@b.co" and mime["Subject"] == "Hello"
    assert "no-reply@example.com" in mime["From"]
    assert mime.is_multipart()
