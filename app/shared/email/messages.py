"""Provider-neutral message type, sender protocol and delivery errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmailDeliveryError(Exception):
    """The provider failed to accept the message (outage, auth failure,
    rejected recipient, timeout). Never raised for "success", never swallowed
    into success: callers must treat it as "not delivered".

    The message must not contain the message body or any link/token.
    """


class EmailNotConfiguredError(EmailDeliveryError):
    """No usable email provider is configured (`EMAIL_PROVIDER=none` or an
    incomplete SMTP configuration)."""


@dataclass(frozen=True)
class EmailMessage:
    to: str
    subject: str
    text: str
    html: str | None = None

    def __repr__(self) -> str:  # never let a stray log/traceback print a body or link
        return f"EmailMessage(to=<redacted>, subject={self.subject!r}, body=<redacted>)"


class EmailSender(Protocol):
    async def send(self, message: EmailMessage) -> None:
        """Deliver `message` or raise `EmailDeliveryError`."""
        ...
