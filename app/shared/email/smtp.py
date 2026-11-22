"""SMTP delivery using only the standard library (no new dependency).

`smtplib` is blocking, so the actual exchange runs in a worker thread via
`asyncio.to_thread` to keep the event loop free. Transport security: STARTTLS
(default) or implicit TLS; certificate verification is on (default SSL
context). Plaintext (`smtp_security=none`) is refused unless the host is
loopback (a local Mailpit/MailHog used in development).
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.message import EmailMessage as MimeMessage
from email.utils import formataddr

from app.shared.config.logging import get_logger
from app.shared.email.messages import (
    EmailDeliveryError,
    EmailMessage,
    EmailNotConfiguredError,
)

logger = get_logger(__name__)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class SmtpEmailSender:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        security: str,
        from_address: str,
        from_name: str,
        timeout: float,
    ) -> None:
        if not host or not from_address:
            raise EmailNotConfiguredError("SMTP_HOST and EMAIL_FROM_ADDRESS are required.")
        if security == "none" and host.lower() not in _LOOPBACK_HOSTS:
            raise EmailNotConfiguredError(
                "Plaintext SMTP is only permitted for a loopback host; use starttls or ssl."
            )
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._security = security
        self._from_address = from_address
        self._from_name = from_name
        self._timeout = timeout

    def _build(self, message: EmailMessage) -> MimeMessage:
        mime = MimeMessage()
        mime["From"] = formataddr((self._from_name, self._from_address))
        mime["To"] = message.to
        mime["Subject"] = message.subject
        mime.set_content(message.text)
        if message.html:
            mime.add_alternative(message.html, subtype="html")
        return mime

    def _send_blocking(self, message: EmailMessage) -> None:
        mime = self._build(message)
        context = ssl.create_default_context()
        if self._security == "ssl":
            client: smtplib.SMTP = smtplib.SMTP_SSL(
                self._host, self._port, timeout=self._timeout, context=context
            )
        else:
            client = smtplib.SMTP(self._host, self._port, timeout=self._timeout)
        with client:
            client.ehlo()
            if self._security == "starttls":
                client.starttls(context=context)
                client.ehlo()
            if self._username:
                client.login(self._username, self._password)
            client.send_message(mime)

    async def send(self, message: EmailMessage) -> None:
        try:
            await asyncio.to_thread(self._send_blocking, message)
        except (smtplib.SMTPException, OSError, ssl.SSLError, TimeoutError) as exc:
            # Class name only: the exception text can echo the recipient or
            # server banner, and must never carry body content.
            raise EmailDeliveryError(f"SMTP delivery failed ({type(exc).__name__}).") from exc
