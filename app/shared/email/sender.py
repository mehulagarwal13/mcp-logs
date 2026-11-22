"""Provider selection from `Settings`."""

from __future__ import annotations

from app.shared.config.settings import get_settings
from app.shared.email.messages import EmailNotConfiguredError, EmailSender
from app.shared.email.smtp import SmtpEmailSender


def email_delivery_configured() -> bool:
    """Cheap, I/O-free readiness check: is there a provider that could send?

    Used by the signup/resend endpoints to fail 503 for *every* caller when
    delivery cannot work (so the status never depends on the submitted
    email), instead of creating pending rows nobody can verify.
    """
    settings = get_settings()
    if settings.email_provider == "smtp":
        return bool(settings.smtp_host and settings.email_from_address)
    return False


def get_email_sender() -> EmailSender:
    settings = get_settings()
    if settings.email_provider == "smtp":
        return SmtpEmailSender(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password.get_secret_value(),
            security=settings.smtp_security,
            from_address=settings.email_from_address,
            from_name=settings.email_from_name,
            timeout=settings.smtp_timeout_seconds,
        )
    raise EmailNotConfiguredError("EMAIL_PROVIDER is 'none'; no email can be sent.")
