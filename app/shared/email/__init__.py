"""Provider-independent outgoing email (shared/: cross-module, like shared/security).

Three separable pieces so future mail types (password reset, invitations)
reuse everything but a template:

* `messages` -- `EmailMessage`, the provider-neutral value object, plus the
  errors a sender may raise.
* `templates` -- renders subject/text/html. Pure functions, no I/O.
* `smtp` / `get_email_sender` -- delivery. Selected by `EMAIL_PROVIDER`.

Nothing here ever logs a message body, a link, or a token: callers log a
verification row id, never the content.
"""

from __future__ import annotations

from app.shared.email.messages import (
    EmailDeliveryError,
    EmailMessage,
    EmailNotConfiguredError,
    EmailSender,
)
from app.shared.email.sender import email_delivery_configured, get_email_sender

__all__ = [
    "EmailDeliveryError",
    "EmailMessage",
    "EmailNotConfiguredError",
    "EmailSender",
    "email_delivery_configured",
    "get_email_sender",
]
