"""Message rendering, separate from delivery. Pure functions."""

from __future__ import annotations

from html import escape

from app.shared.email.messages import EmailMessage


def verification_link(frontend_base_url: str, raw_token: str) -> str:
    """`<base>/verify-email#token=...` -- the token rides in the URL
    *fragment*, which browsers never send to servers or in `Referer`, so it
    stays out of access logs and proxies. The SPA reads it from
    `location.hash`, removes it from the address bar, and POSTs it.
    """
    return f"{frontend_base_url.rstrip('/')}/verify-email#token={raw_token}"


def render_verification_email(
    *, to: str, link: str, ttl_minutes: int, kind: str
) -> EmailMessage:
    hours = max(1, round(ttl_minutes / 60))
    what = {
        "registration": "create your EKIP account",
        "invitation": "accept your EKIP organization invitation",
        "existing_user": "confirm the email address on your EKIP account",
    }.get(kind, "confirm your email address")
    subject = "Confirm your email address for EKIP"
    text = (
        f"Someone (hopefully you) asked to {what} using this email address.\n\n"
        f"Confirm it by opening this link (valid for about {hours} hour(s), single use):\n\n"
        f"{link}\n\n"
        "If you did not request this, ignore this email - nothing will happen and "
        "no account is created without this confirmation.\n"
    )
    safe_link = escape(link, quote=True)
    html = (
        f"<p>Someone (hopefully you) asked to {escape(what)} using this email address.</p>"
        f"<p><a href=\"{safe_link}\">Confirm your email address</a></p>"
        f"<p>This link is valid for about {hours} hour(s) and can be used once.</p>"
        "<p>If you did not request this, ignore this email &mdash; nothing will happen and "
        "no account is created without this confirmation.</p>"
    )
    return EmailMessage(to=to, subject=subject, text=text, html=html)
