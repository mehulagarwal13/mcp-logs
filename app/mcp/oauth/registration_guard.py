"""Per-IP throttle for the unauthenticated OAuth dynamic-registration endpoint.

`POST /register` is public by design (RFC 7591 / the MCP authorization spec),
and every call costs a KMS encryption plus a database write. The `mcp` SDK's
registration handler has no hook for the caller's address, so the throttle is a
small pure-ASGI middleware in front of the SDK app. The in-process token bucket
matches `app.mcp.rate_limit` (one MCP server process); the global rolling
registration cap in `EkipOAuthProvider.register_client` is the cross-IP
backstop.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Receive, Scope, Send

from app.shared.rate_limiter import TokenBucketRateLimiter

REGISTRATION_PATH = "/register"


class RegistrationRateLimitMiddleware:
    def __init__(self, app: ASGIApp, *, per_ip_per_hour: int) -> None:
        self.app = app
        self._per_hour = float(per_ip_per_hour)
        self._limiter = TokenBucketRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == REGISTRATION_PATH
        ):
            client = scope.get("client")
            ip = client[0] if client else "unknown"
            allowed = await self._limiter.try_acquire(
                f"mcp.register:ip:{ip}", self._per_hour / 3600.0, capacity=self._per_hour
            )
            if not allowed:
                await self._reject(send)
                return
        await self.app(scope, receive, send)

    async def _reject(self, send: Send) -> None:
        retry_after = max(1, int(3600 / self._per_hour))
        body = json.dumps(
            {
                "error": "temporarily_unavailable",
                "error_description": (
                    "Too many client registrations from this address; retry later."
                ),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(retry_after).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
