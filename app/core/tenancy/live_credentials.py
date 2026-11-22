"""Internal-only carrier for a *decrypted* connector credential, used by the
Investigation Agent's live evidence lookups.

This is deliberately NOT a Pydantic model and NOT a dataclass:

- Pydantic's `ValidationError` text echoes `input_value`, and FastAPI's
  `jsonable_encoder` / `dataclasses.asdict` would happily serialize every
  field of a model or dataclass -- including a token.
- A plain class with `__slots__` has no `__dict__`, is not a dataclass, and
  is not a Pydantic model, so none of those generic serializers can reach the
  token. The only way to read it is the explicit `reveal_token()` call, which
  is easy to grep for in review.

`repr`/`str` never include the token, and copying/pickling is refused so the
secret cannot be accidentally persisted (for example into a task queue or a
cache). It must never be returned from a router, MCP tool, or stored.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, NoReturn


class LiveConnectorCredential:
    """One active connector, with its credential decrypted in memory."""

    __slots__ = (
        "config",
        "connector_config_id",
        "organization_id",
        "project_id",
        "source",
        "_token",
    )

    def __init__(
        self,
        *,
        connector_config_id: uuid.UUID,
        organization_id: uuid.UUID,
        project_id: uuid.UUID | None,
        source: str,
        config: Mapping[str, Any],
        token: str,
    ) -> None:
        self.connector_config_id = connector_config_id
        self.organization_id = organization_id
        self.project_id = project_id
        self.source = source
        # Non-secret, user-visible settings only (repos, channels, ...);
        # the resolver strips underscore-prefixed worker state.
        self.config: dict[str, Any] = dict(config)
        self._token = token

    def reveal_token(self) -> str:
        """The plaintext credential. Call only at the moment of the outbound
        request; never log, store, or return the result.
        """
        return self._token

    def __repr__(self) -> str:
        return (
            f"LiveConnectorCredential(source={self.source!r}, "
            f"connector_config_id={self.connector_config_id}, token=<redacted>)"
        )

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("LiveConnectorCredential cannot be copied or pickled.")


class LiveConnectorSkip:
    """A connector the resolver could not hand to a live source, with a
    fixed-vocabulary `reason` (never exception text).
    """

    __slots__ = ("connector_config_id", "reason", "source")

    def __init__(self, *, connector_config_id: uuid.UUID, source: str, reason: str) -> None:
        self.connector_config_id = connector_config_id
        self.source = source
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"LiveConnectorSkip(source={self.source!r}, "
            f"connector_config_id={self.connector_config_id}, reason={self.reason!r})"
        )


class LiveConnectorResolution:
    """Result of `core.tenancy.service.resolve_live_evidence_connectors`."""

    __slots__ = ("credentials", "skipped")

    def __init__(
        self, credentials: list[LiveConnectorCredential], skipped: list[LiveConnectorSkip]
    ) -> None:
        self.credentials = credentials
        self.skipped = skipped
