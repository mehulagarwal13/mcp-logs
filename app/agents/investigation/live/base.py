"""`LiveEvidenceSource` -- the protocol every live, targeted evidence source
the Investigation Agent can query directly during an active investigation
implements (`github_live.py`, `slack_live.py`, `monitoring_live.py`).

Structurally parallel to `app.ingestion.connectors.base.Connector`, but
deliberately a *separate*, simpler protocol, not a reuse of that one: a live
lookup is a single-shot, narrow, incident-scoped query ("give me what's
relevant right now" -- a handful of requests, no pagination cursor, no
persistence, no chunk/embed pipeline), fundamentally different from a full,
resumable, paginated sync destined for `retrieval.upsert`
(`ingestion.processors.pipeline`). Keeping this protocol here, inside
`agents/investigation/`, rather than importing or extending anything under
`app.ingestion`, is also what keeps `agents/` honoring the import-linter
contract that forbids it from depending on `app.ingestion` at all
(`pyproject.toml`'s `[tool.importlinter]`, "agents does not depend on mcp or
ingestion internals") -- ingestion remains the *only* place that owns the
long-term fetch-clean-chunk-embed-store pipeline; a `LiveEvidenceSource`
only ever produces `EvidenceItem`s directly, never a `RawDocument`, and
never writes anything to `documents`/`document_metadata`/any chunk table.

Every live source is handed a `LiveConnectorCredential`
(`app.core.tenancy.live_credentials` -- core/tenancy's internal carrier for a
connector whose credential has been decrypted in memory) for the specific
connector it's querying. Read the token with `connector.reveal_token()` only
when building the outbound request, and the non-secret settings (repos,
channels, ...) from `connector.config`. NOT the public `ConnectorConfig`:
that one is always redacted (its `credential_ref` is a placeholder), which is
exactly why live lookups used to authenticate with a bogus token.
`agents.investigation.evidence._gather_live_evidence` is the only caller that
resolves connectors (via `core.tenancy.service.
resolve_live_evidence_connectors`, which enforces tenant isolation and
incident-read authorization) and dispatches by `connector.source`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.core.tenancy.live_credentials import LiveConnectorCredential
from app.shared.schemas import EvidenceItem


class LiveSourceError(Exception):
    """A live lookup failed as a whole (every request failed). Messages must be
    generic: they are never shown to callers, but must still never carry a
    credential or provider response body.
    """


class LiveSourceAuthError(LiveSourceError):
    """The provider rejected the connector's credential (revoked, expired, or
    invalid). Not worth retrying; surfaced as `auth_failed` in the
    investigation's `LiveEvidenceStatus`.
    """


@runtime_checkable
class LiveEvidenceSource(Protocol):
    """One external source the Investigation Agent can query live. See
    module docstring.
    """

    source_name: str
    # False for a placeholder that has no real backend yet. Such a source is
    # never called and its connectors' credentials are never decrypted; the
    # investigation reports it as `not_implemented` rather than as a success.
    # Optional: a source that does not define it is treated as implemented.
    implemented: bool

    async def fetch_live_evidence(
        self,
        *,
        connector: LiveConnectorCredential,
        query: str,
        since: datetime,
        limit: int,
    ) -> list[EvidenceItem]:
        """Return up to `limit` fresh `EvidenceItem`s relevant to `query`, no
        older than `since`, fetched directly from the external API -- never
        from anything already ingested/indexed by `retrieval.search()`.

        Every returned item's `metadata` must include
        `"retrieval_mode": "live"` (see
        `agents.investigation.evidence._gather_live_evidence`'s docstring
        for why: it is how sub-stage B's hypothesis prompt --
        `agents.investigation.hypothesis._format_evidence_line`, unchanged
        by this feature -- and any future reader of `EvidenceItem` can tell
        this apart from indexed evidence, without a dedicated schema field).

        Must not raise for a single failed sub-request (e.g. one repo's
        search failing, one channel's history call failing) -- that should
        be logged and skipped internally, the same "an individual source's
        failure is logged and skipped, never fatal" discipline
        `agents.investigation.evidence`'s module docstring already applies
        to every other evidence source. It is acceptable to let a *total*
        failure (e.g. an invalid credential) raise, since the caller already
        wraps every `fetch_live_evidence` call in its own try/except, exactly
        like every other source in this module.

        Raise `LiveSourceAuthError` when the provider rejects the credential,
        and `LiveSourceError` when every request failed -- never return `[]`
        for those, or a broken connector is indistinguishable from "nothing
        relevant found".
        """
        ...
