"""Token-based authentication and tenant/user scope enforcement.

Development mode (AUTH_MODE=disabled): no tokens required, but tenant/user
 scoping in queries still applies. NOT for production.

Token mode (AUTH_MODE=token): clients must send
 ``Authorization: Bearer <token>``. Tokens are configured via API_TOKENS as
 ``TOKEN:tenant/user`` entries (``*`` = any user within that tenant).
 The authorized scope is derived from the token — a request whose
 tenant_id/user_id falls outside the token scope is rejected with 403.

Limitations (documented): static bearer tokens, no expiry/rotation, no
identity-provider integration. Suitable for service-to-service and local
agents; use JWT/OAuth2 in front of this service for real multi-user prod.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

_bearer = HTTPBearer(auto_error=False)


@dataclass
class Identity:
    tenant_id: str | None  # None = not authenticated / dev mode
    user_id: str | None
    wildcard_user: bool = False


def parse_token_scopes(raw: str) -> dict[str, Identity]:
    """Parse a ``TOKEN:tenant/user`` list into a lookup table.

    Malformed entries (empty token, empty tenant, empty user, extra ``/``) are
    **skipped**, never silently reinterpreted as privileged identities. Run
    :func:`validate_token_config` at startup to reject a bad configuration
    outright instead of dropping entries at request time.
    """
    table: dict[str, Identity] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        token, scope = (part.strip() for part in entry.split(":", 1))
        if not token or not scope:
            continue
        if "/" in scope:
            tenant, user = (part.strip() for part in scope.split("/", 1))
        else:
            tenant, user = scope, "*"
        # A token with no tenant must never become an identity: that would grant
        # unrestricted cross-tenant access.
        if not tenant or not user or "/" in user:
            continue
        table[token] = Identity(
            tenant_id=tenant,
            user_id=None if user == "*" else user,
            wildcard_user=user == "*",
        )
    return table


def validate_token_config(
    raw: str, *, require_non_empty: bool = False, source: str = "API_TOKENS"
) -> dict[str, str]:
    """Validate a token configuration, raising ``ValueError`` on any bad entry.

    Returns ``{token: scope}``. Call this at startup (and whenever HTTP MCP auth
    is enabled) so a misconfiguration fails fast instead of degrading silently.
    """
    entries = [e.strip() for e in (raw or "").split(",") if e.strip()]
    if require_non_empty and not entries:
        raise ValueError(
            f"{source} must define at least one 'TOKEN:tenant/user' entry when "
            "authentication is enabled"
        )
    seen: dict[str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"{source} entry {entry!r} is invalid; expected 'TOKEN:tenant/user'")
        token, scope = (part.strip() for part in entry.split(":", 1))
        if not token:
            raise ValueError(f"{source} contains an entry with an empty token")
        if not scope:
            raise ValueError(f"{source} token {token!r} has an empty scope")
        if "/" in scope:
            tenant, user = (part.strip() for part in scope.split("/", 1))
        else:
            tenant, user = scope, "*"
        if not tenant:
            raise ValueError(
                f"{source} token {token!r} has an empty tenant; refusing to start "
                "because a tenant-less token would grant cross-tenant access"
            )
        if not user:
            raise ValueError(f"{source} token {token!r} has an empty user scope")
        if "/" in user:
            raise ValueError(f"{source} token {token!r} has an invalid scope {scope!r}")
        if token in seen:
            raise ValueError(f"{source} defines token {token!r} more than once")
        seen[token] = scope
    return seen


def get_identity(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Identity:
    if settings.AUTH_MODE == "disabled":
        return Identity(tenant_id=None, user_id=None)
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    table = parse_token_scopes(settings.API_TOKENS)
    identity = table.get(creds.credentials)
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return identity


def enforce_scope(identity: Identity, tenant_id: str, user_id: str) -> None:
    """Reject cross-tenant / cross-user access for authenticated callers.

    Fails **closed**: in token mode an authenticated identity without a tenant
    scope is denied rather than allowed (a missing tenant must never bypass
    tenant isolation).
    """
    if settings.AUTH_MODE == "disabled":
        return
    if identity.tenant_id is None:
        raise HTTPException(
            status_code=403,
            detail="Authenticated token has no tenant scope",
        )
    if tenant_id != identity.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access denied")
    if not identity.wildcard_user and user_id != identity.user_id:
        raise HTTPException(status_code=403, detail="Cross-user access denied")
