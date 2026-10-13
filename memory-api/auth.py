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


def _parse_tokens() -> dict[str, Identity]:
    table: dict[str, Identity] = {}
    for entry in settings.API_TOKENS.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        token, scope = entry.split(":", 1)
        token, scope = token.strip(), scope.strip()
        if "/" in scope:
            tenant, user = scope.split("/", 1)
        else:
            tenant, user = scope, "*"
        table[token] = Identity(
            tenant_id=tenant.strip() or None,
            user_id=None if user.strip() == "*" else user.strip(),
            wildcard_user=user.strip() == "*",
        )
    return table


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
    table = _parse_tokens()
    identity = table.get(creds.credentials)
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return identity


def enforce_scope(identity: Identity, tenant_id: str, user_id: str) -> None:
    """Reject cross-tenant / cross-user access for authenticated callers."""
    if settings.AUTH_MODE == "disabled":
        return
    if identity.tenant_id is None:
        return
    if tenant_id != identity.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access denied")
    if not identity.wildcard_user and user_id != identity.user_id:
        raise HTTPException(status_code=403, detail="Cross-user access denied")
