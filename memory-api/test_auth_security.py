"""Production security-readiness regression tests for ``AUTH_MODE`` & scope isolation.

These tests exercise the authentication/authorization helpers directly and do
**not** require Elasticsearch. Each test temporarily patches global settings and
restores them via the ``auth_settings`` fixture.

They cover the fixes for two previously confirmed gaps:

  1. ``enforce_scope`` used to short-circuit when an "authenticated" token
     resolved to an empty tenant (``API_TOKENS=tok:/*``), granting unrestricted
     cross-tenant access. Now such entries are rejected at startup and scope
     enforcement fails closed.
  2. The MCP HTTP transport used to authenticate no one; that is covered by
     ``test_mcp_http_auth.py``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from auth import (
    Identity,
    enforce_scope,
    get_identity,
    parse_token_scopes,
    validate_token_config,
)
from config import settings


def _creds(token: str | None) -> HTTPAuthorizationCredentials | None:
    if token is None:
        return None
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture
def auth_settings():
    """Temporarily override auth settings; always restore the originals."""
    previous = (settings.AUTH_MODE, settings.API_TOKENS)
    yield
    settings.AUTH_MODE, settings.API_TOKENS = previous


# --------------------------------------------------------------------------- #
# Token mode: requests must authenticate
# --------------------------------------------------------------------------- #
def test_token_mode_requires_bearer(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/user1"
    with pytest.raises(HTTPException) as exc:
        get_identity(None)
    assert exc.value.status_code == 401


def test_token_mode_rejects_unknown_token(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/user1"
    with pytest.raises(HTTPException) as exc:
        get_identity(_creds("not-a-real-token"))
    assert exc.value.status_code == 401


def test_token_mode_maps_token_to_scope(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/user1"
    ident = get_identity(_creds("tok-A"))
    assert (ident.tenant_id, ident.user_id) == ("tenantA", "user1")


# --------------------------------------------------------------------------- #
# Token mode: crossing tenant/user boundaries is denied
# --------------------------------------------------------------------------- #
def test_cross_tenant_denied(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/user1"
    ident = get_identity(_creds("tok-A"))
    enforce_scope(ident, "tenantA", "user1")  # same scope -> allowed
    with pytest.raises(HTTPException) as exc:
        enforce_scope(ident, "tenantB", "user1")
    assert exc.value.status_code == 403


def test_cross_user_denied(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/user1"
    ident = get_identity(_creds("tok-A"))
    with pytest.raises(HTTPException) as exc:
        enforce_scope(ident, "tenantA", "user2")
    assert exc.value.status_code == 403


def test_wildcard_user_allowed_within_tenant_only(auth_settings):
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok-A:tenantA/*"
    ident = get_identity(_creds("tok-A"))
    enforce_scope(ident, "tenantA", "any-user")  # wildcard user allowed
    with pytest.raises(HTTPException) as exc:
        enforce_scope(ident, "tenantB", "any-user")
    assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# Fixed gap 1: empty-tenant tokens can never bypass isolation
# --------------------------------------------------------------------------- #
def test_empty_tenant_token_is_not_a_valid_identity(auth_settings):
    """``tok:/*`` must never resolve to a privileged identity."""
    settings.AUTH_MODE = "token"
    settings.API_TOKENS = "tok:/*"
    # The malformed entry is dropped, so the token is simply unknown -> 401.
    with pytest.raises(HTTPException) as exc:
        get_identity(_creds("tok"))
    assert exc.value.status_code == 401


def test_enforce_scope_fails_closed_without_tenant(auth_settings):
    """Even a hand-built tenant-less identity is denied in token mode."""
    settings.AUTH_MODE = "token"
    with pytest.raises(HTTPException) as exc:
        enforce_scope(Identity(tenant_id=None, user_id=None), "tenantB", "user1")
    assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# Disabled mode: documents the (dev-only) default so it cannot regress silently
# --------------------------------------------------------------------------- #
def test_disabled_mode_is_anonymous_and_unscoped(auth_settings):
    """AUTH_MODE=disabled performs no auth and no scope check.

    This is exactly why production must set ``AUTH_MODE=token``; the assertion
    documents the current default rather than endorsing it.
    """
    settings.AUTH_MODE = "disabled"
    ident = get_identity(None)
    assert ident.tenant_id is None and ident.user_id is None
    enforce_scope(ident, "any_tenant", "any_user")  # no exception in dev mode


# --------------------------------------------------------------------------- #
# Startup configuration validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        "no-colon-here",          # missing ':' separator
        "tok:",                   # empty scope
        ":tenantA/user1",         # empty token
        "tok:/user1",             # empty tenant
        "tok:tenantA/",           # empty user
        "tok:tenantA/user/extra", # extra '/' in user scope
    ],
)
def test_validate_token_config_rejects_malformed_entries(raw):
    with pytest.raises(ValueError):
        validate_token_config(raw, require_non_empty=True)


def test_validate_token_config_requires_non_empty_when_enabled():
    with pytest.raises(ValueError):
        validate_token_config("", require_non_empty=True)
    # Allowed when auth is not required (e.g. disabled dev mode).
    assert validate_token_config("", require_non_empty=False) == {}


def test_validate_token_config_rejects_duplicate_tokens():
    with pytest.raises(ValueError):
        validate_token_config("tok:tenantA/user1,tok:tenantB/user2")


def test_validate_token_config_accepts_valid_entries():
    table = validate_token_config(
        "tok-A:tenantA/user1,tok-B:tenantB/*", require_non_empty=True
    )
    assert table == {"tok-A": "tenantA/user1", "tok-B": "tenantB/*"}


def test_parse_token_scopes_drops_malformed_but_keeps_valid():
    table = parse_token_scopes("good:tenantA/user1,bad:/*,nocolon")
    assert set(table) == {"good"}
    assert table["good"].tenant_id == "tenantA"
