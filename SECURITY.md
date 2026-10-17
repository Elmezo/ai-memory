# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately by opening a
[GitHub Security Advisory](https://github.com/Elmezo/ai-memory/security/advisories/new)
or by contacting the maintainers through the repository's issue tracker with a
request for a private channel.

Do **not** open a public issue for an unpatched vulnerability. Include:

- a description of the issue and its impact,
- steps to reproduce (proof of concept if possible),
- affected version / commit,
- any suggested remediation.

We aim to acknowledge reports within a few days and to coordinate disclosure
with the reporter.

## Security Model

`ai-memory` is a self-hosted, single-tenant-per-deployment memory service. The
default configuration is tuned for **local development** and makes the
following assumptions explicit.

### Authentication & authorization

- `AUTH_MODE=disabled` (the default) performs **no** token check. It is intended
  for local use only. Tenant/user scoping is still enforced on every query, so
  one scope cannot read another scope's memories by accident, but any caller can
  choose their own scope.
- `AUTH_MODE=token` requires `Authorization: Bearer <token>`. Tokens are static
  and mapped to a `tenant/user` scope via `API_TOKENS`. Requests outside the
  token's scope are rejected with `403`.
- Limitations: static bearer tokens, no expiry/rotation, no external identity
  provider. Put an authenticating reverse proxy (or OAuth2/OIDC) in front of the
  service for real multi-user production use.

### Tenant / user isolation

Every read and write is filtered server-side by `tenant_id` and `user_id`. Memory
IDs alone are never sufficient to access a document outside the caller's scope.

### MCP scope injection

When the MCP server runs with `MEMORY_TENANT_ID` / `MEMORY_USER_ID` set (the
default in `.env.example`), the server **overrides** any tenant/user supplied by
the agent and acts only within that fixed scope, preventing an agent from
forging identity to reach other scopes.

### Elasticsearch

- The bundled `docker-compose.yml` runs Elasticsearch with
  `xpack.security.enabled=false` for a frictionless local setup. Its port is
  bound to `127.0.0.1` only. **Do not** expose it to an untrusted network.
- For shared/production deployments, enable Elasticsearch security and TLS, and
  configure the client with `ES_USERNAME`, `ES_PASSWORD`, `ES_CA_CERTS`, and
  `ES_VERIFY_CERTS`.
- All queries are built as structured Elasticsearch DSL; user input is never
  string-interpolated into query JSON. The target index is fixed by
  `ES_INDEX`; callers cannot select an arbitrary index.

### Container

- The Docker image runs as an unprivileged non-root user (`appuser`, uid 10001).
- Compose services use `no-new-privileges` and publish ports on loopback by
  default.

### Secrets

- Configuration is read from environment variables / a local `.env` file which
  is git-ignored. `.env.example` contains placeholders only.
- Never commit real credentials, tokens, or `.env` files.

## Supported Versions

Security fixes are applied to the latest revision of the `main` branch.
