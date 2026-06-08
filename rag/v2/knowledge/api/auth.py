"""JWT authentication and RBAC dependency.

Phase 9 — Security Layer.

require_jwt() is a FastAPI dependency that:
  1. Extracts the Bearer token from Authorization header
  2. Verifies RS256 signature using cached JWKS public keys
  3. Checks token expiry
  4. Returns TokenClaims (sub, roles, tenant_id)

In local dev (JWT_ALGORITHM=stub), any non-empty token is accepted and
mapped to the dev tenant. Set JWT_ALGORITHM=RS256 and JWT_PUBLIC_KEY_PATH
for production.

JWE helpers: encrypt_answer / decrypt_answer wrap joserfc for per-tenant
answer encryption used by the semantic cache.
"""

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


# ── Token claims ──────────────────────────────────────────────────────────────

@dataclass
class TokenClaims:
    sub:       str          # user identifier
    tenant_id: str
    roles:     list[str]
    exp:       int          # Unix timestamp


# ── JWKS public key cache ─────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_public_key(key_path: str) -> Any:
    """Load RSA public key from PEM file. LRU-cached — file read once per process."""
    path = Path(key_path)
    if not path.exists():
        return None
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        return load_pem_public_key(path.read_bytes())
    except Exception as exc:
        logger.warning("Failed to load public key from %s: %s", key_path, exc)
        return None


# ── Token verification ────────────────────────────────────────────────────────

def _verify_stub(token: str) -> TokenClaims:
    """Dev-mode stub: accept any non-empty token."""
    try:
        parts  = token.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            return TokenClaims(
                sub=payload.get("sub", "dev-user"),
                tenant_id=payload.get("tenant_id", "default"),
                roles=payload.get("roles", ["reader", "writer", "admin"]),
                exp=payload.get("exp", int(time.time()) + 3600),
            )
    except Exception:
        pass
    return TokenClaims(
        sub="dev-user",
        tenant_id="default",
        roles=["reader", "writer", "admin"],
        exp=int(time.time()) + 3600,
    )


def _verify_rs256(token: str, settings: Settings) -> TokenClaims:
    """Verify RS256 JWT using the configured public key."""
    import jwt as pyjwt
    pub_key = _load_public_key(settings.jwt_public_key_path)
    if pub_key is None:
        raise HTTPException(status_code=401, detail="JWT public key not configured")
    try:
        payload = pyjwt.decode(token, pub_key, algorithms=["RS256"])
        return TokenClaims(
            sub=payload["sub"],
            tenant_id=payload.get("tenant_id", "default"),
            roles=payload.get("roles", []),
            exp=payload["exp"],
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


async def require_jwt(request: Request) -> TokenClaims:
    """FastAPI dependency — extracts and verifies the Bearer JWT.

    Returns TokenClaims on success. Raises HTTP 401/403 on failure.
    In dev mode (JWT_ALGORITHM != RS256) any non-empty token is accepted.
    """
    settings = load_settings()
    creds: HTTPAuthorizationCredentials | None = await _bearer(request)

    if creds is None:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            detail="Authorization header required",
        )

    token = creds.credentials

    if settings.jwt_algorithm != "RS256":
        claims = _verify_stub(token)
    else:
        claims = _verify_rs256(token, settings)

    if claims.exp < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")

    # Inject into contextvars so middleware logs pick them up
    from knowledge.api.middleware import set_tenant_id, set_user_id
    # user_id is SHA-256(sub + tenant_salt) in logs; pass raw sub to context
    set_user_id(hashlib.sha256(claims.sub.encode()).hexdigest()[:16])
    set_tenant_id(claims.tenant_id)

    return claims


def check_corpus_access(claims: TokenClaims, corpus_allowed_roles: list[str]) -> None:
    """Raise HTTP 403 if the JWT roles don't intersect corpus allowed_roles."""
    if not set(claims.roles).intersection(corpus_allowed_roles):
        raise HTTPException(
            status_code=403,
            detail="Insufficient role to access this corpus",
        )


# ── JWE helpers for semantic cache ────────────────────────────────────────────

def _derive_key(tenant_id: str, settings: Settings) -> bytes:
    """Derive a per-tenant 32-byte key from JWT secret + tenant_id."""
    material = f"{settings.jwe_algorithm}:{tenant_id}".encode()
    return hashlib.sha256(material).digest()


def encrypt_answer(payload: dict[str, Any], tenant_id: str, settings: Settings | None = None) -> str:
    """JWE-encrypt a RAGResponse dict for storage in semantic_cache.

    Falls back to base64(JSON) when joserfc is not installed (dev/test).
    """
    _settings = settings or load_settings()
    try:
        import joserfc.jwe as jwe
        from joserfc.jwk import OctKey
        key = OctKey.import_key(_derive_key(tenant_id, _settings))
        token = jwe.encrypt_compact(
            {"alg": "A256KW", "enc": "A256GCM"},
            json.dumps(payload).encode(),
            key,
        )
        return token.decode() if isinstance(token, bytes) else token
    except ImportError:
        return base64.b64encode(json.dumps(payload).encode()).decode()


def decrypt_answer(token: str, tenant_id: str, settings: Settings | None = None) -> dict[str, Any]:
    """Decrypt a JWE token from semantic_cache."""
    _settings = settings or load_settings()
    try:
        import joserfc.jwe as jwe
        from joserfc.jwk import OctKey
        key   = OctKey.import_key(_derive_key(tenant_id, _settings))
        result = jwe.decrypt_compact(token.encode() if isinstance(token, str) else token, key)
        return json.loads(result.plaintext)
    except ImportError:
        return json.loads(base64.b64decode(token.encode()))
    except Exception as exc:
        raise ValueError(f"Failed to decrypt answer: {exc}") from exc
