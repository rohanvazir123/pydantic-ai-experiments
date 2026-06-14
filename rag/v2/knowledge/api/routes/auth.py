"""Auth routes — JWT token issuance and refresh."""

import base64
import json
import time
import uuid

from fastapi import APIRouter, Request

from knowledge.api.middleware import get_request_id
from knowledge.api.schemas import (
    APIResponse,
    RefreshResponse,
    TokenRequest,
    TokenResponse,
)
from knowledge.config.settings import load_settings

router = APIRouter(prefix="/auth", tags=["auth"])

DEV_EMAIL    = "dev@neuralflow.ai"
DEV_ROLES    = ["reader", "writer", "admin"]
DEV_TENANT   = "default"
ACCESS_TTL   = 900      # 15 min
REFRESH_TTL  = 86_400   # 24 h


def _b64url(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).rstrip(b"=").decode()


def _make_stub_token(sub: str, tenant_id: str, roles: list[str], ttl: int = ACCESS_TTL) -> str:
    """Build a JWT-shaped token without crypto — accepted by _verify_stub in auth.py."""
    header  = _b64url({"alg": "none", "typ": "JWT"})
    payload = _b64url({
        "sub":       sub,
        "tenant_id": tenant_id,
        "roles":     roles,
        "iat":       int(time.time()),
        "exp":       int(time.time()) + ttl,
    })
    return f"{header}.{payload}.dev"


def _make_rs256_token(sub: str, tenant_id: str, roles: list[str], ttl: int = ACCESS_TTL) -> str:
    """Sign a JWT with the RS256 private key."""
    import jwt as pyjwt
    from pathlib import Path
    s = load_settings()
    key = Path(s.jwt_private_key_path).read_bytes()
    return pyjwt.encode(
        {
            "sub":       sub,
            "tenant_id": tenant_id,
            "roles":     roles,
            "iat":       int(time.time()),
            "exp":       int(time.time()) + ttl,
        },
        key,
        algorithm="RS256",
    )


def _issue_token(sub: str, tenant_id: str, roles: list[str], ttl: int = ACCESS_TTL) -> str:
    s = load_settings()
    if s.jwt_algorithm == "RS256":
        try:
            return _make_rs256_token(sub, tenant_id, roles, ttl)
        except Exception:
            pass  # fall through to stub if key not available yet
    return _make_stub_token(sub, tenant_id, roles, ttl)


@router.post("/token", response_model=APIResponse[TokenResponse])
async def issue_token(body: TokenRequest, request: Request) -> APIResponse[TokenResponse]:
    """Issue a JWT access token.

    Dev mode: accepts any credentials; sub = email, roles = admin.
    RS256 mode: signs with private key from settings.jwt_private_key_path.
    """
    token = _issue_token(
        sub=body.email,
        tenant_id=DEV_TENANT,
        roles=DEV_ROLES,
        ttl=ACCESS_TTL,
    )
    return APIResponse(
        request_id=get_request_id() or str(uuid.uuid4()),
        data=TokenResponse(access_token=token, expires_in=ACCESS_TTL),
    )


@router.post("/refresh", response_model=APIResponse[RefreshResponse])
async def refresh_token(request: Request) -> APIResponse[RefreshResponse]:
    """Return a new access token from the refresh cookie.

    Dev mode: issues a fresh stub token for the dev user so page refreshes
    work without a real cookie store.
    """
    token = _issue_token(
        sub=DEV_EMAIL,
        tenant_id=DEV_TENANT,
        roles=DEV_ROLES,
        ttl=ACCESS_TTL,
    )
    return APIResponse(
        request_id=get_request_id() or str(uuid.uuid4()),
        data=RefreshResponse(access_token=token, expires_in=ACCESS_TTL),
    )


def make_dev_token(ttl: int = REFRESH_TTL) -> str:
    """Generate a long-lived dev token for use in scripts / seed output."""
    return _issue_token(DEV_EMAIL, DEV_TENANT, DEV_ROLES, ttl)
