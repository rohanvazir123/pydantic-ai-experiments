"""Auth routes — JWT token issuance and refresh.

Phase 9 (Security) adds real JWT signing with RS256. This stub returns
a placeholder token so other routes can be tested end-to-end before
Phase 9 is complete.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request

from knowledge.api.schemas import APIResponse, RefreshResponse, TokenRequest, TokenResponse
from knowledge.api.middleware import get_request_id

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=APIResponse[TokenResponse])
async def issue_token(body: TokenRequest, request: Request) -> APIResponse[TokenResponse]:
    """Issue a JWT access token.

    Phase 9 TODO: verify credentials against tenant store, sign RS256 JWT,
    set httpOnly refresh token cookie.
    """
    # Stub: accept any credentials; return a placeholder token
    token = f"stub.{uuid.uuid4().hex}.token"
    return APIResponse(
        request_id=get_request_id(),
        data=TokenResponse(access_token=token, expires_in=900),
    )


@router.post("/refresh", response_model=APIResponse[RefreshResponse])
async def refresh_token(request: Request) -> APIResponse[RefreshResponse]:
    """Rotate refresh token and return new access token.

    Phase 9 TODO: read httpOnly refresh cookie, verify server-side Redis entry,
    rotate and issue new RS256 JWT + refresh cookie.
    """
    token = f"stub.{uuid.uuid4().hex}.refreshed"
    return APIResponse(
        request_id=get_request_id(),
        data=RefreshResponse(access_token=token, expires_in=900),
    )
