// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

/**
 * Auth helpers — JWT login, logout, token decode.
 *
 * Access token: stored in memory (_accessToken in api.ts).
 *   - Never localStorage (XSS risk) or sessionStorage (tab-isolated).
 *   - Lost on full page refresh — tryRefresh() restores it from the
 *     httpOnly refresh token cookie automatically.
 *
 * Refresh token: httpOnly cookie set by the server on POST /auth/token.
 *   - The browser sends it automatically on POST /auth/refresh.
 *   - JavaScript cannot read it — this is intentional (CSRF mitigation).
 */

import { api, setAccessToken, clearTokens } from './api'

export interface TokenClaims {
  sub:       string     // user ID (hashed in server logs, raw here for client display)
  tenant_id: string
  roles:     string[]
  exp:       number     // Unix timestamp
}

export interface LoginResponse {
  access_token: string
  token_type:   'bearer'
  expires_in:   number
}

/** Log in with email + password. Stores the access token in memory. */
export async function login(email: string, password: string): Promise<void> {
  const res = await api.post<LoginResponse>('/auth/token', { email, password })
  setAccessToken(res.access_token)
}

/** Log out: clear in-memory token and ask the server to invalidate the refresh cookie. */
export async function logout(): Promise<void> {
  clearTokens()
  await fetch('/api/v2/auth/logout', { method: 'POST' }).catch(() => null)
}

/**
 * Attempt to restore the access token from the refresh cookie.
 * Called once on app load (in root layout) to survive page refreshes.
 * Returns true if a new access token was obtained.
 */
export async function tryRestoreSession(): Promise<boolean> {
  try {
    const res = await fetch('/api/v2/auth/refresh', { method: 'POST' })
    if (!res.ok) return false
    const json = await res.json()
    if (json.data?.access_token) {
      setAccessToken(json.data.access_token)
      return true
    }
    return false
  } catch {
    return false
  }
}

/** Decode JWT payload without verification (client-side display only). */
export function decodeToken(token: string): TokenClaims | null {
  try {
    const [, payload] = token.split('.')
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'))
    return JSON.parse(json) as TokenClaims
  } catch {
    return null
  }
}

/** Returns true if the token is present and not expired. */
export function isTokenValid(token: string | null): boolean {
  if (!token) return false
  const claims = decodeToken(token)
  if (!claims) return false
  return claims.exp > Date.now() / 1000
}
