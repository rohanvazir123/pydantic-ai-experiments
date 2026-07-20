// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

/**
 * Typed API client.
 *
 * All calls go to /api/v2/* — a relative URL that works in both environments:
 *   - Dev: Vite proxy forwards /api/v2/* → http://127.0.0.1:7100
 *   - Production: Nginx proxies /api/v2/* → API container
 *
 * Never hardcode http://localhost:7100 here. Always use the relative path.
 */

const BASE = '/api/v2'

export class APIError extends Error {
  code: string
  status: number
  retryAfterS?: number
  constructor(code: string, message: string, status: number, retryAfterS?: number) {
    super(message)
    this.name = 'APIError'
    this.code = code
    this.status = status
    this.retryAfterS = retryAfterS
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const token = getAccessToken()
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...init.headers,
  }

  const res = await fetch(`${BASE}${path}`, { ...init, headers })

  if (res.status === 401) {
    // Try refreshing the token once
    const refreshed = await tryRefresh()
    if (refreshed) {
      return request(path, init)   // retry with new token
    }
    throw new APIError('UNAUTHORIZED', 'Session expired. Please log in again.', 401)
  }

  const json = await res.json()

  if (!res.ok) {
    const err = json.error ?? {}
    throw new APIError(
      err.code ?? 'UNKNOWN_ERROR',
      err.message ?? 'An unexpected error occurred.',
      res.status,
      err.retry_after_s,
    )
  }

  return json.data as T
}

// ── Token management ─────────────────────────────────────────────────────────
// Access token: in memory only (not localStorage — XSS risk).
// Refresh token: httpOnly cookie set by the server on POST /auth/token.

let _accessToken: string | null = null

export function setAccessToken(token: string): void {
  _accessToken = token
}

export function getAccessToken(): string | null {
  return _accessToken
}

export function clearTokens(): void {
  _accessToken = null
  // Refresh token is an httpOnly cookie — cleared by the server on logout
}

async function tryRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/auth/refresh`, { method: 'POST' })
    if (!res.ok) return false
    const json = await res.json()
    setAccessToken(json.data.access_token)
    return true
  } catch {
    return false
  }
}

// ── Public API methods ────────────────────────────────────────────────────────

export const api = {
  get:    <T>(path: string) => request<T>(path),
  post:   <T>(path: string, body: unknown) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
  patch:  <T>(path: string, body: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
  clearCache: () =>
    request<{ cache_keys_deleted: number; message: string }>('/corpus/cache/clear', { method: 'POST' }),
}
