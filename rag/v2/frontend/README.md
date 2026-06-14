# frontend/

## Table of Contents

- [How the Frontend Connects to the Backend](#how-the-frontend-connects-to-the-backend)
  - [Production — Nginx proxy (same origin)](#production--nginx-proxy-same-origin)
  - [Local dev — Next.js rewrite proxy](#local-dev--nextjs-rewrite-proxy)
  - [Why not CORS?](#why-not-cors)
- [Connection Files](#connection-files)
- [SSE Streaming](#sse-streaming)
- [Auth Flow](#auth-flow)
- [Running Locally](#running-locally)
- [Directory Layout](#directory-layout)

---

## How the Frontend Connects to the Backend

The frontend **never** calls `http://localhost:8000` directly. All API calls use the relative path `/api/v1/*`. How that resolves depends on the environment.

### Production — Nginx proxy (same origin)

```
Browser → https://app.example.com/api/v1/chat  (same origin as the page)
  └── Nginx proxies internally:  api:8000/api/v1/chat
```

Nginx is the single entry point on port 443. Both the frontend (`frontend:3000`) and the API (`api:8000`) sit behind it. Since the browser only ever talks to one host, there is **no CORS** and **no cross-origin**.

`NEXT_PUBLIC_API_BASE_URL` is set to `""` (empty string) at Docker build time — all fetch calls are relative URLs like `fetch('/api/v1/chat')`.

### Local dev — Next.js rewrite proxy

```
Browser → http://localhost:3000/api/v1/chat  (Next.js dev server)
  └── next.config.ts rewrite:  http://localhost:8000/api/v1/chat
```

When `NODE_ENV !== 'production'`, `next.config.ts` adds a rewrite rule that proxies `/api/v1/*` server-side to the API on port 8000. The browser still makes a same-origin request to `:3000` — Next.js dev server forwards it. **No CORS, no browser proxy extension needed.**

```ts
// next.config.ts (abridged)
async rewrites() {
  if (process.env.NODE_ENV === 'production') return []
  const apiBase = process.env.API_BASE_URL ?? 'http://localhost:8000'
  return [{ source: '/api/v1/:path*', destination: `${apiBase}/api/v1/:path*` }]
}
```

Set `API_BASE_URL` in `.env.local` if the API runs on a different host or port.

### Why not CORS?

CORS headers on the API and direct cross-origin calls from the browser would work, but:
- The `Authorization: Bearer <token>` header must be set per-request — simpler via the `api.ts` wrapper
- SSE streams via `EventSource` don't support custom headers
- httpOnly refresh token cookies must be same-origin (or same-site) to work

The Nginx-as-gateway pattern solves all three without any CORS configuration.

---

## Connection Files

All connection logic lives in `src/lib/`:

| File | Purpose |
|------|---------|
| `api.ts` | `api.get/post/patch/delete<T>()` — typed fetch wrapper; adds `Authorization: Bearer <token>`; auto-refreshes on 401; throws `APIError` with `code` + `message` on failure |
| `sse.ts` | `async function* streamSSE(url, init)` — reads `ReadableStream`, yields parsed SSE event objects by type |
| `auth.ts` | `login()`, `logout()`, `tryRestoreSession()` (called on app load), `decodeToken()`, `isTokenValid()` |

**Token storage**:
- Access token: **in memory** (`_accessToken` in `api.ts`) — never `localStorage` (XSS risk)
- Refresh token: **httpOnly cookie** set by the server — JavaScript cannot read it; browser sends it automatically on `POST /api/v1/auth/refresh`

---

## SSE Streaming

Chat streaming (`POST /api/v1/chat/stream`) uses `fetch` + `ReadableStream`, **not** `EventSource`.

`EventSource` is GET-only — it cannot send a JSON body. Our streaming endpoint is POST because the `ChatRequest` body carries `session_id`, `corpus_ids`, and `model_tier`.

```ts
// src/hooks/useChat.ts (example usage)
const controller = new AbortController()

for await (const event of streamSSE('/api/v1/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ query, session_id, corpus_ids }),
}, controller.signal)) {
  if ('delta' in event)    appendToken(event.delta)
  if ('done' in event)     setCitations(event.citations)
  if ('error' in event)    handleError(event.error)
  if ('abstained' in event) setAbstention(event)
}
```

SSE event types emitted by the API:

| Event | Fields | When |
|-------|--------|------|
| Token delta | `{ delta: string }` | Each token from the LLM |
| Done | `{ done: true, citations: Citation[] }` | Stream complete |
| Abstention | `{ abstained: true, layer: 1\|2\|3, reason: string }` | Pipeline gate fired |
| Error | `{ error: string }` | Server-side exception |

---

## Auth Flow

```
1. User submits login form
   POST /api/v1/auth/token  { email, password }
   ← { access_token, expires_in }  (+ httpOnly refresh cookie set by server)

2. access_token stored in memory (api.ts: _accessToken)
   Every subsequent fetch adds: Authorization: Bearer <token>

3. On 401 response:
   POST /api/v1/auth/refresh  (browser auto-sends httpOnly refresh cookie)
   ← { access_token }  (new access token + rotated refresh cookie)
   Original request retried with new token

4. On page refresh:
   tryRestoreSession() called in root layout
   POST /api/v1/auth/refresh  (httpOnly cookie still present)
   ← new access_token  → setAccessToken()  → user stays logged in

5. On logout:
   clearTokens()  (wipes in-memory access token)
   POST /api/v1/auth/logout  (server clears httpOnly refresh cookie)
```

---

## Running Locally

```bash
cd rag/v2/frontend

# Install dependencies
npm install

# Copy env file
cp .env.local.example .env.local
# API_BASE_URL defaults to http://localhost:8000

# Start the API first (in a separate terminal)
cd ../  &&  uv run uvicorn knowledge.api.app:app --reload --port 8000

# Start the frontend
npm run dev
# → http://localhost:3000
```

---

## Directory Layout

```
frontend/
├── next.config.ts          ← rewrite proxy for local dev; standalone output for Docker
├── src/
│   ├── app/                ← Next.js 15 App Router pages
│   ├── components/         ← React components
│   ├── lib/
│   │   ├── api.ts          ← typed fetch wrapper (READ THIS FIRST)
│   │   ├── sse.ts          ← SSE streaming via fetch + ReadableStream
│   │   └── auth.ts         ← login / logout / token restore
│   ├── hooks/              ← useChat, useIngest, useConversations, useMemories, …
│   ├── store/              ← Zustand: chatStore (conversations, session_id, corpus)
│   └── types/              ← TypeScript types mirroring API schemas
└── .env.local.example
```
