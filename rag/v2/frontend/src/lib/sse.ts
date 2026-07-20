// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

/**
 * SSE streaming helper.
 *
 * Uses fetch + ReadableStream, NOT EventSource.
 * EventSource is GET-only — it cannot send a JSON body.
 * Our streaming endpoints use POST (ChatRequest body carries
 * session_id, corpus_ids, model_tier, etc.).
 *
 * Usage:
 *   for await (const event of streamSSE('/api/v2/chat/stream', { method: 'POST', body: ... })) {
 *     if (event.delta)   appendToken(event.delta)
 *     if (event.done)    setCitations(event.citations)
 *     if (event.error)   handleError(event.error)
 *   }
 */

export interface SSEDelta      { delta: string }
export interface SSEDone       { done: true; citations: unknown[] }
export interface SSEError      { error: string }
export interface SSEAbstained  { abstained: true; layer: number; reason: string }
export type SSEEvent = SSEDelta | SSEDone | SSEError | SSEAbstained

export async function* streamSSE(
  url: string,
  init: RequestInit,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const token = (await import('./api')).getAccessToken()
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(init.headers ?? {}),
  }

  const res = await fetch(url, { ...init, headers, signal })

  if (!res.ok) {
    const json = await res.json().catch(() => ({}))
    const err = json.error ?? {}
    throw new Error(err.message ?? `SSE request failed: ${res.status}`)
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split('\n\n')

      // The last element may be an incomplete event — keep it in the buffer
      buffer = parts.pop() ?? ''

      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6)
        if (payload === '[DONE]') return
        try {
          yield JSON.parse(payload) as SSEEvent
        } catch {
          // Malformed line — skip silently
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
