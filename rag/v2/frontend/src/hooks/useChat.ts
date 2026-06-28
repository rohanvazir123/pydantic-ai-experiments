
import { useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { streamSSE } from '@/lib/sse'

// Session-only Q&A cache: dies on reload (sessionStorage) and has a TTL.
const CACHE_TTL_MS = 30 * 60 * 1000  // 30 minutes

interface CacheEntry { answer: string; citations: any[]; expiry: number }

// Client-side: lowercase + collapse whitespace. Server normalizes fully with spaCy.
function normalizeQuery(query: string): string {
  return query.toLowerCase().trim().replace(/\s+/g, ' ')
}

function cacheKey(query: string, corpusIds: string[], tier: string): string {
  return `qa:${JSON.stringify({ query: normalizeQuery(query), corpusIds, tier })}`
}

function cacheGet(key: string): CacheEntry | null {
  try {
    const raw = sessionStorage.getItem(key)
    if (!raw) return null
    const entry: CacheEntry = JSON.parse(raw)
    if (Date.now() > entry.expiry) { sessionStorage.removeItem(key); return null }
    return entry
  } catch { return null }
}

function cacheSet(key: string, entry: Omit<CacheEntry, 'expiry'>): void {
  try {
    sessionStorage.setItem(key, JSON.stringify({ ...entry, expiry: Date.now() + CACHE_TTL_MS }))
  } catch { /* sessionStorage full — silently skip */ }
}

export function useChat() {
  const abortRef = useRef<AbortController | null>(null)

  async function sendMessage(query: string) {
    // Always read fresh state via getState() — the hook snapshot may be stale
    // after a newConversation() call in the same event handler.
    const getStore = useChatStore.getState

    let convId = getStore().activeId
    if (!convId) convId = getStore().newConversation()

    // Re-read after possible creation
    const conv = getStore().conversations.find(c => c.id === convId)
    if (!conv) return   // should never happen

    getStore().addUserMessage(convId, query)

    const corpusIds = getStore().selectedCorpusIds
    const tier      = getStore().modelTier
    const key       = cacheKey(query, corpusIds, tier)
    const cached    = cacheGet(key)

    if (cached) {
      // Serve from session cache instantly — no spinner needed
      getStore().finaliseMessage(convId, {
        content:   cached.answer,
        citations: cached.citations,
      })
      return
    }

    getStore().appendToken(convId, '')   // thinking cursor

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    let fullAnswer  = ''
    let citations: any[] = []

    try {
      for await (const event of streamSSE(
        '/api/v2/chat/stream',
        {
          method: 'POST',
          body: JSON.stringify({
            query,
            corpus_ids:  corpusIds,
            session_id:  conv.session_id,
            model_tier:  tier,
          }),
        },
        abortRef.current.signal,
      )) {
        if ('delta' in event) {
          fullAnswer += event.delta
          getStore().appendToken(convId, event.delta)
        } else if ('done' in event && event.done) {
          citations = (event as any).citations ?? []
          getStore().finaliseMessage(convId, {
            citations,
            prompt_tokens:     (event as any).prompt_tokens,
            completion_tokens: (event as any).completion_tokens,
          })
          if (fullAnswer) cacheSet(key, { answer: fullAnswer, citations })
        } else if ('abstained' in event) {
          getStore().finaliseMessage(convId, {
            status:            `abstained_${(event as any).layer === 1 ? 'retrieval' : 'judge'}` as any,
            abstention_layer:  (event as any).layer,
            abstention_reason: (event as any).reason,
            content:           'No relevant information found. Try rephrasing your question.',
          })
        } else if ('error' in event) {
          getStore().finaliseMessage(convId, { content: `Error: ${(event as any).error}` })
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        getStore().finaliseMessage(convId, { content: 'Connection error. Please try again.' })
      }
    }
  }

  function stop() { abortRef.current?.abort() }

  return { sendMessage, stop }
}
