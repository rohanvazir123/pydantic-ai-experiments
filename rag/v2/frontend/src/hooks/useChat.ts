'use client'
import { useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { streamSSE } from '@/lib/sse'

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
    getStore().appendToken(convId, '')   // thinking cursor

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      for await (const event of streamSSE(
        '/api/v2/chat/stream',
        {
          method: 'POST',
          body: JSON.stringify({
            query,
            corpus_ids:  getStore().selectedCorpusIds,
            session_id:  conv.session_id,
            model_tier:  getStore().modelTier,
          }),
        },
        abortRef.current.signal,
      )) {
        if ('delta' in event) {
          getStore().appendToken(convId, event.delta)
        } else if ('done' in event && event.done) {
          getStore().finaliseMessage(convId, {
            citations:         (event as any).citations ?? [],
            prompt_tokens:     (event as any).prompt_tokens,
            completion_tokens: (event as any).completion_tokens,
          })
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
