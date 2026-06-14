'use client'
import { useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { streamSSE } from '@/lib/sse'

export function useChat() {
  const store    = useChatStore()
  const abortRef = useRef<AbortController | null>(null)

  async function sendMessage(query: string) {
    let convId = store.activeId
    if (!convId) convId = store.newConversation()
    const conv = store.conversations.find(c => c.id === convId)!

    store.addUserMessage(convId, query)
    // Show thinking cursor immediately — before first token arrives
    store.appendToken(convId, '')

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      for await (const event of streamSSE(
        '/api/v2/chat/stream',
        {
          method: 'POST',
          body: JSON.stringify({
            query,
            corpus_ids:  store.selectedCorpusIds,
            session_id:  conv.session_id,
            model_tier:  store.modelTier,
          }),
        },
        abortRef.current.signal,
      )) {
        if ('delta' in event) {
          store.appendToken(convId, event.delta)
        } else if ('done' in event && event.done) {
          store.finaliseMessage(convId, {
            citations:        (event as any).citations ?? [],
            prompt_tokens:    (event as any).prompt_tokens,
            completion_tokens: (event as any).completion_tokens,
          })
        } else if ('abstained' in event) {
          store.finaliseMessage(convId, {
            status:            `abstained_${(event as any).layer === 1 ? 'retrieval' : 'judge'}` as any,
            abstention_layer:  (event as any).layer,
            abstention_reason: (event as any).reason,
            content:           'No relevant information found. Try rephrasing your question.',
          })
        } else if ('error' in event) {
          store.finaliseMessage(convId, { content: `Error: ${(event as any).error}` })
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        store.finaliseMessage(convId, { content: 'Connection error. Please try again.' })
      }
    }
  }

  function stop() { abortRef.current?.abort() }

  return { sendMessage, stop }
}
