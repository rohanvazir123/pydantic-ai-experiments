'use client'
import { useState, useEffect, useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { useChat }      from '@/hooks/useChat'
import { MessageBubble } from '@/components/chat/MessageBubble'
import { CitationPanel } from '@/components/chat/CitationPanel'
import { InputBar }      from '@/components/chat/InputBar'
import type { Citation } from '@/types/chat'

export default function ChatPage() {
  const store         = useChatStore()
  const { sendMessage, stop } = useChat()
  const [loading, setLoading] = useState(false)
  const [debug,   setDebug]   = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const conversation  = store.conversations.find(c => c.id === store.activeId)
  const lastMsg       = conversation?.messages.at(-1)
  const citations: Citation[] = lastMsg?.role === 'assistant' ? lastMsg.citations ?? [] : []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages.length])

  async function handleSend(query: string) {
    if (!store.activeId) store.newConversation()
    setLoading(true)
    try {
      await sendMessage(query)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-[var(--bg)]">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-[var(--border)] flex flex-col bg-[var(--surface)]">
        <div className="p-4 border-b border-[var(--border)]">
          <button
            onClick={() => store.newConversation()}
            className="w-full bg-[var(--accent)] text-white rounded-lg px-3 py-2 text-sm hover:bg-[#3d5de6] transition-colors"
          >
            + New Chat
          </button>
        </div>
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {store.conversations.map(c => (
            <button
              key={c.id}
              onClick={() => store.setActive(c.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
                c.id === store.activeId
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--border)]'
              }`}
            >
              {c.title ?? 'New conversation'}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-[var(--border)] flex items-center justify-between">
          <span className="text-xs text-[var(--text-muted)]">Debug</span>
          <button
            onClick={() => setDebug(v => !v)}
            className={`w-8 h-4 rounded-full transition-colors ${debug ? 'bg-[var(--accent)]' : 'bg-[var(--border)]'}`}
          />
        </div>
      </aside>

      {/* Chat area */}
      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-col flex-1 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-4 max-w-3xl mx-auto w-full">
            {!conversation?.messages.length && (
              <p className="text-center text-[var(--text-muted)] mt-20 text-sm">
                Start a conversation — your knowledge base is ready.
              </p>
            )}
            {conversation?.messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} debugMode={debug} />
            ))}
            <div ref={bottomRef} />
          </div>

          <InputBar onSend={handleSend} onStop={stop} loading={loading} />
        </main>

        {/* Citation panel */}
        <CitationPanel citations={citations} />
      </div>
    </div>
  )
}
