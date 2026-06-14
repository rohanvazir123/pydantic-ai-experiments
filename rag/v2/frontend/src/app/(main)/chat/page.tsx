'use client'
import { useState, useEffect, useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { useChat }      from '@/hooks/useChat'
import { MessageBubble } from '@/components/chat/MessageBubble'
import { CitationPanel } from '@/components/chat/CitationPanel'
import { InputBar }      from '@/components/chat/InputBar'
import { api }           from '@/lib/api'
import { Database }      from 'lucide-react'
import type { Citation } from '@/types/chat'

interface CorpusInfo {
  id:           string
  display_name: string
}

export default function ChatPage() {
  const store         = useChatStore()
  const { sendMessage, stop } = useChat()
  const [loading,  setLoading]  = useState(false)
  const [debug,    setDebug]    = useState(false)
  const [corpora,  setCorpora]  = useState<CorpusInfo[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  const conversation = store.conversations.find(c => c.id === store.activeId)
  const lastMsg      = conversation?.messages.at(-1)
  const citations: Citation[] = lastMsg?.role === 'assistant' ? lastMsg.citations ?? [] : []

  // Fetch corpora once on mount, auto-select first
  useEffect(() => {
    api.get<CorpusInfo[]>('/corpus').then(list => {
      if (!list?.length) return
      setCorpora(list)
      if (store.selectedCorpusIds.length === 0) {
        store.setCorpusIds([list[0].id])
      }
    }).catch(() => {
      // API not up yet — default to 'default' corpus so chat still works
      store.setCorpusIds(['default'])
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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

  const selectedCorpus = corpora.find(c => c.id === store.selectedCorpusIds[0])

  return (
    <div className="flex h-screen bg-[var(--bg)]">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-[var(--border)] flex flex-col bg-[var(--surface)]">

        {/* Corpus selector */}
        <div className="p-3 border-b border-[var(--border)]">
          <label className="text-xs text-[var(--text-muted)] flex items-center gap-1.5 mb-1.5">
            <Database size={11} /> Corpus
          </label>
          {corpora.length > 0 ? (
            <select
              value={store.selectedCorpusIds[0] ?? ''}
              onChange={e => store.setCorpusIds([e.target.value])}
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            >
              {corpora.map(c => (
                <option key={c.id} value={c.id}>{c.display_name}</option>
              ))}
            </select>
          ) : (
            <div className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-sm text-[var(--text-muted)]">
              {store.selectedCorpusIds[0] ?? 'Loading…'}
            </div>
          )}
        </div>

        {/* New chat */}
        <div className="p-3 border-b border-[var(--border)]">
          <button
            onClick={() => store.newConversation()}
            className="w-full bg-[var(--accent)] text-white rounded-lg px-3 py-2 text-sm hover:bg-[#3d5de6] transition-colors"
          >
            + New Chat
          </button>
        </div>

        {/* Conversation list */}
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

        {/* Debug toggle */}
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
              <div className="text-center mt-20">
                <p className="text-[var(--text-muted)] text-sm">
                  Ask anything about{' '}
                  <span className="text-[var(--text)]">
                    {selectedCorpus?.display_name ?? store.selectedCorpusIds[0] ?? 'your knowledge base'}
                  </span>
                </p>
              </div>
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
