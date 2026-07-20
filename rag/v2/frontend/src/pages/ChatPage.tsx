// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

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

export function ChatPage() {
  const store         = useChatStore()
  const { sendMessage, stop } = useChat()
  const [loading,   setLoading]   = useState(false)
  const [debug,     setDebug]     = useState(true)
  const [corpora,   setCorpora]   = useState<CorpusInfo[]>([])
  const [questions, setQuestions] = useState<string[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  const conversation = store.conversations.find(c => c.id === store.activeId)
  const lastMsg      = conversation?.messages.at(-1)
  const citations: Citation[] = lastMsg?.role === 'assistant' ? lastMsg.citations ?? [] : []

  // Fetch suggested questions from public JSON (editable without touching code)
  useEffect(() => {
    fetch('/suggested-questions.json')
      .then(r => r.json())
      .then((data: { q: string }[]) => setQuestions(data.map(d => d.q)))
      .catch(() => {})
  }, [])

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

  void corpora // selectedCorpus unused — corpus shown in dropdown only

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

        {/* Conversation list — only show titled conversations */}
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {store.conversations.filter(c => c.title).map(c => (
            <button
              key={c.id}
              onClick={() => store.setActive(c.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
                c.id === store.activeId
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--border)]'
              }`}
            >
              {c.title}
            </button>
          ))}
        </nav>

        {/* Debug toggle */}
        <div className="p-3 border-t border-[var(--border)] flex items-center justify-between">
          <span className="text-xs text-[var(--text-muted)]">Debug</span>
          <button
            role="switch"
            aria-checked={debug}
            onClick={() => setDebug(v => !v)}
            className={`relative inline-flex w-9 h-5 rounded-full transition-colors focus:outline-none ${debug ? 'bg-[var(--accent)]' : 'bg-gray-300'}`}
          >
            <span className={`inline-block w-4 h-4 bg-white rounded-full shadow transform transition-transform mt-0.5 ${debug ? 'translate-x-4' : 'translate-x-0.5'}`} />
          </button>
        </div>
      </aside>

      {/* Chat area */}
      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-col flex-1 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto min-h-0">
            <div className="max-w-3xl mx-auto px-6 py-4">
              {conversation?.messages.map(msg => (
                <MessageBubble key={msg.id} message={msg} debugMode={debug} />
              ))}
              <div ref={bottomRef} />
            </div>
          </div>

          <InputBar onSend={handleSend} onStop={stop} loading={loading} />
        </main>

        {/* Right panel: citations when present, suggested questions always */}
        <div className="w-64 shrink-0 border-l border-[var(--border)] flex flex-col bg-[var(--surface)] overflow-y-auto">
          <CitationPanel citations={citations} />

          <div className={`p-4 ${citations.length ? 'border-t border-[var(--border)]' : ''}`}>
            <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Suggested Questions
            </h3>
            <div className="flex flex-col gap-2">
              {questions.map(q => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  disabled={loading}
                  className="text-left px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] hover:border-[var(--accent)] hover:bg-blue-50 text-xs text-[var(--text-muted)] hover:text-[var(--text)] transition-colors leading-snug disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
