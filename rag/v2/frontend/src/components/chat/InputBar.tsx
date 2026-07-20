// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.


import { useState, useRef, type KeyboardEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { useChatStore } from '@/store/chatStore'

interface Props {
  onSend:  (query: string) => void
  onStop:  () => void
  loading: boolean
}

const TIER_OPTIONS = ['auto', 'small', 'large'] as const

export function InputBar({ onSend, onStop, loading }: Props) {
  const [text, setText] = useState('')
  const { selectedCorpusIds, modelTier, setModelTier } = useChatStore()
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const q = text.trim()
    if (!q || loading) return
    onSend(q)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleInput() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  return (
    <div className="border-t border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        {/* Corpus indicator */}
        {selectedCorpusIds.length > 0 && (
          <span className="text-xs text-[var(--text-muted)] shrink-0 mb-2">
            {selectedCorpusIds[0]}
          </span>
        )}

        {/* Model tier picker */}
        <select
          value={modelTier}
          onChange={e => setModelTier(e.target.value as typeof modelTier)}
          className="text-xs bg-[var(--bg)] border border-[var(--border)] rounded px-2 py-1.5 text-[var(--text-muted)] shrink-0"
        >
          {TIER_OPTIONS.map(t => <option key={t} value={t}>{t}</option>)}
        </select>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => { setText(e.target.value); handleInput() }}
          onKeyDown={handleKeyDown}
          placeholder="Ask your knowledge base..."
          rows={1}
          className="flex-1 resize-none bg-[var(--bg)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm text-[var(--text)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] transition-colors"
        />

        {/* Send / Stop */}
        <button
          onClick={loading ? onStop : submit}
          disabled={!loading && !text.trim()}
          className="shrink-0 p-2.5 rounded-xl bg-[var(--accent)] hover:bg-[#3d5de6] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading
            ? <Loader2 size={16} className="text-white animate-spin" />
            : <Send    size={16} className="text-white" />
          }
        </button>
      </div>
    </div>
  )
}
