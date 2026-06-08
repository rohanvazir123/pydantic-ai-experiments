'use client'
import { useState } from 'react'
import type { Message } from '@/types/chat'

interface Props { message: Message }

export function DebugPanel({ message }: Props) {
  const [open, setOpen] = useState(false)

  const hasDebugInfo = message.latency_ms || message.request_id || message.trace_url ||
                       message.cache_hit  || message.confidence != null

  if (!hasDebugInfo) return null

  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(v => !v)}
        className="text-xs text-[var(--text-muted)] hover:text-[var(--text)] underline"
      >
        {open ? 'Hide debug' : 'Debug ▾'}
      </button>

      {open && (
        <div className="mt-2 p-3 rounded bg-[var(--surface)] border border-[var(--border)] text-xs font-mono space-y-1">
          {message.request_id && (
            <div className="flex gap-2">
              <span className="text-[var(--text-muted)]">request_id</span>
              <span className="text-[var(--text)] break-all">{message.request_id}</span>
              <button
                onClick={() => navigator.clipboard.writeText(message.request_id!)}
                className="text-[var(--accent)] shrink-0"
              >copy</button>
            </div>
          )}
          {message.cache_hit && (
            <div><span className="text-[var(--text-muted)]">cache_hit </span><span>{message.cache_hit}</span></div>
          )}
          {message.confidence != null && (
            <div><span className="text-[var(--text-muted)]">confidence </span><span>{message.confidence.toFixed(3)}</span></div>
          )}
          {message.latency_ms && Object.entries(message.latency_ms).map(([stage, ms]) => (
            <div key={stage}>
              <span className="text-[var(--text-muted)]">{stage.padEnd(14)} </span>
              <span className={ms > 2000 ? 'text-[var(--error)]' : ms > 500 ? 'text-[var(--warning)]' : 'text-[var(--success)]'}>
                {ms}ms
              </span>
            </div>
          ))}
          {message.trace_url && (
            <div>
              <a
                href={message.trace_url}
                target="_blank"
                rel="noreferrer"
                className="text-[var(--accent)] hover:underline"
              >
                Open in Langfuse →
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
