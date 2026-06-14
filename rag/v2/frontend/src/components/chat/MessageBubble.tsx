'use client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '@/types/chat'
import { CostBadge }  from './CostBadge'
import { DebugPanel } from './DebugPanel'

interface Props { message: Message; debugMode?: boolean }

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  answered:             { label: 'Answered',              color: 'text-[var(--success)]' },
  abstained_retrieval:  { label: 'No results found',      color: 'text-[var(--warning)]' },
  abstained_citation:   { label: 'Could not cite sources', color: 'text-[var(--warning)]' },
  abstained_judge:      { label: 'Could not verify',      color: 'text-[var(--warning)]' },
}

export function MessageBubble({ message, debugMode = false }: Props) {
  const isUser = message.role === 'user'
  const meta   = message.status ? STATUS_LABELS[message.status] : null

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? 'bg-[var(--accent)] text-white' : 'bg-[var(--surface)] text-[var(--text)]'} rounded-xl px-4 py-3 shadow`}>

        {/* Streaming cursor */}
        {message.streaming ? (
          <span>{message.content}<span className="animate-pulse">▋</span></span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert prose-sm max-w-none">
            {message.content}
          </ReactMarkdown>
        )}

        {/* Low confidence warning */}
        {message.low_confidence_warning && !message.streaming && (
          <p className="mt-2 text-xs text-[var(--warning)] border border-[var(--warning)] rounded px-2 py-1">
            ⚠ This answer may be incomplete based on available context.
          </p>
        )}

        {/* Pipeline status badge */}
        {meta && !isUser && !message.streaming && (
          <p className={`mt-1 text-xs ${meta.color}`}>{meta.label}</p>
        )}

        {/* Cost badge (below message, small) */}
        {!isUser && !message.streaming && (
          <CostBadge message={message} adminMode={debugMode} />
        )}

        {/* Debug panel */}
        {!isUser && !message.streaming && debugMode && (
          <DebugPanel message={message} />
        )}
      </div>
    </div>
  )
}
