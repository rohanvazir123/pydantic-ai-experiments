'use client'
import type { Citation } from '@/types/chat'

interface Props { citations: Citation[] }

function ConfidenceBadge({ score }: { score: number }) {
  const pct   = Math.round(score * 100)
  const color = score >= 0.7 ? 'bg-green-500' : score >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-1 text-xs">
      <div className="w-16 h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[var(--text-muted)]">{pct}%</span>
    </div>
  )
}

export function CitationPanel({ citations }: Props) {
  if (!citations.length) return null

  return (
    <div className="w-64 shrink-0 border-l border-[var(--border)] p-4 overflow-y-auto">
      <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
        Sources
      </h3>
      <div className="space-y-3">
        {citations.map(c => (
          <div key={c.chunk_id} className="bg-[var(--surface)] rounded-lg p-3 text-xs">
            <p className="font-medium text-[var(--text)] truncate" title={c.document_title}>
              {c.document_title}
            </p>
            <p className="text-[var(--text-muted)] truncate text-[10px] mb-1">{c.document_source}</p>
            <ConfidenceBadge score={c.relevance_score} />
            <p className="mt-1 text-[var(--text-muted)] line-clamp-2">{c.excerpt}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
