'use client'
import type { Message } from '@/types/chat'

interface Props { message: Message; adminMode?: boolean }

export function CostBadge({ message, adminMode = false }: Props) {
  const { estimated_cost_usd, prompt_tokens, completion_tokens, model_tier_used, latency_ms } = message

  // Only show when there's something to display
  if (!estimated_cost_usd && !prompt_tokens && !latency_ms) return null
  // Hide from non-admin users unless cost is non-zero (free models)
  if (!adminMode && !estimated_cost_usd) return null

  const totalMs = latency_ms
    ? Object.values(latency_ms).reduce((a, b) => a + b, 0)
    : null

  return (
    <div className="flex items-center gap-3 text-xs text-[var(--text-muted)] mt-1 font-mono">
      {estimated_cost_usd != null && estimated_cost_usd > 0 && (
        <span title="Estimated cost">${estimated_cost_usd.toFixed(4)}</span>
      )}
      {(prompt_tokens || completion_tokens) && (
        <span title="Token usage">
          {((prompt_tokens ?? 0) + (completion_tokens ?? 0)).toLocaleString()} tok
        </span>
      )}
      {model_tier_used && (
        <span title="Model tier" className="capitalize">{model_tier_used}</span>
      )}
      {totalMs && (
        <span title="Total pipeline latency">{totalMs}ms</span>
      )}
    </div>
  )
}
