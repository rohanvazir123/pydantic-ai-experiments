import type { Message } from './types'

interface Props {
  message: Message
  adminMode: boolean
}

export function CostBadge({ message, adminMode }: Props) {
  if (!adminMode || message.cost_usd == null) return null

  return <p className="cost-badge">${message.cost_usd.toFixed(4)}</p>
}
