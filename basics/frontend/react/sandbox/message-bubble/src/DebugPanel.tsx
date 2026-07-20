import type { Message } from './types'

interface Props {
  message: Message
}

export function DebugPanel({ message }: Props) {
  const debugInfo = { status: message.status ?? null, citations: message.citations ?? [] }

  return <pre className="debug-panel">{JSON.stringify(debugInfo, null, 2)}</pre>
}
