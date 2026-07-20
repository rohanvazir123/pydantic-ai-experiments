import { useState } from 'react'
import { MessageBubble } from './MessageBubble'
import type { Message } from './types'

const sampleMessages: Message[] = [
  {
    id: '1',
    role: 'user',
    content: 'What does NeuralFlow AI do?',
  },
  {
    id: '2',
    role: 'assistant',
    content: 'NeuralFlow AI builds a **multi-tenant RAG platform** for enterprise knowledge search.',
    status: 'answered',
    cost_usd: 0.0021,
    citations: [{ id: 'c1', title: 'company-overview.md' }],
  },
  {
    id: '3',
    role: 'assistant',
    content: 'Based on partial context, PTO appears to accrue monthly, but the source is unclear.',
    status: 'abstained_citation',
    low_confidence_warning: true,
    cost_usd: 0.0009,
  },
  {
    id: '4',
    role: 'assistant',
    content: 'Thinking',
    streaming: true,
  },
]

export default function App() {
  const [debugMode, setDebugMode] = useState(false)

  return (
    <div className="app">
      <label className="debug-toggle">
        <input type="checkbox" checked={debugMode} onChange={e => setDebugMode(e.target.checked)} />
        debugMode
      </label>

      {sampleMessages.map(msg => (
        <MessageBubble key={msg.id} message={msg} debugMode={debugMode} />
      ))}

      <p className="note">
        Try deleting <code>debugMode={'{debugMode}'}</code> from one of the
        <code>&lt;MessageBubble /&gt;</code> calls above — nothing breaks, because the prop
        is optional and falls back to <code>false</code>.
      </p>
    </div>
  )
}
