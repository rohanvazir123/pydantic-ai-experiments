import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from './types'
import { CostBadge } from './CostBadge'
import { DebugPanel } from './DebugPanel'

interface Props {
  message: Message
  debugMode?: boolean
}

const STATUS_LABELS: Record<string, { label: string; className: string }> = {
  answered:            { label: 'Answered',              className: 'status-ok' },
  abstained_retrieval: { label: 'No results found',      className: 'status-warn' },
  abstained_citation:  { label: 'Could not cite sources', className: 'status-warn' },
  abstained_judge:     { label: 'Could not verify',       className: 'status-warn' },
}

export function MessageBubble({ message, debugMode = false }: Props) {
  const isUser = message.role === 'user'
  const meta = message.status ? STATUS_LABELS[message.status] : null

  return (
    <div className={`message-row ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
        {message.streaming ? (
          <span className="streaming-text">
            {message.content}
            <span className="typing-dots">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </span>
          </span>
        ) : (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}

        {message.low_confidence_warning && !message.streaming && (
          <p className="low-confidence-warning">
            This answer may be incomplete based on available context.
          </p>
        )}

        {meta && !isUser && !message.streaming && (
          <p className={`status-label ${meta.className}`}>{meta.label}</p>
        )}

        {!isUser && !message.streaming && <CostBadge message={message} adminMode={debugMode} />}

        {!isUser && !message.streaming && debugMode && <DebugPanel message={message} />}
      </div>
    </div>
  )
}
