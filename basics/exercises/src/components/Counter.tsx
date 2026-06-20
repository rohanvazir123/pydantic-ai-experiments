import { useState } from 'react'

interface Props {
  label: string
  startAt?: number
}

export function Counter({ label, startAt = 0 }: Props) {
  const [count, setCount] = useState(startAt)

  // derived value — do NOT put this in useState
  const isNegative = count < 0

  return (
    <div style={{
      background: 'var(--card-bg, #f9f9f9)',
      border: '1px solid var(--border, #e0e0e0)',
      borderRadius: '12px',
      padding: '20px 24px',
    }}>
      <p style={{ margin: '0 0 4px', fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.5 }}>
        {label}
      </p>
      <p style={{ margin: '0 0 16px', fontSize: '3rem', fontWeight: 700, lineHeight: 1 }}>
        {count}
      </p>
      <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
        <button onClick={() => setCount(prev => prev - 1)} style={{ width: '40px' }}>−</button>
        <button onClick={() => setCount(0)} style={{ padding: '0.6em 1em', fontSize: '0.85em' }}>Reset</button>
        <button onClick={() => setCount(prev => prev + 1)} style={{ width: '40px' }}>+</button>
      </div>
      {isNegative && (
        <p style={{ margin: '12px 0 0', fontSize: '13px', color: '#ef4444' }}>Gone negative!</p>
      )}
    </div>
  )
}
