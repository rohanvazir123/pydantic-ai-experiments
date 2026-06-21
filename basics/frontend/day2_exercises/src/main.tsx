import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { App1, App2, App3 } from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div style={{ display: 'flex', gap: '1rem', padding: '1.5rem', alignItems: 'flex-start', minHeight: '100vh', background: '#334155', borderRadius: '16px' }}>
      <div style={{ flex: 1 }}><App1 /></div>
      <div style={{ flex: 1 }}><App2 /></div>
      <div style={{ flex: 1 }}><App3 /></div>
    </div>
  </StrictMode>,
)
