// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { login, tryRestoreSession } from '@/lib/auth'
import { getAccessToken } from '@/lib/api'

const DEV_EMAIL    = 'dev@neuralflow.ai'
const DEV_PASSWORD = 'devpass'

export function LoginPage() {
  const navigate = useNavigate()
  const [email,    setEmail]    = useState(DEV_EMAIL)
  const [password, setPassword] = useState(DEV_PASSWORD)
  const [loading,  setLoading]  = useState(false)
  const [checking, setChecking] = useState(true)
  const [error,    setError]    = useState<string | null>(null)

  useEffect(() => {
    if (getAccessToken()) { navigate('/chat', { replace: true }); return }
    tryRestoreSession().then(ok => {
      if (ok) navigate('/chat', { replace: true })
      else    setChecking(false)
    })
  }, [navigate])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await login(email, password)
      navigate('/chat')
    } catch (err: any) {
      setError(err.message ?? 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  if (checking) return null

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-[var(--accent)] mb-4">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <h1 className="text-xl font-semibold text-[var(--text)]">Knowledge</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">RAG v2 — Sign in to continue</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 space-y-4">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1.5">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--accent)] transition-colors" />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1.5">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--accent)] transition-colors" />
          </div>
          {error && (
            <p className="text-xs text-[var(--error)] bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>
          )}
          <button type="submit" disabled={loading}
            className="w-full bg-[var(--accent)] hover:bg-[#3d5de6] disabled:opacity-50 text-white rounded-lg py-2.5 text-sm font-medium transition-colors">
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
          <p className="text-center text-xs text-[var(--text-muted)]">Dev credentials are pre-filled — just click Sign in</p>
        </form>
      </div>
    </div>
  )
}
