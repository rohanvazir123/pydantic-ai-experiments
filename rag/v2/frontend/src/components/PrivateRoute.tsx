// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

import { useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import { getAccessToken } from '@/lib/api'
import { tryRestoreSession } from '@/lib/auth'

export function PrivateRoute({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<'checking' | 'ok' | 'unauth'>('checking')

  useEffect(() => {
    if (getAccessToken()) { setStatus('ok'); return }
    tryRestoreSession().then(ok => setStatus(ok ? 'ok' : 'unauth'))
  }, [])

  if (status === 'checking') return <div className="min-h-screen bg-[var(--bg)]" />
  if (status === 'unauth')   return <Navigate to="/login" replace />
  return <>{children}</>
}
