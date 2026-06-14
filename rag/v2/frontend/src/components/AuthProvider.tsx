'use client'
import { useEffect, useRef } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { tryRestoreSession } from '@/lib/auth'
import { getAccessToken } from '@/lib/api'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router   = useRouter()
  const pathname = usePathname()
  const restored = useRef(false)

  useEffect(() => {
    if (restored.current) return
    restored.current = true

    if (pathname === '/login') return

    // If we already have an in-memory token, nothing to do
    if (getAccessToken()) return

    // Try to restore from refresh cookie; if fails, send to login
    tryRestoreSession().then(ok => {
      if (!ok) router.replace('/login')
    })
  }, [pathname, router])

  return <>{children}</>
}
