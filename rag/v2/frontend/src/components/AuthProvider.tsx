'use client'
import { useEffect, useRef, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { tryRestoreSession } from '@/lib/auth'
import { getAccessToken } from '@/lib/api'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router   = useRouter()
  const pathname = usePathname()
  const checked  = useRef(false)
  const [ready, setReady] = useState(false)   // block render until auth resolved

  useEffect(() => {
    if (checked.current) return
    checked.current = true

    if (getAccessToken()) { setReady(true); return }

    tryRestoreSession().then(ok => {
      if (ok) { setReady(true) }
      else    { router.replace('/login') }
    })
  }, [pathname, router])

  // Render a blank screen while auth resolves — must NOT return null,
  // as Next.js 15 treats null from a layout as not-found and shows the 404 boundary.
  if (!ready) return <div className="min-h-screen bg-[var(--bg)]" />

  return <>{children}</>
}
