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

  // Render nothing until we know the user is authenticated.
  // This prevents clicks on child elements before auth is confirmed.
  if (!ready) return null

  return <>{children}</>
}
