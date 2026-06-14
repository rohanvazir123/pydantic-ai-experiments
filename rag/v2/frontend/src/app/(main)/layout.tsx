'use client'
import { NavBar }       from '@/components/NavBar'
import { AuthProvider } from '@/components/AuthProvider'

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      {/* Fixed 56px icon nav on the left */}
      <NavBar />
      {/* Offset content so the nav doesn't overlap */}
      <div className="pl-14 min-h-screen">
        {children}
      </div>
    </AuthProvider>
  )
}
