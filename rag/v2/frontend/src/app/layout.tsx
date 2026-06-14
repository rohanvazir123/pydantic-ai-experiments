import type { Metadata } from 'next'
import './globals.css'
import { Toaster } from 'react-hot-toast'

export const metadata: Metadata = {
  title: 'Knowledge — RAG v2',
  description: 'Multi-corpus knowledge assistant',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[var(--bg)] text-[var(--text)] min-h-screen">
        {children}
        <Toaster position="bottom-right" toastOptions={{
          style: { background: '#ffffff', color: '#111827', border: '1px solid #e1e4eb' },
        }} />
      </body>
    </html>
  )
}
