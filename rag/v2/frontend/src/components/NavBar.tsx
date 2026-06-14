'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare, Upload, Brain, Terminal, LogOut } from 'lucide-react'
import { logout } from '@/lib/auth'
import { useRouter } from 'next/navigation'

const NAV_ITEMS = [
  { href: '/chat',      icon: MessageSquare, label: 'Chat'      },
  { href: '/ingest',    icon: Upload,         label: 'Ingest'    },
  { href: '/memories',  icon: Brain,          label: 'Memories'  },
  { href: '/logs',      icon: Terminal,       label: 'Logs'      },
]

export function NavBar() {
  const pathname = usePathname()
  const router   = useRouter()

  if (pathname === '/login') return null

  async function handleLogout() {
    await logout()
    router.push('/login')
  }

  return (
    <nav className="fixed left-0 top-0 bottom-0 w-14 bg-[var(--surface)] border-r border-[var(--border)] flex flex-col items-center py-3 z-50">
      {/* Logo mark */}
      <div className="w-8 h-8 rounded-lg bg-[var(--accent)] flex items-center justify-center mb-6 shrink-0">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
        </svg>
      </div>

      {/* Nav links */}
      <div className="flex flex-col items-center gap-1 flex-1">
        {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
          const active = pathname.startsWith(href)
          return (
            <Link
              key={href}
              href={href}
              title={label}
              className={`w-10 h-10 rounded-lg flex items-center justify-center transition-colors group relative ${
                active
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--border)] hover:text-[var(--text)]'
              }`}
            >
              <Icon size={18} />
              {/* Tooltip */}
              <span className="absolute left-12 px-2 py-1 bg-[var(--surface)] border border-[var(--border)] text-xs text-[var(--text)] rounded whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                {label}
              </span>
            </Link>
          )
        })}
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        title="Sign out"
        className="w-10 h-10 rounded-lg flex items-center justify-center text-[var(--text-muted)] hover:bg-[var(--border)] hover:text-[var(--error)] transition-colors group relative"
      >
        <LogOut size={18} />
        <span className="absolute left-12 px-2 py-1 bg-[var(--surface)] border border-[var(--border)] text-xs text-[var(--text)] rounded whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
          Sign out
        </span>
      </button>
    </nav>
  )
}
