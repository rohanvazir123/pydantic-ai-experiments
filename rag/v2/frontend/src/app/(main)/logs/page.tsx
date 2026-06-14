'use client'
import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '@/lib/api'
import { RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'

interface LogEntry {
  level:             string
  timestamp?:        string
  message?:          string
  service?:          string
  request_id?:       string
  session_id?:       string
  user_id?:          string
  tenant_id?:        string
  route?:            string
  status?:           number
  latency_ms?:       number
  stage?:            string
  duration_ms?:      number
  chunk_count?:      number
  job_id?:           string
  corpus_id?:        string
  pipeline_status?:  string
  [key: string]: unknown
}

// ── Level config ──────────────────────────────────────────────────────────────

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const
type Level = typeof LEVELS[number]

const LEVEL_STYLE: Record<Level, { badge: string; row: string; label: string }> = {
  DEBUG:    { badge: 'bg-[var(--border)] text-[var(--text-muted)]',    row: '',                     label: 'DBG' },
  INFO:     { badge: 'bg-blue-500/20 text-blue-400',                   row: '',                     label: 'INF' },
  WARNING:  { badge: 'bg-yellow-500/20 text-yellow-400',               row: 'bg-yellow-500/5',      label: 'WRN' },
  ERROR:    { badge: 'bg-red-500/20 text-[var(--error)]',              row: 'bg-red-500/5',         label: 'ERR' },
  CRITICAL: { badge: 'bg-red-700/30 text-red-300 font-bold',           row: 'bg-red-700/10',        label: 'CRT' },
}

function LatencyCell({ ms }: { ms?: number }) {
  if (ms == null) return <span className="text-[var(--text-muted)]">—</span>
  const color = ms > 5000 ? 'text-red-400' : ms > 2000 ? 'text-yellow-400' : ms > 500 ? 'text-orange-400' : 'text-green-400'
  return <span className={color}>{ms.toLocaleString()}ms</span>
}

// ── Expandable row ────────────────────────────────────────────────────────────

function LogRow({ log, idx }: { log: LogEntry; idx: number }) {
  const [open, setOpen] = useState(false)
  const level = (log.level?.toUpperCase() ?? 'INFO') as Level
  const style = LEVEL_STYLE[level] ?? LEVEL_STYLE.INFO

  return (
    <>
      <tr
        onClick={() => setOpen(v => !v)}
        className={`border-b border-[var(--border)] cursor-pointer transition-colors hover:bg-[var(--surface)] ${style.row}`}
      >
        <td className="px-2 py-1.5 text-[var(--text-muted)] whitespace-nowrap w-4">
          {open
            ? <ChevronDown size={12} className="text-[var(--text-muted)]" />
            : <ChevronRight size={12} className="text-[var(--text-muted)]" />}
        </td>
        <td className="px-2 py-1.5 text-[var(--text-muted)] whitespace-nowrap tabular-nums">
          {log.timestamp ? new Date(log.timestamp).toLocaleTimeString([], { hour12: false }) : '—'}
        </td>
        <td className="px-2 py-1.5">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${style.badge}`}>
            {style.label}
          </span>
        </td>
        <td className="px-2 py-1.5 text-[var(--text-muted)] whitespace-nowrap">{log.service ?? '—'}</td>
        <td className="px-2 py-1.5 text-[var(--text)] max-w-xs truncate" title={log.message ?? ''}>
          {log.message ?? log.route ?? log.stage ?? '—'}
        </td>
        <td className="px-2 py-1.5 whitespace-nowrap">
          <LatencyCell ms={log.latency_ms ?? log.duration_ms} />
        </td>
        <td className="px-2 py-1.5">
          {log.status != null
            ? <span className={log.status >= 400 ? 'text-[var(--error)]' : 'text-green-400'}>{log.status}</span>
            : log.pipeline_status
            ? <span className={log.pipeline_status === 'answered' ? 'text-green-400' : 'text-[var(--warning)]'}>{log.pipeline_status}</span>
            : '—'}
        </td>
        <td className="px-2 py-1.5 text-[var(--text-muted)] truncate max-w-[180px] font-mono text-[10px]"
          title={log.request_id ?? log.job_id ?? ''}>
          {log.request_id ?? log.job_id ?? '—'}
        </td>
      </tr>

      {/* Expanded raw JSON transcript */}
      {open && (
        <tr className="border-b border-[var(--border)] bg-[var(--surface)]">
          <td colSpan={8} className="px-4 py-3">
            <pre className="text-[11px] font-mono text-[var(--text-muted)] whitespace-pre-wrap break-all max-h-64 overflow-y-auto leading-relaxed">
              {JSON.stringify(log, null, 2)}
            </pre>
          </td>
        </tr>
      )}
    </>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

const AUTO_REFRESH_OPTIONS = [
  { label: 'Off',  value: 0    },
  { label: '5s',   value: 5000 },
  { label: '15s',  value: 15000 },
  { label: '30s',  value: 30000 },
]

export default function LogsPage() {
  const [logs,         setLogs]         = useState<LogEntry[]>([])
  const [loading,      setLoading]      = useState(false)
  const [activeLevels, setActiveLevels] = useState<Set<Level>>(new Set(['INFO', 'WARNING', 'ERROR', 'CRITICAL']))
  const [service,      setService]      = useState('')
  const [rid,          setRid]          = useState('')
  const [limit,        setLimit]        = useState(200)
  const [autoMs,       setAutoMs]       = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    try {
      // Always fetch DEBUG and above — level toggle is client-side for instant response
      const params = new URLSearchParams({ level: 'DEBUG', limit: String(limit) })
      if (service) params.set('service', service)
      if (rid)     params.set('request_id', rid)
      const data = await api.get<LogEntry[]>(`/logs?${params}`)
      setLogs(data ?? [])
    } catch {
      // silently ignore — API may not be running yet
    } finally {
      setLoading(false)
    }
  }, [limit, service, rid])

  // Initial fetch + re-fetch when filters change
  useEffect(() => { fetchLogs() }, [fetchLogs])

  // Auto-refresh interval
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (autoMs > 0) {
      intervalRef.current = setInterval(fetchLogs, autoMs)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [autoMs, fetchLogs])

  function toggleLevel(lvl: Level) {
    setActiveLevels(prev => {
      const next = new Set(prev)
      next.has(lvl) ? next.delete(lvl) : next.add(lvl)
      return next
    })
  }

  // Client-side level filter (no API call needed)
  const visibleLogs = logs.filter(l => activeLevels.has((l.level?.toUpperCase() ?? 'INFO') as Level))

  return (
    <div className="p-5 h-screen flex flex-col bg-[var(--bg)] gap-3">

      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <h1 className="text-base font-semibold">System Logs</h1>
        <div className="flex items-center gap-2">
          {/* Auto-refresh */}
          <div className="flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
            <span>Auto-refresh</span>
            <div className="flex rounded-lg border border-[var(--border)] overflow-hidden">
              {AUTO_REFRESH_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setAutoMs(opt.value)}
                  className={`px-2.5 py-1 text-xs transition-colors ${
                    autoMs === opt.value
                      ? 'bg-[var(--accent)] text-white'
                      : 'bg-[var(--surface)] text-[var(--text-muted)] hover:bg-[var(--border)]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Manual refresh */}
          <button
            onClick={fetchLogs}
            disabled={loading}
            className="flex items-center gap-1.5 text-sm bg-[var(--surface)] border border-[var(--border)] px-3 py-1.5 rounded-lg hover:border-[var(--accent)] transition-colors"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Level toggles + filters */}
      <div className="flex items-center gap-3 flex-wrap shrink-0">

        {/* Level toggle chips */}
        <div className="flex items-center gap-1.5">
          {LEVELS.map(lvl => {
            const on = activeLevels.has(lvl)
            const s  = LEVEL_STYLE[lvl]
            return (
              <button
                key={lvl}
                onClick={() => toggleLevel(lvl)}
                title={`Toggle ${lvl} logs`}
                className={`px-2.5 py-1 rounded-full text-[11px] font-mono border transition-all ${
                  on
                    ? `${s.badge} border-transparent`
                    : 'bg-transparent text-[var(--text-muted)] border-[var(--border)] opacity-40'
                }`}
              >
                {lvl}
              </button>
            )
          })}
        </div>

        <div className="w-px h-5 bg-[var(--border)]" />

        {/* Service filter */}
        <input
          value={service}
          onChange={e => setService(e.target.value)}
          placeholder="service filter"
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-2.5 py-1.5 text-xs w-36 focus:outline-none focus:border-[var(--accent)]"
        />

        {/* request_id filter */}
        <input
          value={rid}
          onChange={e => setRid(e.target.value)}
          placeholder="request_id or job_id"
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-2.5 py-1.5 text-xs w-56 font-mono focus:outline-none focus:border-[var(--accent)]"
        />

        {/* Limit */}
        <select
          value={limit}
          onChange={e => setLimit(Number(e.target.value))}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-2.5 py-1.5 text-xs"
        >
          {[100, 200, 500, 1000, 2000].map(n => <option key={n} value={n}>{n} entries</option>)}
        </select>

        <span className="text-xs text-[var(--text-muted)] ml-auto">
          {visibleLogs.length} / {logs.length} shown
        </span>
      </div>

      {/* Log table */}
      <div className="flex-1 overflow-auto rounded-lg border border-[var(--border)] min-h-0">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-[var(--surface)] border-b border-[var(--border)] z-10">
            <tr>
              <th className="px-2 py-2 w-4" />
              {['Time', 'Level', 'Service', 'Message / Route', 'Latency', 'Status', 'request_id / job_id'].map(h => (
                <th key={h} className="px-2 py-2 text-left text-[var(--text-muted)] font-normal whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleLogs.map((log, i) => (
              <LogRow key={i} log={log} idx={i} />
            ))}
            {!visibleLogs.length && !loading && (
              <tr>
                <td colSpan={8} className="px-4 py-10 text-center text-[var(--text-muted)]">
                  {logs.length === 0 ? 'No logs yet' : 'No logs match the active level filters'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
