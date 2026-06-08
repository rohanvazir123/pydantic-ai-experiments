'use client'
import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import { RefreshCw } from 'lucide-react'

interface LogEntry {
  level:        string
  timestamp:    string
  service?:     string
  request_id?:  string
  session_id?:  string
  user_id?:     string
  tenant_id?:   string
  route?:       string
  status?:      number
  latency_ms?:  number
  // Ingestion / retrieval stage latencies
  stage?:       string
  duration_ms?: number
  chunk_count?: number
  job_id?:      string
  corpus_id?:   string
  pipeline_status?: string
  [key: string]: unknown
}

const LEVEL_COLOR: Record<string, string> = {
  DEBUG:    'text-[var(--text-muted)]',
  INFO:     'text-blue-400',
  WARNING:  'text-[var(--warning)]',
  ERROR:    'text-[var(--error)]',
  CRITICAL: 'text-red-300 font-bold',
}

function LatencyCell({ ms }: { ms?: number }) {
  if (ms == null) return <span className="text-[var(--text-muted)]">—</span>
  const color = ms > 5000 ? 'text-red-400'
              : ms > 2000 ? 'text-yellow-400'
              : ms > 500  ? 'text-orange-400'
              : 'text-green-400'
  return <span className={color}>{ms.toLocaleString()}ms</span>
}

export default function LogsPage() {
  const [logs,    setLogs]    = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [level,   setLevel]   = useState('INFO')
  const [service, setService] = useState('')
  const [rid,     setRid]     = useState('')
  const [limit,   setLimit]   = useState(100)

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ level, limit: String(limit) })
      if (service) params.set('service', service)
      if (rid)     params.set('request_id', rid)
      const data = await api.get<LogEntry[]>(`/logs?${params}`)
      setLogs(data ?? [])
    } finally {
      setLoading(false)
    }
  }, [level, service, rid, limit])

  useEffect(() => { fetchLogs() }, [fetchLogs])

  return (
    <div className="p-6 h-screen flex flex-col bg-[var(--bg)]">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-semibold">System Logs</h1>
        <button
          onClick={fetchLogs}
          disabled={loading}
          className="flex items-center gap-2 text-sm bg-[var(--surface)] border border-[var(--border)] px-3 py-1.5 rounded-lg hover:border-[var(--accent)] transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <select value={level} onChange={e => setLevel(e.target.value)}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1.5 text-sm">
          {['DEBUG','INFO','WARNING','ERROR','CRITICAL'].map(l =>
            <option key={l}>{l}</option>)}
        </select>
        <input value={service} onChange={e => setService(e.target.value)}
          placeholder="service (api / ingest-worker)"
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1.5 text-sm w-48" />
        <input value={rid} onChange={e => setRid(e.target.value)}
          placeholder="request_id"
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1.5 text-sm w-80 font-mono" />
        <select value={limit} onChange={e => setLimit(Number(e.target.value))}
          className="bg-[var(--surface)] border border-[var(--border)] rounded px-3 py-1.5 text-sm">
          {[50, 100, 200, 500].map(n => <option key={n}>{n}</option>)}
        </select>
      </div>

      {/* Log table */}
      <div className="flex-1 overflow-auto rounded-lg border border-[var(--border)]">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-[var(--surface)] border-b border-[var(--border)]">
            <tr>
              {['Time','Level','Service','Route / Stage','Latency','Status','request_id / job_id'].map(h => (
                <th key={h} className="px-3 py-2 text-left text-[var(--text-muted)] font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {logs.map((log, i) => (
              <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--surface)] transition-colors">
                <td className="px-3 py-1.5 text-[var(--text-muted)] whitespace-nowrap">
                  {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '—'}
                </td>
                <td className={`px-3 py-1.5 ${LEVEL_COLOR[log.level] ?? ''}`}>{log.level}</td>
                <td className="px-3 py-1.5 text-[var(--text-muted)]">{log.service ?? '—'}</td>
                <td className="px-3 py-1.5">{log.route ?? log.stage ?? '—'}</td>
                <td className="px-3 py-1.5">
                  <LatencyCell ms={log.latency_ms ?? log.duration_ms} />
                </td>
                <td className="px-3 py-1.5">
                  {log.status != null
                    ? <span className={log.status >= 400 ? 'text-[var(--error)]' : 'text-green-400'}>{log.status}</span>
                    : log.pipeline_status
                    ? <span className={log.pipeline_status === 'answered' ? 'text-green-400' : 'text-[var(--warning)]'}>{log.pipeline_status}</span>
                    : '—'}
                </td>
                <td className="px-3 py-1.5 text-[var(--text-muted)] truncate max-w-[200px]" title={log.request_id ?? log.job_id ?? ''}>
                  {log.request_id ?? log.job_id ?? '—'}
                </td>
              </tr>
            ))}
            {!logs.length && !loading && (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-[var(--text-muted)]">No logs</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
