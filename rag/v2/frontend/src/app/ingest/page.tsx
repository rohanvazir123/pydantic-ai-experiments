'use client'
import { useState } from 'react'
import { api } from '@/lib/api'
import toast from 'react-hot-toast'
import { Upload } from 'lucide-react'

interface JobStatus { job_id: string; status: string; progress: number; chunks_ingested?: number; error?: string }

export default function IngestPage() {
  const [corpusId,  setCorpusId]  = useState('neuralflow')
  const [sourcePath, setSourcePath] = useState('../../rag/documents')
  const [mode,      setMode]      = useState<'incremental' | 'full'>('incremental')
  const [jobs,      setJobs]      = useState<JobStatus[]>([])
  const [loading,   setLoading]   = useState(false)

  async function submitIngest() {
    setLoading(true)
    try {
      const res = await api.post<{ job_id: string; status: string }>('/ingest', {
        corpus_id: corpusId, source_path: sourcePath, mode,
        enable_graph_extraction: false,
      })
      setJobs(prev => [{ job_id: res.job_id, status: 'queued', progress: 0 }, ...prev])
      toast.success(`Job ${res.job_id.slice(0, 8)}… queued`)
      pollStatus(res.job_id)
    } catch (e: any) {
      toast.error(e.message ?? 'Ingest failed')
    } finally {
      setLoading(false)
    }
  }

  async function pollStatus(jobId: string) {
    const interval = setInterval(async () => {
      try {
        const s = await api.get<JobStatus>(`/ingest/${jobId}/status`)
        setJobs(prev => prev.map(j => j.job_id === jobId ? s : j))
        if (['completed', 'failed'].includes(s.status)) clearInterval(interval)
      } catch { clearInterval(interval) }
    }, 2000)
  }

  return (
    <div className="p-6 bg-[var(--bg)] min-h-screen">
      <h1 className="text-lg font-semibold mb-6">Ingestion</h1>

      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-xl p-6 max-w-xl space-y-4">
        <div>
          <label className="text-xs text-[var(--text-muted)] block mb-1">Corpus ID</label>
          <input value={corpusId} onChange={e => setCorpusId(e.target.value)}
            className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm" />
        </div>
        <div>
          <label className="text-xs text-[var(--text-muted)] block mb-1">Source Path</label>
          <input value={sourcePath} onChange={e => setSourcePath(e.target.value)}
            className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm font-mono" />
        </div>
        <div className="flex gap-3">
          {(['incremental', 'full'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${mode === m ? 'bg-[var(--accent)] text-white' : 'bg-[var(--bg)] border border-[var(--border)] text-[var(--text-muted)]'}`}>
              {m}
            </button>
          ))}
        </div>
        <button onClick={submitIngest} disabled={loading}
          className="flex items-center gap-2 bg-[var(--accent)] text-white px-4 py-2 rounded-lg text-sm hover:bg-[#3d5de6] disabled:opacity-50 transition-colors">
          <Upload size={16} /> Submit Ingest Job
        </button>
      </div>

      {/* Job list */}
      {jobs.length > 0 && (
        <div className="mt-6 space-y-3 max-w-xl">
          <h2 className="text-sm font-medium text-[var(--text-muted)]">Recent Jobs</h2>
          {jobs.map(j => (
            <div key={j.job_id} className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-mono text-xs text-[var(--text-muted)]">{j.job_id.slice(0, 16)}…</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${j.status === 'completed' ? 'bg-green-500/20 text-green-400' : j.status === 'failed' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                  {j.status}
                </span>
              </div>
              <div className="w-full bg-[var(--border)] rounded-full h-1.5">
                <div className="bg-[var(--accent)] h-1.5 rounded-full transition-all" style={{ width: `${j.progress}%` }} />
              </div>
              {j.chunks_ingested != null && (
                <p className="text-xs text-[var(--text-muted)] mt-1">{j.chunks_ingested.toLocaleString()} chunks</p>
              )}
              {j.error && <p className="text-xs text-[var(--error)] mt-1">{j.error}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
