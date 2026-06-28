
import { useState, useEffect } from 'react'
import { api } from '@/lib/api'
import { Trash2, Plus } from 'lucide-react'
import toast from 'react-hot-toast'

interface Memory { id: string; content: string; created_at?: string }

export function MemoriesPage() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [newText,  setNewText]  = useState('')
  const [loading,  setLoading]  = useState(false)

  useEffect(() => { load() }, [])

  async function load() {
    setLoading(true)
    try { setMemories((await api.get<Memory[]>('/memories')) ?? []) }
    finally { setLoading(false) }
  }

  async function add() {
    if (!newText.trim()) return
    try {
      await api.post('/memories', { content: newText })
      setNewText('')
      toast.success('Memory added')
      load()
    } catch (e: any) { toast.error(e.message) }
  }

  async function del(id: string) {
    try {
      await api.delete(`/memories/${id}`)
      setMemories(prev => prev.filter(m => m.id !== id))
    } catch (e: any) { toast.error(e.message) }
  }

  async function delAll() {
    if (!confirm('Delete ALL memories? This cannot be undone.')) return
    try { await api.delete('/memories'); setMemories([]); toast.success('All memories deleted') }
    catch (e: any) { toast.error(e.message) }
  }

  return (
    <div className="p-6 bg-[var(--bg)] min-h-screen max-w-2xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-lg font-semibold">My Memories</h1>
        {memories.length > 0 && (
          <button onClick={delAll} className="text-xs text-[var(--error)] hover:underline">Delete all</button>
        )}
      </div>

      {/* Add memory */}
      <div className="flex gap-2 mb-6">
        <input value={newText} onChange={e => setNewText(e.target.value)}
          placeholder="Add a fact about yourself..."
          onKeyDown={e => e.key === 'Enter' && add()}
          className="flex-1 bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm" />
        <button onClick={add} className="bg-[var(--accent)] text-white p-2 rounded-lg hover:bg-[#3d5de6] transition-colors">
          <Plus size={16} />
        </button>
      </div>

      {loading && <p className="text-[var(--text-muted)] text-sm">Loading…</p>}

      <div className="space-y-3">
        {memories.map(m => (
          <div key={m.id} className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-4 py-3 flex items-start gap-3">
            <span className="text-sm flex-1">{m.content}</span>
            <button onClick={() => del(m.id)} className="text-[var(--text-muted)] hover:text-[var(--error)] transition-colors shrink-0">
              <Trash2 size={14} />
            </button>
          </div>
        ))}
        {!loading && !memories.length && (
          <p className="text-[var(--text-muted)] text-sm text-center py-8">No memories yet. The system extracts them automatically from conversations.</p>
        )}
      </div>
    </div>
  )
}
