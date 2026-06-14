'use client'
import { useState, useEffect, useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { useChat }      from '@/hooks/useChat'
import { MessageBubble } from '@/components/chat/MessageBubble'
import { CitationPanel } from '@/components/chat/CitationPanel'
import { InputBar }      from '@/components/chat/InputBar'
import { api }           from '@/lib/api'
import { Database }      from 'lucide-react'
import type { Citation } from '@/types/chat'

interface CorpusInfo {
  id:           string
  display_name: string
}

const SUGGESTED_QUESTIONS = [
  // Company overview
  'What does NeuralFlow AI do?',
  'When was NeuralFlow AI founded and what is its mission?',
  'What industries does NeuralFlow AI serve?',
  'What are NeuralFlow AI\'s main products and services?',
  'Who are NeuralFlow AI\'s typical clients?',
  'What makes NeuralFlow AI different from competitors?',
  'What is the company vision for the next 3 years?',

  // Team & culture
  'What are the core principles and company culture?',
  'What does the onboarding process look like for new hires?',
  'How is performance evaluated at NeuralFlow AI?',
  'What is the management structure?',
  'How does the company support professional development?',
  'What are the team collaboration tools and practices?',
  'How does the company handle remote and hybrid work?',
  'What is the hiring process?',
  'What values guide decision-making at NeuralFlow AI?',

  // HR & policies
  'What is the PTO and leave policy?',
  'What are the working hours and flexibility options?',
  'What benefits does NeuralFlow AI offer?',
  'What is the parental leave policy?',
  'How does the company handle sick leave?',
  'What is the expense reimbursement policy?',
  'What is the code of conduct?',
  'How is compensation structured?',
  'What is the bonus and equity policy?',
  'What are the travel and accommodation policies?',

  // Business performance
  'What were the key outcomes from the latest business review?',
  'What were the Q4 financial highlights?',
  'How did revenue compare to targets?',
  'What are the key growth metrics for the year?',
  'What new clients were acquired recently?',
  'Which business units performed best?',
  'What were the main challenges discussed in the review?',
  'What are the strategic priorities going forward?',
  'What investments were made in the last quarter?',
  'What cost savings were achieved?',
  'What is the revenue forecast?',

  // Technology & architecture
  'What technologies and tools does the team use?',
  'What is the technology stack for AI projects?',
  'How does the company approach data privacy and security?',
  'What cloud infrastructure is used?',
  'What are the main software development practices?',
  'How are AI models deployed to production?',
  'What is the QA and testing approach?',
  'How does the team handle technical debt?',
  'What monitoring and observability tools are used?',
  'How is the data pipeline structured?',
  'What are the API design standards?',

  // Projects & clients
  'What projects are currently in progress?',
  'What was the GlobalFinance client review about?',
  'What were the outcomes of recent client engagements?',
  'How does the company manage project delivery?',
  'What is the typical project lifecycle?',
  'How are client requirements gathered?',
  'What is the escalation process for project issues?',
  'How are project milestones tracked?',

  // Meetings & decisions
  'What were the key decisions from the January 8 meeting?',
  'What were the key decisions from the January 15 meeting?',
  'What action items came out of the latest team meeting?',
  'What topics were discussed in the recent all-hands?',
  'Who owns the follow-up items from the last meeting?',
  'What were the blockers discussed in the latest standup?',

  // Research & knowledge
  'What is the CLIP model and how is it used?',
  'What is Retrieval Augmented Generation (RAG)?',
  'What does the BIS annual report say about global inflation?',
  'What are the key findings from the Tesla Q4 2023 report?',
  'How does the company stay current with AI research?',
  'What papers or research guides the team\'s technical decisions?',

  // Implementation & delivery
  'What does the implementation playbook cover?',
  'What are the standard phases of an AI implementation project?',
  'How does the company approach change management?',
  'What risk management practices are followed?',
  'What are the quality standards for deliverables?',
  'How is client training and enablement handled?',
  'What is the go-live checklist?',
  'How are post-implementation reviews conducted?',

  // Mission & strategy
  'What are the company\'s goals and objectives for this year?',
  'What markets is NeuralFlow AI targeting for expansion?',
  'How does the company prioritise between projects?',
  'What is the partnership and vendor strategy?',
  'How does the company approach AI ethics and responsible AI?',
  'What is the competitive landscape?',
  'How does NeuralFlow measure business impact for clients?',
  'What is the product roadmap?',

  // Operational
  'What are the security and compliance requirements?',
  'How is data handled and protected?',
  'What is the incident response process?',
  'How are software releases managed?',
  'What are the SLA commitments to clients?',
  'How is the support process structured?',
  'What communication channels are used internally?',
  'How are decisions escalated and approved?',
]

export default function ChatPage() {
  const store         = useChatStore()
  const { sendMessage, stop } = useChat()
  const [loading,  setLoading]  = useState(false)
  const [debug,    setDebug]    = useState(true)
  const [corpora,  setCorpora]  = useState<CorpusInfo[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  const conversation = store.conversations.find(c => c.id === store.activeId)
  const lastMsg      = conversation?.messages.at(-1)
  const citations: Citation[] = lastMsg?.role === 'assistant' ? lastMsg.citations ?? [] : []

  // Fetch corpora once on mount, auto-select first
  useEffect(() => {
    api.get<CorpusInfo[]>('/corpus').then(list => {
      if (!list?.length) return
      setCorpora(list)
      if (store.selectedCorpusIds.length === 0) {
        store.setCorpusIds([list[0].id])
      }
    }).catch(() => {
      // API not up yet — default to 'default' corpus so chat still works
      store.setCorpusIds(['default'])
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages.length])

  async function handleSend(query: string) {
    if (!store.activeId) store.newConversation()
    setLoading(true)
    try {
      await sendMessage(query)
    } finally {
      setLoading(false)
    }
  }

  const selectedCorpus = corpora.find(c => c.id === store.selectedCorpusIds[0])

  return (
    <div className="flex h-screen bg-[var(--bg)]">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-[var(--border)] flex flex-col bg-[var(--surface)]">

        {/* Corpus selector */}
        <div className="p-3 border-b border-[var(--border)]">
          <label className="text-xs text-[var(--text-muted)] flex items-center gap-1.5 mb-1.5">
            <Database size={11} /> Corpus
          </label>
          {corpora.length > 0 ? (
            <select
              value={store.selectedCorpusIds[0] ?? ''}
              onChange={e => store.setCorpusIds([e.target.value])}
              className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            >
              {corpora.map(c => (
                <option key={c.id} value={c.id}>{c.display_name}</option>
              ))}
            </select>
          ) : (
            <div className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-sm text-[var(--text-muted)]">
              {store.selectedCorpusIds[0] ?? 'Loading…'}
            </div>
          )}
        </div>

        {/* Conversation list */}
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {store.conversations.map(c => (
            <button
              key={c.id}
              onClick={() => store.setActive(c.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
                c.id === store.activeId
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--border)]'
              }`}
            >
              {c.title ?? 'New chat'}
            </button>
          ))}
        </nav>

        {/* Debug toggle */}
        <div className="p-3 border-t border-[var(--border)] flex items-center justify-between">
          <span className="text-xs text-[var(--text-muted)]">Debug</span>
          <button
            role="switch"
            aria-checked={debug}
            onClick={() => setDebug(v => !v)}
            className={`relative inline-flex w-9 h-5 rounded-full transition-colors focus:outline-none ${debug ? 'bg-[var(--accent)]' : 'bg-gray-300'}`}
          >
            <span className={`inline-block w-4 h-4 bg-white rounded-full shadow transform transition-transform mt-0.5 ${debug ? 'translate-x-4' : 'translate-x-0.5'}`} />
          </button>
        </div>
      </aside>

      {/* Chat area */}
      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-col flex-1 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto min-h-0">
            <div className="max-w-3xl mx-auto px-6 py-4">
              {conversation?.messages.map(msg => (
                <MessageBubble key={msg.id} message={msg} debugMode={debug} />
              ))}
              <div ref={bottomRef} />
            </div>
          </div>

          <InputBar onSend={handleSend} onStop={stop} loading={loading} />
        </main>

        {/* Right panel: citations when present, suggested questions always */}
        <div className="w-64 shrink-0 border-l border-[var(--border)] flex flex-col bg-[var(--surface)] overflow-y-auto">
          <CitationPanel citations={citations} />

          <div className={`p-4 ${citations.length ? 'border-t border-[var(--border)]' : ''}`}>
            <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Suggested Questions
            </h3>
            <div className="flex flex-col gap-2">
              {SUGGESTED_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  disabled={loading}
                  className="text-left px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] hover:border-[var(--accent)] hover:bg-blue-50 text-xs text-[var(--text-muted)] hover:text-[var(--text)] transition-colors leading-snug disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
