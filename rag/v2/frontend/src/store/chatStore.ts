
import { create } from 'zustand'
import type { Conversation, Message } from '@/types/chat'

interface ChatState {
  conversations:       Conversation[]
  activeId:            string | null
  selectedCorpusIds:   string[]
  modelTier:           'auto' | 'small' | 'large'
  isDark:              boolean

  // Actions
  newConversation:     () => string
  setActive:           (id: string) => void
  appendToken:         (convId: string, delta: string) => void
  finaliseMessage:     (convId: string, msg: Partial<Message>) => void
  addUserMessage:      (convId: string, content: string) => void
  setCorpusIds:        (ids: string[]) => void
  setModelTier:        (t: 'auto' | 'small' | 'large') => void
  toggleDark:          () => void
  loadConversations:   (convs: Conversation[]) => void
}

export const useChatStore = create<ChatState>((set, _get) => ({
  conversations:     [],
  activeId:          null,
  selectedCorpusIds: [],
  modelTier:         'auto',
  isDark:            false,

  newConversation: () => {
    const id = crypto.randomUUID()
    const conv: Conversation = {
      id,
      session_id: crypto.randomUUID(),   // one UUID per conversation thread
      turn_count: 0,
      messages:   [],
    }
    set(s => ({ conversations: [conv, ...s.conversations], activeId: id }))
    return id
  },

  setActive: id => set({ activeId: id }),

  addUserMessage: (convId, content) => {
    const msg: Message = { id: crypto.randomUUID(), role: 'user', content }
    set(s => ({
      conversations: s.conversations.map(c =>
        c.id === convId ? { ...c, messages: [...c.messages, msg] } : c
      ),
    }))
  },

  appendToken: (convId, delta) => set(s => ({
    conversations: s.conversations.map(c => {
      if (c.id !== convId) return c
      const msgs = [...c.messages]
      const last = msgs[msgs.length - 1]
      if (last?.streaming) {
        msgs[msgs.length - 1] = { ...last, content: last.content + delta }
      } else {
        msgs.push({ id: crypto.randomUUID(), role: 'assistant', content: delta, streaming: true })
      }
      return { ...c, messages: msgs }
    }),
  })),

  finaliseMessage: (convId, partial) => set(s => ({
    conversations: s.conversations.map(c => {
      if (c.id !== convId) return c
      const msgs = [...c.messages]
      const last = msgs[msgs.length - 1]
      if (last?.streaming) {
        msgs[msgs.length - 1] = { ...last, ...partial, streaming: false }
      }
      return { ...c, messages: msgs, turn_count: c.turn_count + 1 }
    }),
  })),

  setCorpusIds:      ids  => set({ selectedCorpusIds: ids }),
  setModelTier:      t    => set({ modelTier: t }),
  toggleDark:        ()   => set(s => ({ isDark: !s.isDark })),
  loadConversations: convs => set({ conversations: convs }),
}))
