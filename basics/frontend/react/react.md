# React Tutorial — Patterns from a Production RAG Chat App

## Table of Contents

- [Part 1 — Core React](#part-1--core-react)
  - [JSX: What it actually is](#jsx-what-it-actually-is)
  - [Functional components and props](#functional-components-and-props)
  - [useState: state vs derived values](#usestate-state-vs-derived-values)
  - [useEffect: after render, cleanup, and the dependency array](#useeffect-after-render-cleanup-and-the-dependency-array)
  - [useRef: DOM refs and mutable values](#useref-dom-refs-and-mutable-values)
  - [Conditional rendering](#conditional-rendering)
  - [Lists and keys](#lists-and-keys)
  - [Event handling](#event-handling)
  - [Lifting state up vs colocating](#lifting-state-up-vs-colocating)
- [Part 2 — Patterns from the project](#part-2--patterns-from-the-project)
  - [Custom hooks](#custom-hooks)
  - [Context vs props vs global state](#context-vs-props-vs-global-state)
  - [Zustand: global state without the boilerplate](#zustand-global-state-without-the-boilerplate)
  - [React Router v7](#react-router-v7)
  - [Async in components: loading and error state](#async-in-components-loading-and-error-state)
  - [Aborting async work with AbortController](#aborting-async-work-with-abortcontroller)
  - [Performance: useMemo and useCallback](#performance-usememo-and-usecallback)
  - [TypeScript with React](#typescript-with-react)
- [Part 3 — Production patterns](#part-3--production-patterns)
  - [File and folder structure](#file-and-folder-structure)
  - [PrivateRoute: auth guards as components](#privateroute-auth-guards-as-components)
  - [Layout components](#layout-components)
  - [Avoiding prop drilling with Zustand](#avoiding-prop-drilling-with-zustand)
  - [sessionStorage cache in a custom hook](#sessionstorage-cache-in-a-custom-hook)

---

## Part 1 — Core React

### JSX: What it actually is

JSX is not HTML inside JavaScript. It is syntactic sugar that the TypeScript/Babel compiler rewrites into `React.createElement()` calls before the browser ever sees it.

```tsx
// What you write
const el = <div className="chat">Hello</div>

// What it compiles to
const el = React.createElement('div', { className: 'chat' }, 'Hello')
```

**Rules you cannot break:**

1. **One root element.** A component must return a single node. Wrap siblings in a `<div>`, or use a fragment `<>...</>` when you don't want extra DOM nodes.

2. **`className`, not `class`.** `class` is a reserved word in JavaScript.

3. **All tags must be closed.** Self-close void elements: `<input />`, `<img />`, `<br />`.

4. **Expressions go inside `{}`**. Statements (if, for) do not. Use ternary or `&&` instead.

```tsx
// Fragment avoids an extra wrapper div
export function MessageBubble({ message }: Props) {
  return (
    <>
      <div className="bubble">{message.content}</div>
      <span className="role">{message.role}</span>
    </>
  )
}
```

---

### Functional components and props

A component is a function that returns JSX. Props are its arguments — always typed with a TypeScript interface.

```tsx
// src/components/chat/MessageBubble.tsx

interface Props {
  message: Message
  debugMode?: boolean   // optional prop with ?
}

export function MessageBubble({ message, debugMode = false }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] rounded-xl px-4 py-3`}>
        {message.content}
      </div>
    </div>
  )
}
```

Key points:
- Destructure props directly in the parameter list.
- Default values go in the destructuring: `debugMode = false`.
- `React.FC` is no longer the recommended pattern. Just write a plain typed function.
- Component names must start with an uppercase letter so JSX can distinguish them from HTML tags.

---

### useState: state vs derived values

`useState` makes a value reactive: when it changes, React re-renders the component.

```tsx
// src/components/chat/InputBar.tsx
import { useState, useRef, type KeyboardEvent } from 'react'

export function InputBar({ onSend, onStop, loading }: Props) {
  const [text, setText] = useState('')   // '' is the initial value

  function submit() {
    const q = text.trim()
    if (!q || loading) return
    onSend(q)
    setText('')   // state update triggers a re-render
  }

  return (
    <textarea
      value={text}
      onChange={e => setText(e.target.value)}
      placeholder="Ask your knowledge base..."
    />
  )
}
```

**When NOT to use state:**

- If you can compute the value from existing state or props, it is a *derived value* — just compute it inline. Storing derived values in `useState` creates sync bugs.

```tsx
// Wrong — duplicated state that can go out of sync
const [messages, setMessages] = useState<Message[]>([])
const [messageCount, setMessageCount] = useState(0)

// Right — derive it
const [messages, setMessages] = useState<Message[]>([])
const messageCount = messages.length  // computed on every render, always correct
```

- If the value changes but you don't need the component to re-render, use `useRef`.

---

### useEffect: after render, cleanup, and the dependency array

`useEffect` runs *after* the browser has painted. It is for side effects: fetching data, subscribing to events, syncing with external systems.

```tsx
// src/components/PrivateRoute.tsx
import { useEffect, useState } from 'react'
import { tryRestoreSession } from '@/lib/auth'

export function PrivateRoute({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<'checking' | 'ok' | 'unauth'>('checking')

  useEffect(() => {
    // Runs once after the first render (empty dependency array)
    if (getAccessToken()) { setStatus('ok'); return }
    tryRestoreSession().then(ok => setStatus(ok ? 'ok' : 'unauth'))
  }, [])   // <-- dependency array

  if (status === 'checking') return <div className="min-h-screen" />
  if (status === 'unauth')   return <Navigate to="/login" replace />
  return <>{children}</>
}
```

**The dependency array controls when the effect re-runs:**

| Dependency array | Effect runs |
|---|---|
| Omitted | After every render |
| `[]` (empty) | Once, after the first render (on mount) |
| `[value]` | After any render where `value` changed |

**Cleanup.** If your effect subscribes to something, return a function that unsubscribes. React calls it before the next effect run and on unmount.

```tsx
useEffect(() => {
  const handler = () => console.log('scroll')
  window.addEventListener('scroll', handler)
  return () => window.removeEventListener('scroll', handler)  // cleanup
}, [])
```

**You cannot make the effect callback itself `async`.** React expects the callback to return either nothing or a cleanup function — not a Promise. Instead, define an async function inside the callback and call it:

```tsx
useEffect(() => {
  async function load() {
    const data = await fetchSomething()
    setData(data)
  }
  load()
}, [])
```

---

### useRef: DOM refs and mutable values

`useRef` gives you a box (`{ current: value }`) that persists across renders but does *not* trigger re-renders when changed.

**Two uses:**

1. **DOM ref** — get a direct handle to a DOM node.

```tsx
// src/components/chat/InputBar.tsx
const textareaRef = useRef<HTMLTextAreaElement>(null)

function handleInput() {
  const el = textareaRef.current
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'  // auto-grow
}

return <textarea ref={textareaRef} onChange={handleInput} />
```

2. **Mutable value without re-render** — ideal for storing an `AbortController`, a timer ID, or any value you need to read in an event handler without causing renders.

```tsx
// src/hooks/useChat.ts
const abortRef = useRef<AbortController | null>(null)

async function sendMessage(query: string) {
  abortRef.current?.abort()             // cancel the previous request
  abortRef.current = new AbortController()
  // ... pass abortRef.current.signal to fetch
}

function stop() { abortRef.current?.abort() }
```

---

### Conditional rendering

Three patterns, each with a different use case:

```tsx
// 1. &&  — render something or nothing
{message.streaming && <span className="typing-dot" />}

// 2. Ternary — render one of two things
<div className={isUser ? 'justify-end' : 'justify-start'}>

// 3. Early return — most readable for complex guards
export function MessageBubble({ message }: Props) {
  if (!message.content) return null   // render nothing

  return <div>{message.content}</div>
}
```

Pitfall with `&&`: if the left side is `0`, React renders `0`. Use an explicit boolean:

```tsx
// Wrong — renders "0" when count is 0
{count && <Badge count={count} />}

// Right
{count > 0 && <Badge count={count} />}
```

---

### Lists and keys

When rendering arrays, every element needs a stable `key`. React uses keys to match old and new lists — without them it re-creates every DOM node on every change.

```tsx
// src/components/chat/InputBar.tsx
const TIER_OPTIONS = ['auto', 'small', 'large'] as const

{TIER_OPTIONS.map(t => (
  <option key={t} value={t}>{t}</option>
))}
```

**Never use array index as key when the list can reorder or have items added/removed in the middle.** The key must be stable and uniquely identify the data, not the position.

```tsx
// Wrong — key tied to position
{conversations.map((conv, i) => <Item key={i} conv={conv} />)}

// Right — key tied to the data's identity
{conversations.map(conv => <Item key={conv.id} conv={conv} />)}
```

In `chatStore.ts`, every conversation and message gets `id: crypto.randomUUID()` at creation time — that UUID is the key.

---

### Event handling

React wraps native DOM events in a `SyntheticEvent` for cross-browser consistency. The pattern is the same as DOM events but camelCase.

```tsx
// src/components/chat/InputBar.tsx
import { type KeyboardEvent } from 'react'

function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()   // stop the textarea from inserting a newline
    submit()
  }
}

return <textarea onKeyDown={handleKeyDown} onChange={e => setText(e.target.value)} />
```

**Inline vs named handler:** Use inline arrows for simple one-liners (`onChange={e => setText(e.target.value)}`). Extract a named function when the logic is more than a single expression — it keeps JSX readable and avoids deeply nested lambdas.

---

### Lifting state up vs colocating

**Colocate** state as low in the tree as possible. If only one component needs `text`, keep `useState('')` inside that component.

**Lift up** when two sibling components need the same value. Move state to their closest common ancestor and pass it down as props.

```tsx
// ChatPage owns 'loading' because both InputBar and the message list
// need to know whether streaming is in progress.
function ChatPage() {
  const [loading, setLoading] = useState(false)

  return (
    <>
      <MessageList loading={loading} />
      <InputBar loading={loading} onSend={...} onStop={...} />
    </>
  )
}
```

When lifting state becomes unwieldy (many levels of prop passing), that is the signal to reach for Zustand — see Part 2.

---

## Part 2 — Patterns from the project

### Custom hooks

A custom hook is a plain function whose name starts with `use` and which calls other hooks inside. It lets you extract stateful logic out of a component so it can be reused and tested independently.

**Rules of hooks** — the linter enforces these:
- Call hooks only at the top level of a function (not inside if, loops, or nested functions).
- Call hooks only from React function components or other hooks.

`useChat.ts` is the main example: it encapsulates the entire streaming lifecycle — abort control, cache lookup, SSE iteration, and store mutations — behind a single `sendMessage(query)` call. The component does not need to know anything about `AbortController` or SSE parsing.

```tsx
// src/hooks/useChat.ts (simplified)
import { useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import { streamSSE } from '@/lib/sse'

export function useChat() {
  const abortRef = useRef<AbortController | null>(null)

  async function sendMessage(query: string) {
    const getStore = useChatStore.getState   // non-reactive read — see Zustand section

    let convId = getStore().activeId
    if (!convId) convId = getStore().newConversation()

    getStore().addUserMessage(convId, query)
    getStore().appendToken(convId, '')       // triggers the typing cursor

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      for await (const event of streamSSE('/api/v2/chat/stream', { method: 'POST', body: ... }, abortRef.current.signal)) {
        if ('delta' in event)    getStore().appendToken(convId, event.delta)
        else if ('done' in event) getStore().finaliseMessage(convId, { citations: event.citations })
        else if ('error' in event) getStore().finaliseMessage(convId, { content: `Error: ${event.error}` })
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        getStore().finaliseMessage(convId, { content: 'Connection error. Please try again.' })
      }
    }
  }

  function stop() { abortRef.current?.abort() }

  return { sendMessage, stop }
}
```

The component side becomes trivial:

```tsx
// In ChatPage.tsx
const { sendMessage, stop } = useChat()
```

---

### Context vs props vs global state

| Mechanism | Good for | Drawbacks |
|---|---|---|
| Props | Data that flows naturally parent → child, one or two levels | Becomes "prop drilling" across many levels |
| React Context | Infrequently-changing values (theme, locale, auth user) | Every consumer re-renders on any context change |
| Zustand | High-frequency global state (conversations, messages, streaming tokens) | External dependency; overkill for simple apps |

In this project:
- `modelTier` and `selectedCorpusIds` are needed by both `InputBar` and the backend request — they live in the Zustand store.
- `debugMode` is only used in `ChatPage` and passed to `MessageBubble` — it stays as a prop.
- Theme (dark mode) is in the store because it is set in `NavBar` and consumed by every component via CSS variables.

---

### Zustand: global state without the boilerplate

Zustand creates a store as a hook. Any component that calls the hook subscribes to that slice of state.

```tsx
// src/store/chatStore.ts
import { create } from 'zustand'
import type { Conversation, Message } from '@/types/chat'

interface ChatState {
  conversations:     Conversation[]
  activeId:          string | null
  modelTier:         'auto' | 'small' | 'large'
  appendToken:       (convId: string, delta: string) => void
  finaliseMessage:   (convId: string, msg: Partial<Message>) => void
  // ... more actions
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeId:      null,
  modelTier:     'auto',

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
  // ...
}))
```

**Using the store in a component:**

```tsx
// Subscribes to only 'modelTier' and 'setModelTier' — rerenders only when those change
const { modelTier, setModelTier } = useChatStore()
```

**`getState()` for non-reactive reads.** Inside an async function (like `sendMessage`), the hook's snapshot captured at render time can be stale by the time the async code runs. Use `useChatStore.getState()` to always read the current value:

```tsx
// src/hooks/useChat.ts
async function sendMessage(query: string) {
  // useChatStore() here would be a stale closure — wrong
  // getState() reads the live store — right
  const getStore = useChatStore.getState

  let convId = getStore().activeId
  if (!convId) convId = getStore().newConversation()
  // ...
}
```

---

### React Router v7

The app uses `BrowserRouter` (HTML5 history API, no hash in URLs). All route configuration lives in `src/router.tsx`.

```tsx
// src/router.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

export function Router() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login"    element={<LoginPage />} />
        <Route path="/chat"     element={<Private><ChatPage /></Private>} />
        <Route path="/logs"     element={<Private><LogsPage /></Private>} />
        <Route path="*"         element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
```

`Private` is a local wrapper that composes `PrivateRoute` (auth guard) and `MainLayout` (shell):

```tsx
function Private({ children }: { children: React.ReactNode }) {
  return (
    <PrivateRoute>
      <MainLayout>{children}</MainLayout>
    </PrivateRoute>
  )
}
```

**Useful Router hooks:**

```tsx
import { useNavigate, useLocation } from 'react-router-dom'

const navigate = useNavigate()
navigate('/chat')          // programmatic navigation
navigate('/login', { replace: true })  // replace instead of push

const location = useLocation()
location.pathname          // '/chat'
```

**`<Navigate>`** is a component that redirects declaratively — used inside `PrivateRoute` to send unauthenticated users to `/login`.

---

### Async in components: loading and error state

The standard pattern for async data in a component:

```tsx
// src/pages/IngestPage.tsx (pattern)
const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
const [errorMsg, setErrorMsg] = useState<string | null>(null)

async function handleSubmit() {
  setStatus('loading')
  setErrorMsg(null)
  try {
    await api.post('/v2/ingest', payload)
    setStatus('done')
  } catch (err: any) {
    setErrorMsg(err.message ?? 'Unknown error')
    setStatus('error')
  }
}

return (
  <>
    <button onClick={handleSubmit} disabled={status === 'loading'}>
      {status === 'loading' ? 'Ingesting…' : 'Start Ingest'}
    </button>
    {errorMsg && <p className="text-red-500">{errorMsg}</p>}
  </>
)
```

Keep `loading`, `error`, and `data` as separate state values. Do not collapse them into a single string — it makes conditions awkward.

---

### Aborting async work with AbortController

Long-running fetches must be cancellable. The browser's `AbortController` API integrates directly with `fetch` and with the SSE reader.

```tsx
// src/hooks/useChat.ts
const abortRef = useRef<AbortController | null>(null)

async function sendMessage(query: string) {
  abortRef.current?.abort()               // cancel any in-flight request
  abortRef.current = new AbortController()

  try {
    for await (const event of streamSSE(url, options, abortRef.current.signal)) {
      // handle events
    }
  } catch (err: any) {
    if (err?.name !== 'AbortError') {
      // Only show an error for unexpected failures, not user-initiated stops
      getStore().finaliseMessage(convId, { content: 'Connection error.' })
    }
  }
}

function stop() { abortRef.current?.abort() }
```

Why `useRef` and not `useState` for the controller? The `AbortController` instance is never displayed in the UI — there is no reason to re-render when it changes. `useRef` stores it without triggering renders.

In `useEffect`, abort in the cleanup function to cancel the request when the component unmounts mid-fetch:

```tsx
useEffect(() => {
  const ctrl = new AbortController()
  fetchData(ctrl.signal).then(setData)
  return () => ctrl.abort()   // component unmounted — cancel the request
}, [])
```

---

### Performance: useMemo and useCallback

**`useMemo`** memoizes the result of an expensive calculation so it is not recomputed on every render.

**`useCallback`** memoizes a function reference so it is stable across renders (useful when passing callbacks to memoized child components).

**When they actually help:**
- An expensive pure computation (sorting, filtering a large list) that is called on every render.
- A callback passed as a prop to a `React.memo`-wrapped child — stable reference prevents unnecessary child re-renders.

**When they are premature optimisation:**
- Anywhere the computation is trivial (string concatenation, a lookup in a small array).
- When the component re-renders for another reason anyway — memoizing the callback does nothing.

In this codebase, `useMemo` appears for the filtered/sorted conversation list in `ChatPage`:

```tsx
const titledConversations = useMemo(
  () => conversations.filter(c => c.title),
  [conversations],
)
```

This avoids re-filtering on every keystroke in the input bar (which triggers a re-render via `setText`) since `conversations` itself hasn't changed.

---

### TypeScript with React

**Component props:**

```tsx
// Explicit interface — preferred over inline type
interface Props {
  message: Message
  debugMode?: boolean
}

export function MessageBubble({ message, debugMode = false }: Props) { ... }
```

**`useState` generics** — TypeScript infers the type from the initial value, but be explicit when the initial value is ambiguous:

```tsx
const [status, setStatus] = useState<'checking' | 'ok' | 'unauth'>('checking')
const [error, setError]   = useState<string | null>(null)
```

**Event types:**

```tsx
import { type KeyboardEvent, type ChangeEvent } from 'react'

function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) { ... }
function handleChange(e: ChangeEvent<HTMLInputElement>) { ... }
```

**`React.ReactNode`** is the correct type for `children`:

```tsx
interface Props {
  children: React.ReactNode
}
```

**`as const`** for literal arrays used in JSX:

```tsx
const TIER_OPTIONS = ['auto', 'small', 'large'] as const
// Type is readonly ['auto', 'small', 'large'], not string[]
```

---

## Part 3 — Production patterns

### File and folder structure

```
src/
├── components/         # Reusable UI components
│   ├── NavBar.tsx
│   ├── MainLayout.tsx
│   ├── PrivateRoute.tsx
│   └── chat/           # Feature-scoped sub-folder
│       ├── MessageBubble.tsx
│       ├── InputBar.tsx
│       ├── CitationPanel.tsx
│       ├── CostBadge.tsx
│       └── DebugPanel.tsx
├── hooks/              # Custom hooks (useChat.ts)
├── lib/                # Pure utilities (api.ts, auth.ts, sse.ts)
├── pages/              # One file per route
│   ├── ChatPage.tsx
│   ├── LoginPage.tsx
│   ├── IngestPage.tsx
│   └── LogsPage.tsx
├── store/              # Zustand stores
│   └── chatStore.ts
├── types/              # Shared TypeScript interfaces
│   └── chat.ts
├── router.tsx          # All route definitions in one place
└── main.tsx            # Entry point — mounts Router, nothing else
```

**Naming conventions:**
- Components: `PascalCase.tsx`
- Hooks: `camelCase.ts`, always prefixed with `use`
- Utilities: `camelCase.ts`
- Types: `camelCase.ts`

Keep `main.tsx` minimal — it should only mount the root component. Route definitions go in `router.tsx`, not scattered through pages.

---

### PrivateRoute: auth guards as components

A `PrivateRoute` wraps protected pages and redirects unauthenticated users. It is stateful: it must wait for session restoration before deciding which way to route.

```tsx
// src/components/PrivateRoute.tsx
import { useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import { getAccessToken } from '@/lib/api'
import { tryRestoreSession } from '@/lib/auth'

export function PrivateRoute({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<'checking' | 'ok' | 'unauth'>('checking')

  useEffect(() => {
    if (getAccessToken()) { setStatus('ok'); return }
    // Try refresh-cookie-based session restoration before giving up
    tryRestoreSession().then(ok => setStatus(ok ? 'ok' : 'unauth'))
  }, [])

  if (status === 'checking') return <div className="min-h-screen bg-[var(--bg)]" />
  if (status === 'unauth')   return <Navigate to="/login" replace />
  return <>{children}</>
}
```

The three-state approach (`checking | ok | unauth`) prevents a flash of the login page during session restoration. While `checking`, render a blank screen that matches the app background instead of immediately redirecting.

In `router.tsx` the pattern is composed with `MainLayout`:

```tsx
function Private({ children }: { children: React.ReactNode }) {
  return (
    <PrivateRoute>
      <MainLayout>{children}</MainLayout>
    </PrivateRoute>
  )
}
```

---

### Layout components

`MainLayout` is a thin shell that renders the persistent `NavBar` and positions the page content. Pages do not import `NavBar` directly — the layout handles it.

```tsx
// src/components/MainLayout.tsx
import { NavBar } from './NavBar'

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <NavBar />
      <div className="pl-14 min-h-screen">
        {children}
      </div>
    </>
  )
}
```

`NavBar` is fixed-width (`w-14` = 56px). Every page gets `pl-14` padding-left so its content does not sit under the nav. Pages never need to know about this offset — `MainLayout` owns it.

This pattern means adding a new page is just:
1. Create `src/pages/NewPage.tsx`.
2. Add `<Route path="/new" element={<Private><NewPage /></Private>} />` in `router.tsx`.

No changes to `NavBar` or layout components needed.

---

### Avoiding prop drilling with Zustand

Without global state, `modelTier` would need to travel: `ChatPage` → `InputBar` → `onSend` callback → back up to `useChat` → into the API request. Instead, every consumer reads directly from the store:

```tsx
// InputBar reads and writes the tier directly — no props needed
const { selectedCorpusIds, modelTier, setModelTier } = useChatStore()

// useChat reads it at request time via getState()
const tier = useChatStore.getState().modelTier
```

The rule: if more than two unrelated components in different subtrees need the same value, put it in the store. If it is data that only flows parent → child, keep it as props.

---

### sessionStorage cache in a custom hook

The Q&A cache in `useChat.ts` stores answers in `sessionStorage` with a TTL. This gives instant responses for repeated questions within the same browser tab, without hitting the backend.

```tsx
// src/hooks/useChat.ts
const CACHE_TTL_MS = 30 * 60 * 1000   // 30 minutes

interface CacheEntry { answer: string; citations: any[]; expiry: number }

function cacheKey(query: string, corpusIds: string[], tier: string): string {
  return `qa:${JSON.stringify({ query, corpusIds, tier })}`
}

function cacheGet(key: string): CacheEntry | null {
  try {
    const raw = sessionStorage.getItem(key)
    if (!raw) return null
    const entry: CacheEntry = JSON.parse(raw)
    if (Date.now() > entry.expiry) { sessionStorage.removeItem(key); return null }
    return entry
  } catch { return null }
}

function cacheSet(key: string, entry: Omit<CacheEntry, 'expiry'>): void {
  try {
    sessionStorage.setItem(key, JSON.stringify({ ...entry, expiry: Date.now() + CACHE_TTL_MS }))
  } catch { /* sessionStorage full — silently skip */ }
}
```

Usage inside `sendMessage`:

```tsx
const cached = cacheGet(key)
if (cached) {
  getStore().finaliseMessage(convId, { content: cached.answer, citations: cached.citations })
  return   // skip the API call entirely
}
// ... stream from API, then:
if (fullAnswer) cacheSet(key, { answer: fullAnswer, citations })
```

Design notes:
- The cache key includes the query, corpus IDs, and model tier — different corpora or tiers produce different keys.
- `sessionStorage` dies on tab close, which is the desired behaviour (no stale answers across sessions).
- Both `cacheGet` and `cacheSet` are wrapped in `try/catch` because `sessionStorage` can throw if storage is full or if running in a sandboxed iframe.
- The TTL check happens on read, not on a timer — simple and correct.
