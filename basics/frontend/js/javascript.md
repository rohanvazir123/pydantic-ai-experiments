# Modern JavaScript & TypeScript

A practical reference for a developer who built a production RAG chat app (Vite + React 19 + TypeScript + Zustand on the front, FastAPI + Redis + Pydantic AI on the back). No Hello World — every example maps to a real pattern from the codebase.

## Table of Contents

- [Part 1 — JavaScript Basics](#part-1--javascript-basics)
  - [1.1 Variables: var / let / const](#11-variables-var--let--const)
  - [1.2 Primitive Types](#12-primitive-types)
  - [1.3 Functions](#13-functions)
  - [1.4 Template Literals](#14-template-literals)
  - [1.5 Destructuring](#15-destructuring)
  - [1.6 Spread and Rest](#16-spread-and-rest)
  - [1.7 Optional Chaining and Nullish Coalescing](#17-optional-chaining-and-nullish-coalescing)
  - [1.8 Array Methods](#18-array-methods)
  - [1.9 Object Methods](#19-object-methods)
  - [1.10 Modules: import / export](#110-modules-import--export)
  - [1.11 Classes (brief)](#111-classes-brief)
- [Part 2 — What We Actually Used](#part-2--what-we-actually-used)
  - [2.1 Promises and async/await](#21-promises-and-asyncawait)
  - [2.2 Error Handling in Async Code](#22-error-handling-in-async-code)
  - [2.3 AbortController — Cancellable Fetch](#23-abortcontroller--cancellable-fetch)
  - [2.4 Fetch API with Streaming (ReadableStream + TextDecoder)](#24-fetch-api-with-streaming-readablestream--textdecoder)
  - [2.5 sessionStorage with TTL Pattern](#25-sessionstorage-with-ttl-pattern)
  - [2.6 Closures and React Hooks](#26-closures-and-react-hooks)
  - [2.7 WeakMap / WeakSet and Memory Management](#27-weakmap--weakset-and-memory-management)
  - [2.8 Generators and Async Generators](#28-generators-and-async-generators)
  - [2.9 TypeScript Essentials](#29-typescript-essentials)
  - [2.10 Module Patterns and Barrel Exports](#210-module-patterns-and-barrel-exports)
  - [2.11 Event-Driven Patterns](#211-event-driven-patterns)

---

## Part 1 — JavaScript Basics

### 1.1 Variables: var / let / const

`var` is function-scoped and hoisted — it causes subtle bugs in loops and closures. `let` is block-scoped and reassignable. `const` is block-scoped and cannot be reassigned, but its contents (array items, object properties) are still mutable. **Default to `const`; use `let` only when you need to reassign.**

```js
const CACHE_TTL_MS = 30 * 60 * 1000  // never reassigned — const is correct
let fullAnswer = ''                    // accumulated token-by-token — needs let
// var fullAnswer = ''                 // avoid: leaks out of blocks, hoisted weirdly
```

`const` prevents `fullAnswer = 'oops'` but not `fullAnswer += token` — you own that reassignment explicitly via `let`.

---

### 1.2 Primitive Types

JavaScript has seven primitives: `string`, `number`, `bigint`, `boolean`, `undefined`, `null`, `symbol`. Everything else is an object (arrays, functions, plain objects).

```js
const query  = 'What is the PTO policy?'   // string
const score  = 0.92                          // number (always float64)
const active = true                          // boolean
const token  = null                          // explicit absence of value
let   result                                 // undefined — declared but not assigned

typeof query   // 'string'
typeof score   // 'number'
typeof null    // 'object' — historic JS bug, check with === null instead
```

`null` vs `undefined`: use `null` to signal intentional emptiness (like `_accessToken = null` after logout); `undefined` means a value was never set.

---

### 1.3 Functions

Three forms — each has a use case.

```js
// Declaration — hoisted, can be called before its definition
function cacheKey(query, corpusIds, tier) {
  return `qa:${JSON.stringify({ query, corpusIds, tier })}`
}

// Arrow function — compact, inherits `this` from surrounding scope (important in React)
const stop = () => { abortRef.current?.abort() }

// Default parameters — used in the API client's request() helper
async function request(path, init = {}) {
  const headers = { 'Content-Type': 'application/json', ...init.headers }
  return fetch(path, { ...init, headers })
}
```

Arrow functions do **not** have their own `this`, which is why React event handlers and callbacks are almost always arrows.

---

### 1.4 Template Literals

Backtick strings support embedded expressions and multi-line content without concatenation.

```js
// From cacheKey() in useChat.ts
const key = `qa:${JSON.stringify({ query, corpusIds, tier })}`

// Multi-line (useful for constructing SQL in tests or prompts)
const prompt = `
  You are a helpful assistant.
  Query: ${query}
  Context: ${context}
`.trim()
```

---

### 1.5 Destructuring

Extract named properties from objects or positional items from arrays without temporary variables.

```js
// Object destructuring — from Zustand set callbacks
const { conversations, activeId } = useChatStore.getState()

// With rename
const { done: isDone, value: chunk } = await reader.read()

// Array destructuring
const [first, ...rest] = citations

// Destructuring in function parameters
function addUserMessage(convId, content) {
  // vs receiving { convId, content } — same syntax, object form
}

// Nested destructuring (use sparingly — harms readability past two levels)
const { error: { message, code } = {} } = json
```

---

### 1.6 Spread and Rest

Spread (`...`) expands an iterable into individual items. Rest (`...`) collapses remaining items into an array. Same syntax, opposite direction.

```js
// Spread — merge objects (right wins on duplicate keys)
const headers = {
  'Content-Type': 'application/json',
  ...(token ? { Authorization: `Bearer ${token}` } : {}),  // conditional spread
  ...init.headers,   // caller overrides come last
}

// Spread — clone an array before mutation (Zustand immutability rule)
const msgs = [...c.messages]
msgs[msgs.length - 1] = { ...last, streaming: false }

// Rest — gather remaining arguments
function log(level, ...messages) {
  console[level](messages.join(' '))
}
```

The Zustand store uses spread extensively because you must never mutate state in-place — always return a new object.

---

### 1.7 Optional Chaining and Nullish Coalescing

`?.` short-circuits to `undefined` if the left side is `null`/`undefined`, avoiding TypeError crashes. `??` returns the right side only when the left is `null` or `undefined` (unlike `||`, which also fires on `0`, `''`, `false`).

```js
// From useChat.ts — abort only if a controller exists
abortRef.current?.abort()

// From sse.ts — safe fallback when JSON parse returns no error field
const err = json.error ?? {}
const message = err.message ?? `SSE request failed: ${res.status}`

// Difference between ?? and ||
const tier = config.tier ?? 'auto'    // 'auto' only if tier is null/undefined
const tier2 = config.tier || 'auto'   // 'auto' also if tier is '' or 0 — usually wrong
```

---

### 1.8 Array Methods

All are non-mutating (they return new arrays). Chain them to build data pipelines.

```js
// map — transform every element; used when rendering message list in ChatPage
const bubbles = messages.map(msg => ({ ...msg, text: msg.content.trim() }))

// filter — keep matching elements
const answered = messages.filter(m => m.status === 'answered')

// find — first match or undefined (used to locate active conversation)
const conv = conversations.find(c => c.id === convId)

// some / every — boolean checks
const hasStreaming = messages.some(m => m.streaming)
const allDone = messages.every(m => !m.streaming)

// reduce — fold to a single value (token count total)
const totalTokens = messages.reduce((sum, m) => sum + (m.prompt_tokens ?? 0), 0)

// flat / flatMap — flatten nested arrays
const allChunks = corpora.flatMap(corpus => corpus.chunks)
// equivalent to corpora.map(...).flat(1)
```

---

### 1.9 Object Methods

```js
const msg = { id: '1', role: 'user', content: 'hello', streaming: true }

Object.keys(msg)    // ['id', 'role', 'content', 'streaming']
Object.values(msg)  // ['1', 'user', 'hello', true]
Object.entries(msg) // [['id','1'], ['role','user'], ...]

// Object.assign — shallow merge (prefer spread in modern code)
const merged = Object.assign({}, defaults, overrides)

// Spread merge — same result, cleaner
const merged2 = { ...defaults, ...overrides }

// Object.fromEntries — rebuild from entries (useful after filtering keys)
const publicMsg = Object.fromEntries(
  Object.entries(msg).filter(([k]) => k !== 'streaming')
)
```

---

### 1.10 Modules: import / export

ES modules are the standard. Each file is its own scope; you must explicitly export what should be visible.

```js
// Named export — caller must use the exact name (or alias with `as`)
export function cacheGet(key) { ... }
export const CACHE_TTL_MS = 1_800_000

// Default export — caller picks the name; one per file
export default function App() { ... }

// Named import
import { streamSSE } from '@/lib/sse'
import { useChatStore } from '@/store/chatStore'

// Default import
import App from './App'

// Namespace import — rarely needed, but useful for dynamic aliasing
import * as api from '@/lib/api'

// Dynamic import — lazy load a module at runtime (used in sse.ts to avoid circular dep)
const { getAccessToken } = await import('./api')
```

Named exports are preferred in libraries — they support tree-shaking and IDE rename refactors cleanly.

---

### 1.11 Classes (brief)

Classes exist in JS but modern React prefers functions + hooks. Use classes when you need inheritance or when wrapping a stateful external resource.

```js
// From api.ts — custom error class for typed catch blocks
class APIError extends Error {
  constructor(code, message, status) {
    super(message)   // must call super first in derived class
    this.name = 'APIError'
    this.code = code
    this.status = status
  }
}

// Usage
try { await api.get('/health') }
catch (err) {
  if (err instanceof APIError && err.status === 401) logout()
}
```

---

## Part 2 — What We Actually Used

### 2.1 Promises and async/await

A Promise represents a value that will be available in the future. `async/await` is syntactic sugar over Promises — it makes async code read like synchronous code.

```js
// Promise.all — run independent requests concurrently, await all
const [user, corpora] = await Promise.all([
  api.get('/auth/me'),
  api.get('/corpora'),
])

// Promise.race — first settled wins (timeout pattern)
const result = await Promise.race([
  fetch('/api/v2/chat/stream', opts),
  new Promise((_, reject) =>
    setTimeout(() => reject(new Error('timeout')), 10_000)
  ),
])

// Promise.allSettled — get all results even if some reject
const results = await Promise.allSettled(uploadPromises)
const failed = results.filter(r => r.status === 'rejected')
```

---

### 2.2 Error Handling in Async Code

`try/catch/finally` works with `await` exactly like with synchronous code. `finally` always runs — use it to clean up (remove loaders, release locks).

```js
// Pattern from useChat.ts sendMessage()
try {
  for await (const event of streamSSE(url, opts, signal)) {
    if ('delta' in event) appendToken(event.delta)
    if ('done' in event)  finalise(event)
  }
} catch (err) {
  // AbortError is intentional (user clicked Stop) — don't show an error
  if (err?.name !== 'AbortError') {
    finalise({ content: 'Connection error. Please try again.' })
  }
} finally {
  setLoading(false)  // always clear spinner
}
```

Always distinguish between expected errors (abort, 401) and unexpected errors. Re-throw or surface unexpected ones; silently swallow the expected ones.

---

### 2.3 AbortController — Cancellable Fetch

`AbortController` gives you a signal you can attach to `fetch`. Calling `.abort()` cancels the in-flight request and rejects the promise with an `AbortError`. The pattern in `useChat.ts` is: store the controller in a `useRef` so it persists across renders, then abort the previous request before starting a new one.

```js
// From useChat.ts — exact pattern used for SSE streaming
const abortRef = useRef(null)          // persists across renders without causing re-renders

async function sendMessage(query) {
  abortRef.current?.abort()            // cancel any in-flight request
  abortRef.current = new AbortController()

  const res = await fetch('/api/v2/chat/stream', {
    method: 'POST',
    body: JSON.stringify({ query }),
    signal: abortRef.current.signal,   // attach to fetch
  })
}

function stop() { abortRef.current?.abort() }
```

`useRef` instead of `useState` because you don't want React to re-render when the controller changes — it's imperative infrastructure, not UI state.

---

### 2.4 Fetch API with Streaming (ReadableStream + TextDecoder)

`fetch` returns a `Response` whose `.body` is a `ReadableStream<Uint8Array>`. You read it chunk-by-chunk with `.getReader()`. `TextDecoder` converts raw bytes into strings. **We use this instead of `EventSource` because EventSource is GET-only — it cannot send a JSON body.**

```js
// Simplified core of sse.ts streamSSE()
const res = await fetch(url, { ...init, signal })
const reader = res.body.getReader()
const decoder = new TextDecoder()
let buffer = ''

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  // { stream: true } tells the decoder more chunks are coming — don't flush
  buffer += decoder.decode(value, { stream: true })

  // SSE events are separated by double newlines
  const parts = buffer.split('\n\n')
  buffer = parts.pop() ?? ''          // last part may be incomplete — keep it

  for (const part of parts) {
    if (!part.startsWith('data: ')) continue
    const payload = part.slice(6)
    if (payload === '[DONE]') return
    yield JSON.parse(payload)         // this is inside an async generator — see 2.8
  }
}
reader.releaseLock()   // always release in finally block
```

---

### 2.5 sessionStorage with TTL Pattern

`sessionStorage` is a key-value store that survives page refreshes but is cleared when the tab closes. Values must be strings, so you JSON-serialise them. Because `sessionStorage` has no native TTL, you store the expiry timestamp inside the value and check it on read.

```js
// Exact pattern from useChat.ts cacheGet / cacheSet
const CACHE_TTL_MS = 30 * 60 * 1000   // 30 minutes

function cacheSet(key, { answer, citations }) {
  try {
    sessionStorage.setItem(key, JSON.stringify({
      answer,
      citations,
      expiry: Date.now() + CACHE_TTL_MS,   // store absolute expiry time
    }))
  } catch { /* quota exceeded — silently skip */ }
}

function cacheGet(key) {
  try {
    const raw = sessionStorage.getItem(key)
    if (!raw) return null
    const entry = JSON.parse(raw)
    if (Date.now() > entry.expiry) {
      sessionStorage.removeItem(key)   // evict expired entry
      return null
    }
    return entry
  } catch { return null }
}
```

Wrapping in `try/catch` handles two cases: `JSON.parse` failures from corrupt data, and `setItem` throwing `QuotaExceededError` when storage is full.

---

### 2.6 Closures and React Hooks

A closure is a function that captures variables from its enclosing scope. Every React hook callback closes over the render's snapshot of state — if state changes after the function is created but before it runs, the function sees the **old** value. This caused the bug fixed in commit `9a8a5bd`.

```js
// Stale closure bug — the danger
function BadComponent() {
  const [count, setCount] = useState(0)
  const handleClick = () => {
    // count is captured at the time this function was created
    setTimeout(() => console.log(count), 1000)  // always logs 0
  }
}

// Fix: read fresh state via getState() instead of trusting the closure snapshot
// From useChat.ts — this is exactly why useChatStore.getState() is called inside sendMessage
async function sendMessage(query) {
  const getStore = useChatStore.getState   // getState is stable — it's a function reference
  const convId = getStore().activeId       // fresh read at call time, not closure time
  // ...
}
```

The rule: **if you need the latest value of something inside a callback or async function, read it via a ref or a `getState()` call rather than closing over the state variable.**

---

### 2.7 WeakMap / WeakSet and Memory Management

`WeakMap` and `WeakSet` hold object keys weakly — if the key object is garbage collected, the entry is automatically removed. They don't prevent GC and aren't iterable. Use them to associate metadata with objects without creating memory leaks.

```js
// WeakMap: associate per-object state without preventing GC
const abortControllers = new WeakMap()

function attachController(requestObj, controller) {
  abortControllers.set(requestObj, controller)  // key is the object
}

// When requestObj is GC'd, the entry disappears automatically
// Can't iterate WeakMap — that's by design

// WeakSet: track membership without holding references
const seen = new WeakSet()
function processOnce(node) {
  if (seen.has(node)) return
  seen.add(node)
  // process...
}
```

In our app, `useRef` plays the same role for the `AbortController` — the ref holds the controller without exposing it to React's render cycle.

---

### 2.8 Generators and Async Generators

A generator function (marked `function*`) pauses at each `yield` and resumes on the next `.next()` call. An **async generator** (`async function*`) combines this with `await`, letting you `yield` values from an async data source one at a time. This is the foundation of `streamSSE` in `sse.ts`.

```js
// sync generator — simple example
function* range(start, end) {
  for (let i = start; i < end; i++) yield i
}
for (const n of range(0, 3)) console.log(n)  // 0, 1, 2

// async generator — exactly the pattern in sse.ts
async function* streamSSE(url, init, signal) {
  const res = await fetch(url, { ...init, signal })
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // ... parse SSE events from buffer ...
      yield parsedEvent    // caller receives one event at a time
    }
  } finally {
    reader.releaseLock()   // finally runs whether we break, return, or throw
  }
}

// Caller in useChat.ts — for-await-of consumes the async generator
for await (const event of streamSSE('/api/v2/chat/stream', opts, signal)) {
  if ('delta' in event) appendToken(event.delta)
}
```

The generator owns the reader; the caller owns the event loop. Neither needs to know how the other is implemented.

---

### 2.9 TypeScript Essentials

TypeScript adds a type layer that is erased at compile time. It prevents entire classes of runtime errors.

```ts
// Type alias — name a shape (prefer for unions and primitives)
type ModelTier = 'auto' | 'small' | 'large'
type PipelineStatus = 'answered' | 'abstained_retrieval' | 'abstained_judge'

// Interface — name an object shape (prefer for extendable contracts)
// From types/chat.ts
interface Message {
  id:         string
  role:       'user' | 'assistant'   // string literal union
  content:    string
  citations?: Citation[]             // ? means optional (can be undefined)
  streaming?: boolean
}

// Generics — parameterise over a type
async function request<T>(path: string): Promise<T> {
  const res = await fetch(path)
  return res.json() as T             // `as` cast — tells TS to trust you
}
// Usage
const messages = await request<Message[]>('/api/v2/chat/history')

// Union types + type guards — discriminate SSE event variants
type SSEEvent = SSEDelta | SSEDone | SSEError | SSEAbstained

// 'in' operator is a type guard — narrows the union inside the block
if ('delta' in event) {
  // TS knows event is SSEDelta here
  appendToken(event.delta)
} else if ('done' in event) {
  // TS knows event is SSEDone here
  setCitations(event.citations)
}

// Omit — derive a type by removing keys (from useChat.ts cacheSet)
type CacheEntry = { answer: string; citations: Citation[]; expiry: number }
function cacheSet(key: string, entry: Omit<CacheEntry, 'expiry'>): void { ... }

// Partial — make all keys optional (common in update functions)
function finaliseMessage(convId: string, msg: Partial<Message>): void { ... }
```

---

### 2.10 Module Patterns and Barrel Exports

A barrel file (`index.ts`) re-exports from multiple files so callers use one import path instead of knowing where each thing lives.

```ts
// src/lib/index.ts — barrel for the lib/ folder
export { api, APIError } from './api'
export { streamSSE }     from './sse'
export { getAccessToken, setAccessToken, clearTokens } from './api'

// Caller — single import regardless of which file things live in
import { api, streamSSE, APIError } from '@/lib'
```

The access-token in `api.ts` is a module-level variable (`let _accessToken = null`). This is the **module singleton pattern** — one instance per module load, shared across all importers. It is intentionally not in React state or localStorage to reduce XSS surface.

```ts
// Module singleton — private by convention (underscore prefix)
let _accessToken: string | null = null

export function setAccessToken(token: string): void { _accessToken = token }
export function getAccessToken(): string | null     { return _accessToken }
export function clearTokens(): void                 { _accessToken = null }
```

---

### 2.11 Event-Driven Patterns

Browsers use `EventTarget` / `EventEmitter` (Node) for decoupled communication. In React apps, Zustand's `subscribe` and custom event busses play the same role. The SSE stream is itself an event-driven pattern — the server pushes events; the client reacts.

```ts
// Native browser CustomEvent — useful for cross-component signalling without prop drilling
window.dispatchEvent(new CustomEvent('auth:expired', { detail: { reason: 'token_ttl' } }))
window.addEventListener('auth:expired', (e) => {
  logout()
  showToast((e as CustomEvent).detail.reason)
})

// Zustand subscribe — react to store changes outside React components
// (useful for side effects like persisting to localStorage)
useChatStore.subscribe(
  state => state.activeId,     // selector — only fire when activeId changes
  (activeId) => {
    if (activeId) document.title = `Chat — ${activeId.slice(0, 8)}`
  }
)

// The SSE generator is itself an event source: server pushes, client pulls
// The for-await-of loop IS the event loop for our chat stream
for await (const event of streamSSE(url, opts, signal)) {
  // each iteration = one server-sent event
  dispatch(event)
}
```

The key insight: SSE over `fetch` + an async generator gives you a typed, cancellable, POST-compatible event stream — something `EventSource` cannot do.
