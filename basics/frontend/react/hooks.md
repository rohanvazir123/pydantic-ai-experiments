# React Hooks with TypeScript

## Table of Contents

- [Basic Hooks](#basic-hooks)
  - [useState](#usestate)
  - [useEffect](#useeffect)
    - [Complete example — typed API fetch with loading/error states](#complete-example--typed-api-fetch-with-loadingerror-states)
  - [useContext](#usecontext)
- [Ref Hooks](#ref-hooks)
  - [useRef](#useref)
  - [useImperativeHandle](#useimperativehandle)
- [Performance Hooks](#performance-hooks)
  - [useMemo](#usememo)
  - [useCallback](#usecallback)
- [Transition and Deferral Hooks](#transition-and-deferral-hooks)
  - [useTransition](#usetransition)
  - [useDeferredValue](#usedeferredvalue)
- [Sync and ID Hooks](#sync-and-id-hooks)
  - [useId](#useid)
  - [useSyncExternalStore](#usesyncexternalstore)
- [Effect Variant Hooks](#effect-variant-hooks)
  - [useLayoutEffect](#uselayouteffect)
  - [useInsertionEffect](#useinsertioneffect)
- [Custom Hooks](#custom-hooks)
  - [Rules and Patterns](#rules-and-patterns)
  - [useChat — Full Project Breakdown](#usechat--full-project-breakdown)
  - [useLocalStorage — Generic Example](#uselocalstorage--generic-example)

---

## Basic Hooks

### useState

**Signature:** `useState<T>(initialValue: T | (() => T)): [T, Dispatch<SetStateAction<T>>]`

Manages local component state. `T` is usually inferred — annotate explicitly only when TypeScript cannot infer from the initial value.

```tsx
const [text, setText] = useState('')                        // inferred string
const [user, setUser] = useState<User | null>(null)         // explicit — null is ambiguous
const [items, setItems] = useState<string[]>(() =>          // lazy initialiser — runs once
  JSON.parse(localStorage.getItem('items') ?? '[]'))

setCount(prev => prev + 1)   // functional update — safe when updates batch
```

**Use:** values that, when changed, should re-render the component.
**Avoid:** for values that do not affect rendering — use `useRef` instead.

---

### useEffect

**Signature:** `useEffect(setup: () => (() => void) | void, deps?: unknown[]): void`

Runs a side effect after the browser paints. Return a cleanup function or nothing.

```tsx
useEffect(() => {
  const controller = new AbortController()
  fetch('/api/data', { signal: controller.signal }).then(r => r.json()).then(setData)
  return () => controller.abort()   // cleanup on unmount or before next run
}, [url])
```

**Why you cannot async the callback** — `useEffect` expects `void | (() => void)`, not `Promise<void>`. Define async inside and call immediately:

```tsx
useEffect(() => {
  async function load() { setData(await fetchData()) }
  load()
}, [])
```

**Dependency array:** omit — every render; `[]` — once on mount; `[a, b]` — when a or b changes.

**Complete example — typed API fetch with loading/error states:**

```tsx
import React, { useState, useEffect } from "react";

// 1. Define TypeScript types for the API response shape
interface PexelsPhoto {
  id: number;
  photographer: string;
  avg_color: string;
  src: { medium: string; large: string };
}

interface PexelsResponse {
  page: number;
  per_page: number;
  photos: PexelsPhoto[];
  total_results: number;
}

export function PexelsGallery() {
  // 2. Initialise state with explicit TypeScript types
  const [photos, setPhotos] = useState<PexelsPhoto[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 3. Define the async function INSIDE the hook — never make the callback itself async
    const fetchPhotos = async () => {
      try {
        setLoading(true);
        const response = await fetch("https://api.pexels.com/v1/curated", {
          headers: { Authorization: "YOUR_API_KEY" },
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data: PexelsResponse = await response.json();
        setPhotos(data.photos);
      } catch (err) {
        // err is `unknown` — narrow it before using .message
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);  // always clears loading, even on error
      }
    };

    // 4. Invoke immediately
    fetchPhotos();
  }, []); // 5. Empty array → runs exactly once on mount

  // 6. Handle each UI state before the happy path
  if (loading) return <div>Loading images...</div>;
  if (error)   return <div>Error: {error}</div>;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px" }}>
      {photos.map((photo) => (
        <div key={photo.id} style={{ backgroundColor: photo.avg_color, padding: "8px", borderRadius: "8px" }}>
          <img src={photo.src.medium} alt={`Shot by ${photo.photographer}`} style={{ width: "100%" }} />
          <p style={{ margin: "4px 0 0", fontSize: "12px" }}>📸 {photo.photographer}</p>
        </div>
      ))}
    </div>
  );
}
```

Key patterns this example demonstrates:
- Interfaces mirror the exact API response shape — no `any`
- `useState<PexelsPhoto[]>([])` — explicit generic, not inferred from `[]`
- `err instanceof Error` — the correct way to narrow `unknown` in a catch block
- `finally` — `setLoading(false)` runs whether the fetch succeeded or threw
- Early returns for loading/error keep the happy-path JSX clean

---

### useContext

**Signature:** `useContext<T>(context: React.Context<T>): T`

Reads the nearest matching `Context.Provider` value.

```tsx
const ThemeContext = React.createContext<{ theme: string; toggle: () => void } | null>(null)

function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be inside ThemeProvider')
  return ctx
}
```

**Why this project uses Zustand instead:** Context re-renders every consumer when the value reference changes. Zustand's `useSyncExternalStore` re-renders only components subscribed to the changed slice.

---

## Ref Hooks

### useRef

**Signature:** `useRef<T>(initialValue: T): MutableRefObject<T>`

Two distinct uses — the type annotation makes the intent explicit.

**DOM ref** — attach to an element, read the live DOM node after mount:

```tsx
// From InputBar.tsx — auto-resize textarea
const textareaRef = useRef<HTMLTextAreaElement>(null)

function handleInput() {
  const el = textareaRef.current
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}

return <textarea ref={textareaRef} onChange={handleInput} />
```

**Mutable value ref** — persist a value across renders without triggering re-render:

```tsx
// From useChat.ts — cancel in-flight SSE stream before starting a new one
const abortRef = useRef<AbortController | null>(null)

abortRef.current?.abort()
abortRef.current = new AbortController()

function stop() { abortRef.current?.abort() }
```

The abort ref is the canonical pattern for hooks driving async streams. It survives re-renders, does not cause them, and is always readable from event handlers.

---

### useImperativeHandle

**Signature:** `useImperativeHandle<T>(ref, createHandle: () => T, deps?): void`

Exposes a custom imperative API from a child to a parent ref — use when `forwardRef` alone exposes too much (e.g. you want `reset()` without exposing the raw DOM node).

```tsx
interface DialogHandle { open: () => void; close: () => void }

const Modal = React.forwardRef<DialogHandle, { children: React.ReactNode }>(({ children }, ref) => {
  const [open, setOpen] = useState(false)
  useImperativeHandle(ref, () => ({ open: () => setOpen(true), close: () => setOpen(false) }))
  return open ? <div className="modal">{children}</div> : null
})

// Parent
const modalRef = useRef<DialogHandle>(null)
modalRef.current?.open()
```

**Avoid** for anything achievable with props and callbacks.

---

## Performance Hooks

### useMemo

**Signature:** `useMemo<T>(factory: () => T, deps: unknown[]): T`

Memoizes an expensive computed value; recomputes only when `deps` changes.

```tsx
const filtered = useMemo<Conversation[]>(
  () => conversations.filter(c => c.title.toLowerCase().includes(query.toLowerCase())),
  [conversations, query]
)
```

**Helps:** genuinely expensive factory (large sort, regex over thousands of items) + frequent unrelated re-renders.
**Does not help:** cheap arithmetic; deps contain objects recreated each render (memo never hits).

---

### useCallback

**Signature:** `useCallback<F extends Function>(fn: F, deps: unknown[]): F`

Memoizes a function reference; recreates only when `deps` changes.

```tsx
const handleSend = useCallback((query: string) => sendMessage(query), [sendMessage])
return <InputBar onSend={handleSend} onStop={stop} loading={loading} />
```

**Prevents re-renders only when:** the child is wrapped in `React.memo` AND the function is a prop — both conditions must be true.
**Premature:** for inline handlers not passed to memoized children.

---

## Transition and Deferral Hooks

### useTransition

**Signature:** `useTransition(): [boolean, TransitionStartFunction]`

Marks a state update as non-urgent. React can interrupt it for higher-priority work. `isPending` is `true` during the transition.

```tsx
const [isPending, startTransition] = useTransition()

startTransition(() => setActiveTab(tab))   // non-urgent — defer rendering

{isPending ? <Spinner /> : <TabContent tab={activeTab} />}
```

**Use:** tab switches, navigation, filtering large lists.
**Avoid:** for urgent updates (text input, button feedback) — those must be immediate.

---

### useDeferredValue

**Signature:** `useDeferredValue<T>(value: T): T`

Returns a lagging copy of `value` during busy renders, keeping the real value responsive.

```tsx
const deferredQuery = useDeferredValue(query)   // typed as string
const results = useMemo(() => heavySearch(items, deferredQuery), [items, deferredQuery])
```

**vs useTransition:** use `useDeferredValue` when you receive the value (prop or external state). Use `useTransition` when you own the setter.

---

## Sync and ID Hooks

### useId

**Signature:** `useId(): string`

Generates a stable, unique ID per component instance — survives SSR hydration.

```tsx
function FormField({ label }: { label: string }) {
  const id = useId()   // e.g. ':r1:'
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <input id={id} />
    </>
  )
}
```

**Use:** `htmlFor`/`id` pairs, `aria-labelledby`, `aria-describedby`.
**Avoid for list keys** — use data IDs for that.

---

### useSyncExternalStore

**Signature:** `useSyncExternalStore<T>(subscribe, getSnapshot: () => T, getServerSnapshot?: () => T): T`

The hook Zustand uses internally. Re-renders only when the snapshot changes — safe under concurrent rendering.

```tsx
function useWindowWidth(): number {
  return useSyncExternalStore(
    cb => { window.addEventListener('resize', cb); return () => window.removeEventListener('resize', cb) },
    () => window.innerWidth,
    () => 1024,   // SSR default
  )
}
```

Reach for this when connecting React to an external event source (WebSocket, browser API, third-party store) that has no React state.

---

## Effect Variant Hooks

### useLayoutEffect

**Signature:** Same as `useEffect`.

Runs synchronously after React mutates the DOM but before the browser paints. Use for layout measurements to avoid a visible flash.

```tsx
const ref = useRef<HTMLDivElement>(null)
const [height, setHeight] = useState(0)

useLayoutEffect(() => {
  if (ref.current) setHeight(ref.current.getBoundingClientRect().height)
}, [])
```

**Use:** DOM measurements, tooltip positioning, reading scroll offset before paint.
**Avoid:** data fetching, subscriptions — `useLayoutEffect` blocks painting. Use `useEffect` for those.

---

### useInsertionEffect

**Signature:** Same as `useEffect`.

Runs before any DOM mutations. CSS-in-JS libraries (styled-components, Emotion) use it to inject `<style>` tags before render.

**You almost certainly do not need this.** It exists for library authors building CSS-in-JS runtimes.

---

## Custom Hooks

### Rules and Patterns

- Name must start with `use`.
- Call only at the top level of a component or another hook — never inside conditions, loops, or nested functions.
- Return a tuple for two values (mirrors `useState`); return a named object for three or more.

```tsx
function useToggle(initial = false): [boolean, () => void] {
  const [value, setValue] = useState(initial)
  return [value, () => setValue(v => !v)]
}
```

---

### useChat — Full Project Breakdown

`useChat` at `rag/v2/frontend/src/hooks/useChat.ts` combines `useRef`, Zustand, and an async generator to stream SSE responses.

```ts
export function useChat() {
  const abortRef = useRef<AbortController | null>(null)

  async function sendMessage(query: string) {
    // Read Zustand via getState() — avoids stale closure from hook render time
    const getStore = useChatStore.getState

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      for await (const event of streamSSE(
        '/api/v2/chat/stream',
        { method: 'POST', body: JSON.stringify({ query, ... }) },
        abortRef.current.signal,
      )) {
        if ('delta' in event)             getStore().appendToken(convId, event.delta)
        else if ('done' in event && event.done) getStore().finaliseMessage(convId, { citations: ... })
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError')
        getStore().finaliseMessage(convId, { content: 'Connection error.' })
    }
  }

  function stop() { abortRef.current?.abort() }
  return { sendMessage, stop }
}
```

Key patterns:
1. `useRef<AbortController | null>(null)` — lets `stop()` cancel the stream from any event handler, no shared state.
2. `useChatStore.getState` inside async — reads current store state, not the stale snapshot captured at hook render time.
3. `for await` on an async generator — each SSE event is yielded one at a time.
4. No `useState` or `useEffect` — all UI state lives in Zustand; the hook stays stateless and easy to test.

---

### useLocalStorage — Generic Example

```tsx
function useLocalStorage<T>(key: string, defaultValue: T): [T, (value: T) => void] {
  const [stored, setStored] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key)
      return item ? (JSON.parse(item) as T) : defaultValue
    } catch { return defaultValue }
  })

  function setValue(value: T) {
    try { setStored(value); localStorage.setItem(key, JSON.stringify(value)) }
    catch { /* storage full or SSR — silently skip */ }
  }

  return [stored, setValue]
}

const [theme, setTheme] = useLocalStorage('theme', 'dark')          // T inferred as string
const [user,  setUser]  = useLocalStorage<User | null>('user', null) // T explicit
```

The lazy initialiser reads storage once on mount; `try/catch` handles corrupted JSON and environments where `localStorage` is unavailable (SSR, strict private browsing).
