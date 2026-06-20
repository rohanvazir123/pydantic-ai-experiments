# Day 4 — Hooks Step-by-Step Walkthrough

You will extend the app from Day 3 with every standard React hook, learning exactly when and why each one exists. By the end you have a solid mental model of the full hook system.

**Total time:** ~4 hours  
**Reference doc:** [hooks.md](hooks.md)  
**Prerequisites:** Day 3 done — you have `Counter.tsx`, `UserCard.tsx`, `UserList.tsx`, and a working `App.tsx` that fetches users

## Table of Contents

- [Before You Start](#before-you-start)
- [Step 1 — useRef for DOM focus](#step-1--useref-for-dom-focus)
- [Step 2 — useRef for a mutable value](#step-2--useref-for-a-mutable-value)
- [Step 3 — useContext — theme toggle](#step-3--usecontext--theme-toggle)
- [Step 4 — useMemo — filtered list](#step-4--usememo--filtered-list)
- [Step 5 — useCallback — stable handlers](#step-5--usecallback--stable-handlers)
- [Step 6 — useReducer — replace multiple useState](#step-6--usereducer--replace-multiple-usestate)
- [Step 7 — useId — accessible form](#step-7--useid--accessible-form)
- [Step 8 — useTransition — non-urgent update](#step-8--usetransition--non-urgent-update)
- [Step 9 — useDeferredValue — responsive input](#step-9--usedeferredvalue--responsive-input)
- [Step 10 — Custom hook — useUserSearch](#step-10--custom-hook--useusersearch)
- [End-of-Day Checklist](#end-of-day-checklist)

---

## Before You Start

```bash
cd basics/exercises
npm run dev                  # terminal 1
npx tsc --noEmit --watch     # terminal 2
```

You should have from Day 3:
```
src/
├── components/
│   ├── Counter.tsx
│   ├── UserCard.tsx
│   └── UserList.tsx
└── App.tsx
```

If you do not, scaffold `App.tsx` now with the fetch example from [day3.md Step 7](day3.md#step-7--read-useeffect-then-add-a-data-fetch) before continuing.

---

## Step 1 — useRef for DOM focus

**Read** "useRef" in `hooks.md` (~5 min).

`useRef` has two jobs — DOM references and mutable values. This step covers DOM refs.

Add auto-scroll to the bottom of the user list when users load. Create `basics/exercises/src/components/ScrollList.tsx`:

```tsx
import { useRef, useEffect } from 'react'
import { UserCard } from './UserCard'

interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface Props {
  users: User[]
}

export function ScrollList({ users }: Props) {
  // DOM ref — attach to the last element so we can scroll to it
  const bottomRef = useRef<HTMLDivElement>(null)

  // Scroll into view whenever the users array changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [users])

  return (
    <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #ccc' }}>
      {users.map(u => <UserCard key={u.id} user={u} />)}
      <div ref={bottomRef} />  {/* invisible sentinel — we scroll to this */}
    </div>
  )
}
```

Replace `<UserList>` in `App.tsx` with `<ScrollList users={visibleUsers} />`. Scroll the list — the bottom is always visible after updates.

**What to notice:**
- `useRef<HTMLDivElement>(null)` — the generic matches the element type, initial value is always `null` for DOM refs
- `?.scrollIntoView` — guard with optional chaining because `current` is null until mount
- The ref does not cause a re-render when it is set — that is the point

---

## Step 2 — useRef for a mutable value

`useRef` stores values that should survive re-renders without causing them — the classic use case is storing a timer ID or an `AbortController`.

Add a debounced search: wait 300ms after the user stops typing before filtering. Update `App.tsx`:

```tsx
import { useState, useEffect, useRef } from 'react'

export default function App() {
  const [rawQuery, setRawQuery]   = useState('')
  const [query, setQuery]         = useState('')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounce: clear the previous timer, start a new one
  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value
    setRawQuery(value)   // update the input display immediately

    if (timerRef.current) clearTimeout(timerRef.current)  // cancel previous timer

    timerRef.current = setTimeout(() => {
      setQuery(value)    // only update the filter after 300ms of silence
    }, 300)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  // ...rest of App
  return (
    <>
      <input
        value={rawQuery}
        onChange={handleInputChange}
        placeholder="Filter users (debounced 300ms)..."
        style={{ padding: '8px', width: '100%', marginBottom: '12px' }}
      />
      {/* rest of JSX */}
    </>
  )
}
```

Type quickly — notice the filter only updates after you pause.

**Why `useRef` and not `useState` for the timer?** The timer ID is not displayed — there is no reason to re-render when it changes. `useRef` stores it silently.

---

## Step 3 — useContext — theme toggle

**Read** "useContext" in `hooks.md` (~5 min).

Create a theme context so any component can read and toggle dark/light mode without prop drilling.

Create `basics/exercises/src/context/ThemeContext.tsx`:

```tsx
import { createContext, useContext, useState } from 'react'

interface ThemeContextValue {
  theme: 'light' | 'dark'
  toggle: () => void
}

// null initial value — we will throw if used outside the provider
const ThemeContext = createContext<ThemeContextValue | null>(null)

// Custom hook — throws a clear error if used outside provider
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>')
  return ctx
}

// Provider — wraps the whole app
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const toggle = () => setTheme(t => t === 'light' ? 'dark' : 'light')

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      <div style={{
        minHeight: '100vh',
        background: theme === 'dark' ? '#1a1a1a' : '#ffffff',
        color: theme === 'dark' ? '#ffffff' : '#000000',
        padding: '16px',
      }}>
        {children}
      </div>
    </ThemeContext.Provider>
  )
}
```

Wrap `App` with the provider in `basics/exercises/src/main.tsx`:

```tsx
import { ThemeProvider } from './context/ThemeContext'
import App from './App'
import ReactDOM from 'react-dom/client'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <ThemeProvider>
    <App />
  </ThemeProvider>
)
```

Add a toggle button to `App.tsx`:

```tsx
import { useTheme } from './context/ThemeContext'

export default function App() {
  const { theme, toggle } = useTheme()

  return (
    <>
      <button onClick={toggle}>
        Switch to {theme === 'light' ? 'dark' : 'light'} mode
      </button>
      {/* rest of JSX */}
    </>
  )
}
```

Click the button. The background switches instantly — `useContext` re-renders every consumer when the value changes.

---

## Step 4 — useMemo — filtered list

**Read** "useMemo" in `hooks.md` (~5 min).

Replace the inline `visibleUsers` derivation with `useMemo` so the filter only re-runs when `users` or `query` actually changes:

```tsx
import { useMemo } from 'react'

// Inside App():
const visibleUsers = useMemo(() => {
  return users.filter(u =>
    u.name.toLowerCase().includes(query.toLowerCase()) ||
    u.email.toLowerCase().includes(query.toLowerCase())
  )
}, [users, query])   // only re-runs when users or query changes
```

**When does this actually matter?** Right now with 10 users — never. `useMemo` helps when:
1. The computation is genuinely expensive (sorting thousands of items)
2. Something else causes frequent re-renders (typing, animations)

The habit to build: always derive data inline first. Add `useMemo` only after profiling shows a slowdown.

---

## Step 5 — useCallback — stable handlers

**Read** "useCallback" in `hooks.md` (~5 min).

Create a `UserCardMemo.tsx` that only re-renders when its specific user changes:

```tsx
import { memo } from 'react'
import { UserCard } from './UserCard'

interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface Props {
  user: User
  onSelect: (id: number) => void
}

// memo — only re-renders if props changed (shallow comparison)
export const UserCardMemo = memo(function UserCardMemo({ user, onSelect }: Props) {
  console.log(`Rendering ${user.name}`)   // watch this in console
  return (
    <div onClick={() => onSelect(user.id)} style={{ cursor: 'pointer' }}>
      <UserCard user={user} />
    </div>
  )
})
```

In `App.tsx`, use `useCallback` to keep the handler reference stable:

```tsx
import { useCallback, useState } from 'react'
import { UserCardMemo } from './components/UserCardMemo'

const [selectedId, setSelectedId] = useState<number | null>(null)

// Without useCallback: new function reference on every render → memo never helps
// With useCallback: same reference as long as deps don't change → memo works
const handleSelect = useCallback((id: number) => {
  setSelectedId(id)
}, [])   // no deps — function never changes

// In JSX:
{visibleUsers.map(u => (
  <UserCardMemo
    key={u.id}
    user={u}
    onSelect={handleSelect}
    highlighted={u.id === selectedId}
  />
))}
```

Open DevTools console. Typing in the filter input — only the components that changed should log. Without `useCallback`, ALL would log on every keystroke.

**The rule:** `useCallback` only prevents re-renders when the child is wrapped in `memo` AND the function is passed as a prop. Both conditions must be true.

---

## Step 6 — useReducer — replace multiple useState

**Read** the optional "useReducer" section if it is in `hooks.md`, otherwise follow this pattern.

When you have multiple state values that always change together, `useReducer` is cleaner than separate `useState` calls.

Create `basics/exercises/src/components/SearchBar.tsx`:

```tsx
import { useReducer } from 'react'

interface State {
  query: string
  activeFilter: 'all' | 'admin' | 'viewer'
  sortBy: 'name' | 'email'
}

type Action =
  | { type: 'SET_QUERY';  query: string }
  | { type: 'SET_FILTER'; filter: State['activeFilter'] }
  | { type: 'SET_SORT';   sortBy: State['sortBy'] }
  | { type: 'RESET' }

const initialState: State = { query: '', activeFilter: 'all', sortBy: 'name' }

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'SET_QUERY':  return { ...state, query: action.query }
    case 'SET_FILTER': return { ...state, activeFilter: action.filter }
    case 'SET_SORT':   return { ...state, sortBy: action.sortBy }
    case 'RESET':      return initialState
  }
}

interface Props {
  onChange: (state: State) => void
}

export function SearchBar({ onChange }: Props) {
  const [state, dispatch] = useReducer(reducer, initialState)

  function update(action: Action) {
    const next = reducer(state, action)   // compute next state locally
    dispatch(action)                       // update component state
    onChange(next)                         // notify parent
  }

  return (
    <div style={{ marginBottom: '16px' }}>
      <input
        value={state.query}
        onChange={e => update({ type: 'SET_QUERY', query: e.target.value })}
        placeholder="Search..."
        style={{ padding: '8px', marginRight: '8px' }}
      />
      <select
        value={state.activeFilter}
        onChange={e => update({ type: 'SET_FILTER', filter: e.target.value as State['activeFilter'] })}
      >
        <option value="all">All roles</option>
        <option value="admin">Admin only</option>
        <option value="viewer">Viewer only</option>
      </select>
      <button onClick={() => update({ type: 'RESET' })} style={{ marginLeft: '8px' }}>
        Reset
      </button>
    </div>
  )
}
```

Replace the manual filter input in `App.tsx` with `<SearchBar onChange={...} />`.

---

## Step 7 — useId — accessible form

**Read** "useId" in `hooks.md` (~3 min).

Create `basics/exercises/src/components/FormField.tsx`:

```tsx
import { useId } from 'react'

interface Props {
  label: string
  value: string
  onChange: (value: string) => void
  type?: 'text' | 'email' | 'password'
}

export function FormField({ label, value, onChange, type = 'text' }: Props) {
  // Stable unique ID — survives re-renders, safe for SSR
  const id = useId()

  return (
    <div style={{ marginBottom: '12px' }}>
      {/* htmlFor links label to input — clicking the label focuses the input */}
      <label htmlFor={id} style={{ display: 'block', marginBottom: '4px' }}>
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        aria-labelledby={id}
        style={{ padding: '8px', width: '100%' }}
      />
    </div>
  )
}
```

Add two `<FormField>` components to `App.tsx` and click their labels — the inputs should focus.

**Why not just hardcode `id="name-field"`?** If the component renders twice on the same page, you get duplicate IDs — which breaks accessibility. `useId` guarantees uniqueness per instance.

---

## Step 8 — useTransition — non-urgent update

**Read** "useTransition" in `hooks.md` (~5 min).

Wrap the filter state update in a transition so React can interrupt it for higher-priority work (like keeping the input responsive):

```tsx
import { useTransition } from 'react'

// Inside App():
const [isPending, startTransition] = useTransition()
const [query, setQuery] = useState('')

function handleSearch(value: string) {
  // The input update is urgent — happens immediately
  setRawQuery(value)

  // The filter update is non-urgent — React can delay it
  startTransition(() => {
    setQuery(value)
  })
}

// Show a subtle indicator while the transition runs
{isPending && <p style={{ color: '#999', fontSize: '12px' }}>Filtering...</p>}
```

**When does this matter?** Only when the update causes expensive rendering (a list of thousands of items). With 10 users it makes no visible difference — the value is the pattern itself.

---

## Step 9 — useDeferredValue — responsive input

**Read** "useDeferredValue" in `hooks.md` (~5 min).

`useDeferredValue` is the alternative to `useTransition` when you receive the value rather than owning the setter:

```tsx
import { useDeferredValue, useMemo } from 'react'

// Inside App():
const [query, setQuery] = useState('')

// query updates on every keystroke (urgent — keeps input responsive)
// deferredQuery lags behind (non-urgent — React schedules it when idle)
const deferredQuery = useDeferredValue(query)

const visibleUsers = useMemo(() => {
  return users.filter(u =>
    u.name.toLowerCase().includes(deferredQuery.toLowerCase())
  )
}, [users, deferredQuery])   // use deferredQuery, not query

// The input always shows the latest query
// The list shows slightly older results while catching up
```

**`useDeferredValue` vs `useTransition`:**
- Use `useTransition` when you own the state setter
- Use `useDeferredValue` when the value comes from outside (props, another hook)

---

## Step 10 — Custom hook — useUserSearch

**Read** "Custom Hooks" in `hooks.md` — the `useChat` breakdown and `useLocalStorage` example (~10 min).

Extract everything fetch-related from `App.tsx` into a reusable hook. Create `basics/exercises/src/hooks/useUserSearch.ts`:

```ts
import { useState, useEffect, useRef, useMemo } from 'react'

interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface ApiUser {
  id: number
  name: string
  email: string
}

interface UseUserSearchReturn {
  users: User[]
  visibleUsers: User[]
  loading: boolean
  error: string | null
  query: string
  setQuery: (q: string) => void
}

export function useUserSearch(): UseUserSearchReturn {
  const [users, setUsers]     = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [query, setQuery]     = useState('')
  const abortRef              = useRef<AbortController | null>(null)

  useEffect(() => {
    // Cancel any in-flight request
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    async function fetch_() {
      try {
        setLoading(true)
        setError(null)
        const res = await fetch('https://jsonplaceholder.typicode.com/users', {
          signal: abortRef.current!.signal,
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: ApiUser[] = await res.json()
        setUsers(data.map(u => ({ ...u, role: 'viewer' as const })))
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(err instanceof Error ? err.message : 'Unknown error')
        }
      } finally {
        setLoading(false)
      }
    }

    fetch_()

    return () => abortRef.current?.abort()
  }, [])

  const visibleUsers = useMemo(() =>
    users.filter(u =>
      u.name.toLowerCase().includes(query.toLowerCase()) ||
      u.email.toLowerCase().includes(query.toLowerCase())
    ),
    [users, query]
  )

  return { users, visibleUsers, loading, error, query, setQuery }
}
```

Now `App.tsx` becomes trivially simple:

```tsx
import { useUserSearch } from './hooks/useUserSearch'
import { ScrollList } from './components/ScrollList'
import { useTheme } from './context/ThemeContext'

export default function App() {
  const { visibleUsers, loading, error, query, setQuery, users } = useUserSearch()
  const { theme, toggle } = useTheme()

  if (loading) return <p>Loading...</p>
  if (error)   return <p style={{ color: 'red' }}>Error: {error}</p>

  return (
    <>
      <button onClick={toggle}>
        {theme === 'light' ? 'Dark' : 'Light'} mode
      </button>
      <h1>Users ({visibleUsers.length} of {users.length})</h1>
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Filter users..."
        style={{ padding: '8px', width: '100%', marginBottom: '12px' }}
      />
      <ScrollList users={visibleUsers} />
    </>
  )
}
```

The component has zero fetch logic — it just calls a hook and renders. This is the pattern for every real React app.

---

## End-of-Day Checklist

Close `hooks.md`. Answer from memory:

- [ ] What are the two different uses of `useRef`? Give an example of each.
- [ ] When does `useMemo` actually help? State the two conditions.
- [ ] For `useCallback` to prevent a re-render, what two things must both be true?
- [ ] What is `useContext` for? What is its main performance downside?
- [ ] What is the difference between `useTransition` and `useDeferredValue`? When do you use each?
- [ ] What does `useId` solve that a hardcoded string id does not?
- [ ] What are the Rules of Hooks? Why do they exist?
- [ ] Write the signature of a custom hook that returns `{ data, loading, error }` from memory.

Your `src/` should now have:
```
src/
├── components/
│   ├── Counter.tsx
│   ├── FormField.tsx
│   ├── ScrollList.tsx
│   ├── SearchBar.tsx
│   ├── UserCard.tsx
│   ├── UserCardMemo.tsx
│   └── UserList.tsx
├── context/
│   └── ThemeContext.tsx
├── hooks/
│   └── useUserSearch.ts
└── App.tsx
```

**Tomorrow (Day 5):** Build the full Todo app and photo search capstone — [practice/exercises.md](../../practice/exercises.md).
