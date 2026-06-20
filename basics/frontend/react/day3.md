# Day 3 — React Step-by-Step Walkthrough

You will build 5 small components today, each teaching one concept. By the end you have a working mini-app and a solid mental model of React. Do every step in order — do not skip ahead.

**Total time:** ~4 hours  
**Reference doc:** [react.md](react.md) — open it in a second tab

## Table of Contents

- [Sample Component — Read This First](#sample-component--read-this-first)
- [Before You Start](#before-you-start)
- [Step 1 — Read JSX, then wipe App.tsx](#step-1--read-jsx-then-wipe-apptsx)
- [Step 2 — Build Counter (useState)](#step-2--build-counter-usestate)
- [Step 3 — Add conditional rendering to Counter](#step-3--add-conditional-rendering-to-counter)
- [Step 4 — Build UserCard (typed props)](#step-4--build-usercard-typed-props)
- [Step 5 — Build UserList (lists + keys)](#step-5--build-userlist-lists--keys)
- [Step 6 — Add event handling — filterable list](#step-6--add-event-handling--filterable-list)
- [Step 7 — Read useEffect, then add a data fetch](#step-7--read-useeffect-then-add-a-data-fetch)
- [Step 8 — Read useRef, then wire up auto-focus](#step-8--read-useref-then-wire-up-auto-focus)
- [Step 9 — Lift state up — connect Counter and UserList](#step-9--lift-state-up--connect-counter-and-userlist)
- [Step 10 — Read Parts 2 and 3 (no coding)](#step-10--read-parts-2-and-3-no-coding)
- [End-of-Day Checklist](#end-of-day-checklist)

---

## Sample Component — Read This First

Before diving into the steps, read through this complete, production-style component. Every pattern used here is explained in the steps below — this gives you the target to aim for.

**File:** `basics/exercises/src/components/UserCard.tsx`

```tsx
import React, { useState } from 'react';

// 1. Define an interface for the component props
interface UserCardProps {
  name: string;
  age: number;
  email?: string;                          // optional prop — has a default below
  role: 'admin' | 'user' | 'guest';       // union literal — only these three strings are valid
  onStatusChange: (status: string) => void; // function prop — parent handles the side effect
}

// 2. Standard function with destructured, typed props
//    Prefer this over React.FC — simpler, no implicit children prop
export const UserCard = ({
  name,
  age,
  email = 'No email provided',             // default value for optional prop
  role,
  onStatusChange,
}: UserCardProps): React.JSX.Element => {

  // 3. useState — explicit generic keeps TypeScript honest
  const [isActive, setIsActive] = useState<boolean>(true);

  // 4. Typed event handler — MouseEvent generic matches the element it is attached to
  const handleToggle = (event: React.MouseEvent<HTMLButtonElement>): void => {
    event.preventDefault();
    const newStatus = !isActive ? 'Active' : 'Inactive';
    setIsActive(!isActive);
    onStatusChange(newStatus);             // call back to the parent
  };

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', borderRadius: '8px' }}>
      <h2>{name}</h2>
      <p>Age: {age}</p>
      <p>Email: {email}</p>
      <p>Role: <strong>{role}</strong></p>
      <p>Status: {isActive ? '🟢 Active' : '🔴 Inactive'}</p>
      <button onClick={handleToggle}>Toggle Status</button>
    </div>
  );
};
```

**How to use it — `App.tsx`:**

```tsx
import { useState } from 'react';
import { UserCard } from './components/UserCard';

export default function App() {
  // Parent owns the log of status changes — lifted up from UserCard
  const [log, setLog] = useState<string[]>([]);

  function handleStatusChange(status: string) {
    setLog(prev => [...prev, `Status changed to: ${status}`]);
  }

  return (
    <div style={{ padding: '24px', maxWidth: '400px' }}>
      <h1>User Management</h1>

      {/* All required props provided */}
      <UserCard
        name="Ada Lovelace"
        age={36}
        email="ada@example.com"
        role="admin"
        onStatusChange={handleStatusChange}
      />

      {/* email omitted — uses default "No email provided" */}
      <UserCard
        name="Alan Turing"
        age={41}
        role="user"
        onStatusChange={handleStatusChange}
      />

      {/* Status change log */}
      {log.length > 0 && (
        <div style={{ marginTop: '16px', fontSize: '13px', color: '#666' }}>
          <strong>Activity log:</strong>
          <ul>
            {log.map((entry, i) => <li key={i}>{entry}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
```

**What each pattern demonstrates:**

| Pattern | Where | Why |
|---------|-------|-----|
| `interface UserCardProps` | Props definition | TypeScript enforces every caller passes the right shape |
| `email?: string` with default | Props + destructuring | Optional prop — parent can omit it |
| `'admin' \| 'user' \| 'guest'` | Role prop | Literal union — `role="superuser"` is a compile error |
| `(status: string) => void` | Callback prop | Parent decides what to do; child just calls back |
| `useState<boolean>(true)` | Local state | Explicit generic — avoids ambiguity |
| `React.MouseEvent<HTMLButtonElement>` | Event handler | Narrowed to the exact element type |
| `[...prev, entry]` | Log update | Spread creates a new array — never mutate state |
| `key={i}` in the log | List rendering | Fine here because the log only appends, never reorders |

Paste this into your project and run it before starting Step 1. Once you can see it in the browser, you are ready.

---

## Before You Start

Open two terminals:

```bash
# Terminal 1 — dev server
cd basics/exercises
npm run dev
# Opens http://localhost:5173 — keep this running all day

# Terminal 2 — type checker (catches errors as you save)
cd basics/exercises
npx tsc --noEmit --watch
```

Open `http://localhost:5173` in your browser. Keep it visible alongside your editor. Every time you save a file, the browser updates instantly.

Create the components folder:

```bash
mkdir -p basics/exercises/src/components
```

---

## Step 1 — Read JSX, then wipe App.tsx

**Read** the "JSX: What it actually is" section in `react.md` (~5 min).

Three rules to remember:
- One root element per component (use `<>...</>` fragment to avoid extra `<div>`)
- `className`, not `class`
- Expressions inside `{}`, not statements

Now replace everything in `basics/exercises/src/App.tsx` with this:

```tsx
export default function App() {
  return (
    <>
      <h1>React Day 3</h1>
    </>
  )
}
```

Save. The browser should show "React Day 3". If the type-checker shows no errors, move on.

---

## Step 2 — Build Counter (useState)

**Read** the "Functional components and props" and "useState: state vs derived values" sections (~10 min).

Create `basics/exercises/src/components/Counter.tsx`:

```tsx
import { useState } from 'react'

interface Props {
  label: string
  startAt?: number   // optional — defaults to 0
}

export function Counter({ label, startAt = 0 }: Props) {
  const [count, setCount] = useState(startAt)

  // derived value — do NOT put this in useState
  const isNegative = count < 0

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', marginBottom: '8px' }}>
      <h2>{label}: {count}</h2>
      <button onClick={() => setCount(prev => prev + 1)}>+</button>
      <button onClick={() => setCount(prev => prev - 1)} style={{ margin: '0 8px' }}>−</button>
      <button onClick={() => setCount(0)}>Reset</button>
      {isNegative && <p style={{ color: 'red' }}>Gone negative!</p>}
    </div>
  )
}
```

Wire it up in `App.tsx`:

```tsx
import { Counter } from './components/Counter'

export default function App() {
  return (
    <>
      <h1>React Day 3</h1>
      <Counter label="Apples" />
      <Counter label="Oranges" startAt={5} />
    </>
  )
}
```

Save. Click the buttons. Both counters are **independent** — they each have their own `useState`. This is the fundamental unit of React state.

**What to notice:**
- `setCount(prev => prev + 1)` uses a functional update — always do this when the new value depends on the old one
- `isNegative` is computed inline, not stored in `useState` — it is always in sync

---

## Step 3 — Add conditional rendering to Counter

**Read** the "Conditional rendering" section (~5 min).

Three patterns — update `Counter.tsx` to practice all three:

```tsx
export function Counter({ label, startAt = 0 }: Props) {
  const [count, setCount] = useState(startAt)
  const isNegative = count < 0
  const isZero = count === 0

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', marginBottom: '8px' }}>
      <h2>{label}: {count}</h2>

      {/* Pattern 1: && — render something or nothing */}
      {isNegative && <p style={{ color: 'red' }}>Gone negative!</p>}

      {/* Pattern 2: ternary — render one of two things */}
      <p>{isZero ? 'Counter is at zero' : `${Math.abs(count)} away from zero`}</p>

      {/* Pattern 3: early return is for the whole component — demonstrated below */}
      <button onClick={() => setCount(prev => prev + 1)}>+</button>
      <button onClick={() => setCount(prev => prev - 1)} style={{ margin: '0 8px' }}>−</button>
      <button onClick={() => setCount(0)}>Reset</button>
    </div>
  )
}
```

Save and test. The message below the title should update as you click.

**Common mistake to avoid:** `{count && <p>...</p>}` renders `0` when count is 0. Use `{count !== 0 && <p>...</p>}` instead.

---

## Step 4 — Build UserCard (typed props)

**Read** "Functional components and props" again, focusing on the TypeScript parts (~5 min).

Create `basics/exercises/src/components/UserCard.tsx`:

```tsx
interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface Props {
  user: User
  highlighted?: boolean
}

export function UserCard({ user, highlighted = false }: Props) {
  // Early return — render nothing if name is somehow empty
  if (!user.name) return null

  return (
    <div
      style={{
        border: `2px solid ${highlighted ? 'blue' : '#ccc'}`,
        padding: '12px',
        marginBottom: '8px',
        borderRadius: '8px',
      }}
    >
      <strong>{user.name}</strong>
      <p style={{ margin: '4px 0', fontSize: '14px', color: '#666' }}>{user.email}</p>
      <span
        style={{
          fontSize: '12px',
          background: user.role === 'admin' ? '#fde68a' : '#e0e7ff',
          padding: '2px 8px',
          borderRadius: '4px',
        }}
      >
        {user.role}
      </span>
    </div>
  )
}
```

Add it to `App.tsx` to see it:

```tsx
import { Counter } from './components/Counter'
import { UserCard } from './components/UserCard'

const SAMPLE_USER = { id: 1, name: 'Ada Lovelace', email: 'ada@example.com', role: 'admin' as const }

export default function App() {
  return (
    <>
      <h1>React Day 3</h1>
      <Counter label="Apples" />
      <UserCard user={SAMPLE_USER} highlighted />
      <UserCard user={{ id: 2, name: 'Grace Hopper', email: 'grace@example.com', role: 'viewer' }} />
    </>
  )
}
```

**What to notice:**
- TypeScript errors immediately if you pass a wrong prop (try removing `name` — the editor underlines it)
- `role: 'admin' as const` — needed because `'admin'` would widen to `string` without it
- `highlighted` with no value means `highlighted={true}`

---

## Step 5 — Build UserList (lists + keys)

**Read** "Lists and keys" (~5 min).

Create `basics/exercises/src/components/UserList.tsx`:

```tsx
import { UserCard } from './UserCard'

interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
}

interface Props {
  users: User[]
  highlightedId?: number
}

export function UserList({ users, highlightedId }: Props) {
  if (users.length === 0) {
    return <p>No users found.</p>
  }

  return (
    <div>
      {users.map(user => (
        <UserCard
          key={user.id}             // key must be stable and unique — never use array index here
          user={user}
          highlighted={user.id === highlightedId}
        />
      ))}
    </div>
  )
}
```

Update `App.tsx`:

```tsx
import { Counter } from './components/Counter'
import { UserList } from './components/UserList'

const USERS = [
  { id: 1, name: 'Ada Lovelace',  email: 'ada@example.com',   role: 'admin'  as const },
  { id: 2, name: 'Grace Hopper',  email: 'grace@example.com', role: 'viewer' as const },
  { id: 3, name: 'Alan Turing',   email: 'alan@example.com',  role: 'viewer' as const },
]

export default function App() {
  return (
    <>
      <h1>React Day 3</h1>
      <Counter label="Apples" />
      <hr />
      <UserList users={USERS} highlightedId={1} />
    </>
  )
}
```

You should see all three users, with Ada highlighted in blue.

**Why `key` matters:** Remove `key={user.id}` from `UserList.tsx` — the browser console shows a warning. React uses keys to match old and new elements when the list changes. Without them, React re-creates every element on every update.

---

## Step 6 — Add event handling — filterable list

**Read** "Event handling" and "Lifting state up vs colocating" (~10 min).

This step introduces a controlled input — the pattern used for every text field in React.

Update `App.tsx`:

```tsx
import { useState } from 'react'
import { Counter } from './components/Counter'
import { UserList } from './components/UserList'

const USERS = [
  { id: 1, name: 'Ada Lovelace',  email: 'ada@example.com',   role: 'admin'  as const },
  { id: 2, name: 'Grace Hopper',  email: 'grace@example.com', role: 'viewer' as const },
  { id: 3, name: 'Alan Turing',   email: 'alan@example.com',  role: 'viewer' as const },
]

export default function App() {
  const [query, setQuery] = useState('')

  // derived value — filter inline, no separate useState
  const visibleUsers = USERS.filter(u =>
    u.name.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <>
      <h1>React Day 3</h1>
      <Counter label="Apples" />
      <hr />
      <input
        type="text"
        placeholder="Filter users..."
        value={query}                           // controlled — value driven by state
        onChange={e => setQuery(e.target.value)} // update state on every keystroke
        style={{ marginBottom: '12px', padding: '8px', width: '100%' }}
      />
      <p>{visibleUsers.length} user(s) found</p>
      <UserList users={visibleUsers} />
    </>
  )
}
```

Save. Type in the box — the list filters instantly.

**What to notice:**
- `value={query}` + `onChange` = controlled input. The input shows what React says, not what the browser wants.
- `visibleUsers` is derived from `query` and `USERS` — no extra `useState`. This is a critical habit.
- State lives in `App` because both the `<input>` and `<UserList>` need it — this is "lifting state up".

---

## Step 7 — Read useEffect, then add a data fetch

**Read** "useEffect: after render, cleanup, and the dependency array" carefully (~15 min). Pay attention to the dependency array table and the cleanup example.

Now add a real API call. Update `App.tsx` to fetch users from a public API instead of the hardcoded array:

```tsx
import { useState, useEffect } from 'react'
import { UserList } from './components/UserList'

// Shape of what the API returns
interface ApiUser {
  id: number
  name: string
  email: string
}

// Convert to our internal User type
function toUser(api: ApiUser) {
  return { id: api.id, name: api.name, email: api.email, role: 'viewer' as const }
}

export default function App() {
  const [users, setUsers]     = useState<ReturnType<typeof toUser>[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [query, setQuery]     = useState('')

  useEffect(() => {
    const controller = new AbortController()

    async function fetchUsers() {
      try {
        setLoading(true)
        const res = await fetch('https://jsonplaceholder.typicode.com/users', {
          signal: controller.signal,
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: ApiUser[] = await res.json()
        setUsers(data.map(toUser))
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(err instanceof Error ? err.message : 'Unknown error')
        }
      } finally {
        setLoading(false)
      }
    }

    fetchUsers()

    return () => controller.abort()   // cleanup: cancel the request on unmount
  }, [])  // empty array = run once on mount

  const visibleUsers = users.filter(u =>
    u.name.toLowerCase().includes(query.toLowerCase())
  )

  if (loading) return <p>Loading users...</p>
  if (error)   return <p style={{ color: 'red' }}>Error: {error}</p>

  return (
    <>
      <h1>React Day 3</h1>
      <input
        type="text"
        placeholder="Filter users..."
        value={query}
        onChange={e => setQuery(e.target.value)}
        style={{ marginBottom: '12px', padding: '8px', width: '100%' }}
      />
      <p>{visibleUsers.length} of {users.length} users</p>
      <UserList users={visibleUsers} />
    </>
  )
}
```

Save. You should briefly see "Loading users..." then 10 users appear from the API.

**What every line does:**
- `useEffect(() => { ... }, [])` — runs once after the first render
- `async function fetchUsers()` inside the effect — you can not make the callback itself async
- `controller.abort()` in the return — if the component unmounts before the fetch finishes, this cancels it
- `(err as Error).name !== 'AbortError'` — abort is not a real error, so we ignore it
- Early returns (`if (loading)`, `if (error)`) before the happy-path JSX — keeps the render clean

---

## Step 8 — Read useRef, then wire up auto-focus

**Read** "useRef: DOM refs and mutable values" (~10 min).

Add auto-focus to the search input so it is focused when the page loads. Update `App.tsx`:

```tsx
import { useState, useEffect, useRef } from 'react'

// ... (keep everything else the same)

export default function App() {
  // ... existing state

  const inputRef = useRef<HTMLInputElement>(null)   // DOM ref — always null until mount

  // Auto-focus the input once users have loaded
  useEffect(() => {
    if (!loading) {
      inputRef.current?.focus()   // ?. because current is null before mount
    }
  }, [loading])   // runs when loading changes from true to false

  // ... existing useEffect for fetch

  if (loading) return <p>Loading users...</p>
  if (error)   return <p style={{ color: 'red' }}>Error: {error}</p>

  return (
    <>
      <h1>React Day 3</h1>
      <input
        ref={inputRef}               // attach the ref to the DOM node
        type="text"
        placeholder="Filter users..."
        value={query}
        onChange={e => setQuery(e.target.value)}
        style={{ marginBottom: '12px', padding: '8px', width: '100%' }}
      />
      <p>{visibleUsers.length} of {users.length} users</p>
      <UserList users={visibleUsers} />
    </>
  )
}
```

Save. After the data loads, the input should be focused automatically.

**Why `useRef` and not `useState` for the input element?** The DOM node never changes — storing it in state would cause a pointless re-render every time the ref is set. `useRef` stores it without triggering any renders.

---

## Step 9 — Lift state up — connect Counter and UserList

**Read** "Lifting state up vs colocating" (~5 min).

Add the `Counter` back, but this time use its count to show only that many users — this forces you to lift state up.

```tsx
// In App.tsx — add Counter back with a shared count
import { Counter } from './components/Counter'

// Inside App():
const [limit, setLimit] = useState(5)

// Replace visibleUsers derivation with:
const visibleUsers = users
  .filter(u => u.name.toLowerCase().includes(query.toLowerCase()))
  .slice(0, limit)

// In the JSX:
<Counter label="Show users" startAt={5} />
```

Wait — this won't work yet. `Counter` owns its own state, so `App` cannot read it. This is the problem lifting state up solves.

Move the count out of `Counter` and into `App` by passing it as props:

Update `Counter.tsx` to be **controlled** — it receives `count` and `onChange` from the parent:

```tsx
interface Props {
  label: string
  count: number
  onChange: (newCount: number) => void
}

export function Counter({ label, count, onChange }: Props) {
  const isNegative = count < 0

  return (
    <div style={{ border: '1px solid #ccc', padding: '16px', marginBottom: '8px' }}>
      <h2>{label}: {count}</h2>
      <button onClick={() => onChange(count + 1)}>+</button>
      <button onClick={() => onChange(count - 1)} style={{ margin: '0 8px' }}>−</button>
      <button onClick={() => onChange(0)}>Reset</button>
      {isNegative && <p style={{ color: 'red' }}>Gone negative!</p>}
      <p>{count === 0 ? 'Zero' : `${Math.abs(count)} away from zero`}</p>
    </div>
  )
}
```

Update `App.tsx`:

```tsx
const [limit, setLimit] = useState(5)

const visibleUsers = users
  .filter(u => u.name.toLowerCase().includes(query.toLowerCase()))
  .slice(0, limit)

// In JSX:
<Counter label="Show N users" count={limit} onChange={setLimit} />
```

Now `App` owns the count, `Counter` just displays and calls back. Increase the counter — more users appear. Decrease it — fewer users. **This is lifting state up.**

---

## Step 10 — Read Parts 2 and 3 (no coding)

You have built enough for today. Now read the conceptual sections so they land with context:

1. **"Custom hooks"** — see how `useChat` wraps everything you did today into one reusable function
2. **"Context vs props vs global state"** — understand when to use Zustand vs props
3. **"Zustand"** — read the store example; you will use this on Day 5
4. **"File and folder structure"** — this is the layout your practice project will use

You do not need to build anything for these. Read and let it sink in.

---

## End-of-Day Checklist

You are done with Day 3 if you can answer these without opening any docs:

- [ ] What is JSX? What does `<div className="x">` compile to?
- [ ] What is the difference between `useState` and a regular variable? Why does React care?
- [ ] What is a derived value? Give an example of something you should NOT put in `useState`.
- [ ] What does the `key` prop do in a list? Why can't you use the array index?
- [ ] What is a controlled input? What two props does it need?
- [ ] Write the pattern for an async `useEffect` fetch from memory (async inner function, error handling, cleanup).
- [ ] What is lifting state up? When do you do it?

Your `basics/exercises/src/` should now have:

```
src/
├── components/
│   ├── Counter.tsx
│   ├── UserCard.tsx
│   └── UserList.tsx
└── App.tsx
```

**Tomorrow (Day 4):** TSX deep-dive + all React hooks with TypeScript types. See [tsx/README.md](../tsx/README.md) and [hooks.md](hooks.md).
