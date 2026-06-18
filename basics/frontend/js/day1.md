# Day 1 — JavaScript Step-by-Step Walkthrough

You will work through modern JS concepts using the browser DevTools console and scratch files. No framework, no build step — just the language. By the end you will have the JS muscle memory that makes React feel natural.

**Total time:** ~3 hours  
**Reference doc:** [javascript.md](javascript.md)

## Table of Contents

- [Before You Start](#before-you-start)
- [Step 1 — let, const, and types](#step-1--let-const-and-types)
- [Step 2 — Arrow functions](#step-2--arrow-functions)
- [Step 3 — Destructuring](#step-3--destructuring)
- [Step 4 — Spread and rest](#step-4--spread-and-rest)
- [Step 5 — Optional chaining and nullish coalescing](#step-5--optional-chaining-and-nullish-coalescing)
- [Step 6 — Array methods](#step-6--array-methods)
- [Step 7 — ES modules](#step-7--es-modules)
- [Step 8 — Async / await and fetch](#step-8--async--await-and-fetch)
- [Step 9 — AbortController](#step-9--abortcontroller)
- [Step 10 — Async generators](#step-10--async-generators)
- [End-of-Day Checklist](#end-of-day-checklist)

---

## Before You Start

Open your browser (Chrome or Firefox). Press **F12** to open DevTools. Click the **Console** tab. Every code block in Steps 1–7 can be pasted directly here — no setup needed.

For Steps 8–10, you will use a scratch file in the Vite project:

```bash
cd basics/basics
npm run dev    # keep running — you do not need this until Step 8
```

Create `basics/basics/src/scratch-day1.ts` — paste code there, then import it once in `main.tsx` to run it:

```tsx
// basics/basics/src/main.tsx — add this line temporarily
import './scratch-day1'
```

---

## Step 1 — let, const, and types

**Read** the "Basics" opening of `javascript.md` (~3 min).

Paste into DevTools console, one block at a time:

```js
// const — cannot be reassigned
const name = 'Ada'
name = 'Grace'   // TypeError — try it, see the error

// let — can be reassigned
let count = 0
count = 1        // fine

// typeof — JS's runtime type system
typeof 'hello'   // 'string'
typeof 42        // 'number'
typeof true      // 'boolean'
typeof null      // 'object'  ← famous JS quirk
typeof undefined // 'undefined'
typeof {}        // 'object'
typeof []        // 'object'  ← arrays are objects

// Equality — always use ===, never ==
0 == false    // true  ← == coerces types, dangerous
0 === false   // false ← === checks type AND value, safe
```

**Rule:** Use `const` by default. Use `let` only when you need to reassign. Never use `var`.

---

## Step 2 — Arrow functions

**Read** the "Arrow functions" section (~5 min).

```js
// Traditional function
function add(a, b) { return a + b }

// Arrow function — same thing
const add = (a, b) => a + b    // implicit return when body is a single expression

// Multi-line arrow — need explicit return and braces
const greet = (name) => {
  const message = `Hello, ${name}`
  return message
}

// Single param — parentheses optional
const double = n => n * 2

// No params — empty parens required
const getRandom = () => Math.random()

// Returning an object literal — wrap in parens or JS thinks { is a block
const makeUser = (name) => ({ name, createdAt: Date.now() })

// Test them
add(2, 3)           // 5
double(7)           // 14
makeUser('Ada')     // { name: 'Ada', createdAt: ... }
```

**Why arrow functions matter in React:** Every event handler and every callback in React is an arrow function. `onClick={() => setCount(c => c + 1)}` — that is an arrow returning an arrow.

---

## Step 3 — Destructuring

**Read** the "Destructuring" section (~5 min).

```js
// Object destructuring
const user = { id: 1, name: 'Ada', role: 'admin', address: { city: 'London' } }

const { name, role } = user          // pick what you need
const { name: userName } = user      // rename: userName = 'Ada'
const { address: { city } } = user   // nested destructure
const { age = 25 } = user            // default if key is missing

console.log(name, role, userName, city, age)  // Ada admin Ada London 25

// Array destructuring
const nums = [10, 20, 30, 40]
const [first, second] = nums         // first=10, second=20
const [, , third] = nums             // skip with empty comma: third=30
const [head, ...rest] = nums         // head=10, rest=[20,30,40]

// Destructuring in function parameters — THIS is how React props work
function greet({ name, role = 'viewer' }) {
  return `${name} is a ${role}`
}
greet({ name: 'Grace' })            // 'Grace is a viewer'
greet({ name: 'Ada', role: 'admin' }) // 'Ada is an admin'
```

**Why this matters in React:** Every React component destructures its props: `function Button({ label, onClick, disabled = false }) { ... }`.

---

## Step 4 — Spread and rest

**Read** the "Spread / rest" section (~5 min).

```js
// Spread arrays — creates a new array, never mutates
const a = [1, 2, 3]
const b = [...a, 4, 5]       // [1,2,3,4,5]
const c = [0, ...a]          // [0,1,2,3]
const copy = [...a]          // [1,2,3] — shallow copy

// Spread objects — creates a new object
const user = { name: 'Ada', role: 'admin' }
const updated = { ...user, role: 'viewer' }   // override role
const withAge = { age: 30, ...user }          // prepend — user props win on conflict

// Rest parameters — collects remaining args into an array
function sum(first, ...rest) {
  return rest.reduce((acc, n) => acc + n, first)
}
sum(1, 2, 3, 4)    // 10

// Rest in destructuring — collect remaining keys
const { name, ...rest2 } = user
console.log(name)    // 'Ada'
console.log(rest2)   // { role: 'admin' }
```

**Why spread matters in React:** State updates must never mutate — always spread:
```js
// Wrong — mutates the array, React does not re-render
todos.push(newTodo)

// Right — new array reference triggers re-render
setTodos(prev => [...prev, newTodo])

// Wrong — mutates the object
user.name = 'Grace'

// Right
setUser(prev => ({ ...prev, name: 'Grace' }))
```

---

## Step 5 — Optional chaining and nullish coalescing

**Read** that section (~3 min).

```js
const user = { profile: { address: { city: 'London' } } }
const empty = {}

// Optional chaining — ?. returns undefined instead of throwing
user.profile.address.city        // 'London'
empty.profile.address.city       // TypeError — crashes
empty?.profile?.address?.city    // undefined — safe

// Works on method calls and array access too
const arr = null
arr?.map(x => x)                 // undefined, not TypeError
arr?.[0]                         // undefined

// Nullish coalescing — ?? returns right side only for null/undefined
const city = empty?.profile?.address?.city ?? 'Unknown'   // 'Unknown'
const zero = 0 ?? 'default'     // 0  ← ?? does NOT treat 0 as falsy
const zero2 = 0 || 'default'    // 'default' ← || treats 0 as falsy

// Combined — very common in React for API data
const displayName = user?.profile?.displayName ?? user?.name ?? 'Anonymous'
```

**Rule:** Use `?.` when accessing nested data that might be null/undefined (API responses, optional props). Use `??` for defaults — prefer it over `||` when 0 or empty string are valid values.

---

## Step 6 — Array methods

**Read** the "Array methods" section (~10 min). These are the most important methods in React.

```js
const users = [
  { id: 1, name: 'Ada',   role: 'admin',  score: 90 },
  { id: 2, name: 'Grace', role: 'viewer', score: 75 },
  { id: 3, name: 'Alan',  role: 'viewer', score: 85 },
  { id: 4, name: 'Linus', role: 'admin',  score: 60 },
]

// map — transform every element, returns new array (same length)
const names = users.map(u => u.name)
// ['Ada', 'Grace', 'Alan', 'Linus']

// filter — keep elements matching condition, returns new array (shorter)
const admins = users.filter(u => u.role === 'admin')
// [{ id:1, ... }, { id:4, ... }]

// find — first match or undefined
const ada = users.find(u => u.name === 'Ada')
// { id: 1, name: 'Ada', ... }

// some — true if any element matches
const hasAdmin = users.some(u => u.role === 'admin')   // true

// every — true if all elements match
const allAdmins = users.every(u => u.role === 'admin') // false

// reduce — accumulate to a single value
const totalScore = users.reduce((sum, u) => sum + u.score, 0)   // 310
const avgScore = totalScore / users.length                       // 77.5

// Chaining — very common in React
const topViewerNames = users
  .filter(u => u.role === 'viewer')
  .sort((a, b) => b.score - a.score)
  .map(u => u.name)
// ['Alan', 'Grace']

// includes — check if value is in array
[1, 2, 3].includes(2)   // true
```

**Critical:** `map`, `filter`, `find`, `some`, `every`, `reduce` all return a **new array/value** — they do not modify the original. This is essential for React state updates.

---

## Step 7 — ES modules

**Read** "ES modules" (~5 min).

You cannot paste module syntax in DevTools. Create two files:

`basics/basics/src/utils.ts`:
```ts
// Named export — can have many per file
export function formatName(first: string, last: string): string {
  return `${first} ${last}`
}

export const MAX_USERS = 100

// Default export — one per file, usually the main thing
export default function greet(name: string): string {
  return `Hello, ${name}!`
}
```

`basics/basics/src/scratch-day1.ts`:
```ts
// Named imports — use exact names
import { formatName, MAX_USERS } from './utils'

// Default import — any name you like
import greet from './utils'

// Both at once
import greet2, { formatName as fmt } from './utils'

console.log(formatName('Ada', 'Lovelace'))  // 'Ada Lovelace'
console.log(MAX_USERS)                       // 100
console.log(greet('Grace'))                  // 'Hello, Grace!'
console.log(fmt('Alan', 'Turing'))           // 'Alan Turing'
```

Import the scratch file in `main.tsx` temporarily:
```tsx
import './scratch-day1'
```

Open the browser console — you should see the four log lines.

**Rule:** Prefer named exports. Default exports make refactoring harder (the name is not enforced by the module system). React components are the exception — they are conventionally default-exported from their file.

---

## Step 8 — Async / await and fetch

**Read** "Async / await" and the `fetch` example (~10 min).

Update `basics/basics/src/scratch-day1.ts`:

```ts
// A function that returns a Promise — async keyword makes it return one automatically
async function fetchUser(id: number) {
  const res = await fetch(`https://jsonplaceholder.typicode.com/users/${id}`)

  // Always check ok before parsing — a 404 does not throw, it just has ok=false
  if (!res.ok) {
    throw new Error(`HTTP error: ${res.status}`)
  }

  const data = await res.json()
  return data
}

// Call it — async functions return a Promise, so you need .then() or await
fetchUser(1).then(user => console.log('Fetched:', user.name))

// Or in an async IIFE (immediately invoked function expression)
;(async () => {
  try {
    const user = await fetchUser(1)
    console.log('Name:', user.name)
    console.log('Email:', user.email)
  } catch (err) {
    // err is unknown in TypeScript — narrow before use
    const message = err instanceof Error ? err.message : String(err)
    console.error('Failed:', message)
  }
})()

// Promise.all — run multiple requests in parallel, wait for all
;(async () => {
  const [user1, user2, user3] = await Promise.all([
    fetchUser(1),
    fetchUser(2),
    fetchUser(3),
  ])
  console.log(user1.name, user2.name, user3.name)
})()
```

Save. Check the browser console — you should see names printed from the API.

**Key rules:**
- `await` can only be used inside an `async` function
- Always check `res.ok` — a 404 response does not throw
- `Promise.all` is faster than three sequential `await` calls

---

## Step 9 — AbortController

**Read** "AbortController" section (~5 min).

This is how you cancel a fetch — essential for React's `useEffect` cleanup.

Update `scratch-day1.ts`:

```ts
// Create a controller — one per request
const controller = new AbortController()

async function fetchWithCancel(url: string) {
  try {
    const res = await fetch(url, { signal: controller.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      console.log('Request was cancelled — this is expected, not an error')
      return null
    }
    throw err   // re-throw unexpected errors
  }
}

// Start the fetch
const promise = fetchWithCancel('https://jsonplaceholder.typicode.com/users/1')

// Cancel it immediately (simulates component unmounting before fetch completes)
controller.abort()

promise.then(data => console.log('Result:', data))
// Logs: 'Request was cancelled — this is expected, not an error'
// data will be null
```

Now try without cancelling — remove the `controller.abort()` line:

```ts
const controller2 = new AbortController()
fetchWithCancel('https://jsonplaceholder.typicode.com/users/2')
  .then(data => console.log('Got:', data?.name))
```

**In React**, this pattern appears in `useEffect`:
```ts
useEffect(() => {
  const ctrl = new AbortController()
  fetchData(ctrl.signal).then(setData)
  return () => ctrl.abort()   // called on unmount — cancels in-flight request
}, [])
```

---

## Step 10 — Async generators

**Read** "Async generators" section (~5 min). These power the SSE (server-sent events) streaming in the RAG app.

Update `scratch-day1.ts`:

```ts
// An async generator yields values over time
async function* countUp(from: number, to: number, delayMs: number) {
  for (let i = from; i <= to; i++) {
    await new Promise(resolve => setTimeout(resolve, delayMs))
    yield i    // pause here, give the value to the caller, resume on next iteration
  }
}

// Consume with for await — waits for each yielded value
;(async () => {
  for await (const n of countUp(1, 5, 500)) {
    console.log('Got:', n)   // prints 1, 2, 3, 4, 5 — one every 500ms
  }
  console.log('Done')
})()

// Real-world analogy: SSE stream
// Each SSE event is one yield. The for-await loop processes them one by one.
// When the stream closes, the loop ends naturally.
// When you call controller.abort(), the next yield throws AbortError, ending the loop.
```

---

## End-of-Day Checklist

Close `javascript.md`. Answer these from memory:

- [ ] What is the difference between `==` and `===`? Which do you always use?
- [ ] Write an arrow function that takes two numbers and returns their sum.
- [ ] Write a destructuring line that pulls `name` and `email` out of `{ id, name, email, role }`.
- [ ] What does `[...arr, newItem]` do? Why is this used in React state updates instead of `arr.push(newItem)`?
- [ ] What does `user?.profile?.city ?? 'Unknown'` return when `user` is `null`?
- [ ] Write a `filter` + `map` chain: from an array of users, keep only `role === 'admin'`, then return just their names.
- [ ] What does `res.ok` check? Why can't you just use `try/catch` alone with `fetch`?
- [ ] What is `AbortController` used for? Write the two lines that create one and cancel a request.
- [ ] What keyword do you put before `function` to make it return a Promise?

**Tomorrow (Day 2):** Add types to everything you wrote today. See [ts/day2.md](../ts/day2.md).
