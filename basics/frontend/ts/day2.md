# Day 2 — TypeScript Step-by-Step Walkthrough

You will annotate everything you wrote yesterday and progressively layer on TypeScript's type system. By the end, every variable, function, and data shape in your scratch files is explicitly typed — and you understand why.

**Total time:** ~3 hours  
**Reference doc:** [typescript.md](typescript.md)  
**Prerequisites:** Day 1 done — you understand arrow functions, destructuring, async/await

## Table of Contents

- [Before You Start](#before-you-start)
- [Step 1 — Primitives and annotations](#step-1--primitives-and-annotations)
- [Step 2 — Interfaces](#step-2--interfaces)
- [Step 3 — Union types and literals](#step-3--union-types-and-literals)
- [Step 4 — Functions with types](#step-4--functions-with-types)
- [Step 5 — Generics](#step-5--generics)
- [Step 6 — Utility types](#step-6--utility-types)
- [Step 7 — unknown, any, and narrowing](#step-7--unknown-any-and-narrowing)
- [Step 8 — Type guards](#step-8--type-guards)
- [Step 9 — Typing async fetch](#step-9--typing-async-fetch)
- [Step 10 — import type and discriminated unions](#step-10--import-type-and-discriminated-unions)
- [End-of-Day Checklist](#end-of-day-checklist)

---

## Before You Start

```bash
cd basics/frontend/day1_exercises
npm run dev                  # terminal 1 — keep running
npx tsc --noEmit --watch     # terminal 2 — shows type errors as you save
```

Create `basics/frontend/day1_exercises/src/scratch-day2.ts`. This is where all today's code goes.

Make sure `main.tsx` imports it:
```tsx
import './scratch-day2'
```

Open `http://localhost:5173` and keep the browser console visible.

---

## Step 1 — Primitives and annotations

**Read** "Primitives and annotations" in `typescript.md` (~5 min).

```ts
// Explicit annotations — TypeScript can infer most of these, but writing them builds muscle memory
const name: string = 'Ada'
const age: number = 35
const active: boolean = true
const nothing: null = null
const missing: undefined = undefined

// Arrays
const names: string[] = ['Ada', 'Grace', 'Alan']
const scores: number[] = [90, 75, 85]

// Tuple — fixed length, each position has its own type
const pair: [string, number] = ['Ada', 90]
const [person, score] = pair    // person: string, score: number

// Let TypeScript infer when the initial value makes it obvious — less noise
const inferredName = 'Ada'   // TypeScript knows this is string
const inferredAge = 35       // TypeScript knows this is number

// But annotate when the initial value is ambiguous
const maybeUser: string | null = null     // needs annotation — null alone tells TypeScript nothing
const items: string[] = []                // needs annotation — [] alone infers never[]
```

**Rule:** Annotate when TypeScript cannot infer the right type from the initial value. Let it infer when it can.

---

## Step 2 — Interfaces

**Read** "interface vs type" section (~5 min).

```ts
// Interface — defines the shape of an object
interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'    // literal union — only these two strings are valid
  createdAt?: Date            // optional field — may or may not be present
  readonly token: string      // readonly — cannot be reassigned after creation
}

// Create a value matching the interface
const ada: User = {
  id: 1,
  name: 'Ada Lovelace',
  email: 'ada@example.com',
  role: 'admin',
  token: 'abc123',
}

// TypeScript errors you get for free:
// ada.role = 'superuser'      // error — not in the union
// ada.token = 'xyz'           // error — readonly
// const u: User = { id: 1 }  // error — missing required fields

// Extending an interface
interface AdminUser extends User {
  permissions: string[]
  lastLogin: Date
}

// Inline type alias — for unions and primitives, not object shapes
type ID = string | number
type Status = 'idle' | 'loading' | 'done' | 'error'

const id: ID = 42             // number is valid
const id2: ID = 'user_123'   // string is also valid

// Interface vs type:
// Use interface for object shapes (User, Props, ApiResponse) — better error messages
// Use type for unions, aliases, and mapped types
```

---

## Step 3 — Union types and literals

**Read** "Union and intersection types" and "Discriminated unions" (~10 min).

```ts
// Union — value can be one of several types
type StringOrNumber = string | number
const x: StringOrNumber = 42
const y: StringOrNumber = 'hello'

// Literal types — only specific values allowed
type Direction = 'north' | 'south' | 'east' | 'west'
const move = (d: Direction) => console.log(`Moving ${d}`)
move('north')   // ok
// move('up')   // error — not in the union

// Discriminated union — each variant has a unique 'kind' field
// TypeScript uses this to narrow which variant you have
type AsyncResult<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done';  data: T }
  | { status: 'error'; message: string }

function describe(result: AsyncResult<User>): string {
  switch (result.status) {
    case 'idle':    return 'Not started'
    case 'loading': return 'Loading...'
    case 'done':    return `Got user: ${result.data.name}`   // data is available here
    case 'error':   return `Error: ${result.message}`        // message is available here
  }
  // TypeScript knows all cases are covered — no default needed
}

// Try removing one case — TypeScript errors because the function might not return
```

**This is the pattern for every loading state in React:** `'idle' | 'loading' | 'done' | 'error'`.

---

## Step 4 — Functions with types

**Read** "Annotating functions" (~5 min).

```ts
// Annotate parameters and return type
function add(a: number, b: number): number {
  return a + b
}

// Arrow function
const multiply = (a: number, b: number): number => a * b

// Optional parameter — must come after required params
function greet(name: string, title?: string): string {
  return title ? `${title} ${name}` : name
}

// Default parameter — implies the type, no annotation needed
function greetWithDefault(name: string, title = 'Dr'): string {
  return `${title} ${name}`
}

// Rest parameter
function sum(...nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0)
}

// Function that takes a callback — Callable type
function runTwice(fn: (n: number) => void, value: number): void {
  fn(value)
  fn(value)
}
runTwice(n => console.log(n), 42)   // prints 42 twice

// void vs undefined — void means the return value is not used
// A function returning void can technically return undefined
```

---

## Step 5 — Generics

**Read** "Generics" section carefully (~15 min). This is the most important concept for React (`useState<T>`, `useRef<T>`, `fetch<T>`).

```ts
// Generic function — T is a placeholder for any type
function identity<T>(value: T): T {
  return value
}

identity(42)          // T inferred as number — returns number
identity('hello')     // T inferred as string — returns string
identity<boolean>(true)  // T provided explicitly

// Generic with constraint — T must have a .length property
function longest<T extends { length: number }>(a: T, b: T): T {
  return a.length >= b.length ? a : b
}
longest('hello', 'hi')      // 'hello'
longest([1, 2, 3], [1, 2])  // [1,2,3]
// longest(1, 2)             // error — number has no .length

// Generic interface
interface ApiResponse<T> {
  data: T
  status: number
  message: string
}

// Concrete usage
const userResponse: ApiResponse<User> = {
  data: ada,
  status: 200,
  message: 'ok',
}

const listResponse: ApiResponse<User[]> = {
  data: [ada],
  status: 200,
  message: 'ok',
}

// Generic fetch — type the response you expect
async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

// Usage — T is User[]
const users = await fetchJson<User[]>('https://jsonplaceholder.typicode.com/users')
users[0].name    // TypeScript knows this is a string
```

---

## Step 6 — Utility types

**Read** "Utility types" section (~10 min).

```ts
interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'viewer'
  password: string
}

// Partial<T> — all fields become optional
type UserUpdate = Partial<User>
const patch: UserUpdate = { name: 'Grace' }   // only name, rest omitted

// Required<T> — all fields become required (opposite of Partial)
type StrictUser = Required<User>

// Omit<T, K> — remove specific keys
type PublicUser = Omit<User, 'password'>
const safe: PublicUser = { id: 1, name: 'Ada', email: 'a@b.com', role: 'admin' }
// safe.password   // error — property does not exist

// Pick<T, K> — keep only specific keys
type UserPreview = Pick<User, 'id' | 'name'>
const preview: UserPreview = { id: 1, name: 'Ada' }

// Record<K, V> — object with specific key and value types
type RoleMap = Record<'admin' | 'viewer', string[]>
const permissions: RoleMap = {
  admin:  ['read', 'write', 'delete'],
  viewer: ['read'],
}

// Readonly<T> — all fields become readonly
type ImmutableUser = Readonly<User>

// ReturnType<T> — extract the return type of a function
function getUser() { return { id: 1, name: 'Ada' } }
type UserShape = ReturnType<typeof getUser>   // { id: number; name: string }

// Awaited<T> — unwrap a Promise type
type UserData = Awaited<ReturnType<typeof fetchJson<User>>>  // User
```

---

## Step 7 — unknown, any, and narrowing

**Read** "unknown vs any vs never" (~5 min).

```ts
// any — disables type checking entirely. Avoid it.
let bad: any = 42
bad = 'now a string'      // no error
bad.doesNotExist()        // no error — will crash at runtime

// unknown — safe version of any. Must narrow before use.
let value: unknown = 42

// You must check the type before doing anything with it
if (typeof value === 'number') {
  value + 1    // safe — TypeScript knows it's number here
}

if (typeof value === 'string') {
  value.toUpperCase()   // safe — TypeScript knows it's string here
}

// Narrowing in a catch block — err is unknown
async function riskyFetch(url: string) {
  try {
    const res = await fetch(url)
    return await res.json()
  } catch (err) {
    // Wrong — err might not have .message
    // console.error(err.message)

    // Right — narrow first
    const message = err instanceof Error ? err.message : String(err)
    console.error('Fetch failed:', message)
    return null
  }
}

// never — a value that can never exist
function assertNever(x: never): never {
  throw new Error(`Unexpected value: ${x}`)
}

// In a switch, TypeScript knows all cases are covered because the default arm receives never
type Color = 'red' | 'green' | 'blue'
function toHex(c: Color): string {
  switch (c) {
    case 'red':   return '#ff0000'
    case 'green': return '#00ff00'
    case 'blue':  return '#0000ff'
    default:      return assertNever(c)   // error if you add 'purple' to Color without handling it
  }
}
```

---

## Step 8 — Type guards

**Read** "Type guards" section (~5 min).

```ts
interface Dog { kind: 'dog'; breed: string }
interface Cat { kind: 'cat'; indoor: boolean }
type Pet = Dog | Cat

// Type guard with 'in'
function describeByIn(pet: Pet): string {
  if ('breed' in pet) return `Dog: ${pet.breed}`
  return `Cat, indoor: ${pet.indoor}`
}

// Type guard with discriminant (preferred — more explicit)
function describe(pet: Pet): string {
  if (pet.kind === 'dog') return `Dog: ${pet.breed}`
  return `Cat, indoor: ${pet.indoor}`
}

// Custom type guard — the 'is' return type tells TypeScript what narrowing to do
function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'name' in value &&
    'email' in value
  )
}

// Usage — useful for validating API responses
const raw: unknown = await fetchJson<unknown>('https://jsonplaceholder.typicode.com/users/1')
if (isUser(raw)) {
  console.log(raw.name)    // safe — TypeScript knows raw is User inside this block
}
```

---

## Step 9 — Typing async fetch

**Read** "Project Patterns → generic fetch client" (~5 min).

Combine everything from today into a properly typed fetch wrapper:

```ts
// Generic typed fetch — what you will use in React components
async function apiFetch<T>(
  url: string,
  options?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(url, { ...options, signal })
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

// Typed response interfaces
interface Post {
  id: number
  title: string
  body: string
  userId: number
}

// Usage — T is inferred from the generic argument
;(async () => {
  try {
    const post = await apiFetch<Post>('https://jsonplaceholder.typicode.com/posts/1')
    console.log(post.title)   // TypeScript knows this is a string

    const posts = await apiFetch<Post[]>('https://jsonplaceholder.typicode.com/posts?_limit=3')
    console.log(posts.map(p => p.title))   // TypeScript knows posts is Post[]
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    console.error(msg)
  }
})()
```

---

## Step 10 — import type and discriminated unions

**Read** "import type" and "satisfies" (~5 min).

```ts
// import type — erased at compile time, never in the JS bundle
// Use for interfaces and types that only exist at compile time
import type { User } from './types'   // hypothetical — zero runtime cost

// satisfies — validates a value against a type without widening it
const config = {
  host: 'localhost',
  port: 8080,
  debug: true,
} satisfies { host: string; port: number; debug: boolean }

// Without satisfies: const config: { host: string; port: number; debug: boolean }
// With satisfies:    TypeScript still knows config.port is number (not just number)

// Putting it all together — a discriminated union for an entire async state machine
type FetchState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; data: T }
  | { status: 'error'; message: string }

// This is what you will use in React useState:
// const [state, setState] = useState<FetchState<User[]>>({ status: 'idle' })
//
// setState({ status: 'loading' })
// setState({ status: 'done', data: users })
// setState({ status: 'error', message: 'Network error' })
```

---

## End-of-Day Checklist

Close `typescript.md`. Answer from memory:

- [ ] When do you use `interface`? When do you use `type`?
- [ ] What is `Partial<User>`? Write it out without the utility — what does the raw type look like?
- [ ] What is `Omit<User, 'password'>` used for?
- [ ] Write a generic function `wrap<T>(value: T): { value: T }` from memory.
- [ ] Why is `any` dangerous? What should you use at a system boundary instead?
- [ ] In a `catch (err)` block, what type is `err`? How do you safely read `.message`?
- [ ] What does a discriminated union give you that a plain union does not?
- [ ] What is `import type` for? Why not just use `import`?

**Tomorrow (Day 3):** Build React components using all of today's types. See [react/day3.md](../react/day3.md).
