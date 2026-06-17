# TypeScript — Day 2

TypeScript is JavaScript with a type checker. Once it clicks, you will not want to write JS without it — the compiler finds bugs before you run a single line.

## Table of Contents

- [Setup](#setup)
- [What to Study](#what-to-study)
- [Day 2 Schedule](#day-2-schedule)
- [What to Build](#what-to-build)
- [Checklist](#checklist)
- [Common Errors and What They Mean](#common-errors-and-what-they-mean)

---

## Setup

### 1 — Node environment (do once per machine)

**macOS**
```bash
brew install nvm

# Add to ~/.zshrc:
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"

source ~/.zshrc
nvm install --lts
nvm use --lts
nvm alias default node

node --version    # 18.x or 20.x
```

**Windows (PowerShell as Administrator)**
```powershell
winget install CoreyButler.NVMforWindows
# Close and reopen terminal:
nvm install lts
nvm use lts
node --version
```

### 2 — Install packages (do once per clone)

```bash
cd basics/basics
npm install                                 # core deps
npm install zustand react-router-dom clsx   # extras used in exercises
```

### 3 — Start working

```bash
cd basics/basics
npm run dev                  # http://localhost:5173
npx tsc --noEmit --watch     # type-checker in a second terminal — re-runs on save
```

Create scratch files at `basics/basics/src/scratch-ts.ts`. The dev server ignores unimported files, so you can write freely without breaking the app.

**VS Code:** Install **ESLint** and **TypeScript + JavaScript Language Features**. Red squiggles = type errors — fix them as you go.

---

## What to Study

**File:** [typescript.md](typescript.md)

| Section | Why it matters |
|---------|---------------|
| Primitives and annotations | Every variable, param, and return value gets a type |
| `interface` vs `type` | Interfaces for object shapes (props, API responses); `type` for unions |
| Union and intersection types | `string \| null`, `loading \| done \| error` — daily usage |
| Discriminated unions | How you model state machines safely |
| `unknown` vs `any` vs `never` | `any` is a lie; learn to narrow `unknown` in catch blocks |
| `readonly` | Prevents accidental mutation of props |
| Generics | `useState<T>`, `fetch<T>`, `useRef<T>` — everything in React is generic |
| Utility types | `Partial`, `Omit`, `Pick`, `Record` — you use these constantly |
| `import type` | Type-only imports that esbuild strips completely |
| Type guards | `instanceof`, `in`, custom `is` predicates |
| `satisfies` | Validate a value against a type without widening it |

---

## Day 2 Schedule

```
Morning (2 h)
  Read: Type System + Generics sections
  Do:   In basics/basics/src/scratch-ts.ts, annotate 10 small functions from scratch

Afternoon (1.5 h)
  Read: Advanced section (import type, type guards, mapped types, satisfies)
  Do:   Implement the exercises below

Evening (30 min)
  Open typescript.md — Project Patterns section
  Read the Zustand store typing + generic fetch client examples
  These patterns appear in every exercise from Day 3 onward
```

---

## What to Build

Create `basics/basics/src/scratch-ts.ts`:

```ts
// 1. Define an interface ApiUser with: id, name, email, role: 'admin' | 'viewer'

// 2. Write a generic function identity<T>(value: T): T

// 3. Write fetchTyped<T>(url: string): Promise<T>
//    Uses fetch, checks response.ok, returns parsed JSON typed as T

// 4. Define a discriminated union for async state:
//    type AsyncState<T> = { status: 'idle' } | { status: 'loading' } | { status: 'done'; data: T } | { status: 'error'; message: string }

// 5. Write a function that accepts AsyncState<ApiUser[]> and returns a string
//    describing the current state. TypeScript should REQUIRE you to handle all 4 cases.

// 6. Write a type guard: function isApiUser(value: unknown): value is ApiUser
//    Check that value is an object with id (number), name (string), email (string)

// 7. Use Partial<ApiUser> to write updateUser(existing: ApiUser, patch: Partial<ApiUser>): ApiUser

// 8. Use Omit<ApiUser, 'id'> to define a CreateUserPayload type

// 9. Use Record<ApiUser['role'], string[]> to define rolePermissions — a map of role to list of allowed actions

// 10. Write a catch block that narrows unknown to Error:
//     try { ... } catch (err) { const msg = ??? }
```

---

## Checklist

Before moving to Day 3:

- [ ] What is the difference between `interface` and `type`? When do you use each?
- [ ] Why is `any` dangerous? What should you use instead at system boundaries?
- [ ] In a `catch (err)` block, what type is `err`? How do you safely get `.message` from it?
- [ ] What does `Partial<T>` do? Write it out from memory.
- [ ] What does `Omit<T, K>` do? What is a real use case?
- [ ] Why do you write `import type { Foo }` instead of `import { Foo }` for interfaces?
- [ ] What is a discriminated union? Write a 3-variant example from memory.
- [ ] Can a generic function infer its type parameter, or do you always have to provide `<T>` explicitly?

---

## Common Errors and What They Mean

| Error | Cause | Fix |
|-------|-------|-----|
| `Type 'string \| null' is not assignable to type 'string'` | You forgot to handle the null case | Check for null first: `if (x !== null)` or use `??` |
| `Object is possibly 'undefined'` | Optional chaining needed | `obj?.prop` instead of `obj.prop` |
| `Property 'message' does not exist on type 'unknown'` | `err` in catch is `unknown` | `err instanceof Error ? err.message : String(err)` |
| `Type '{}' is not assignable to type 'T'` | TypeScript can't infer the generic | Provide it explicitly: `useState<User \| null>(null)` |
| `'X' is declared but its value is never read` | Unused import or variable | Delete it, or prefix with `_` to signal intentional |
