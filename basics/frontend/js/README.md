# JavaScript — Day 1

Modern JavaScript is the foundation everything else builds on. Before you touch React or TypeScript, these concepts must be second nature.

## Table of Contents

- [Setup](#setup)
- [What to Study](#what-to-study)
- [**Day 1 Step-by-Step Walkthrough →**](day1.md)
- [Day 1 Schedule](#day-1-schedule)
- [What to Build](#what-to-build)
- [Checklist](#checklist)

---

## Setup

### 1 — Node environment (do once per machine)

Node version manager is the frontend equivalent of Python's `venv` — it isolates your Node version per project.

**macOS**
```bash
# Install nvm via Homebrew
brew install nvm

# Add to ~/.zshrc (zsh) or ~/.bash_profile (bash) — paste both lines:
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"

# Reload shell, then install Node LTS
source ~/.zshrc
nvm install --lts
nvm use --lts
nvm alias default node   # make it the default for new terminals

node --version            # 18.x or 20.x
npm --version             # 9.x or 10.x
```

**Windows (PowerShell as Administrator)**
```powershell
# Install nvm-windows via winget
winget install CoreyButler.NVMforWindows

# Close and reopen terminal, then:
nvm install lts
nvm use lts

node --version            # 18.x or 20.x
npm --version
```

### 2 — Scaffold the project (do once per machine)

```bash
# macOS / Windows — run from the basics/ directory
cd basics

# Node 20+
npm create vite@latest basics -- --template react-ts

# Node 18 fallback
npm create vite@5 basics -- --template react-ts
```

### 3 — Install packages (do once per clone)

```bash
cd basics/basics
npm install                       # installs all deps into node_modules/
npm install zustand react-router-dom clsx   # extra packages used in exercises
```

`node_modules/` is the package environment — equivalent to a Python `venv/`. It is gitignored; everyone runs `npm install` after cloning.

### 4 — Start working

```bash
cd basics/basics
npm run dev          # dev server at http://localhost:5173 — hot-reloads on save
npx tsc --noEmit     # type-check (run this in a second terminal while you code)
```

For Day 1 you can also paste JS snippets directly into the browser DevTools console (F12) — no file needed.

---

## What to Study

**File:** [javascript.md](javascript.md)

Read in order. Every section is something you will use daily in React:

| Section | Why it matters |
|---------|---------------|
| `let`/`const` | You never use `var` again |
| Arrow functions | Every React callback is an arrow function |
| Destructuring | Props are always destructured |
| Spread / rest | State updates always spread: `{ ...prev, key: value }` |
| Optional chaining `?.` and nullish coalescing `??` | Guards against null in component data |
| Array methods | `map`, `filter`, `reduce` are how you build lists in JSX |
| ES modules | Every React file is a module |
| Async / await | All API calls and effects use this |
| `AbortController` | How you cancel fetch calls in React effects |
| `ReadableStream` + SSE | How the RAG chat streams tokens |
| Async generators | How `streamSSE` works under the hood |

---

## Day 1 Schedule

```
Morning (2 h)
  Read: Basics section (let/const → array methods → modules)
  Do:   Open browser DevTools console, type every example by hand

Afternoon (1.5 h)
  Read: Advanced section (async/await → AbortController → async generators)
  Do:   In basics/basics/src/, create scratch.ts and implement the exercises below

Evening (30 min)
  Review: Close the doc and write from memory — arrow function, destructuring,
           an async fetch with error handling, and a filter+map chain
```

---

## What to Build

Create `basics/basics/src/scratch-js.ts` and implement each of these without looking at the doc:

```ts
// 1. Destructure this object and rename 'id' to 'userId'
const user = { id: 42, name: 'Ada', role: 'admin' }

// 2. Write an async function fetchUser(id: number) that:
//    - fetches https://jsonplaceholder.typicode.com/users/{id}
//    - returns the parsed JSON
//    - throws a descriptive error if response.ok is false

// 3. Write fetchWithCancel(url: string) that:
//    - creates an AbortController
//    - passes its signal to fetch
//    - returns { data: Promise<any>, cancel: () => void }

// 4. Write a function that takes string[] and returns only
//    items longer than 5 chars, uppercased, sorted A→Z

// 5. Write an async generator function* pollEverySecond(url: string)
//    that fetches the URL every second and yields the JSON response.
//    Stop after 5 iterations.
```

---

## Checklist

Before moving to Day 2, you should be able to answer these without notes:

- [ ] What is the difference between `null` and `undefined`? What does `??` do with each?
- [ ] What does `?.` return when the left side is `null`?
- [ ] Why can you not use `await` at the top level of a regular `.ts` file?
- [ ] What happens if you do not call `controller.abort()` when a component unmounts?
- [ ] What is the difference between `Array.prototype.map` and `Array.prototype.forEach`? Which one do you use in JSX and why?
- [ ] Write a one-liner that takes `items: { id: string; value: number }[]` and returns the sum of all `value` fields.
