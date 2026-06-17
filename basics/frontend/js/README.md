# JavaScript — Day 1

Modern JavaScript is the foundation everything else builds on. Before you touch React or TypeScript, these concepts must be second nature.

## Table of Contents

- [Setup](#setup)
- [What to Study](#what-to-study)
- [Day 1 Schedule](#day-1-schedule)
- [What to Build](#what-to-build)
- [Checklist](#checklist)

---

## Setup

No install needed for Day 1 — all examples run directly in the browser console or in a `.ts` scratch file. When you are ready to run TypeScript files, use the Vite project:

```bash
# macOS / Windows — from the basics/basics/ directory
cd basics/basics
npm install          # only needed once
npm run dev          # starts dev server at http://localhost:5173

# Type-check scratch files without running them
npx tsc --noEmit
```

Create scratch files in `basics/basics/src/scratch-js.ts` and import them in `main.tsx` temporarily to test in the browser, or run them with:

```bash
# macOS
node --input-type=module < src/scratch-js.js   # after tsc compiles it

# Simpler: just open DevTools console (F12) and paste JS snippets directly
```

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
