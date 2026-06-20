# React — Days 3 and 4

React is a UI library. Everything is a function that returns JSX, and React decides when to call it. Days 3–4 cover the core + every hook you will use day-to-day.

## Table of Contents

- [Setup](#setup)
- [What to Study](#what-to-study)
- [**Day 3 Step-by-Step Walkthrough →**](day3.md)
- [**Day 4 Step-by-Step Walkthrough →**](day4.md)
- [Day 3 Schedule — Core React](#day-3-schedule--core-react)
- [Day 4 Schedule — Hooks Deep Dive](#day-4-schedule--hooks-deep-dive)
- [What to Build Each Day](#what-to-build-each-day)
- [Mental Models Worth Internalising](#mental-models-worth-internalising)
- [Checklist](#checklist)

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
npm --version
```

**Windows (PowerShell as Administrator)**
```powershell
winget install CoreyButler.NVMforWindows
# Close and reopen terminal:
nvm install lts
nvm use lts
node --version
```

> **Windows tip:** If `nvm use` fails, right-click PowerShell → "Run as Administrator" for the first call only. Node will then work in any normal terminal.

### 2 — Install packages (do once per clone)

```bash
cd basics/exercises
npm install                                 # React, TypeScript, Vite, ESLint
npm install zustand react-router-dom clsx   # state, routing, classnames
```

### 3 — Start working

```bash
cd basics/exercises
npm run dev                  # http://localhost:5173 — hot-reloads on save
npx tsc --noEmit --watch     # type-checker in a second terminal
npx eslint src/              # lint on demand
```

**Recommended VS Code extensions:**
- ESLint
- Prettier (`"editor.formatOnSave": true`)
- ES7+ React/Redux/React-Native snippets (`rfce` → full component scaffold)

---

## What to Study

Two files, read in order:

| File | When | What |
|------|------|------|
| [react.md](react.md) | Day 3 | Components, props, state, effects, routing, Zustand, production patterns |
| [hooks.md](hooks.md) | Day 4 | Every standard hook — signature, purpose, typed example |

---

## Day 3 Schedule — Core React

```
Morning (2 h)
  Read: react.md — Part 1 (Core React)
  Topics: JSX, components, useState, useEffect, useRef, conditional rendering,
          lists + keys, event handling, lifting state up

  Do: In basics/exercises/src/, create:
      - components/Counter.tsx     (useState + button)
      - components/UserCard.tsx    (typed props, conditional rendering)
      - components/NumberList.tsx  (map with keys, filter)

Afternoon (2 h)
  Read: react.md — Part 2 (Patterns) + Part 3 (Production)
  Topics: Custom hooks, Zustand, React Router, async state, AbortController,
          useMemo/useCallback, file structure, PrivateRoute, layout components

  Do: In basics/exercises/src/:
      - hooks/useFetch.ts          (generic fetch hook — see exercise below)
      - pages/HomePage.tsx         (renders UserCard list from the hook)
      - App.tsx                    (add BrowserRouter + Routes)

Evening (30 min)
  Re-read the Zustand section — draw the data flow on paper:
  Store → hook → component → action → store
```

---

## Day 4 Schedule — Hooks Deep Dive

```
Morning (2 h)
  Read: hooks.md — Basic Hooks through Performance Hooks
  Topics: useState, useEffect (Pexels example), useContext, useRef,
          useImperativeHandle, useMemo, useCallback

  Do: Implement the Pexels example from the doc from scratch (no copy-paste).
      Build: PexelsGallery.tsx — typed API fetch, loading/error states, grid layout.

Afternoon (1.5 h)
  Read: hooks.md — Transition/Deferral through Custom Hooks
  Topics: useTransition, useDeferredValue, useId, useSyncExternalStore,
          useLayoutEffect, useInsertionEffect, useChat breakdown, useLocalStorage

  Do: Implement useLocalStorage<T> from memory.
      Wire it into a ThemeToggle component that persists 'light' | 'dark'.

Evening (30 min)
  Read: tsx.md (the TSX file — see tsx/ folder)
  Focus on: event types, generic components, forwardRef, how Vite processes TSX
```

---

## What to Build Each Day

### Day 3 — useFetch hook

```ts
// src/hooks/useFetch.ts
// Generic data-fetching hook with loading, error, and AbortController cleanup.

// Return type:
// { data: T | null; loading: boolean; error: string | null }

// Requirements:
// - Accept a URL string.
// - Re-run when the URL changes (put url in the dependency array).
// - Cancel the in-flight request on cleanup (AbortController in useEffect return).
// - Narrow the error in the catch block — err is unknown.
// - Do NOT set state after abort (guard with AbortError check).
```

### Day 4 — PexelsGallery (type it yourself)

```ts
// src/components/PexelsGallery.tsx
// Build the gallery without looking at hooks.md.
// You need: PexelsPhoto interface, PexelsResponse interface,
//           useState for photos/loading/error, useEffect with async inner function,
//           early returns for loading and error, grid of <img> tags.
// After you finish, open hooks.md and compare.
```

---

## Mental Models Worth Internalising

**Re-render contract:** React calls your component function whenever state or props change. Everything inside the function runs fresh. `useRef` and `useState` are the two things that survive re-renders.

**`useState` vs `useRef`:** If the value changing should update the UI → `useState`. If you need to store it but the UI does not care → `useRef`.

**Derived values:** If you can compute a value from state or props, do it inline. Never put a derived value in `useState` — it will go out of sync.

**Effect dependency array:** "Run this effect when these values change." If you lie about the deps (omit something that the effect reads), you get stale data bugs. The linter catches this — listen to it.

**Cleanup:** Every subscription, interval, or fetch inside `useEffect` must be cleaned up in the return function. If you do not, you get state updates on unmounted components and memory leaks.

**Zustand `getState()` vs hook:** Inside an async function, use `store.getState()` to read live values. The hook's snapshot was captured at render time and may be stale by the time the async code runs.

---

## Checklist

Before Day 5 (practice):

- [ ] Can you write a typed functional component with optional props and defaults from memory?
- [ ] Can you write a `useEffect` that fetches data, handles loading/error, and cancels on cleanup?
- [ ] What is the difference between `[]`, `[value]`, and no dependency array in `useEffect`?
- [ ] Can you write a `useRef` for a DOM node and a `useRef` for a mutable value — and explain why each one does not use `useState`?
- [ ] When does `useMemo` actually help? Give a real condition (not "when it's expensive").
- [ ] What does `React.memo` do? What must ALSO be true for it to prevent a re-render?
- [ ] What is prop drilling, and what are two ways to avoid it?
- [ ] Can you explain the Zustand `getState()` vs hook pattern in one sentence?
