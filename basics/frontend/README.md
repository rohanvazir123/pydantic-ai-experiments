# Frontend Engineering Reference

Tutorials covering the full frontend stack used in the RAG v2 project. Each file is self-contained — read in order or jump to what you need.

## Table of Contents

- [JavaScript](#javascript)
- [TypeScript](#typescript)
- [React](#react)
- [TSX](#tsx)
- [React Hooks](#react-hooks)
- [Practice](#practice)

---

## JavaScript

**[js/javascript.md](js/javascript.md)**

Covers modern JS from fundamentals through the advanced patterns used in this codebase.

| Section | Topics |
|---------|--------|
| Basics | `let`/`const`, types, arrow functions, destructuring, spread/rest, `?.`, `??`, array methods, modules |
| Advanced | Async/await, `Promise.race`, `AbortController`, `ReadableStream` + SSE, `sessionStorage` TTL cache, closures, async generators, TypeScript interop |

---

## TypeScript

**[ts/typescript.md](ts/typescript.md)**

Everything needed to write production TypeScript — from the type system to the project-specific patterns.

| Section | Topics |
|---------|--------|
| Type System | Primitives, `interface` vs `type`, union/intersection, discriminated unions, `unknown`/`any`/`never`, `readonly` |
| Generics | Generic functions, utility types (`Partial`, `Omit`, `Pick`, `Record`, `Awaited`, `ReturnType`) |
| Advanced | `import type`, `verbatimModuleSyntax`, `erasableSyntaxOnly`, type guards, mapped types, `satisfies` |
| Project Patterns | Zustand store typing, generic fetch client, `AbortController`, SSE stream types, `tsconfig` options |

---

## React

**[react/react.md](react/react.md)**

React 19 fundamentals and the production patterns used in the chat app.

| Section | Topics |
|---------|--------|
| Core | JSX, components, `useState`, `useEffect`, `useRef`, conditional rendering, lists + keys, events |
| Project Patterns | Custom hooks, Zustand vs Context, React Router v7, async state, `AbortController` cleanup |
| Production | File structure, `PrivateRoute`, `MainLayout`, avoiding prop drilling, sessionStorage cache hook |

---

## TSX

**[tsx/tsx.md](tsx/tsx.md)**

What `.tsx` adds on top of `.jsx` and TypeScript — prop typing, event types, refs, generic components, and the Vite build pipeline.

| Topic | Detail |
|-------|--------|
| Types | `ReactNode` vs `ReactElement` vs `JSX.Element`, props interfaces |
| Events | `MouseEvent`, `ChangeEvent`, `FormEvent`, `KeyboardEvent<T>` |
| Refs | DOM refs, mutable value refs, `forwardRef` |
| Build | Vite + esbuild (transform only) vs `tsc -b` (type check), path aliases |

---

## React Hooks

**[react/hooks.md](react/hooks.md)**

Every standard React hook with TypeScript types, purpose, and a real example from the project.

| Hook | Purpose |
|------|---------|
| `useState<T>` | Typed local state, lazy init, functional update |
| `useEffect` | Side effects, cleanup, async pattern |
| `useContext<T>` | Typed context — and when to use Zustand instead |
| `useRef<T>` | DOM refs + mutable values that skip re-render |
| `useMemo<T>` | Memoised derived values |
| `useCallback<F>` | Stable function references |
| `useTransition` | Non-urgent state updates |
| `useDeferredValue<T>` | Deferred expensive renders |
| `useId` | Stable unique IDs |
| `useSyncExternalStore<T>` | External store subscriptions (what Zustand uses) |
| `useLayoutEffect` | Synchronous post-DOM-mutation effects |
| `useImperativeHandle` | Expose methods via ref |
| Custom hooks | `useChat` breakdown, `useLocalStorage<T>` pattern |

---

## Practice

**[practice/exercises.md](practice/exercises.md)**

Eight progressive exercises that build a Todo app from scratch, followed by a capstone photo search app. Each exercise targets one concept from the tutorials above and uses the Vite scaffold at `basics/basics/`.

| Exercise | Concept |
|----------|---------|
| 1 | TypeScript interfaces and union types |
| 2 | Functional components and typed props |
| 3 | `useState`, lists, derived values |
| 4 | `useEffect` — load and sync to `localStorage` |
| 5 | Custom hook — extract logic from a component |
| 6 | `useRef` — DOM focus |
| 7 | `useMemo`, `useCallback`, `React.memo` |
| 8 | Async fetch with loading/error state |
| Capstone | Photo search app — all concepts combined |
