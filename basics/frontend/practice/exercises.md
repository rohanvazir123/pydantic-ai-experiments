# React + TypeScript Practice Exercises

All exercises use the Vite project at `basics/basics/`. Run `npm run dev` from that directory before you start.

## Table of Contents

- [Exercise 1 — TypeScript Types](#exercise-1--typescript-types)
- [Exercise 2 — Your First Component](#exercise-2--your-first-component)
- [Exercise 3 — State and Lists](#exercise-3--state-and-lists)
- [Exercise 4 — useEffect and Persistence](#exercise-4--useeffect-and-persistence)
- [Exercise 5 — Custom Hook](#exercise-5--custom-hook)
- [Exercise 6 — useRef](#exercise-6--useref)
- [Exercise 7 — useMemo and useCallback](#exercise-7--usememo-and-usecallback)
- [Exercise 8 — API Fetch with Loading and Error States](#exercise-8--api-fetch-with-loading-and-error-states)
- [Capstone — Photo Search App](#capstone--photo-search-app)

---

## Exercise 1 — TypeScript Types

**Goal:** Define the types your entire Todo app will use.

**File:** `src/types/todo.ts` (create it)

**Task:** Fill in every `TODO` comment.

```ts
// src/types/todo.ts

// TODO: Define a Todo interface with these fields:
//   id:        string
//   text:      string
//   done:      boolean
//   createdAt: number   (timestamp from Date.now())

// TODO: Define a union type Filter = 'all' | 'active' | 'done'

// TODO: Define a TodosState interface:
//   todos:     Todo[]
//   filter:    Filter
```

**Check:** No TypeScript errors when you import from `'./types/todo'` in `App.tsx`.

**Reference:** [TypeScript tutorial → Type System](../ts/typescript.md#type-system)

---

## Exercise 2 — Your First Component

**Goal:** Build a `TodoItem` component that displays one todo and lets the user toggle it done or delete it.

**File:** `src/components/TodoItem.tsx` (create it)

**Starter:**

```tsx
// src/components/TodoItem.tsx
import type { Todo } from '../types/todo'

// TODO: Define a Props interface:
//   todo:     Todo
//   onToggle: (id: string) => void
//   onDelete: (id: string) => void

// TODO: Implement the component.
//   - Strike through the text when todo.done is true.
//   - A checkbox toggles done. Its checked value is todo.done.
//     onClick calls onToggle(todo.id).
//   - A delete button calls onDelete(todo.id).

export function TodoItem(/* props */) {
  // your implementation
}
```

**Expected output** (when rendered with a sample todo):

```
[ ] Buy milk        [Delete]
[x] ~~Walk dog~~    [Delete]
```

**Reference:** [React tutorial → Functional components and props](../react/react.md#functional-components-and-props), [TSX → Typing Component Props](../tsx/tsx.md#typing-component-props)

---

## Exercise 3 — State and Lists

**Goal:** Build a `TodoList` component that manages the full list — add, toggle, delete, filter.

**File:** `src/components/TodoList.tsx` (create it), then render it from `App.tsx`.

**Starter:**

```tsx
// src/components/TodoList.tsx
import { useState } from 'react'
import { TodoItem } from './TodoItem'
import type { Todo, Filter } from '../types/todo'

export function TodoList() {
  // TODO: useState for todos (Todo[]) — start with []
  // TODO: useState for filter (Filter) — start with 'all'
  // TODO: useState for inputText (string) — start with ''

  // TODO: addTodo()
  //   - If inputText.trim() is empty, return early.
  //   - Create a new Todo: id via crypto.randomUUID(), text from inputText, done: false, createdAt: Date.now()
  //   - Append to todos with functional update: setTodos(prev => [...prev, newTodo])
  //   - Clear inputText.

  // TODO: toggleTodo(id: string)
  //   - Map over todos; for the matching id, return { ...todo, done: !todo.done }

  // TODO: deleteTodo(id: string)
  //   - Filter out the todo with the matching id.

  // TODO: visibleTodos — derive from todos and filter (no separate useState!)
  //   'all'    → todos
  //   'active' → todos where !done
  //   'done'   → todos where done

  return (
    <div>
      {/* TODO: Input + Add button */}
      {/* TODO: Filter buttons: All | Active | Done — highlight the active filter */}
      {/* TODO: Map visibleTodos → <TodoItem> — use todo.id as key */}
      {/* TODO: Show "No todos yet." when visibleTodos is empty */}
    </div>
  )
}
```

**In `App.tsx`:** Replace the starter content with `<TodoList />`.

**Check:** You can add todos, mark them done, delete them, and filter.

**Reference:** [React tutorial → useState](../react/react.md#usestate-state-vs-derived-values), [React tutorial → Lists and keys](../react/react.md#lists-and-keys)

---

## Exercise 4 — useEffect and Persistence

**Goal:** Persist todos to `localStorage` so they survive a page reload.

**Where:** Inside `TodoList.tsx`.

**Task:**

```tsx
// Load from localStorage on mount — runs once
useEffect(() => {
  // TODO: Read the item 'todos' from localStorage.
  //       If it exists, JSON.parse it and call setTodos with the result.
  //       Wrap in try/catch — JSON.parse can throw on corrupted data.
}, [])

// Save to localStorage whenever todos change
useEffect(() => {
  // TODO: JSON.stringify todos and write it to localStorage key 'todos'.
}, [todos]) // <-- depend on todos
```

**Check:** Add a todo, reload the page — it should still be there.

**What to understand:**
- The first effect (`[]`) runs exactly once on mount — ideal for loading saved data.
- The second effect (`[todos]`) runs after every render where `todos` changed — ideal for syncing to an external store.
- If you put both in one effect, you would overwrite storage with an empty array on the first render before the load effect could run. Two effects, two responsibilities.

**Reference:** [Hooks tutorial → useEffect](../react/hooks.md#useeffect)

---

## Exercise 5 — Custom Hook

**Goal:** Extract all todo logic out of `TodoList.tsx` into a `useTodos` hook.

**File:** `src/hooks/useTodos.ts` (create it)

**Task:** Move every `useState`, `useEffect`, and handler out of `TodoList.tsx` and into this hook. The hook returns everything the component needs.

```ts
// src/hooks/useTodos.ts
import { useState, useEffect } from 'react'
import type { Todo, Filter } from '../types/todo'

export function useTodos() {
  // TODO: Move all state here (todos, filter, inputText)
  // TODO: Move both useEffects here
  // TODO: Move addTodo, toggleTodo, deleteTodo here
  // TODO: Compute visibleTodos here

  return {
    // TODO: Return everything TodoList needs:
    //   todos, filter, inputText, visibleTodos,
    //   setFilter, setInputText, addTodo, toggleTodo, deleteTodo
  }
}
```

**`TodoList.tsx` after the refactor:**

```tsx
import { useTodos } from '../hooks/useTodos'
import { TodoItem } from './TodoItem'

export function TodoList() {
  const { visibleTodos, filter, inputText, setFilter, setInputText, addTodo, toggleTodo, deleteTodo } = useTodos()

  return (
    // ... exactly the same JSX as before
  )
}
```

**Check:** Everything still works, but `TodoList.tsx` contains no `useState` or `useEffect` — only JSX and the hook call.

**Reference:** [React tutorial → Custom hooks](../react/react.md#custom-hooks), [Hooks tutorial → Custom Hooks](../react/hooks.md#custom-hooks)

---

## Exercise 6 — useRef

**Goal:** Auto-focus the input field on mount, and auto-focus it again after adding a todo so the user can keep typing without clicking.

**Where:** `TodoList.tsx`

**Task:**

```tsx
import { useRef } from 'react'

// TODO: Create inputRef = useRef<HTMLInputElement>(null)

// TODO: Attach ref={inputRef} to your <input> element.

// TODO: In addTodo (or call it from the hook's return), after clearing the text,
//       call inputRef.current?.focus()

// TODO: Also call inputRef.current?.focus() in a useEffect with [] so it
//       focuses on initial mount.
```

**What to understand:** `inputRef.current` is `null` until the component mounts (React sets it when the DOM node is created). Checking `?.` before calling `.focus()` guards against that window.

**Reference:** [Hooks tutorial → useRef](../react/hooks.md#useref)

---

## Exercise 7 — useMemo and useCallback

**Goal:** Memoize the filtered list and the stable callbacks passed to `TodoItem`.

**Where:** `TodoList.tsx` (or inside `useTodos.ts`)

**Task A — useMemo:**

```tsx
import { useMemo } from 'react'

// Replace the inline visibleTodos derivation with:
const visibleTodos = useMemo<Todo[]>(() => {
  // TODO: same filter logic as before
}, [todos, filter])
```

**Task B — useCallback:**

```tsx
import { useCallback } from 'react'

// TODO: Wrap toggleTodo in useCallback — deps: [setTodos]
// TODO: Wrap deleteTodo in useCallback — deps: [setTodos]
```

**Task C — React.memo:**

```tsx
// In TodoItem.tsx — wrap the export:
export const TodoItem = React.memo(function TodoItem({ todo, onToggle, onDelete }: Props) {
  // ... same implementation
})
```

**Check:** Open React DevTools Profiler — with `React.memo` + `useCallback`, toggling one todo should highlight only that `TodoItem`, not all of them.

**When it actually matters:** Your list has a handful of items — `useMemo` and `useCallback` have no measurable impact here. This exercise is about learning the pattern, not optimising a real bottleneck. Do not reach for these in production until you have profiled and confirmed a re-render problem.

**Reference:** [Hooks tutorial → useMemo](../react/hooks.md#usememo), [Hooks tutorial → useCallback](../react/hooks.md#usecallback)

---

## Exercise 8 — API Fetch with Loading and Error States

**Goal:** Add a "Random todo idea" button that fetches a suggestion from a public API.

**API:** `https://jsonplaceholder.typicode.com/todos/{id}` — returns `{ id, title, completed }`. Pick a random id 1–200.

**File:** `src/components/TodoSuggestion.tsx` (create it)

**Task:**

```tsx
// src/components/TodoSuggestion.tsx
import { useState } from 'react'

interface JsonPlaceholderTodo {
  // TODO: type the response — id: number, title: string, completed: boolean
}

interface Props {
  onAdd: (text: string) => void   // called when the user accepts the suggestion
}

export function TodoSuggestion({ onAdd }: Props) {
  // TODO: useState for suggestion (string | null) — null = none fetched yet
  // TODO: useState for loading (boolean)
  // TODO: useState for error (string | null)

  async function fetchSuggestion() {
    // TODO: setLoading(true), setError(null)
    // TODO: Pick a random id: Math.floor(Math.random() * 200) + 1
    // TODO: fetch the URL, check response.ok (throw if not), parse JSON
    // TODO: setLoading(false), setSuggestion(data.title)
    // TODO: In catch: setError(narrowed error message), setLoading(false)
    //       Remember: err is unknown — check instanceof Error before .message
  }

  return (
    <div>
      <button onClick={fetchSuggestion} disabled={loading}>
        {loading ? 'Fetching…' : 'Random idea'}
      </button>
      {error      && <p style={{ color: 'red' }}>{error}</p>}
      {suggestion && (
        <p>
          Suggestion: <em>{suggestion}</em>{' '}
          <button onClick={() => onAdd(suggestion)}>Add it</button>
        </p>
      )}
    </div>
  )
}
```

**Wire it up:** In `TodoList.tsx`, render `<TodoSuggestion onAdd={addTodo} />`. `addTodo` currently takes the text from `inputText` — refactor it to accept an optional `text` parameter, or add a separate `addTodoText(text: string)` that does not depend on input state.

**Reference:** [Hooks tutorial → useEffect complete example](../react/hooks.md#complete-example--typed-api-fetch-with-loadingerror-states)

---

## Capstone — Photo Search App

**Goal:** Build a standalone photo search page in `App.tsx` using everything from the tutorial.

**API:** `https://api.pexels.com/v1/search?query={query}&per_page=12` with header `Authorization: YOUR_API_KEY`. Get a free key at [pexels.com/api](https://www.pexels.com/api/).

**What to build:**

```
[ Search input            ] [Search]

Loading... / Error: ... / (empty state)

┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  <photo>    │  │  <photo>    │  │  <photo>    │
│ Photographer│  │ Photographer│  │ Photographer│
└─────────────┘  └─────────────┘  └─────────────┘
```

**Requirements — implement each one:**

1. **Types** (`src/types/pexels.ts`)
   - `PexelsPhoto`: `id`, `photographer`, `avg_color`, `src.medium`
   - `PexelsResponse`: `photos: PexelsPhoto[]`, `total_results`

2. **Custom hook** (`src/hooks/usePhotoSearch.ts`)
   - State: `photos`, `loading`, `error`, `query`
   - `search(q: string)` — async, sets loading/error/photos
   - Returns `{ photos, loading, error, query, search }`
   - Cancel in-flight requests with `AbortController` + `useRef`
   - Guard `err.name !== 'AbortError'` before setting error state

3. **Photo card component** (`src/components/PhotoCard.tsx`)
   - Props: `photo: PexelsPhoto`
   - Renders `<img src={photo.src.medium}>`, photographer name
   - Background color: `photo.avg_color`
   - `React.memo` wrapped

4. **Search bar component** (`src/components/SearchBar.tsx`)
   - Props: `onSearch: (q: string) => void`, `loading: boolean`
   - Controlled input (`useState`)
   - Submit on Enter (`KeyboardEvent<HTMLInputElement>`) or button click
   - Disable input and button while loading

5. **App.tsx** — compose them:
   ```tsx
   const { photos, loading, error, search } = usePhotoSearch()

   return (
     <>
       <SearchBar onSearch={search} loading={loading} />
       {loading && <p>Loading…</p>}
       {error   && <p>Error: {error}</p>}
       <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
         {photos.map(p => <PhotoCard key={p.id} photo={p} />)}
       </div>
     </>
   )
   ```

6. **Bonus — useDeferredValue** (from the hooks tutorial)
   - Add a local `query` state that updates on every keystroke.
   - Wrap it: `const deferredQuery = useDeferredValue(query)`.
   - Only call `search(deferredQuery)` in a `useEffect([deferredQuery])`.
   - This lets the input stay responsive while the search runs.

**Checklist before you call it done:**
- [ ] No TypeScript errors (`tsc --noEmit` in `basics/basics/`)
- [ ] Loading state shows while fetch is in progress
- [ ] Error state shows on a bad API key or network failure
- [ ] Typing in the search box does not freeze the UI
- [ ] Switching queries quickly does not flash stale photos (AbortController cancels the previous request)
- [ ] `PhotoCard` only re-renders when its `photo` prop changes (check with React DevTools)
