# TSX — Day 4 Supplement

TSX is what you write every time you make a React component. This folder covers the TypeScript-specific additions that `.tsx` brings over plain `.jsx` — prop typing, event types, refs, generic components, and how Vite actually processes your files.

## Table of Contents

- [Setup](#setup)
- [What to Study](#what-to-study)
- [When to Read This](#when-to-read-this)
- [Key Rules to Memorise](#key-rules-to-memorise)
- [Quick Reference](#quick-reference)
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
cd basics/exercises
npm install                                 # Vite, React, TypeScript
npm install zustand react-router-dom clsx   # extras
```

### 3 — Start working

```bash
cd basics/exercises
npm run dev          # http://localhost:5173
npx tsc --noEmit     # IMPORTANT: esbuild does NOT type-check — run this separately
```

Every `.tsx` component you write goes in `basics/exercises/src/components/`. Import it in `App.tsx` to see it in the browser.

---

## What to Study

**File:** [tsx.md](tsx.md)

| Section | What you learn |
|---------|---------------|
| What TSX Is | Why `.tsx` ≠ `.ts`, why esbuild does not type-check |
| ReactNode vs ReactElement | Which type to use for `children` and return types |
| Typing Component Props | `interface Props`, optional props, destructuring defaults |
| Event Types | `MouseEvent<T>`, `ChangeEvent<T>`, `KeyboardEvent<T>`, `FormEvent<T>` |
| Typing Refs | DOM refs (`null` initial), mutable value refs, which to use when |
| Generic Components | `function List<T,>` — the trailing comma explained |
| The `as` Prop | Polymorphic components — advanced, read but do not over-apply |
| `forwardRef` | When a parent needs the child's DOM node |
| How Vite Processes TSX | esbuild transforms, `tsc -b` for type checking, path aliases |

---

## When to Read This

Read **tsx.md** on Day 4 evening, after you have finished the React and Hooks tutorials. You will recognise every pattern from the components you already wrote — this doc just gives them proper names and TypeScript types.

---

## Key Rules to Memorise

**1. Use `interface` for props, not `type`.**
Interfaces give better error messages when you pass the wrong shape. Use `type` for unions (`'loading' | 'done'`) and mapped types.

**2. `children` is always `React.ReactNode`.**
It accepts everything React can render — strings, elements, null, arrays. Never type it as `React.ReactElement` unless you explicitly need to reject null.

**3. Event handler types include the element.**
```tsx
// Wrong — loses the element type
(e: Event) => void

// Right
(e: React.ChangeEvent<HTMLInputElement>) => void
(e: React.KeyboardEvent<HTMLTextAreaElement>) => void
```

**4. DOM refs start as `null`.**
```tsx
const ref = useRef<HTMLInputElement>(null)   // DOM ref — always null initial value
// Access is always guarded:
if (ref.current) ref.current.focus()
```

**5. Generic components need a trailing comma in `.tsx` files.**
```tsx
function Box<T,>({ value }: { value: T }) { ... }
//          ^ without this comma, the parser thinks <T> is JSX
```

**6. `tsc --noEmit` is the type checker. `npm run dev` is not.**
Vite's dev server (esbuild) will happily serve code with type errors. Always run `npx tsc --noEmit` before calling your code correct.

---

## Quick Reference

```tsx
// Props interface
interface Props {
  label:     string
  count?:    number              // optional
  children?: React.ReactNode     // optional children
  onClick:   () => void
  onChange:  (value: string) => void
}

// Event handler types
React.MouseEvent<HTMLButtonElement>
React.ChangeEvent<HTMLInputElement>
React.ChangeEvent<HTMLTextAreaElement>
React.ChangeEvent<HTMLSelectElement>
React.KeyboardEvent<HTMLTextAreaElement>
React.FormEvent<HTMLFormElement>
React.FocusEvent<HTMLInputElement>

// Ref types
useRef<HTMLInputElement>(null)        // DOM ref
useRef<HTMLTextAreaElement>(null)     // DOM ref
useRef<HTMLDivElement>(null)          // DOM ref
useRef<AbortController | null>(null)  // mutable value ref
useRef<number | null>(null)           // timer id

// Return types (usually inferred — annotate only when forcing non-null)
React.ReactNode    // anything renderable
React.ReactElement // exactly one element
```

---

## Checklist

- [ ] What is the difference between `React.ReactNode` and `React.ReactElement`? When do you use each?
- [ ] Why do you write `useRef<HTMLInputElement>(null)` with `null` as the initial value, not `undefined`?
- [ ] Write the correct event handler type for an `<input onChange>` handler from memory.
- [ ] What is the trailing comma in `function Foo<T,>` for?
- [ ] Does `npm run dev` catch TypeScript type errors? How do you actually run the type checker?
- [ ] What is the difference between a path alias (`@/`) in `vite.config.ts` vs `tsconfig.json` — and why do both need to agree?
