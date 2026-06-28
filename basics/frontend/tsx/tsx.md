# TSX — TypeScript + JSX

## Table of Contents

- [What TSX Is](#what-tsx-is)
- [ReactNode vs ReactElement vs JSX.Element](#reactnode-vs-reactelement-vs-jsxelement)
- [Typing Component Props](#typing-component-props)
- [Event Types](#event-types)
- [Typing Refs](#typing-refs)
- [Generic Components](#generic-components)
- [The `as` Prop — Polymorphic Components](#the-as-prop--polymorphic-components)
- [Type-Safe className Patterns](#type-safe-classname-patterns)
- [forwardRef](#forwardref)
- [How Vite Processes TSX](#how-vite-processes-tsx)

---

## What TSX Is

TSX is a `.tsx` file — TypeScript source that contains JSX syntax. It adds two things on top of plain `.jsx`:

1. **Static types** — props, events, refs, and return values are all checked at compile time by `tsc`.
2. **JSX type-checking** — the compiler knows `<div onClick={...}>` expects a `React.MouseEventHandler<HTMLDivElement>`, not an arbitrary function.

The JSX transform (converting `<Component />` to `React.createElement(...)` or the modern `_jsx(...)`) is done by **esbuild** inside Vite — it is not TypeScript's job. TypeScript only checks types; esbuild strips them and emits JS. This matters because you can have a type error that never blocks the dev server.

`.tsx` is required (not `.ts`) any time a file contains angle-bracket JSX. A plain `.ts` file that tries to write JSX will get a parse error.

---

## ReactNode vs ReactElement vs JSX.Element

These three look interchangeable but are distinct:

| Type | What it includes | When to use it |
|---|---|---|
| `React.ReactNode` | `ReactElement`, `string`, `number`, `boolean`, `null`, `undefined`, arrays, fragments | `children` prop type — accepts everything React can render |
| `React.ReactElement` | The object returned by `React.createElement()` / JSX — never `null` | When you need to call `.props` or pass to `React.cloneElement` |
| `JSX.Element` | Alias for `React.ReactElement<any, any>` — same thing, older spelling | Avoid — prefer `React.ReactElement` for clarity |

```tsx
// children should be ReactNode — it covers null, strings, nested elements
interface Props {
  children: React.ReactNode
}

// A function that MUST return an element (not null) uses ReactElement
function RequiredIcon(): React.ReactElement {
  return <span>★</span>
}

// Most component return types can just be inferred — TypeScript figures it out
export function Badge({ children }: Props) {
  return <span className="badge">{children}</span>
}
```

Rule of thumb: annotate `children` as `React.ReactNode`. Let return types be inferred unless you need to enforce non-null.

---

## Typing Component Props

Define props with an `interface` (not `type` — interfaces give better error messages for object shapes).

```tsx
interface Props {
  // required props
  onSend:  (query: string) => void
  onStop:  () => void
  loading: boolean

  // optional props — use ? not `| undefined` explicitly
  label?:    string
  children?: React.ReactNode
}

export function InputBar({ onSend, onStop, loading, label = 'Send' }: Props) {
  // ...
}
```

From `InputBar.tsx` in this project — exactly this pattern, no wrapper type, destructured inline:

```tsx
interface Props {
  onSend:  (query: string) => void
  onStop:  () => void
  loading: boolean
}

export function InputBar({ onSend, onStop, loading }: Props) {
```

**Children types at a glance:**

```tsx
children: React.ReactNode          // anything renderable — most common
children: React.ReactElement       // exactly one element, not null
children: string                   // text only
children?: React.ReactNode         // optional children
children: React.ReactNode[]        // multiple children (rare — ReactNode already covers arrays)
```

---

## Event Types

TypeScript narrows event types to the specific element. The generic is the element the handler is attached to.

```tsx
// MouseEvent — button, div, any clickable element
function handleClick(e: React.MouseEvent<HTMLButtonElement>) {
  e.preventDefault()
  console.log(e.currentTarget.dataset.id)
}

// ChangeEvent — input, textarea, select
function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
  setValue(e.target.value)
}

// FormEvent — form submit
function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
  e.preventDefault()
}

// KeyboardEvent — from InputBar.tsx, exact pattern used in this project
import { type KeyboardEvent } from 'react'

function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submit()
  }
}
```

The `KeyboardEvent` import style (`type KeyboardEvent`) is a type-only import — it gets completely erased by esbuild and adds zero runtime overhead. This is the pattern from `InputBar.tsx`:

```tsx
import { useState, useRef, type KeyboardEvent } from 'react'

function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
  // ...
}
```

**Common pairings:**

| Event | Element generic |
|---|---|
| `React.MouseEvent<T>` | `HTMLButtonElement`, `HTMLDivElement` |
| `React.ChangeEvent<T>` | `HTMLInputElement`, `HTMLTextAreaElement`, `HTMLSelectElement` |
| `React.FormEvent<T>` | `HTMLFormElement` |
| `React.KeyboardEvent<T>` | `HTMLTextAreaElement`, `HTMLInputElement` |
| `React.FocusEvent<T>` | `HTMLInputElement` |

When the element type does not matter, use `React.SyntheticEvent` as a fallback.

---

## Typing Refs

`useRef` has two distinct uses and the type reflects which one you want.

**DOM ref** — attach to an element via the `ref` prop:

```tsx
// Generic is the element type; initial value must be null for DOM refs
const textareaRef = useRef<HTMLTextAreaElement>(null)

// From InputBar.tsx
const textareaRef = useRef<HTMLTextAreaElement>(null)

// Access is always guarded because it's null until mount
if (textareaRef.current) {
  textareaRef.current.style.height = 'auto'
}

return <textarea ref={textareaRef} />
```

**Mutable value ref** — stores a value that persists across renders without triggering re-render:

```tsx
// From useChat.ts — stores an AbortController to cancel in-flight SSE streams
const abortRef = useRef<AbortController | null>(null)

// Cancel previous before starting new
abortRef.current?.abort()
abortRef.current = new AbortController()
```

The difference: DOM refs use `useRef<Element>(null)` and TypeScript will enforce the `ref` prop is compatible. Mutable refs use `useRef<T | null>(null)` and you manage `.current` yourself.

**Other element types you will encounter:**

```tsx
useRef<HTMLDivElement>(null)      // scrollable containers
useRef<HTMLInputElement>(null)    // programmatic focus
useRef<HTMLDialogElement>(null)   // native dialog
useRef<number | null>(null)       // setInterval/setTimeout ids
```

---

## Generic Components

A component can be generic over the type of its data — useful for lists, tables, select inputs.

```tsx
interface ListProps<T> {
  items:    T[]
  getKey:   (item: T) => string
  renderItem: (item: T) => React.ReactNode
}

// The trailing comma in <T,> prevents the parser from treating <T> as JSX
function List<T,>({ items, getKey, renderItem }: ListProps<T>) {
  return (
    <ul>
      {items.map(item => (
        <li key={getKey(item)}>{renderItem(item)}</li>
      ))}
    </ul>
  )
}

// Usage — T is inferred as { id: string; label: string }
<List
  items={corpora}
  getKey={c => c.id}
  renderItem={c => <span>{c.label}</span>}
/>
```

The `<T,>` trailing comma is a TSX parser quirk — without it, the compiler cannot tell `<T>` from an opening JSX tag.

---

## The `as` Prop — Polymorphic Components

Lets a component render as different HTML elements while keeping types correct.

```tsx
type AsProp<C extends React.ElementType> = { as?: C }

type PolymorphicProps<C extends React.ElementType, P = {}> =
  AsProp<C> & Omit<React.ComponentPropsWithoutRef<C>, keyof AsProp<C>> & P

function Text<C extends React.ElementType = 'p'>({
  as,
  ...rest
}: PolymorphicProps<C>) {
  const Tag = as ?? 'p'
  return <Tag {...rest} />
}

// Renders as <p> with paragraph props
<Text>Hello</Text>

// Renders as <h2> with heading props — href would be a type error here
<Text as="h2">Section title</Text>

// Renders as <a> — href is now valid
<Text as="a" href="/docs">Link</Text>
```

Keep this pattern brief in real code — it is complex to maintain. Only reach for it in shared component libraries.

---

## Type-Safe className Patterns

Template literals with TypeScript string literal types catch invalid class construction at compile time.

```tsx
type Size   = 'sm' | 'md' | 'lg'
type Variant = 'primary' | 'ghost'

interface ButtonProps {
  size:    Size
  variant: Variant
}

function Button({ size, variant }: ButtonProps) {
  // TypeScript knows every branch is valid
  const sizeClass:    Record<Size, string>    = { sm: 'px-2 py-1 text-xs', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' }
  const variantClass: Record<Variant, string> = { primary: 'bg-accent text-white', ghost: 'bg-transparent border' }

  return (
    <button className={`${sizeClass[size]} ${variantClass[variant]} rounded`}>
      ...
    </button>
  )
}
```

The `Record<K, string>` lookup pattern is safer than `cn(size === 'sm' && '...')` chains for many variants — TypeScript will error if you add a new union member and forget to handle it.

For conditional classes, the project uses template literals directly:

```tsx
// From InputBar.tsx style
className="flex-1 resize-none bg-[var(--bg)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-[var(--accent)] transition-colors"
```

---

## forwardRef

Used when a parent needs direct access to a child's DOM node (e.g. to call `.focus()`).

```tsx
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string
}

// React.forwardRef<ElementType, PropsType>
const TextInput = React.forwardRef<HTMLInputElement, InputProps>(
  ({ label, ...rest }, ref) => (
    <label>
      {label}
      <input ref={ref} {...rest} />
    </label>
  )
)

TextInput.displayName = 'TextInput'

// Parent usage
const inputRef = useRef<HTMLInputElement>(null)
<TextInput ref={inputRef} label="Search" />
inputRef.current?.focus()
```

In React 19, `ref` becomes a regular prop — `forwardRef` is no longer needed. Until then, keep `displayName` set so React DevTools shows a useful name.

---

## How Vite Processes TSX

1. **esbuild** handles the transform step — it strips TypeScript types and converts JSX to `_jsx()` calls. This is extremely fast (sub-millisecond per file).
2. **No type checking at build time.** `vite build` and `vite dev` will succeed even if you have type errors. This is intentional — esbuild only parses, never type-checks.
3. **Type checking is a separate step.** Run `tsc -b` (or `tsc --noEmit`) to check types. In this project that is wired to the `build` script in `package.json`:
   ```json
   "build": "tsc -b && vite build"
   ```
4. The JSX transform used is the **automatic runtime** (`"jsx": "react-jsx"` in `tsconfig.json`) — you do not need `import React from 'react'` at the top of every file. The compiler inserts the import automatically.
5. **Path aliases** (`@/`) are configured in both `vite.config.ts` (for bundling) and `tsconfig.json` (for type checking) — both must agree or you get runtime resolution without type errors, or type errors without runtime failures.
