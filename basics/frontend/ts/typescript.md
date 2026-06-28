# TypeScript Reference for Production RAG Apps

A focused guide grounded in a Vite + React 19 + TypeScript + FastAPI stack.  
Config baseline: `strict`, `verbatimModuleSyntax`, `erasableSyntaxOnly`, `moduleResolution: bundler`.

---

## Table of Contents

1. [Part 1 — Type System Fundamentals](#part-1--type-system-fundamentals)
   - [Primitive types and inference](#primitive-types-and-inference)
   - [interface vs type](#interface-vs-type)
   - [Unions, intersections, discriminated unions](#unions-intersections-discriminated-unions)
   - [Literal types and as const](#literal-types-and-as-const)
   - [Optional properties and non-null assertion](#optional-properties-and-non-null-assertion)
   - [unknown vs any vs never](#unknown-vs-any-vs-never)
   - [Enums vs string union types](#enums-vs-string-union-types)
   - [readonly and immutability](#readonly-and-immutability)
2. [Part 2 — Generics](#part-2--generics)
   - [Generic functions](#generic-functions)
   - [Generic interfaces and types](#generic-interfaces-and-types)
   - [Constraints: extends, keyof, typeof](#constraints-extends-keyof-typeof)
   - [Utility types](#utility-types)
3. [Part 3 — Advanced Patterns](#part-3--advanced-patterns)
   - [type import and verbatimModuleSyntax](#type-import-and-verbatimmodulesyntax)
   - [erasableSyntaxOnly and parameter properties](#erasablesyntaxonly-and-parameter-properties)
   - [Type guards](#type-guards)
   - [Mapped types and template literal types](#mapped-types-and-template-literal-types)
   - [Declaration merging and module augmentation](#declaration-merging-and-module-augmentation)
   - [satisfies operator](#satisfies-operator)
4. [Part 4 — TypeScript with the Project](#part-4--typescript-with-the-project)
   - [Typing Zustand stores](#typing-zustand-stores)
   - [Typing fetch responses with generics](#typing-fetch-responses-with-generics)
   - [AbortController and AbortSignal](#abortcontroller-and-abortsignal)
   - [ReadableStream and TextDecoder for SSE](#readablestream-and-textdecoder-for-sse)
   - [Record and index signatures](#record-and-index-signatures)
   - [as casting — safe vs unsafe](#as-casting--safe-vs-unsafe)
   - [tsconfig options that matter](#tsconfig-options-that-matter)

---

## Part 1 — Type System Fundamentals

### Primitive types and inference

TypeScript infers types from assignment. Annotate only where inference fails or where an explicit contract matters.

```typescript
// Let TS infer — no annotation needed
const query = 'What is the PTO policy?';   // inferred: string
const maxChunks = 10;                       // inferred: number
const streaming = true;                     // inferred: boolean

// Annotate function signatures — inference doesn't cross function boundaries
function buildHeader(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

// Annotate when the initialiser is misleading
const chunks: string[] = [];   // without annotation: never[]
```

Primitives: `string`, `number`, `boolean`, `bigint`, `symbol`, `null`, `undefined`.

---

### interface vs type

Use `interface` for object shapes (extendable, declaration-mergeable).  
Use `type` for unions, intersections, mapped types, and aliases that aren't plain object shapes.

```typescript
// interface — preferred for API response shapes
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

// interface extends interface
interface StreamingMessage extends Message {
  partial: boolean;
}

// type — preferred for unions and computed shapes
type ModelTier = 'nano' | 'small' | 'large';

type ChatOrSearch = ChatRequest | SearchRequest;   // union — must be type

// Both produce identical object shapes at runtime; choose by use-case
```

---

### Unions, intersections, discriminated unions

```typescript
// Union — value is one of several types
type Corpus = string | null;

// Intersection — value satisfies all types simultaneously
type AuthedRequest = BaseRequest & { token: string };

// Discriminated union — a shared literal field narrows the type
type APIEvent =
  | { kind: 'chunk';   text: string }
  | { kind: 'done';    usage: TokenUsage }
  | { kind: 'error';   message: string };

function handleEvent(event: APIEvent): void {
  switch (event.kind) {
    case 'chunk':
      appendChunk(event.text);    // TS knows event.text exists here
      break;
    case 'done':
      showUsage(event.usage);     // TS knows event.usage exists here
      break;
    case 'error':
      showError(event.message);
      break;
  }
}
```

Discriminated unions are the idiomatic pattern for SSE frame parsing.

---

### Literal types and as const

Literal types narrow `string` to a specific string value.

```typescript
// Without as const: inferred as string[]
const TIERS_MUTABLE = ['nano', 'small', 'large'];

// With as const: inferred as readonly ['nano', 'small', 'large']
const TIERS = ['nano', 'small', 'large'] as const;
type ModelTier = (typeof TIERS)[number];   // 'nano' | 'small' | 'large'

// Object as const — every property becomes a literal type
const ENDPOINTS = {
  chat:   '/v1/chat',
  search: '/v1/search',
  corpus: '/v1/corpus',
} as const;

type Endpoint = (typeof ENDPOINTS)[keyof typeof ENDPOINTS];
// '/v1/chat' | '/v1/search' | '/v1/corpus'
```

---

### Optional properties and non-null assertion

```typescript
interface ChatRequest {
  query: string;
  corpus_id: string;
  model_tier?: ModelTier;     // optional — may be undefined
  conversation_id?: string;
}

// Accessing optional fields safely
function buildPayload(req: ChatRequest): string {
  const tier = req.model_tier ?? 'small';   // nullish coalescing
  return JSON.stringify({ ...req, model_tier: tier });
}

// Non-null assertion (!) — tells TS "I know this is not null/undefined"
// Use only when you have external knowledge TS cannot verify
const el = document.getElementById('chat-input')!;  // safe if the element always exists
el.focus();
```

Use `!` sparingly. If you're reaching for it often, a type guard is usually cleaner.

---

### unknown vs any vs never

```typescript
// any — disables type checking; avoid in production code
const data: any = await fetch('/api').then(r => r.json());
data.nonExistent.field;   // compiles — runtime crash waiting to happen

// unknown — forces you to narrow before use
async function fetchRaw(url: string): Promise<unknown> {
  return fetch(url).then(r => r.json());
}
const raw = await fetchRaw('/v1/health');
if (typeof raw === 'object' && raw !== null && 'status' in raw) {
  console.log((raw as { status: string }).status);
}

// never — a value that can never exist (exhaustive checks)
function assertNever(x: never): never {
  throw new Error(`Unhandled case: ${JSON.stringify(x)}`);
}

function handleTier(tier: ModelTier): string {
  switch (tier) {
    case 'nano':  return 'qwen2.5:0.5b';
    case 'small': return 'llama3.2:3b';
    case 'large': return 'llama3.1:70b';
    default:      return assertNever(tier);   // compile error if a tier is missed
  }
}
```

Rule of thumb: `unknown` at API boundaries, `never` for exhaustiveness, `any` never.

---

### Enums vs string union types

Prefer string union types. Enums carry runtime overhead (they emit JS objects) and interact poorly with `verbatimModuleSyntax`.

```typescript
// Avoid — emits a runtime object, not tree-shakeable
enum Role {
  User = 'user',
  Assistant = 'assistant',
}

// Prefer — zero runtime cost, works with as const
type Role = 'user' | 'assistant';

// const enum is also problematic with bundlers — avoid
```

---

### readonly and immutability

```typescript
interface CorpusInfo {
  readonly id: string;
  readonly name: string;
  description: string;   // mutable
}

// Readonly<T> makes every property readonly
type FrozenCorpus = Readonly<CorpusInfo>;

// readonly arrays
function processChunks(chunks: readonly string[]): number {
  // chunks.push('x')   — compile error
  return chunks.length;
}
```

---

## Part 2 — Generics

### Generic functions

```typescript
// Generic fetch wrapper — T is the expected response shape
async function get<T>(url: string, token: string): Promise<T> {
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

// Caller decides T
const corpora = await get<CorpusInfo[]>('/v1/corpus', token);
const health  = await get<HealthResponse>('/health', token);
```

---

### Generic interfaces and types

```typescript
// Generic wrapper for all API responses
interface APIResponse<T> {
  data: T;
  ok: boolean;
  error?: string;
}

// Generic paginated result
interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

// Usage
type ChatPage = Page<Message>;
type CorpusPage = Page<CorpusInfo>;
```

---

### Constraints: extends, keyof, typeof

```typescript
// T must be an object
function omitKey<T extends object, K extends keyof T>(
  obj: T,
  key: K,
): Omit<T, K> {
  const { [key]: _, ...rest } = obj;
  return rest as Omit<T, K>;
}

// keyof — produces the union of an object's keys
type MessageKey = keyof Message;   // 'id' | 'role' | 'content' | 'timestamp'

// typeof — lifts a runtime value into the type system
const DEFAULT_CONFIG = { stream: true, maxTokens: 2048 } as const;
type Config = typeof DEFAULT_CONFIG;
```

---

### Utility types

```typescript
interface ChatRequest {
  query: string;
  corpus_id: string;
  model_tier: ModelTier;
  conversation_id: string;
}

// Partial — all fields optional (useful for update payloads / defaults)
type ChatDraft = Partial<ChatRequest>;

// Required — all fields mandatory
type StrictRequest = Required<ChatDraft>;

// Pick — keep only listed fields
type SearchPayload = Pick<ChatRequest, 'query' | 'corpus_id'>;

// Omit — drop listed fields
type AnonymousRequest = Omit<ChatRequest, 'conversation_id'>;

// Record — map of keys to values
type ModelMap = Record<ModelTier, string>;
const MODELS: ModelMap = {
  nano:  'qwen2.5:0.5b',
  small: 'llama3.2:3b',
  large: 'llama3.1:70b',
};

// ReturnType and Parameters — introspect function signatures
async function sendChat(req: ChatRequest): Promise<Message> { /* ... */ return {} as Message; }
type SendChatReturn = Awaited<ReturnType<typeof sendChat>>;   // Message
type SendChatArgs  = Parameters<typeof sendChat>;             // [ChatRequest]
```

---

## Part 3 — Advanced Patterns

### type import and verbatimModuleSyntax

With `verbatimModuleSyntax: true`, every import used only as a type MUST use `import type`. The compiler rejects type-only imports written as value imports because it cannot elide them safely across all bundler modes.

```typescript
// Correct — type-only import, erased at compile time
import type { Message, CorpusInfo } from './types';

// Correct — value import (used at runtime)
import { create } from 'zustand';

// Wrong under verbatimModuleSyntax — will error if Message is type-only
import { Message } from './types';   // error: need `import type`

// Inline type import — mix value and type in one line
import { buildPayload, type ChatRequest } from './api';
```

---

### erasableSyntaxOnly and parameter properties

`erasableSyntaxOnly` bans TypeScript-only syntax that needs runtime transformation: parameter properties (`constructor(private x: T)`), enums, and namespaces. Only syntax that can be erased without changing semantics is allowed.

```typescript
// Banned under erasableSyntaxOnly
class APIError extends Error {
  constructor(
    public readonly status: number,   // error: parameter property
    message: string,
  ) {
    super(message);
  }
}

// Correct — explicit field declaration (pure erasable syntax)
class APIError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}
```

---

### Type guards

```typescript
// typeof guard — narrows primitives
function formatValue(v: string | number): string {
  if (typeof v === 'number') return v.toFixed(2);
  return v;
}

// instanceof guard — narrows class instances
function handleError(err: unknown): string {
  if (err instanceof APIError) return `API ${err.status}: ${err.message}`;
  if (err instanceof Error)    return err.message;
  return 'Unknown error';
}

// Custom type predicate — is T
function isMessage(v: unknown): v is Message {
  return (
    typeof v === 'object' &&
    v !== null &&
    'id' in v &&
    'role' in v &&
    'content' in v
  );
}

// Usage — after the guard, TS narrows to Message
const raw = await fetchRaw('/v1/chat');
if (isMessage(raw)) {
  console.log(raw.content);   // typed as string
}
```

---

### Mapped types and template literal types

```typescript
// Mapped type — transform every property
type Nullable<T> = { [K in keyof T]: T[K] | null };

// Template literal type — generate string literal unions from other literals
type EventName = 'chat' | 'search' | 'ingest';
type EventKey = `on_${EventName}`;   // 'on_chat' | 'on_search' | 'on_ingest'

// Combine both — useful for event handler maps
type EventHandlers = {
  [K in EventName as `on_${K}`]: () => void;
};
// { on_chat: () => void; on_search: () => void; on_ingest: () => void }
```

---

### Declaration merging and module augmentation

Interfaces (but not types) support declaration merging. Useful for extending third-party types.

```typescript
// Extend the global Window with a feature flag injected at build time
declare global {
  interface Window {
    __RAG_BUILD_ID__: string;
  }
}

// Augment a module — add a field to an existing interface
declare module 'zustand' {
  interface StoreMutators<S, A> {
    'custom/persist': WithPersist<S>;
  }
}
```

---

### satisfies operator

`satisfies` validates a value against a type without widening it — the inferred type stays narrow.

```typescript
type ModelConfig = { tier: ModelTier; model: string };

// Without satisfies: inferred as ModelConfig — loses literal types
const cfg: ModelConfig = { tier: 'small', model: 'llama3.2:3b' };

// With satisfies: validated against ModelConfig but inferred as literal
const cfg2 = {
  tier: 'small',
  model: 'llama3.2:3b',
} satisfies ModelConfig;

// cfg2.tier is inferred as 'small', not ModelTier
// Useful when you need both type safety AND precise inference downstream
```

---

## Part 4 — TypeScript with the Project

### Typing Zustand stores

```typescript
import { create } from 'zustand';
import type { Message, CorpusInfo } from './types';

interface ChatState {
  messages: Message[];
  conversationId: string | null;
  corpus: CorpusInfo | null;
  streaming: boolean;
  addMessage: (msg: Message) => void;
  setStreaming: (v: boolean) => void;
  reset: () => void;
}

const useChatStore = create<ChatState>((set) => ({
  messages: [],
  conversationId: null,
  corpus: null,
  streaming: false,
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setStreaming: (v) => set({ streaming: v }),
  reset: () => set({ messages: [], conversationId: null, streaming: false }),
}));

// Selector pattern — avoids unnecessary re-renders
const messages  = useChatStore((s) => s.messages);
const streaming = useChatStore((s) => s.streaming);
```

---

### Typing fetch responses with generics

```typescript
import type { CorpusInfo, Message, HealthResponse } from './types';

class APIClient {
  constructor(
    private readonly baseUrl: string,
    private readonly getToken: () => string,
  ) {}

  async get<T>(path: string, signal?: AbortSignal): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      headers: { Authorization: `Bearer ${this.getToken()}` },
      signal,
    });
    if (!res.ok) throw new APIError(res.status, await res.text());
    return res.json() as Promise<T>;
  }

  async post<TBody, TResponse>(
    path: string,
    body: TBody,
    signal?: AbortSignal,
  ): Promise<TResponse> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.getToken()}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new APIError(res.status, await res.text());
    return res.json() as Promise<TResponse>;
  }
}

// Fully typed call sites
const api = new APIClient('http://localhost:8001', getToken);
const corpora = await api.get<CorpusInfo[]>('/v1/corpus');
const health  = await api.get<HealthResponse>('/health');
```

---

### AbortController and AbortSignal

```typescript
// AbortController is a browser built-in — no import needed
function useCancelableChat() {
  const controllerRef = useRef<AbortController | null>(null);

  function startChat(query: string): void {
    // Cancel any in-flight request
    controllerRef.current?.abort();
    controllerRef.current = new AbortController();
    const { signal } = controllerRef.current;   // AbortSignal

    void api.post<ChatRequest, Message>(
      '/v1/chat',
      { query, corpus_id: 'default' },
      signal,
    ).catch((err: unknown) => {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      handleError(err);
    });
  }

  function cancel(): void {
    controllerRef.current?.abort();
  }

  return { startChat, cancel };
}
```

---

### ReadableStream and TextDecoder for SSE

The FastAPI backend streams SSE frames as `data: {...}\n\n`. The browser's `fetch` + `ReadableStream` API handles this natively.

```typescript
async function streamChat(
  query: string,
  token: string,
  onChunk: (text: string) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch('http://localhost:8001/v1/chat', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ query, corpus_id: 'default', stream: true }),
    signal,
  });

  if (!res.body) throw new Error('No response body');

  // ReadableStream<Uint8Array> — typed by the Fetch API
  const reader: ReadableStreamDefaultReader<Uint8Array> =
    res.body.getReader();
  const decoder = new TextDecoder();   // TextDecoder is a browser built-in
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';   // keep incomplete line in buffer

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') return;

      const event = JSON.parse(raw) as APIEvent;
      if (event.kind === 'chunk') onChunk(event.text);
    }
  }
}
```

---

### Record and index signatures

```typescript
// Record<K, V> — all keys have the same value type; keys are known
type ModelMap = Record<ModelTier, string>;

// Index signature — keys are unknown at compile time
interface MetadataMap {
  [key: string]: unknown;   // use unknown, not any
}

// Combining known keys with an index signature via intersection
type ChunkMetadata = {
  source: string;
  page?: number;
} & { [key: string]: unknown };

// Record<string, unknown> is idiomatic for arbitrary JSON objects
function parseMetadata(raw: Record<string, unknown>): ChunkMetadata {
  return {
    source: String(raw['source'] ?? ''),
    page: typeof raw['page'] === 'number' ? raw['page'] : undefined,
    ...raw,
  };
}
```

---

### as casting — safe vs unsafe

```typescript
// Safe — you've already verified the shape (e.g. after a type guard)
const raw: unknown = await res.json();
if (isMessage(raw)) {
  const msg = raw;   // already narrowed — no cast needed
}

// Acceptable — JSON.parse always returns `any`; cast to known shape
const frame = JSON.parse(line) as APIEvent;

// Unsafe — bypasses the type system with no verification
const msg = raw as Message;   // raw might be anything — runtime crash risk

// Double cast (any escape hatch) — almost always wrong
const hack = raw as unknown as Message;   // serious code smell

// Safe pattern: cast + validate with a guard
function parseEvent(line: string): APIEvent | null {
  try {
    const v = JSON.parse(line) as unknown;
    if (
      typeof v === 'object' && v !== null &&
      'kind' in v && typeof (v as { kind: unknown }).kind === 'string'
    ) {
      return v as APIEvent;
    }
    return null;
  } catch {
    return null;
  }
}
```

---

### tsconfig options that matter

```jsonc
// tsconfig.json (Vite project baseline)
{
  "compilerOptions": {
    // Type safety
    "strict": true,                    // enables strictNullChecks, noImplicitAny, etc.
    "noUncheckedIndexedAccess": true,  // arr[0] is T | undefined, not T
    "exactOptionalPropertyTypes": true,// { x?: string } — undefined is not assignable

    // Module system
    "module": "ESNext",
    "moduleResolution": "bundler",     // Vite/esbuild resolution — no .js extensions required
    "verbatimModuleSyntax": true,      // type-only imports must use `import type`

    // Runtime constraints
    "erasableSyntaxOnly": true,        // bans enums, namespaces, parameter properties
    "isolatedModules": true,           // every file must be a module (no global scripts)

    // Output
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "noEmit": true,                    // Vite handles transpilation; tsc is type-check only

    // Paths alias (matches vite.config.ts resolve.alias)
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

Key interactions:

| Option | Effect in this stack |
|---|---|
| `strict` | Enables 8 sub-flags including `strictNullChecks` and `noImplicitAny` |
| `verbatimModuleSyntax` | Forces `import type` for type-only imports; prevents phantom imports |
| `erasableSyntaxOnly` | Enables Node 22 `--strip-only` without a TS compiler step |
| `moduleResolution: bundler` | Allows bare specifiers and `@/` aliases without `.js` extensions |
| `noUncheckedIndexedAccess` | `arr[i]` returns `T \| undefined`; catches off-by-one bugs at compile time |
