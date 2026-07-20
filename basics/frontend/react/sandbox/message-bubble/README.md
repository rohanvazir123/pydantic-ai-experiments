# MessageBubble sandbox

A trimmed, dependency-free copy of `rag/v2/frontend/src/components/chat/MessageBubble.tsx`,
built to run standalone — no Zustand store, no backend, no Tailwind. Same props, same
logic, plain CSS instead of Tailwind utility classes.

## Run it locally first (sanity check)

```bash
cd basics/frontend/react/sandbox/message-bubble
npm install
npm run dev        # http://localhost:5173
```

## Load it into CodeSandbox

**Option A — drag and drop (fastest):**
1. Go to [codesandbox.io/dashboard](https://codesandbox.io/dashboard).
2. Drag the `message-bubble` folder straight onto the page (or use "Import Project" → "Local folder").
3. CodeSandbox detects `package.json` + Vite and installs everything automatically.

**Option B — start from the Vite + React TS template and paste files in:**
1. [codesandbox.io/p/sandbox](https://codesandbox.io/p/sandbox) → pick "Vite" → "React TS".
2. Delete the generated `src/App.css` and default `src/App.tsx` contents.
3. Create each file listed below (same path, same name) and paste in the contents from this folder: `src/types.ts`, `src/CostBadge.tsx`, `src/DebugPanel.tsx`, `src/MessageBubble.tsx`, `src/App.tsx`, `src/index.css`.
4. Open the "Dependencies" panel and add `react-markdown` and `remark-gfm`.

## What to try

- Toggle the `debugMode` checkbox in the running app — watch `CostBadge` and `DebugPanel` appear/disappear. This is the optional-prop behavior from `react.md`.
- Open `src/App.tsx` and delete `debugMode={debugMode}` from one `<MessageBubble />` call. Nothing breaks — TypeScript still compiles, and the component falls back to `debugMode = false`.
- In `src/MessageBubble.tsx`, change the signature from the destructured form back to the longhand `(props: Props)` + `const { message, debugMode = false } = props` and confirm it behaves identically (see `react.md`, "Functional components and props").
- Add a new `Message` field (e.g. `latency_ms?: number`) to `src/types.ts`, then render it conditionally in `MessageBubble.tsx` — practice reading TypeScript errors when you forget the `?`.
- Break something on purpose: remove the `key={msg.id}` in `App.tsx`'s `.map()` and open the browser console to see React's key warning.
