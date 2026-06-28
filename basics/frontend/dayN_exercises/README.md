# dayN_exercises — template

Copy this folder, rename it, install, and go.

## Create a new sandbox

```bash
cp -r basics/frontend/dayN_exercises basics/frontend/day3_exercises
cd basics/frontend/day3_exercises
npm install
```

## Add packages

```bash
npm install zustand
npm install axios
npm install react-query
# etc.
```

## Start dev server

```bash
npm run dev
# opens http://localhost:5173
```

## What's included

| File | Purpose |
|------|---------|
| `src/App.tsx` | Start here — blank slate |
| `src/main.tsx` | React root mount |
| `vite.config.ts` | Vite + React plugin |
| `tsconfig*.json` | Strict TypeScript config |
| `eslint.config.js` | ESLint + React hooks rules |

`node_modules/` and `package-lock.json` are gitignored globally via `basics/frontend/day*/` pattern — no `.gitignore` edits needed for new sandboxes.
