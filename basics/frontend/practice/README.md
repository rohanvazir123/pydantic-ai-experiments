# Practice — Day 5

Hands-on exercises that build on everything from Days 1–4. You write all the code; the docs explain all the concepts. Day 5 is where it clicks.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Day 5 Schedule](#day-5-schedule)
- [Files](#files)
- [Tips for Getting Unstuck](#tips-for-getting-unstuck)

---

## Prerequisites

Before starting these exercises you should have:

- [ ] Read `js/javascript.md` (Day 1)
- [ ] Read `ts/typescript.md` (Day 2)
- [ ] Read `react/react.md` (Day 3)
- [ ] Read `tsx/tsx.md` + `react/hooks.md` (Day 4)
- [ ] A working Vite project at `basics/frontend/day1_exercises/` (see main [README](../README.md) for setup)

---

## Project Setup

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
```

**Windows (PowerShell as Administrator)**
```powershell
winget install CoreyButler.NVMforWindows
# Close and reopen terminal:
nvm install lts
nvm use lts
node --version
```

### 2 — Scaffold the project (do once per machine — basics/frontend/day1_exercises/ is gitignored)

```bash
# Run from the basics/ directory
cd basics

# Node 20+
npm create vite@latest basics -- --template react-ts

# Node 18 fallback
npm create vite@5 basics -- --template react-ts
```

### 3 — Install packages (do once per machine)

```bash
cd basics/frontend/day1_exercises
npm install                                 # core deps (React, TypeScript, Vite)
npm install zustand react-router-dom clsx   # extras used in exercises
```

### 4 — Start working

```bash
cd basics/frontend/day1_exercises
npm run dev                  # http://localhost:5173 — hot-reloads on save
npx tsc --noEmit --watch     # type-checker in a second terminal
```

Work in `basics/frontend/day1_exercises/src/`. The dev server hot-reloads every file save.

**Suggested folder structure as you work through the exercises:**

```
basics/frontend/day1_exercises/src/
├── types/
│   ├── todo.ts          (Exercise 1)
│   └── pexels.ts        (Capstone)
├── components/
│   ├── TodoItem.tsx     (Exercise 2)
│   ├── TodoList.tsx     (Exercises 3–7)
│   ├── TodoSuggestion.tsx (Exercise 8)
│   ├── PhotoCard.tsx    (Capstone)
│   └── SearchBar.tsx    (Capstone)
├── hooks/
│   ├── useTodos.ts      (Exercise 5)
│   └── usePhotoSearch.ts (Capstone)
└── App.tsx              (wire everything together)
```

---

## Day 5 Schedule

```
Morning (2–3 h) — Todo App
  Exercises 1–5: Types → Component → State → Persistence → Custom hook
  Goal: a working Todo list that saves to localStorage

Mid-day (1 h) — Optimisation + Refs
  Exercises 6–7: useRef (auto-focus) + useMemo/useCallback/React.memo
  Goal: understand when these matter (and when they don't)

Afternoon (1 h) — API fetch
  Exercise 8: TodoSuggestion — async fetch, loading/error state
  Goal: pattern you will use for every API call in a real app

Evening (2–3 h) — Capstone
  Photo search app — AbortController, useDeferredValue, React.memo, typed API
  Goal: finish with a working, deployable mini-app that uses every concept
```

---

## Files

| File | What it is |
|------|-----------|
| [exercises.md](exercises.md) | All 8 exercises + capstone — starter stubs with TODOs, reference links, checklists |

---

## Tips for Getting Unstuck

**TypeScript error you don't understand:**
Hover over the red underline in VS Code — the tooltip is usually enough. If not, copy the error text and search for it on the TypeScript docs or Stack Overflow.

**"Cannot find module":**
Check that the file exists and the import path is correct. TypeScript is case-sensitive on Linux/Mac — `TodoItem` and `todoItem` are different files.

**Blank screen / nothing renders:**
Open the browser console (F12 → Console). A JS runtime error will be there.

**State not updating:**
Make sure you are calling the setter, not mutating the array directly.
```tsx
// Wrong — mutates in place, React does not detect the change
todos.push(newTodo)

// Right — new array reference triggers re-render
setTodos(prev => [...prev, newTodo])
```

**useEffect runs too often:**
Check your dependency array. Objects and arrays created inline are new references on every render — move them outside the component or wrap in `useMemo`.

**Type errors after adding a package:**
```bash
npm install -D @types/<package-name>
# e.g. npm install -D @types/react-router-dom
```

**Windows: `npx` not found:**
Make sure Node is on your PATH. Close and reopen the terminal after installing Node via nvm-windows.

**Windows: ESLint or Vite errors about line endings:**
Add a `.editorconfig` at the project root:
```ini
[*]
end_of_line = lf
```
