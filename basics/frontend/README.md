# Frontend Engineering — Setup + Study Guide

Everything you need to go from zero to comfortable with modern frontend in 5 days. Works on macOS and Windows.

## Table of Contents

- [Node.js Setup — macOS](#nodejs-setup--macos)
- [Node.js Setup — Windows](#nodejs-setup--windows)
- [Create the Vite Practice Project](#create-the-vite-practice-project)
- [Install Packages](#install-packages)
- [Verify Everything Works](#verify-everything-works)
- [5-Day Study Plan](#5-day-study-plan)
- [Tutorial Files](#tutorial-files)

---

## Node.js Setup — macOS

Node.js is the JavaScript runtime. Install it via **nvm** (Node Version Manager) so you can switch versions easily — do not use `brew install node` directly.

```bash
# 1. Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install nvm
brew install nvm

# 3. Add nvm to your shell — add these two lines to ~/.zshrc (zsh) or ~/.bash_profile (bash)
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"

# 4. Reload your shell
source ~/.zshrc       # or source ~/.bash_profile

# 5. Install the latest LTS Node
nvm install --lts
nvm use --lts
nvm alias default node   # make it the default for new shells

# 6. Verify
node --version    # 20.x or higher
npm --version     # 10.x or higher
```

---

## Node.js Setup — Windows

Use **nvm-windows** — a separate project from nvm but the same idea.

```powershell
# Option A — winget (Windows 10/11, recommended)
winget install CoreyButler.NVMforWindows

# After install, open a NEW terminal, then:
nvm install lts
nvm use lts

# Option B — manual install
# Download nvm-setup.exe from:
#   https://github.com/coreybutler/nvm-windows/releases/latest
# Run the installer, then in a new PowerShell:
nvm install lts
nvm use lts

# Verify (both options)
node --version    # 20.x or higher
npm --version     # 10.x or higher
```

> **Windows tip:** Run PowerShell as Administrator the first time you call `nvm use` — it needs to create a symlink.

---

## Create the Vite Practice Project

The practice project is already scaffolded at `basics/exercises/`. If you are setting up on a new machine, re-create it from the `basics/` directory:

> **Node version note:** `create-vite@latest` requires Node 20+. If you are on Node 18 (check with `node --version`), use `create-vite@5` instead — same result, older tooling.

**macOS / Linux:**
```bash
cd basics   # the basics/ directory, not basics/exercises/

# Node 20+ (recommended)
npm create vite@latest basics -- --template react-ts

# Node 18 fallback
npm create vite@5 basics -- --template react-ts

cd basics
npm install
```

**Windows (PowerShell):**
```powershell
cd basics   # the basics/ directory, not basics/exercises/

# Node 20+ (recommended)
npm create vite@latest basics -- --template react-ts

# Node 18 fallback
npm create vite@5 basics -- --template react-ts

cd basics
npm install
```

This creates `basics/exercises/` with Vite, React, TypeScript, and ESLint.

---

## Install Packages

The practice exercises use a few additional packages. Install them inside `basics/exercises/` (already done if you cloned this repo and ran `npm install`):

```bash
cd basics/exercises

# State management
npm install zustand

# Routing (types are bundled — no separate @types needed)
npm install react-router-dom

# Utility: conditional CSS classes
npm install clsx
```

**Windows note:** The commands are identical — npm works the same on Windows.

---

## Verify Everything Works

```bash
cd basics/exercises
npm run dev
```

Open `http://localhost:5173` in your browser. You should see the Vite + React starter page.

To run the TypeScript type-checker separately:
```bash
npx tsc --noEmit    # checks types without emitting files
```

To lint:
```bash
npx eslint src/
```

---

## 5-Day Study Plan

Work through the tutorial files in this order. Each day: read the doc, then write the code in `basics/exercises/src/`.

| Day | File | Focus | Time |
|-----|------|-------|------|
| 1 | [js/javascript.md](js/javascript.md) | Modern JS — destructuring, async/await, modules, AbortController | 3–4 h |
| 2 | [ts/typescript.md](ts/typescript.md) | TypeScript type system — interfaces, generics, utility types | 3–4 h |
| 3 | [react/react.md](react/react.md) | React fundamentals + production patterns | 4–5 h |
| 4 | [tsx/tsx.md](tsx/tsx.md) + [react/hooks.md](react/hooks.md) | TSX prop typing + every standard hook | 4–5 h |
| 5 | [practice/exercises.md](practice/exercises.md) | Build the Todo app, then the photo search capstone | 4–6 h |

**How to read each doc:**
1. Skim the Table of Contents first.
2. Read a section, then immediately try to write the example yourself in a scratch file — do not copy-paste.
3. If something is unclear, search for the term in the React or TypeScript docs and read the official explanation.

**The single most important habit:** Every time you write a component, annotate every prop, every state, and every function return type. TypeScript errors are your tutor — read them carefully.

---

## Tutorial Files

| Folder | What it covers |
|--------|---------------|
| [js/](js/) | Modern JavaScript — the language fundamentals every React dev needs |
| [ts/](ts/) | TypeScript — types, generics, utility types, project patterns |
| [react/](react/) | React 19 — components, hooks, routing, Zustand, production structure |
| [tsx/](tsx/) | TSX-specific typing — props, events, refs, generic components, Vite build |
| [practice/](practice/) | Hands-on exercises — Todo app + photo search capstone |
