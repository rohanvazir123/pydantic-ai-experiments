# Node/Express vs. Python/FastAPI — POST/GET Comparison

Same REST resource (`Item`: name, description, price) implemented two ways,
runnable side by side. Source snippets originally sketched in
`../get_post_node_js_and_fast_api_example.md`.

```
node_express/     Express + Drizzle ORM + TypeScript, SQLite (better-sqlite3)
python_fastapi/    FastAPI + SQLModel, SQLite
```

Both expose the same two endpoints — `POST /items` (create) and
`GET /items` (list all) — with a slightly different request path
(`/items` vs `/items/`, FastAPI's default trailing-slash convention).

## Run: node_express

```bash
cd node_express
npm install
npx drizzle-kit push       # creates dev.db from db/schema.ts, no codegen step
npm run dev                 # http://localhost:3000
```

```bash
curl -X POST http://localhost:3000/items \
  -H "Content-Type: application/json" \
  -d '{"name":"Widget","description":"A test widget","price":9.99}'

curl http://localhost:3000/items
```

## Run: python_fastapi

```bash
cd python_fastapi
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/uvicorn main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/items/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Widget","description":"A test widget","price":9.99}'

curl http://localhost:8000/items/
```

Both verified working end-to-end (POST then GET returns the created item).

## Key differences worth noting

| | Express + Drizzle | FastAPI + SQLModel |
|---|---|---|
| Model definition | Plain TypeScript object (`db/schema.ts`), no separate DSL, no codegen step | Plain Python class (`SQLModel` + Pydantic), no codegen step |
| Request validation | Manual (`if (!name \|\| typeof price !== "number")`) | Automatic — FastAPI validates against the `Item` type from the request body before your function runs |
| Response shape | Whatever Drizzle returns (`.returning()`), shaped by hand | Declared via `response_model=Item`, FastAPI filters/serializes automatically |
| DB session lifecycle | Single shared `better-sqlite3` connection wrapped by `drizzle()` | Per-request session via `Depends(get_session)`, a generator-based dependency |
| API docs | Not generated automatically (would need a separate OpenAPI lib) | Free interactive docs at `/docs` (OpenAPI, generated from the same type annotations) |
| Type source of truth | `db/schema.ts` — TypeScript, same language as route code, no build step to get types | One Python class serves as DB model *and* request/response schema |

Drizzle closes most of the ORM-side gap versus Prisma — the schema lives in
a plain `.ts` file, `drizzle-kit push` reads it directly (no `generate`
step, no separate `schema.prisma` DSL). What FastAPI + SQLModel still wins
on is **validation and serialization**: that's a property of FastAPI the
framework, not the ORM, since Express does no request/response
schema-checking on its own. Swapping in Fastify + `drizzle-zod` would close
that remaining gap if you want to see it — a separate exercise from the
ORM swap.

## Migrations in production

Neither project below actually runs this — both still use their "demo"
table-creation step (`create_all()` / `drizzle-kit push`). This is what
each would move to for real schema evolution.

**FastAPI + SQLModel → Alembic.** `SQLModel.metadata.create_all(engine)`
only ever *adds* missing tables; it can't alter an existing column or track
history. Alembic is the standard replacement:

```bash
pip install alembic
alembic init alembic
# edit alembic/env.py: target_metadata = SQLModel.metadata
alembic revision --autogenerate -m "create item table"
alembic upgrade head
```

`alembic upgrade head` is idiomatically run as a **separate deploy step**
(entrypoint script, CI/CD step, or `docker-compose` init container) before
the app starts — not usually called from inside `on_startup()`, though it's
possible via `alembic.command.upgrade(cfg, "head")` if you want it
in-process.

**Node/Express + Drizzle → `drizzle-kit generate` + `migrate()`.** Unlike
Alembic, Drizzle's migrator is designed to run in-process, directly in your
app's startup path:

```bash
npx drizzle-kit generate   # writes versioned SQL into ./drizzle
```

```ts
// server.ts, before app.listen(...)
import { migrate } from "drizzle-orm/better-sqlite3/migrator";
import { db } from "./db/client";

migrate(db, { migrationsFolder: "./drizzle" });
```

That call is idempotent — safe to run on every boot, only applies
migrations that haven't been applied yet. This is the one place the two
stacks' idiomatic patterns actually diverge: Alembic leans toward a
deploy-time CLI step, Drizzle's migrator leans toward an app-startup call.
