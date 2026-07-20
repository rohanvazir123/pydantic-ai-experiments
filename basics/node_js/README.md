# Node.js / Async Practice Notes

Scratch examples while practicing Node.js debugging (VS Code breakpoints, step
over/into/out, Watch expressions) ahead of a debugging-focused interview round.

## Table of Contents

- [Promises settle via the microtask queue, not inline](#promises-settle-via-the-microtask-queue-not-inline)
- [Queue priority: `process.nextTick` > Promise microtasks > `setTimeout`](#queue-priority-processnexttick--promise-microtasks--settimeout)
- [`await` — sugar over `.then()`, not a different mechanism](#await--sugar-over-then-not-a-different-mechanism)
- [Multiple promises at once: sequential vs concurrent](#multiple-promises-at-once-sequential-vs-concurrent)
  - [vs. Python `asyncio` — the real difference isn't "tasks", it's eagerness](#vs-python-asyncio--the-real-difference-isnt-tasks-its-eagerness)
  - [Why Python code never shows a "Promise"](#why-python-code-never-shows-a-promise)
- [Generators (`function*` / `yield`)](#generators-function--yield)
- [Node.js vs Express — and vs FastAPI](#nodejs-vs-express--and-vs-fastapi)

## Promises settle via the microtask queue, not inline

`.then()` callbacks never run synchronously, even for an already-resolved promise —
they're queued as microtasks and run after the current synchronous code finishes.

```js
console.log("1");
Promise.resolve("x").then((v) => console.log("2:", v));
console.log("3");
// logs: 1, 3, 2: x
```

## Queue priority: `process.nextTick` > Promise microtasks > `setTimeout`

Synchronous code always finishes first — everything below is queued, not
executed, when first called. After that, Node drains queues in a fixed
priority order, **not** insertion order:

```js
console.log("Start");

process.nextTick(() => {
  console.log("Next Tick");
});
Promise.resolve().then(() => {
  console.log("Promise");
});
setTimeout(() => {
  console.log("Timeout");
}, 0);

console.log("End");
// logs: Start, End, Next Tick, Promise, Timeout
```

| | `process.nextTick(fn)` | `Promise.resolve().then(fn)` | `setTimeout(fn, 0)` |
|---|---|---|---|
| Queue | Node's own `nextTick` queue | Standard JS microtask queue | Timer (macrotask) phase |
| Drains | Immediately after the current operation, before any other queue | After `nextTick` queue is fully empty, before macrotasks | Only after *all* microtasks (both queues above) are empty |
| Portable to browsers? | No — Node-only | Yes — standard JS | Yes |

`nextTick` isn't part of the ECMAScript microtask spec — it's a Node-specific
queue that runs even before promise callbacks. `async`/`await` continuations
land in the same microtask queue as `.then()`, so they follow the same
"after nextTick, before timers" rule.

## `await` — sugar over `.then()`, not a different mechanism

`await expr` pauses the enclosing `async` function until `expr` (a promise) settles,
then either returns the resolved value or throws the rejection — but it does this
by suspending and resuming the function, not by blocking the thread. Other code
(other timers, other requests) keeps running while an `await` is suspended.

```js
async function run() {
  console.log("1");
  const v = await Promise.resolve("x"); // suspends here, yields to microtask queue
  console.log("2:", v);
}
run();
console.log("3");
// logs: 1, 3, 2: x  (identical ordering to the .then() version above)
```

`await` is exactly equivalent to chaining `.then()` — same microtask timing, same
non-blocking behavior. The difference is only readability: `await` lets you write
async code that *looks* synchronous (top-to-bottom, no nested callbacks), while the
engine still yields control at every `await` point.

Errors: a rejected awaited promise throws inside the `async` function, so plain
`try/catch` works instead of `.catch()`:

```js
async function run() {
  try {
    const data = await fetch(url).then((r) => r.json());
  } catch (err) {
    console.error(err);
  }
}
```

## Multiple promises at once: sequential vs concurrent

Awaiting one at a time runs them **sequentially**, even if they're independent:

```js
const a = await fetch(url1); // waits full round-trip
const b = await fetch(url2); // only starts AFTER a finishes
```

Start both first, *then* await — total time becomes `max(a, b)` instead of `a + b`:

```js
const p1 = fetch(url1); // starts immediately
const p2 = fetch(url2); // also starts immediately, in parallel
const [a, b] = await Promise.all([p1, p2]);
```

The trick: calling `fetch()` starts the request right away — `await` is only where
you pause to wait for the result. Separating "start" from "await" is what unlocks
concurrency.

Combinators for different needs:

| Combinator | Waits for | On failure |
|---|---|---|
| `Promise.all([...])` | all to fulfill | rejects on the **first** rejection (others keep running, results discarded) |
| `Promise.allSettled([...])` | all to settle | never rejects — returns `{status, value\|reason}` per item |
| `Promise.race([...])` | first to settle, win or lose | resolves/rejects with whichever settles first |
| `Promise.any([...])` | first to **fulfill** | ignores rejections; rejects (`AggregateError`) only if *all* reject |

### vs. Python `asyncio` — the real difference isn't "tasks", it's eagerness

JS Promises are **hot**: calling `fetch(url)` (or any async function) starts running
immediately, synchronously, up to its first `await`/suspension point. That's *why*
`Promise.all([fetch(url1), fetch(url2)])` gets concurrency for free — both requests
already started the instant each `fetch()` was called; `Promise.all` just aggregates.

Python coroutines are **cold**: calling `coro_fn()` doesn't run any code — it just
creates a coroutine object that sits inert until something schedules it (`await`,
`asyncio.create_task()`, or `asyncio.gather()`). `asyncio.gather(coro1(), coro2())`
looks like the direct equivalent of `Promise.all`, and behaviorally it is — but only
because `gather` implicitly wraps each argument in a `Task` for you. Without that
wrapping (e.g. plain sequential `await coro1(); await coro2()`), Python coroutines
run one at a time same as sequential `await` in JS — the coroutine doesn't run
ahead just because you "created" it.

### Why Python code never shows a "Promise"

It exists — `asyncio.Future` — Python just hides it. `async def` returns a plain
coroutine object, not a Future; only once the event loop wraps it in a **`Task`**
(a `Future` subclass) does it become the schedulable, stateful thing a JS Promise
is. `await`/`create_task`/`gather` do that wrapping for you, so app code never
touches `.set_result()` / `.add_done_callback()` directly — unlike JS, where
libraries hand you the Promise object itself (`fetch()` returns one).

## Generators (`function*` / `yield`)

Calling a generator function doesn't run its body — it returns a paused Generator
object; each `.next()` resumes it until the next `yield`.

`yield*` delegates to another generator, splicing its yields into the outer one:

```js
function* anotherGenerator(i) {
  yield i + 1;
  yield i + 2;
  yield i + 3;
}

function* generator(i) {
  yield i;
  yield* anotherGenerator(i);
  yield i + 10;
}

const gen = generator(10);
// gen.next() in sequence yields: 10, 11, 12, 13, 20, then { done: true }
```

The outer generator pauses at `yield* anotherGenerator(i)` until the inner one is
fully exhausted (all 3 of its yields consumed) before moving to `yield i + 10`.
Same idea as Python's `yield from`.

## Node.js vs Express — and vs FastAPI

Node.js is the JavaScript runtime itself (lets you run JS outside a browser —
`require`, the file system, networking, etc). Express is a separate library that
runs *on top of* Node to make building web servers easier — routing (`app.get`,
`app.post`), middleware (`app.use`), request/response helpers. You could write a
server with Node's built-in `http` module alone (manual URL parsing, manual body
parsing, much more boilerplate), but Express handles that for you.

Mapping to FastAPI, for transferring existing mental models:

| FastAPI (Python) | Express (Node.js) |
|---|---|
| `uvicorn` running your app | Node.js runtime running `app.js` |
| `FastAPI()` instance | `express()` instance (`app`) |
| `@app.get("/users")` | `app.get('/users', handler)` or `router.get('/', handler)` mounted via `app.use('/users', router)` |
| `APIRouter()` | `express.Router()` — same idea |
| Pydantic body validation (automatic) | `express.json()` middleware just parses JSON into `req.body` — **no validation**, you get whatever was sent (or `undefined` if missing/wrong `Content-Type`) |
| Dependency injection (`Depends(...)`) | Middleware chain via `app.use((req, res, next) => {...})` — used to stash things like a DB handle onto the request object, similar to injecting a DB session as a dependency |
| `async def` route handlers | `async (req, res) => {}` handlers — same async/await, but you must call `res.send()`/`res.json()` yourself; nothing is returned implicitly like a FastAPI response model |
| Path params `{username}` | `:username` in the route string (`/:username/record/:period`) |
| Uvicorn's built-in request validation errors | You write your own — no schema layer by default; bad/missing input (wrong method, missing header, wrong param) just flows through as `undefined`/`null` until it breaks something downstream |

The biggest practical difference for debugging: FastAPI/Pydantic rejects bad
input before your handler ever runs. Express does nothing for you — silent
`undefined` propagation is the default failure mode, not a loud validation error.

