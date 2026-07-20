# Debugging Practice Drills

Five deliberately buggy scenarios for live debugging practice — each in two
forms:

- **`simple/`** — plain Node scripts, no Docker. Fastest way to iterate.
- **`docker/`** — same category of bug, framed closer to a real service,
  run the same way you'd debug a container in the actual interview
  (mirrors `basics/fastify`'s existing `p4.js` setup).

Solutions are in `solutions/` — **don't open them until you've formed a
hypothesis or timed out.** Treat each one as a mini version of the 60-min
format: reproduce → isolate → diagnose → fix, then check yourself against
the write-up.

| # | Scenario | Category |
|---|----------|----------|
| 01 | `memory-leak` | Memory leak |
| 02 | `unhandled-rejection` | Uncaught exception / unhandled rejection |
| 03 | `event-loop-blocking` | Event loop blocking |
| 04 | `async-bug` | Async/await misuse |
| 05 | `data-shape-bug` | Malformed/unexpected data shape |

## Running `simple/`

```bash
cd basics/fastify/debug/simple
node --inspect-brk 01-memory-leak.js
```

In VS Code: open the file, select **"Launch Current File"** in the Run &
Debug dropdown, press F5. No Docker, no port mapping needed — VS Code
starts the process itself under the debugger.

## Running `docker/`

Each scenario folder is self-contained, same pattern as `p4.js`:

```bash
cd basics/fastify/debug/docker/01-memory-leak
docker compose up --build
```

In VS Code's Run & Debug dropdown, pick the matching config —
**"Attach: Debug 01 Memory Leak"**, **"Attach: Debug 02 Unhandled
Rejection"**, etc. (already wired up in `.vscode/launch.json` with the
right `localRoot`/`remoteRoot` for each folder) — then press F5. It'll
pause on entry (`--inspect-brk`); continue from there.

Only run one scenario's container at a time — they all use port 9229.

## Suggested approach per scenario

1. Run it once without the debugger, just to see the (wrong) output.
2. Form a hypothesis for what category of bug it is before opening the
   debugger — you'll usually already suspect one of the five categories
   above from the symptom alone.
3. Set breakpoints / step through to confirm.
4. Fix it, rerun, confirm the output is now correct.
5. Check `solutions/NN-*.md` against your own diagnosis.
