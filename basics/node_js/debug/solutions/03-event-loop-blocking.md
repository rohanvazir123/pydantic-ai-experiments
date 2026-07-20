# 03 — Event Loop Blocking

**Symptom:** the heartbeat/health tick (expected every 200ms) stalls for a
noticeable gap right when the "heavy job" / batch parse kicks in, then
resumes.

**Root cause:** `heavyJob()` / `parseIncomingBatch()` does a large amount of
synchronous, CPU-bound work (a tight numeric loop, or building + JSON-
parsing a huge string) on the main thread. Nothing can run — timers,
I/O callbacks, other requests — until that synchronous call returns.

**How to diagnose:**
- The gap in heartbeat timestamps is visible just from the console output
  (compare the timestamp delta across the gap vs. the normal ~200ms).
- Pause with the debugger and use "step over" through `heavyJob`/
  `parseIncomingBatch` — the debugger itself will feel "stuck" for the
  duration, since nothing else in the process can run either.
- In production you'd reach for `--prof` / the CPU profiler in Chrome
  DevTools to see one long, unbroken stack frame instead of the usual
  short bursts.

**Fix:** move the CPU-bound work off the main thread with `worker_threads`,
or break it into async chunks (e.g. `setImmediate`/batches) that yield back
to the event loop periodically, or replace sync APIs (`JSON.parse` on a
huge blob built synchronously) with a streaming/incremental approach.
