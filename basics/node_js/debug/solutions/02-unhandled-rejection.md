# 02 — Unhandled Rejection

**Symptom:** the process exits (or crashes) partway through the batch/queue
with an `UnhandledPromiseRejection` — "Batch complete" (or "All jobs
complete") never prints, and no other jobs after the bad one run either.

**Root cause:** `fetchRecord(3)` / `callLLM` on the broken job returns a
rejected promise. It's `await`ed inside a plain `for...of` loop with no
`try/catch`, and the outer call (`processBatch(...).then(...)` /
`runWorker()`) has no `.catch()` either. One bad record crashes the whole
batch instead of being handled and the rest continuing.

**How to diagnose:**
- Step through with the debugger — the pause happens at the `await
  fetchRecord(id)` / `await callLLM(job)` line on the failing id, and
  stepping over throws instead of returning a value.
- Node's own stderr output names the unhandled rejection and (Node 15+)
  the exit code — that's the fastest signal before you even open the
  debugger.

**Fix:** wrap the per-item work in `try/catch` inside the loop so one
failure doesn't kill the batch — log/collect the error and continue (or use
`Promise.allSettled` if processing concurrently instead of sequentially).
Also add `process.on('unhandledRejection', ...)` / `process.on
('uncaughtException', ...)` at the process level as a last-resort safety
net — log and exit cleanly, never try to resume from those handlers.
