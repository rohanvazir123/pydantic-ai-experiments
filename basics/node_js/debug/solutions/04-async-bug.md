# 04 — Async / forEach Bug

**Symptom:** "Summary" (simple) / "Classified N of 4" (docker) prints
immediately with an empty or incomplete array, even though each item does
eventually get processed (its own "processed:" log lines print later,
after the summary).

**Root cause:** `Array.prototype.forEach` does not await its callback —
it fires all the async callbacks and returns immediately, regardless of
whether they're `async` functions. `processAll`/`classifyBatch` reads
`results`/`classified` (or returns it) before any of the forEach callbacks
have finished pushing to it.

**How to diagnose:**
- Step into the `forEach` call — note the debugger returns to the line
  right after `forEach` immediately, without waiting, while the async
  callbacks are still pending (visible as separate stack frames appearing
  later, out of the original call order).
- Add a breakpoint on the `console.log("Summary"...)` / `return classified`
  line and inspect the array in the Variables pane — it'll be empty or
  short, then watch it fill in *after* you've already stepped past it.

**Fix:** use `Promise.all(items.map(async (item) => ...))` instead of
`forEach`, so the outer function actually awaits all the promises before
reading/returning the results array. Prefer `Promise.allSettled` if
partial failures should not abort the whole batch.
