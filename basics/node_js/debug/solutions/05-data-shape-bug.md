# 05 — Data Shape Bug

**Simple variant (off-by-one batching):**

**Symptom:** `batches.flat().length` prints less than 10 — items are
silently dropped.

**Root cause:** `items.slice(i, i + batchSize - 1)` — the `- 1` shrinks
every batch's upper bound by one, so each batch only ever contains
`batchSize - 1` items even though the loop still advances `i` by the full
`batchSize`. One item per batch boundary is skipped and never appears in
any batch.

**Fix:** drop the `- 1` — `items.slice(i, i + batchSize)`.

---

**Docker variant (malformed/partial LLM output):**

**Symptom:** crashes with `TypeError: Cannot destructure property 'parties'
of 'undefined'` (or similar) partway through `processExtractions`.

**Root cause:** `doc-3`'s extraction result has no `parties` field at all
(simulating a partial/malformed LLM JSON response). `summarizeParties`
unconditionally destructures `extraction.parties` with
`const [primary, ...others] = extraction.parties`, which throws when
`parties` is `undefined`.

**How to diagnose:** the stack trace itself points at the exact
destructuring line; stepping through with the debugger and inspecting
`extraction` in the Variables pane on the failing iteration shows the
missing field directly.

**Fix:** validate/default the shape before destructuring — e.g.
`const parties = extraction.parties ?? [];` — or use
`Promise.allSettled`-style error isolation per document (in a real
pipeline) so one malformed extraction doesn't take down the whole batch,
same lesson as scenario 02.
