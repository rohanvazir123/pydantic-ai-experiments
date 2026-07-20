# 01 — Memory Leak

**Symptom:** `heapUsed` climbs steadily every tick and never drops or levels off.

**Root cause:** `requestLog` (simple) / `processedDocs` (docker) is an
unbounded collection — every request/document appends a large payload
(`Buffer.alloc(1_000_000)` or a 500 KB string) and nothing ever removes old
entries. Classic unbounded cache / accumulating-array leak.

**How to diagnose:**
- Watch `process.memoryUsage().heapUsed` over time (already logged) — steady
  linear growth with no plateau is the signature.
- With `--inspect` running, open `chrome://inspect` → Memory tab → take two
  heap snapshots a few seconds apart → "Comparison" view shows which
  constructor/array is retaining the most new objects.
- In the Call Stack / Scope panes while paused, note that `requestLog` /
  `processedDocs` is captured by the interval's closure and kept alive for
  the life of the process — nothing ever calls `.shift()`, `.delete()`, or
  sets a TTL/max-size eviction.

**Fix:** cap the collection (LRU with a max size, e.g. evict oldest when
`size > N`), or use a `Map` with TTL-based eviction, or don't retain full
payloads at all if they're not needed after processing.
