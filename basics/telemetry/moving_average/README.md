# Streaming Moving Average (last N seconds)

A data structure that reports the mean of a telemetry metric over a sliding
**time** window, under high throughput and **out-of-order** arrivals. Reads are
O(1); the window sum is memoised and only samples that age out are subtracted.

## Table of Contents

- [Files](#files)
- [Why the fixed file exists](#why-the-fixed-file-exists)
- [How It Works](#how-it-works)
- [Out-of-order arrivals & data-structure choice](#out-of-order-arrivals--data-structure-choice)
- [Complexity](#complexity)
- [Running the Demo](#running-the-demo)
- [Running the Tests](#running-the-tests)

## Files

| File | Purpose |
|------|---------|
| `moving_average_fixed.py` | The implementation — reviewed, corrected, documented. |
| `../tests/test_moving_average_fixed.py` | Pytest suite (fake clock, no `sleep`, no services). |

## Why this file reads as a correction

It supersedes an earlier `rolling_avg.py` draft (since removed). The bugs it
fixed are kept on the record here, because each one is a design lesson that the
[telemetry architecture notes](../README.md) build on:

1. **Stale reads when the metric goes quiet.** The draft only evicted expired
   points inside `add_batch`. If no new data arrived, `get_moving_average` kept
   dividing by points that had already aged out of the window — the "last N
   seconds" contract silently broke. Fixed: **every read evicts first**
   (`get_moving_average` and `window_count`), so the answer is correct even with
   no new data.
2. **Wall clock for windowing.** `time.time()` can step backwards (NTP), which
   corrupts the cutoff. Fixed: default clock is `time.monotonic`, and both the
   clock and an explicit `now` are injectable for deterministic tests.
3. **O(N log N) full re-sort every batch.** The draft called `buffer.sort()` on
   the entire buffer on *every* insert. Fixed: O(k) `extend` on the in-order fast
   path, and `list.sort()` **only when a batch overlaps existing data** — Timsort
   is adaptive, so merging the two already-sorted runs is O(N + k) — see below.
4. **Duplicated count.** The draft tracked `running_count` separately from the
   buffer, so the two could drift apart. Fixed: count is just `len(buffer)`;
   only the **sum** is memoised.
5. **Concurrency contract made explicit (no lock, by design).** The draft's
   ambiguity was "is this safe to share?" This is **single-owner**: one thread,
   or (typically) one `asyncio` loop — the methods are synchronous with no
   `await` inside, so a single loop serialises them for free, and **no lock is
   needed**. We deliberately avoid a `threading.Lock`: mixing one with asyncio
   invites deadlocks (a lock held across an `await` freezes the one loop thread),
   and the GIL is no substitute (it guards CPython internals, not compound ops
   like `running_sum += x`). Share across OS threads → guard externally; need to
   protect a section that spans `await`s → use `asyncio.Lock`. `resync()`
   re-derives the sum with `math.fsum` to clear float drift over long runs.
6. **Loose types.** `dict` points → a typed `Sample` `NamedTuple`.

## How It Works

State is a **buffer sorted ascending by timestamp** plus a memoised
`running_sum`. On `add_batch`:

1. Add the batch's values to `running_sum` (via `math.fsum` for the batch).
2. Keep the buffer sorted — always `extend` (O(k)); **only if** the batch starts
   before the previous tail (out of order) call `list.sort()`. That leaves the
   buffer as two already-sorted runs, which **Timsort** merges in O(N + k), in C
   and in place. `Sample` is a `NamedTuple`, so it sorts by `timestamp` first with
   no `key=`. In-order telemetry never sorts.
3. **Evict from the left** every sample with `timestamp < now - N`, subtracting
   its value from `running_sum`.

On `get_moving_average` / `window_count` it evicts first, then returns
`running_sum / len(buffer)` (or `0.0` for an empty window). Because the sum is
memoised, the read itself is O(1).

`resync()` recomputes the sum exactly from the buffer — call it periodically on
long-lived instances to reset accumulated float error.

```python
from moving_average_fixed import Sample, TelemetryRollingAverage

avg = TelemetryRollingAverage(max_age_seconds=5.0)   # defaults to time.monotonic
avg.add(Sample(timestamp=time.monotonic(), value=42.0))
avg.add_batch([Sample(t0, v0), Sample(t1, v1)])      # batch sorted ascending
print(avg.get_moving_average())
```

## Out-of-order arrivals & data-structure choice

Telemetry from many devices/links arrives **slightly out of order**, so the
buffer can't assume monotonic timestamps. This implementation stays sorted with
`extend` + an adaptive `list.sort()` (Timsort) only when a batch overlaps.
Options considered:

| Structure | Out-of-order insert | Front eviction | Read sum | Notes |
|-----------|--------------------|----------------|----------|-------|
| **`list` + `extend`/`sort`** (chosen) | O(N + k) Timsort (only when overlapping) | O(N) slice-del | O(1) memoised | stdlib, dead simple; in-order fast path is O(k), no sort |
| `list` + `heapq.merge` | O(N + k) but rebuilds a new list, pure-Python generator | O(N) slice-del | O(1) memoised | same big-O, slower constant + extra alloc than in-place `sort` |
| `collections.deque` | O(N) (no middle insert) | **O(1)** popleft | O(1) | great eviction, but poor for reordering |
| `sortedcontainers.SortedList` | **O(log N)** per insert | O(√N) prefix del | O(1) memoised | cleanest for heavy reordering, but a third-party dep |

**Chosen: plain `list` with `extend` + adaptive `sort`.** Timsort detects the two
already-sorted runs (existing buffer + new batch) and merges them in O(N + k) —
in C, in place — so it beats a pure-Python `heapq.merge` on constants and
allocation while being simpler. It exploits the reality that telemetry is
*mostly* in order: the common case is an O(k) `extend` with **no sort at all**;
only a genuinely late batch triggers the O(N + k) sort. If reordering were
frequent *and* buffers large, `SortedList` (or `SortedKeyList` keyed on
timestamp) is the right upgrade — O(log N) inserts with no full rescan and an
efficient `irange`/prefix-delete for eviction — left out only to avoid the
dependency. A `deque` wins on eviction cost but can't absorb out-of-order
inserts, so it's the wrong fit here.

## Complexity

| Operation | This implementation |
|-----------|--------------------|
| `add_batch` (in order) | O(k) |
| `add_batch` (out of order) | O(N + k) |
| `get_moving_average` / `window_count` | O(1) + O(evicted) |
| Memory | O(samples currently in the window) |

## Running the Demo

```bash
.venv/bin/python basics/telemetry/moving_average/moving_average_fixed.py
```

Feeds a burst of readings into a 5-second window and prints the count and
average each tick, then shows both drop to zero after a quiet period (the
read-eviction fix in action).

## Running the Tests

No services needed — the suite uses an injected fake clock, so eviction and
windowing are deterministic without `sleep`. From the repo root:

```bash
.venv/bin/python -m pytest basics/telemetry/tests/test_moving_average_fixed.py -v
```

Dependencies: `pytest` only.
