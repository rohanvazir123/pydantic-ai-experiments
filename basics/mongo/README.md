# MongoDB Python Reference

A companion guide to `basic_mongo.py`. Covers design decisions, gotchas, and
conceptual depth for every major topic in the reference script.

---

## Table of Contents

1. [PyMongo vs Motor vs MongoEngine](#1-pymongo-vs-motor-vs-mongoengine)
2. [Document Model: Embedding vs Referencing](#2-document-model-embedding-vs-referencing)
3. [BSON Types and Python Mapping](#3-bson-types-and-python-mapping)
4. [Aggregation Pipeline](#4-aggregation-pipeline)
5. [Index Types](#5-index-types)
6. [Transactions](#6-transactions)
7. [Write Concerns and Read Concerns](#7-write-concerns-and-read-concerns)
8. [Change Streams](#8-change-streams)
9. [$lookup vs Embedding](#9-lookup-vs-embedding)
10. [Pagination](#10-pagination)
11. [Common Gotchas](#11-common-gotchas)
12. [Schema Design Principles](#12-schema-design-principles)
13. [The 16 MB BSON Document Limit](#13-the-16-mb-bson-document-limit)
14. [Normalized Query Performance Trap](#14-normalized-query-performance-trap)
15. [Write Sharding Strategies](#15-write-sharding-strategies)
16. [Distributed Transactions](#16-distributed-transactions)
17. [Locking Patterns](#17-locking-patterns)
18. [FAQ](#18-faq)

---

## 1. PyMongo vs Motor vs MongoEngine

### When to use each

| Library | Best for |
|---|---|
| **PyMongo** | Synchronous scripts, CLI tools, batch jobs, Flask/Django apps without async |
| **Motor** | FastAPI / asyncio applications; anything using `async def` handlers |
| **MongoEngine** | Teams coming from Django ORM; projects that want model-level validation and signals out of the box |

**PyMongo** is the official low-level driver. Motor wraps PyMongo using a thread pool + futures
to provide a fully async API. MongoEngine is an ODM (Object-Document Mapper) built on top of
PyMongo — it adds class-based document definitions, field validation, reference resolution,
and queryset chaining.

### Trade-offs

- **MongoEngine** adds indirection: raw aggregation requires `__raw__` passthrough, and
  ODM abstraction can hide performance problems.
- **Motor** adds minimal overhead but requires your entire call stack to be async — mixing
  sync code requires `run_sync` helpers or a separate thread pool.
- **PyMongo** gives you the most control and the least magic. Prefer it unless you have a
  specific reason not to.

---

## 2. Document Model: Embedding vs Referencing

### The 16 MB limit

A single MongoDB document (BSON) cannot exceed **16 MB**. This rarely matters for scalar
data but becomes relevant when embedding large arrays (e.g., storing every comment inside
a post document). When an unbounded array could grow large, use a separate collection with a
reference.

### Embedding — denormalization

Store related data inside the parent document.

```python
# Post with embedded comment counts (not the comments themselves)
{"_id": post_id, "title": "...", "comment_count": 42, "top_comment": {"author": "bob", "text": "Great!"}}
```

**Use when:**
- Data is always queried together (one read = complete result).
- The embedded data is bounded in size (e.g., at most 5 tags, not 50 000 comments).
- The embedded data has no independent lifecycle.

### Referencing — normalization

Store the related document in its own collection and hold its `_id`.

```python
# Comment references post by _id
{"_id": comment_id, "post_id": post_id, "author": "bob", "text": "..."}
```

**Use when:**
- Related data is large or unbounded (comments, events, log entries).
- The child is accessed independently.
- Many-to-many relationships.

### Rule of thumb

Embed for "has-one" or small "has-few". Reference for "has-many" beyond a handful or when
the child has its own lifecycle.

---

## 3. BSON Types and Python Mapping

| BSON type | Python type | Notes |
|---|---|---|
| ObjectId | `bson.ObjectId` | 12-byte auto-generated `_id`; not a plain string |
| String | `str` | UTF-8 |
| Int32 | `int` | PyMongo uses Python `int` for both Int32 and Int64 by default |
| Int64 | `bson.Int64` | Force 64-bit with `bson.Int64(n)` |
| Double | `float` | |
| Decimal128 | `bson.Decimal128` | Use for money; `float` loses precision |
| Boolean | `bool` | |
| Date | `datetime.datetime` | **Always use UTC-aware datetimes** |
| Array | `list` | |
| Document | `dict` | |
| Null | `None` | |
| Binary | `bytes` / `bson.Binary` | |
| Regex | `bson.Regex` | For storing regex in documents; queries can use plain Python `re` objects |
| Timestamp | `bson.Timestamp` | Internal MongoDB use; use Date for application timestamps |

### Key pitfall: naive datetimes

MongoDB stores dates in UTC. If you pass a timezone-naive `datetime` object, PyMongo
treats it as UTC but does not convert it. Store `datetime.now(timezone.utc)` consistently.

```python
# Bad — ambiguous
{"created_at": datetime.now()}

# Good
from datetime import datetime, timezone
{"created_at": datetime.now(timezone.utc)}
```

---

## 4. Aggregation Pipeline

### Stage execution order matters

Each stage receives the output of the previous one. The order has correctness and
performance implications:

1. **Put `$match` first** — filter as early as possible so subsequent stages process fewer
   documents. A `$match` at the start can use an index; the same filter after `$group` cannot.
2. **Put `$project` / `$addFields` before heavy stages** — reducing document size early lowers
   memory use.
3. **`$sort` before `$group`** is rarely needed and expensive; `$group` does not preserve order.
4. **`$limit` after `$sort`** is a common pattern and MongoDB optimises it into a "top-k" heap.

### Pipeline optimisation tips

- **Use covered indexes**: `$match` on an indexed field avoids collection scans.
- **Avoid `$unwind` on large arrays early** — it multiplies document count. Filter first.
- **`$lookup` on unindexed foreign fields** is a full collection scan per input document.
  Always index the `foreignField`.
- **`$facet` is a fan-out** — all sub-pipelines run on the full input set; make sure `$match`
  before `$facet` reduces the input sufficiently.
- **`allowDiskUse: True`** (PyMongo: `posts.aggregate(pipeline, allowDiskUse=True)`) is
  needed when intermediate results exceed 100 MB of RAM.
- **Avoid `$where`** — it runs JavaScript per document and cannot use indexes.

### Useful stage cheat sheet

| Stage | Purpose |
|---|---|
| `$match` | Filter documents (use early, uses indexes) |
| `$group` | Aggregate with accumulators (`$sum`, `$avg`, `$push`, `$addToSet`) |
| `$project` | Reshape / include / exclude fields |
| `$addFields` | Add computed fields, keep existing |
| `$replaceRoot` | Promote a sub-document to root |
| `$unwind` | Deconstruct an array into one doc per element |
| `$lookup` | Left outer join from another collection |
| `$sort` | Sort (can use index only as first stage) |
| `$limit` / `$skip` | Pagination |
| `$count` | Emit single doc with count |
| `$sample` | Random subset |
| `$facet` | Multiple sub-pipelines in one pass |
| `$bucket` / `$bucketAuto` | Histogram / auto-histogram |
| `$graphLookup` | Recursive traversal |
| `$setWindowFields` | Window functions (5.0+) |
| `$merge` / `$out` | Write results to a collection |

---

## 5. Index Types

### Single-field

```python
collection.create_index("email", unique=True)
```

General-purpose. Supports equality, range, and sort. The default `_id` index is
single-field unique.

### Compound

```python
collection.create_index([("author_id", 1), ("published_at", -1)])
```

Supports prefix queries — the above index also helps queries on `author_id` alone.
Field order matters for sorting: the index satisfies `sort(author_id, published_at DESC)` but
not `sort(published_at, author_id)`.

### Text

```python
collection.create_index([("title", TEXT), ("body", TEXT)], weights={"title": 10})
```

- Only **one text index per collection**.
- Use `$text` operator for full-text queries; `$regex` does not use the text index.
- Supports language-aware stemming, stop words, and weighted scoring.

### When text vs regex

- **Text index** — searching for words or phrases; relevance ranking needed.
- **Regex** — pattern matching (`^prefix`, `@domain.com$`); use a regular index on the field if
  queries always use a leading-anchor regex (`^`).

### TTL

```python
collection.create_index("expires_at", expireAfterSeconds=0)
```

MongoDB runs a background job every 60 seconds to delete expired documents. Not
real-time — documents may live up to ~60 s past their expiry time. Use for sessions,
tokens, and temporary data. The field must be a `Date` type or an array of `Date`s.

### Sparse

Only indexes documents where the field exists. Saves space when the field is optional.
Sparse indexes cannot be used for queries that must return documents lacking the field
(those documents are invisible to the index).

### Partial

```python
collection.create_index("likes", partialFilterExpression={"published": True})
```

More powerful than sparse — indexes documents matching an arbitrary filter. Smaller than
a full index; saves I/O. MongoDB only uses it when the query predicate implies the filter
(e.g., `find({"published": True, "likes": {"$gt": 100}})` uses the index above).

### 2dsphere

Required for GeoJSON queries (`$near`, `$geoWithin`, `$geoIntersects`). Stores points,
lines, and polygons. The field must contain a valid GeoJSON object or legacy coordinate pair.

---

## 6. Transactions

### Requirements

- **Replica set required.** Transactions are not available on standalone `mongod` instances.
  A single-node replica set (`rs.initiate()` with one member) satisfies the requirement.
- **Sharded clusters** support transactions from MongoDB 4.2, but cross-shard transactions
  have higher overhead.

### Limitations

| Constraint | Detail |
|---|---|
| **Oplog size** | A single transaction's oplog entry cannot exceed 16 MB. Large bulk writes should be batched outside a transaction. |
| **Default timeout** | 60 seconds (`transactionLifetimeLimitSeconds`). Transactions running longer are automatically aborted. |
| **Performance cost** | Locks are held for the duration; long transactions block concurrent writers. Keep transactions short. |
| **No DDL inside transactions** | You cannot create collections or indexes inside a transaction (pre-7.0). |
| **Read your own writes** | Within a session, you see your own uncommitted writes. Other sessions do not. |

### `with_transaction()` vs manual

Prefer `with_transaction(callback)` — it automatically retries the callback on transient
errors (e.g., `TransientTransactionError`) and handles commit with retry logic. Use manual
`start_transaction` / `commit` / `abort` only when you need custom retry logic.

---

## 7. Write Concerns and Read Concerns

### Write concerns (`w=`)

| `w` value | Meaning | Risk |
|---|---|---|
| `0` | Fire-and-forget — no acknowledgement | Data loss if mongod crashes before flush |
| `1` (default) | Primary acknowledges after writing to memory | Data loss on failover before replication |
| `"majority"` | Majority of voting replica set members acknowledge | Durable after failover; slight latency increase |

`j=True` additionally waits for the primary to flush to its journal (WAL) before
acknowledging — adds ~1-10 ms but protects against primary crash before memory flush.

`wtimeout` (milliseconds) causes `WriteConcernError` if acknowledgement is not received in
time — the write may still have succeeded.

### Read concerns

| Level | Meaning |
|---|---|
| `local` (default) | Read whatever the node has; may include data not yet replicated |
| `available` | Same as `local` but used with sharding (may read orphaned chunks) |
| `majority` | Read only data acknowledged by a majority — never rolled back |
| `linearizable` | Strongest — waits to confirm no newer write exists; high latency, single-doc queries only |
| `snapshot` | Used inside transactions — consistent point-in-time view |

For most applications `w=1, readConcern=local` is fine. Use `w=majority, readConcern=majority`
for financial or critical data where durability matters more than latency.

---

## 8. Change Streams

### Delivery guarantee

Change streams provide **at-least-once delivery** — network failures or restarts may replay
an event. Your consumer must be idempotent. Use the `resume_after` token to re-open the
stream from where you left off.

```python
# Persist token to durable storage
resume_token = change["_id"]

# Re-open after restart
collection.watch(resume_after=resume_token)
```

### Requirements

- Replica set or sharded cluster (same as transactions).
- The watched collection must use the WiredTiger storage engine.

### Use cases

- Cache invalidation
- Real-time dashboards
- Audit logs / event sourcing
- Syncing MongoDB data to Elasticsearch or other stores

### `full_document="updateLookup"`

By default, update events only contain the diff (`updateDescription`). Setting
`full_document="updateLookup"` fetches the full document at the time of the event.
Be aware: there is a small window where a subsequent write can make the fetched document
not exactly match the state at the time of the original update.

---

## 9. $lookup vs Embedding

### When to use `$lookup`

- Data is normalised (references) and you need it joined at query time.
- The joined collection is accessed independently far more often than it is joined.
- The joined data is large and does not make sense to duplicate everywhere.

### When to embed instead

- You always need the related data with the parent.
- The related data is small and bounded.
- Read performance is critical and you want to avoid an extra collection scan.

### `$lookup` performance tips

- **Index the `foreignField`** — without an index, `$lookup` does a full collection scan
  for each input document: `O(n × m)`.
- Use pipeline `$lookup` to filter and project on the foreign side before joining,
  reducing data transferred.
- For very high-cardinality joins at scale, consider pre-joining (materialised view via
  `$merge`) during ingestion rather than at query time.

---

## 10. Pagination

### skip + limit (offset pagination)

```python
page = collection.find(query).sort("created_at", -1).skip(page * page_size).limit(page_size)
```

**Simple but has problems at scale:**
- `$skip` N still scans the first N documents — slow on large offsets.
- Results shift if documents are inserted/deleted between pages (phantom reads).

### Range-based cursor pagination (keyset pagination)

```python
# First page
first_page = collection.find(query).sort("created_at", -1).limit(page_size)

# Next page: use the last document's value as the cursor
last_date = first_page[-1]["created_at"]
last_id   = first_page[-1]["_id"]

next_page = collection.find({
    **query,
    "$or": [
        {"created_at": {"$lt": last_date}},
        {"created_at": last_date, "_id": {"$lt": last_id}},  # tie-break
    ]
}).sort([("created_at", -1), ("_id", -1)]).limit(page_size)
```

**Advantages:** O(log n) with an index; stable results even as documents change.
**Disadvantage:** Cannot jump to an arbitrary page number.

**Rule:** Use skip+limit for admin UIs with small collections or where random-access page
jumping is required. Use keyset pagination for user-facing infinite scroll or APIs with
large collections.

---

## 11. Common Gotchas

### ObjectId is not a string

```python
# Wrong — stores a string, not an ObjectId
collection.find_one({"author_id": "507f1f77bcf86cd799439011"})

# Right
from bson import ObjectId
collection.find_one({"author_id": ObjectId("507f1f77bcf86cd799439011")})
```

### Timezone-naive datetimes

```python
# Bad — naive datetime stored as UTC but semantically ambiguous
{"created_at": datetime.now()}

# Good — explicit UTC
{"created_at": datetime.now(timezone.utc)}
```

### Dot notation in queries (nested fields)

```python
# Query nested field with dot notation — do NOT use a dict
collection.find({"address.city": "New York"})  # correct
collection.find({"address": {"city": "New York"}})  # wrong: exact doc match
```

### $set vs replace

```python
# update_one with $set — merges, keeps other fields
collection.update_one({"_id": id}, {"$set": {"name": "Alice"}})

# replace_one — replaces the entire document (keeps _id only)
collection.replace_one({"_id": id}, {"name": "Alice"})  # all other fields gone!
```

### Missing index on $lookup foreignField

Without an index on `foreignField`, `$lookup` performs a full collection scan per input
document. Always check `explain()` and create the index.

```python
# Create index on the foreign side before running $lookup
db["users"].create_index("_id")  # _id is always indexed; use this example for other fields
db["comments"].create_index("post_id")  # index the join field
```

### Field names with dots or dollar signs

MongoDB field names cannot start with `$` or contain `.` in documents.
Use `$rename` to migrate such fields, or encode dots as unicode escapes (not recommended).

### Cursor exhaustion

A PyMongo cursor is a generator — iterating it once exhausts it.
To iterate multiple times, convert to a list first:

```python
results = list(collection.find(query))  # safe to iterate multiple times
```

### `upsert=True` creates unexpected documents

When no document matches the filter, MongoDB creates a new document from the filter +
update. Make sure the filter contains meaningful identifying fields, not just conditions.

```python
# Filter with $gt will not be stored in the new doc — only equality conditions are
collection.update_one({"likes": {"$gt": 0}}, {"$set": {"promoted": True}}, upsert=True)
# If no doc found, creates: {"promoted": True}  — no "likes" field!
```

---

## 12. Schema Design Principles

### 1. Design for your access patterns

Unlike relational databases, MongoDB schema design is driven by how the application reads
data, not by normalisation rules. Identify your most frequent and performance-critical
queries first, then design documents to support them efficiently.

### 2. Embed or reference based on relationship type and size

- One-to-one or one-to-few with bounded data → embed.
- One-to-many with unbounded or large data → reference.
- Many-to-many → reference array on one or both sides, or a junction collection.

### 3. Avoid growing arrays without bound

An array field that grows indefinitely (e.g., appending every click event) will hit the
16 MB document limit and degrade update performance (each `$push` rewrites the document).
Store high-volume events in a separate time-series collection.

### 4. Use schema validation for data integrity

MongoDB is schemaless by default, but you can enforce structure with `$jsonSchema`
validators via `collMod`. Set `validationAction: "error"` in production, `"warn"` during
migrations.

### 5. Index thoughtfully

Every index speeds up reads but slows down writes and consumes disk/RAM. Index fields that
appear in frequent `$match`, `$sort`, or `$lookup` operations. Remove unused indexes
(`db.collection.aggregate([{$indexStats:{}}])`).

### 6. Use the Bucket pattern for time-series data

Group many small documents into one bucket document per time period to reduce index size
and improve range query performance.

```python
# Instead of one doc per reading, group per hour
{"sensor": "A1", "hour": "2024-01-15T14", "readings": [23.1, 23.4, 23.2, ...]}
```

### 7. Avoid deeply nested documents

More than 2-3 levels deep makes queries and updates cumbersome (dot-notation strings
become unwieldy) and harder to index.

### 8. Use `_id` wisely

You can supply a custom `_id` if you have a natural unique key (e.g., `username`, `sku`).
Custom `_id`s can eliminate the need for a separate unique index. Avoid using `ObjectId`
as an application-level identifier if you have a better natural key.

## 13. The 16 MB BSON Document Limit

MongoDB enforces a **hard 16 MB cap on every BSON document**, including all nested
subdocuments and arrays. This is the single most disruptive design constraint in MongoDB
because the entire marketing message is "store rich, nested JSON" — and it works beautifully
until one document crosses the line, then your insert crashes with a cryptic error.

### Why the limit exists

BSON uses a 32-bit signed integer for document length, giving a theoretical max of ~2 GB.
MongoDB deliberately chose 16 MB to keep working sets in RAM and discourage designs that
embed unbounded data. The limit applies uniformly to:
- Documents in any collection
- Aggregation pipeline intermediate documents (`$group` accumulator output)
- Command responses (a single `find` result document)
- Oplog entries — so cross-shard transaction commit records are also capped

### PostgreSQL comparison

| | MongoDB | PostgreSQL |
|---|---|---|
| Per-document / per-row limit | **16 MB hard** | ~1 GB (TOAST, transparent) |
| JSONB field size limit | 16 MB (same — it's the doc) | No practical limit |
| Binary / text field limit | 16 MB inline; GridFS for more | 1 GB per BYTEA / TEXT field |
| Overflow to disk | No — error | Yes — TOAST is automatic |
| Error on overflow | `DocumentTooLarge` at insert | Never — Postgres just stores it |

PostgreSQL's TOAST (The Oversized Attribute Storage Technique) transparently compresses
and stores large values out-of-line. You never need to know it exists. MongoDB has no
equivalent — you have to architect around the limit yourself.

### Common ways to hit the wall

```
1. Growing arrays — $push without pruning
   audit logs, chat messages, comment threads, time-series events
   → starts small, works for months, explodes in production

2. Binary data stored inline
   {"avatar": Binary(jpeg_bytes)}  ← a single 20 MB image crashes the insert

3. User-supplied nested JSON
   accepting arbitrary JSON from API clients with no depth/size guard

4. $group + $push in aggregation
   accumulating all matching documents into one group document
   → silently works on dev datasets, fails at scale
```

### The error you'll see

```
pymongo.errors.DocumentTooLarge: BSON document too large (16793600 > 16777216)

# Or during aggregation:
pymongo.errors.OperationFailure: document too large
```

Neither message tells you *which field* caused it. You have to instrument with
`$bsonSize` to find the offender (see monitoring snippet below).

### Fix A — Capped arrays (`$push` + `$slice`)

Keep only the last N entries. Oldest are silently dropped. Use when you only need
a recent window (e.g., last 500 actions for UX personalization).

```python
db.user_state.update_one(
    {"_id": user_id},
    {"$push": {"recent_events": {
        "$each":  [new_event],
        "$slice": -500,          # keep last 500; drop everything older
    }}},
)
```

### Fix B — Bucketing pattern

Split the unbounded array across many documents, each holding at most `BUCKET_SIZE` entries.
This is what MongoDB's native time-series collections do internally.

```python
BUCKET_SIZE = 200

db.event_buckets.update_one(
    {"user_id": user_id, "count": {"$lt": BUCKET_SIZE}},
    {"$push": {"events": event}, "$inc": {"count": 1}},
    upsert=True,
)

# Query: all events for a user in a time range
db.event_buckets.aggregate([
    {"$match": {"user_id": user_id}},
    {"$unwind": "$events"},
    {"$match": {"events.ts": {"$gte": t0, "$lte": t1}}},
])
```

Each bucket document stays small (bounded at `BUCKET_SIZE` entries). Total data is unlimited.

### Fix C — GridFS for binary data

GridFS splits files into 255 KB chunks stored in `fs.chunks`, with metadata in `fs.files`.
No size limit per file.

```python
import gridfs

fs = gridfs.GridFS(db)

with open("report.pdf", "rb") as f:
    file_id = fs.put(f, filename="report.pdf",
                     content_type="application/pdf",
                     user_id=user_id)

# Retrieve
data = fs.get(file_id).read()

# Query metadata without loading content
meta = db["fs.files"].find_one({"_id": file_id})
```

**Rule**: never store any binary blob inline in a document. A single 20 MB PDF
will crash the insert. Always use GridFS or an object store (S3, GCS).

### Fix D — `$merge` to escape aggregation blowup

`$group` + `$push` accumulates results in memory as one document. Use `$merge` to
stream output into a collection — each result document is small, no accumulation.

```python
# BAD — explodes when the group exceeds 16 MB:
db.raw_events.aggregate([
    {"$match": {"user_id": uid}},
    {"$group": {"_id": "$user_id", "all": {"$push": "$$ROOT"}}},
])

# GOOD — each output doc is a tiny summary:
db.raw_events.aggregate([
    {"$match": {"user_id": uid}},
    {"$group": {
        "_id": {"user_id": "$user_id",
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}}},
        "count":   {"$sum": 1},
        "actions": {"$addToSet": "$action"},
    }},
    {"$merge": {"into": "user_daily_summary", "whenMatched": "merge",
                "whenNotMatched": "insert"}},
])
```

### Monitoring — find documents approaching the limit

Use this before you hit an incident:

```python
# MongoDB 4.4+ — $bsonSize operator
db.your_collection.aggregate([
    {"$project": {"size_bytes": {"$bsonSize": "$$ROOT"}}},
    {"$match":   {"size_bytes": {"$gt": 8 * 1024 * 1024}}},  # warn at 8 MB
    {"$sort":    {"size_bytes": -1}},
    {"$limit":   20},
])

# Pre-4.4 — Python-side check
import bson
for doc in collection.find({}, limit=5000):
    size = len(bson.encode(doc))
    if size > 8 * 1024 * 1024:
        print(f"WARNING: {doc['_id']} is {size / 1024 / 1024:.1f} MB")
```

Add this as a periodic Atlas scheduled trigger or a Prometheus scrape job.

### Pattern comparison

| Pattern | Max single doc | Best for |
|---|---|---|
| Inline array (unbounded) | 16 MB hard cap | Bounded arrays only (<1 k entries) |
| Capped `$slice` | Controlled | Recent-window use cases |
| Bucketing | Unlimited (many docs) | Time-series, event logs, messages |
| GridFS | Unlimited | Binary blobs, files |
| Reference (normalize) | Unlimited | Audits, chat, anything unbounded |
| `$merge` aggregation | Unlimited | Large pipeline outputs |

---

## 14. Normalized Query Performance Trap

MongoDB's aggregation pipeline is powerful, but using multiple `$lookup` stages to
simulate SQL JOINs across normalized collections triggers one of the most common
performance anti-patterns.

### What happens in a multi-$lookup pipeline

```
posts → $match (tag filter)
      → $lookup → pt_authors      (one correlated scan per post)
      → $lookup → pt_comments     (one correlated scan per post)
      → $graphLookup → pt_authors (recursive BFS per comment set)
      → $lookup → pt_likes        (one correlated scan per post)
      → $group
      → $sort / $skip / $limit
```

Each `$lookup` stage performs a nested-loop join between every matched document
and the target collection. Without an index on the `foreignField`, this is a full
collection scan **per document**. With N matched posts and M lookup stages,
the cost is O(N × M × size-of-target-collection).

### Why `$graphLookup` is especially expensive

`$graphLookup` performs breadth-first traversal. At depth D with branching factor B,
it issues B^D individual lookups. On flat org-chart or tag-hierarchy data this
compounds quickly, especially because the working set grows in memory.

### The `$skip` pagination trap

| Approach | Cost |
|---|---|
| `$skip(0)` + `$limit(25)` — page 1 | Process 25 docs, return 25 |
| `$skip(2500)` + `$limit(25)` — page 101 | Process ALL prior stages for 2525 docs, discard 2500, return 25 |

The entire pipeline (all `$lookup` stages) runs for every skipped document.
Page number × page size wasted pipeline executions.

### The fix: embed and cursor-paginate

```python
# Denormalized post document (reads are 10–100× faster):
# {
#   _id, title, body,
#   author: { _id, name },       # snapshot — updated on author rename via $merge
#   tag_labels: ["tech"],         # duplicated labels for O(1) array membership check
#   comment_count: 42,            # counter incremented on comment insert
#   like_count: 17,               # counter incremented on like
# }

# Single-index query + cursor pagination:
db.posts.find(
    {"tag_labels": "tech", "_id": {"$gt": last_seen_id}}
).sort("_id", 1).limit(25)
# Hits compound index { tag_labels: 1, _id: 1 } — O(1) regardless of page.
```

---

## 15. Write Sharding Strategies

Sharding distributes documents across shards (replica sets) using a shard key.
The choice of shard key and sharding strategy determines write distribution,
query routing, and operational overhead.

### Strategy comparison

| Strategy | Write distribution | Range scan | Risk |
|---|---|---|---|
| **Hashed** (`{ user_id: "hashed" }`) | Even (random) | Scatter-gather fan-out | None for writes |
| **Ranged** (`{ ts: 1 }`) | Sequential (hotspot) | Single-shard range scan | Monotonic key hotspot |
| **Ranged + bucket prefix** (`{ bucket: 1, ts: 1 }`) | N buckets (controlled) | Per-bucket range scan | Hotspot within each bucket |
| **Zone sharding** | Pinned to zone | Zone-local only | Manual zone maintenance |

### Hashed sharding

Best for high-cardinality keys with no range query requirement (e.g., `user_id`, `session_id`).

```
sh.shardCollection("myapp.events", { user_id: "hashed" })
```

mongos hashes `user_id` → assigns to a shard. Writes are spread evenly.
Range queries on `user_id` fan out to all shards (scatter-gather).

### Ranged sharding with hotspot mitigation

Monotonically increasing keys (ObjectId, timestamp) always write to the last chunk
until MongoDB splits and migrates it — creating a sustained write hotspot.

Mitigation: prepend a low-cardinality "bucket" field.

```python
import hashlib

def bucket(device_id: str, num_shards: int = 8) -> int:
    return int(hashlib.md5(device_id.encode()).hexdigest(), 16) % num_shards

doc = {
    "bucket": bucket(device_id),   # 0–7
    "ts": datetime.utcnow(),
    "device_id": device_id,
    "value": reading,
}
# Shard key: { bucket: 1, ts: 1 }
# Range query: { bucket: b, ts: { $gte: t0, $lte: t1 } }
```

### Zone sharding (data residency)

```
sh.addShardToZone("shard-eu-west", "EU")
sh.addShardToZone("shard-us-east", "US")
sh.updateZoneKeyRange("myapp.users", { region: "EU" }, { region: "EU~" }, "EU")
sh.updateZoneKeyRange("myapp.users", { region: "US" }, { region: "US~" }, "US")
sh.shardCollection("myapp.users", { region: 1, _id: 1 })
```

EU user documents live only on EU shards. Writes from EU clients never touch US shards.

### Shard key selection rules

| Property | Requirement |
|---|---|
| Cardinality | High — enough distinct values to fill many chunks |
| Frequency | Low — no single value dominates (causes jumbo chunks) |
| Monotonicity | Avoid — monotonic keys cause write hotspots |
| Query coverage | Key should appear in most queries |
| Immutability | Shard key cannot be updated after insert (MongoDB 5.0: partial exception) |

---

## 16. Distributed Transactions

MongoDB distributed transactions use **two-phase commit (2PC)** coordinated by mongos
(for sharded clusters) or the primary (for single replica sets).

### Cross-shard transaction flow

```
1. App opens session → starts transaction
2. Writes sent to relevant shards with session lsid
3. mongos elected coordinator; sends prepareTransaction to each shard
4. Each shard WAL-logs prepare record; votes commit/abort
5. Coordinator writes commit record to config server (durable)
6. Coordinator sends commitTransaction to all shards
7. Shards apply, release locks
```

### Coordinator failure matrix

| Crash point | Outcome |
|---|---|
| Before prepare sent | Transaction aborts (no prepare record exists) |
| After prepare, before commit record | Decision not yet durable → aborts |
| After commit record written | Any mongos reads config server → commits |
| Shard unreachable at prepare | That shard votes abort → coordinator aborts all |
| Network partition during commit | Pending shards retry on reconnect using durable commit record |

### Recommended client settings

```python
client = MongoClient(
    uri,
    w="majority",    # wait for majority of RS members to ack
    journal=True,    # flush to WAL before ack
    wtimeout=5000,   # ms — avoid hanging on node failure
)
```

### `with_transaction` vs manual `start_transaction`

Always prefer `with_transaction` — it handles transient errors
(`TransientTransactionError`, `UnknownTransactionCommitResult`) with automatic
exponential-backoff retry:

```python
with client.start_session() as session:
    session.with_transaction(lambda s: my_operation(s))
```

Manual `start_transaction` / `commit_transaction` is only needed when you need
to inspect the transaction state between operations.

### Limitations

- Max 60-second wall-clock duration (`transactionLifetimeLimitSeconds`)
- Max 16 MB total transaction size (BSON document limit applies to oplog entries)
- No DDL inside transactions (no `createCollection`, `createIndex`)
- Transactions on sharded clusters require all shards to be available at prepare time

---

## 17. Locking Patterns

MongoDB uses WiredTiger row-level (document-level) locking internally. Two concurrent
writes to **different** documents in the same collection do not block each other.
Intent locks (IS, IX) at the collection/database level allow concurrent document-level
operations while preventing conflicting DDL.

User-visible lock patterns are implemented in application code.

### Pessimistic locking

Acquire exclusive access before operating. Best when contention is high or the
operation takes significant time (e.g., checkout flow that reads then debits).

**Single-document — atomic CAS with `findOneAndUpdate`:**

```python
def acquire_lock(col, doc_id, owner):
    return col.find_one_and_update(
        {"_id": doc_id, "locked": False},
        {"$set": {"locked": True, "owner": owner,
                  "locked_at": datetime.utcnow()}},
        return_document=True,
    ) is not None  # False means already locked

def release_lock(col, doc_id, owner):
    col.update_one(
        {"_id": doc_id, "owner": owner},   # only owner can release
        {"$set": {"locked": False, "owner": None}},
    )
```

`findOneAndUpdate` is a single atomic server-side operation — no separate
read-then-write race condition.

**Multi-document — transaction:**

```python
with client.start_session() as s:
    with s.start_transaction():
        for doc_id in to_lock:
            ok = col.find_one_and_update(
                {"_id": doc_id, "locked": False},
                {"$set": {"locked": True}},
                session=s,
            )
            if ok is None:
                s.abort_transaction(); raise RuntimeError("conflict")
        # Both docs are write-locked for the duration of the transaction
        do_work(session=s)
```

**Stale lock risk:** If the holder crashes before `release_lock`, the document
stays locked indefinitely. Mitigate with a `locked_at` TTL check in a background
job, or a TTL index on `locked_at` that resets the field after a timeout.

**Task queue (pessimistic, SKIP LOCKED equivalent):**

```python
task = db.tasks.find_one_and_update(
    {"status": "pending"},
    {"$set": {"status": "running", "worker": worker_id}},
    sort=[("_id", ASCENDING)],  # FIFO
)
```

Atomic claim — no two workers can receive the same task.

### Optimistic locking

No lock is held. Each document carries a `__v` version counter. Writers:
1. Read document + version.
2. Do work (compute new values).
3. Update **only if version unchanged** (CAS).
4. On version mismatch → retry from step 1.

Best for low-contention scenarios (infrequent concurrent writes to the same doc).

```python
def update_price(col, product_id, new_price, max_retries=5):
    for _ in range(max_retries):
        doc = col.find_one({"_id": product_id})
        v = doc["__v"]
        result = col.update_one(
            {"_id": product_id, "__v": v},       # guard on version
            {"$set": {"price": new_price}, "$inc": {"__v": 1}},
        )
        if result.modified_count == 1:
            return True   # success
        # version changed — another writer beat us, retry
    raise RuntimeError("max retries exceeded")
```

### Comparison

| Aspect | Pessimistic | Optimistic |
|---|---|---|
| Best for | High contention, long operations | Low contention, short ops |
| Write throughput | Lower (serialized) | Higher (parallel reads) |
| Conflict cost | No wasted work (blocked upfront) | Retry on conflict |
| Deadlock risk | Yes (multi-doc transactions) | None |
| Requires replica set | Only for multi-doc txn | No |
| Stale lock risk | Yes (crash before release) | No |

---

## 18. FAQ

**Q: When should I use Motor instead of PyMongo?**
A: When your application is built on an async framework (FastAPI, Starlette, aiohttp).
Motor is a thin async wrapper over PyMongo with the same API; switching is mostly
`s/MongoClient/AsyncIOMotorClient/` and adding `await`.

**Q: Can I mix synchronous PyMongo calls inside an async function?**
A: Technically yes, but blocking calls block the event loop. Use Motor or run PyMongo
calls in a thread pool via `asyncio.to_thread(collection.find_one, query)`.

**Q: Why is my aggregation slow even though I have an index?**
A: Common causes: `$match` is not the first stage so the index is not used; the
`$match` filter does not match the index prefix; `$group` or `$unwind` is before
`$match`. Run `.explain("executionStats")` on the aggregation to diagnose.

**Q: How do I handle ObjectId in a REST API (JSON)?**
A: `ObjectId` is not JSON-serializable by default. Serialize it as a string.
In FastAPI use a custom encoder; in Flask use `flask-pymongo`'s JSONEncoder, or convert
manually: `str(doc["_id"])`.

**Q: What is the difference between `$out` and `$merge`?**
A: `$out` replaces the target collection entirely (atomic swap). `$merge` supports
upsert semantics — it can insert, merge, replace, or fail per-document. Prefer `$merge`
for incremental/materialised view updates; use `$out` when you want a clean replace.

**Q: When do I need a transaction?**
A: When you need ACID guarantees across multiple documents or collections — e.g., deducting
credits from one account and adding to another. For single-document operations, MongoDB is
already atomic. For many workflows, careful document design (embedding) eliminates the need
for transactions.

**Q: Can I run a text search and filter by other fields at the same time?**
A: Yes. Combine `$text` with other query operators: `{"$text": {"$search": "python"}, "published": True}`.
The text index must be on the fields you're searching; other fields in the filter can use
their own indexes (MongoDB will choose the best plan).

**Q: Why does `$skip` get slow on large collections?**
A: `$skip` still scans and discards the skipped documents. At large offsets (e.g., skip 100 000)
this is expensive. Switch to keyset (cursor) pagination for user-facing APIs.

**Q: What is `w=0` ("unacknowledged") useful for?**
A: Very high-throughput write scenarios (analytics event ingestion, metrics) where occasional
data loss is acceptable and latency must be minimal. Never use for financial or user data.

**Q: How do change streams handle a replica set failover?**
A: MongoDB change streams are resilient to elections. When the primary steps down, the driver
reconnects to the new primary and the change stream resumes automatically (using an internal
resume token). No special handling is needed in your code.

**Q: Can MongoEngine and PyMongo coexist in the same application?**
A: Yes. MongoEngine uses PyMongo under the hood. You can access the raw PyMongo collection
via `MyDocument._get_collection()` and run raw operations alongside ODM operations.

**Q: What are the gotchas with MongoEngine `ReferenceField`?**
A: By default, accessing a `ReferenceField` triggers an additional query (lazy loading).
Use `select_related()` to pre-fetch references in one query. Also, `reverse_delete_rule`
must be set explicitly (`me.CASCADE`, `me.DENY`, `me.NULLIFY`) — there is no automatic
cascade by default.

**Q: Can I use PyMongo and Motor in the same project?**
