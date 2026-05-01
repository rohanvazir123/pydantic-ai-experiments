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
19. [Read-After-Write Consistency](#19-read-after-write-consistency)

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

### `$` vs `$$` — field references vs variables

Inside aggregation expressions MongoDB uses two distinct sigils:

| Syntax | Meaning | Example |
|---|---|---|
| `"$field"` | Reference a **field** on the current document | `"$price"`, `"$author.name"` |
| `"$$name"` | Reference a **variable** — never a field | `"$$ROOT"`, `"$$post_id"` |

`$$` variables come from two sources:
1. **System variables** — built in, always available
2. **User-defined variables** — declared in `let` (inside `$lookup`) or `$let` / `$map` / `$reduce` expressions

#### System variables

| Variable | Value |
|---|---|
| `$$ROOT` | The entire current document (all fields) |
| `$$CURRENT` | The current document at this point in the pipeline (same as `$$ROOT` at the top level; differs inside `$lookup` sub-pipelines) |
| `$$REMOVE` | Sentinel — when used as a field value in `$project` / `$addFields`, the field is **omitted** from the output entirely |
| `$$NOW` | Current UTC date/time as a `Date` (useful inside expressions, consistent within one command) |
| `$$PRUNE` | Used with `$redact` — removes the current sub-document |
| `$$DESCEND` | Used with `$redact` — keeps the current sub-document and recurses into it |
| `$$KEEP` | Used with `$redact` — keeps the current sub-document and stops recursing |

#### Common `$$ROOT` patterns

```python
# Measure each document's BSON size (MongoDB 4.4+) — safe, no accumulation
{"$project": {"size_bytes": {"$bsonSize": "$$ROOT"}}}

# Promote a nested sub-document to be the new root — safe, one-to-one reshape
{"$replaceRoot": {"newRoot": "$address"}}
# Merge sub-doc fields with parent fields using $$ROOT:
{"$replaceRoot": {"newRoot": {"$mergeObjects": ["$$ROOT", "$address"]}}}
```

#### `$$ROOT` + `$push` — the 16 MB trap

`{"$push": "$$ROOT"}` embeds a **full copy of every matched document** into a single
array on the group result. Each document in the group adds its entire BSON size to the
accumulator. Because the 16 MB limit applies to the *output* document from `$group`,
this blows up silently on small dev datasets and crashes in production once the group
grows large enough.

```python
# DANGEROUS — one group document accumulates ALL matched docs:
db.events.aggregate([
    {"$match": {"user_id": uid}},
    {"$group": {
        "_id": "$user_id",
        "all_events": {"$push": "$$ROOT"},  # each doc is a full copy — adds up fast
    }},
])
# Error once group exceeds 16 MB:
# OperationFailure: document too large

# How fast does it blow up?
# avg doc size 1 KB  → fails after ~16 000 events per user
# avg doc size 10 KB → fails after ~1 600 events per user
# avg doc size 100 KB → fails after ~160 events per user
```

**Alternatives that don't accumulate into one document:**

```python
# 1. Project only the fields you actually need before pushing
{"$group": {
    "_id": "$user_id",
    "events": {"$push": {"ts": "$ts", "action": "$action"}},  # slim projection
}}

# 2. Stream into a collection with $merge — each output doc is a small summary
db.events.aggregate([
    {"$match": {"user_id": uid}},
    {"$group": {
        "_id": {"user": "$user_id",
                "day":  {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}}},
        "count":   {"$sum": 1},
        "actions": {"$addToSet": "$action"},   # set of distinct values only
    }},
    {"$merge": {"into": "user_daily_summary", "whenMatched": "merge",
                "whenNotMatched": "insert"}},
])

# 3. Use $count instead of $push when you only need the total
{"$group": {"_id": "$user_id", "event_count": {"$sum": 1}}}
```

**Rule:** treat `{"$push": "$$ROOT"}` as a code smell. It is occasionally justified
(small bounded groups, debugging pipelines) but almost always means you should either
project first or rethink the pipeline output strategy.

#### `$$REMOVE` — conditional field omission

```python
# Include "discount" field only when it exists; omit it entirely otherwise.
# Without $$REMOVE you'd get "discount": null in the output.
{"$project": {
    "price": 1,
    "discount": {"$cond": [{"$gt": ["$discount", 0]}, "$discount", "$$REMOVE"]},
}}
```

#### User-defined variables with `let`

`let` exposes outer-document fields to the sub-pipeline inside `$lookup`.
`$$name` (double-dollar + name) references the variable inside the pipeline.

```python
{"$lookup": {
    "from": "orders",
    "let":  {"uid": "$_id", "min_val": "$min_order_value"},  # expose outer fields
    "pipeline": [
        {"$match": {"$expr": {
            "$and": [
                {"$eq":  ["$user_id",  "$$uid"]},      # $$uid    = let variable
                {"$gte": ["$total",    "$$min_val"]},   # $$min_val = let variable
            ]
        }}},
    ],
    "as": "qualifying_orders",
}}
# Note: if you only need the equality join with no extra outer-field conditions,
# use localField / foreignField instead — no let / $$ needed.
```

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

Change streams expose MongoDB's oplog as a resumable, ordered cursor of DML events.
They require a **replica set** (or sharded cluster) — the oplog does not exist on standalones.

### Event anatomy

```python
{
    "_id":           <resume_token>,        # opaque dict; persist after every event
    "operationType": "update",              # insert | update | replace | delete | drop | rename | invalidate
    "clusterTime":   Timestamp(1234, 1),    # logical clock, NOT wall-clock
    "ns":            {"db": "blog", "coll": "posts"},
    "documentKey":   {"_id": ObjectId("...")},
    "updateDescription": {
        "updatedFields":   {"likes": 1},    # only changed fields
        "removedFields":   [],
        "truncatedArrays": []
    },
    "fullDocument":  {...}                  # only with full_document="updateLookup"
}
```

### Scope levels

```python
collection.watch(pipeline)   # one collection
db.watch(pipeline)           # all collections in the database
client.watch(pipeline)       # every collection in every database
```

Narrower scope means less oplog traffic the server needs to filter. Use `db`/`client` scope for cross-collection audit logs.

### Resume token

The `_id` of each event is the **resume token** — an opaque BSON dict. Persist it (e.g. to a separate MongoDB collection or Redis) after every successfully processed event. On restart:

```python
# startAfter — resume AFTER the token event (skip it)
collection.watch(startAfter=resume_token)

# resumeAfter — resume FROM the token event (re-delivers it)
collection.watch(resumeAfter=resume_token)

# startAtOperationTime — replay from a cluster Timestamp (e.g. from a backup)
from bson import Timestamp
collection.watch(startAtOperationTime=Timestamp(unix_seconds, 1))
```

Delivery is **at-least-once**: network failures may replay the last event. Your consumer must be idempotent.

### `full_document` options

| Option | Behaviour |
|---|---|
| omit (default) | Update events carry only `updateDescription` (diff) — no extra round-trip |
| `"updateLookup"` | Fetches the full document after the update; may race with a subsequent write |
| `"whenAvailable"` | Returns the doc if still present, `None` if deleted before lookup |
| `"required"` | Errors if the doc was deleted before lookup completes |

Use diff-only for high-throughput streams; `updateLookup` for audit logs where you need the full post-update state.

### Filtering with a pipeline

```python
pipeline = [
    {"$match": {"operationType": {"$in": ["insert", "update"]}}},
    {"$match": {"fullDocument.status": "published"}},    # filter by field value
]
collection.watch(pipeline, full_document="updateLookup")
```

Filtering happens server-side — only matching events are sent over the wire.

### Requirements and limitations

- Replica set or sharded cluster (oplog required).
- WiredTiger storage engine.
- Change stream cursors time out if idle for longer than `cursor.noCursorTimeout` allows — keep the cursor active or re-open on timeout.
- Payload is bounded by the 16 MB BSON document limit per event.

### Common use cases

- Cache invalidation when a product price changes
- Real-time dashboards (order status updates)
- Audit logs / event sourcing
- Syncing to Elasticsearch, Kafka, or another store

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
A: Yes. They connect independently and share no state. Common pattern: Motor for the
async web layer (FastAPI handlers), PyMongo for sync background workers or Celery tasks
in the same repo.

**Q: Is `AsyncIOMotorClient` new?**
A: No. Motor added asyncio support in v1.2 (2015) alongside its original Tornado backend.
`AsyncIOMotorClient` has been the asyncio entry point since then. Motor 2.0 (2019) made
asyncio the primary interface; Motor 3.x dropped the Tornado dependency as a requirement.
The import path (`motor.motor_asyncio`) looks dated but is still correct for current Motor 3.x.

**Q: What are the disadvantages of using Motor (async) over PyMongo (sync)?**
A: Motor is not truly non-blocking at the OS level — it wraps PyMongo's sync driver in a
thread pool executor, so you get concurrency, not epoll-style async I/O. The main disadvantages:

| Disadvantage | Detail |
|---|---|
| Complexity | `async def` / `await` everywhere; harder stack traces |
| Silent bugs | Forgotten `await` returns a coroutine object, not an error |
| Useless in sync contexts | Scripts, notebooks, Flask, Celery — `asyncio.run()` overhead negates any benefit |
| Heavier testing | Requires `pytest-asyncio`, `AsyncMock`, fixture scope management |
| No CPU benefit | Aggregation CPU time on the server is unchanged; async only frees the event loop during I/O waits |
| `threading.local` breaks | Middleware and loggers that rely on thread-local state don't carry across coroutines |

Use Motor only when your framework is already async (FastAPI, Starlette) and you will
actually run concurrent queries. Use PyMongo for everything else — simpler code, identical throughput.

**Q: What happens to aggregation performance when a collection is sharded?**
A: Every aggregation over a sharded collection becomes a **distributed query** regardless
of how simple it looks. mongos sends the pipeline to every shard that holds chunks for the
collection (scatter-gather), collects the partial results, and merges them — typically on a
randomly chosen shard called the *merger shard* (or on mongos itself for memory-heavy stages).

The hidden cost is that any stage which cannot be pushed down to individual shards forces
mongos to act as a **coordinator**, which involves the same two-phase protocol (2PC) used
by distributed transactions:

```
Client → mongos (coordinator)
            ├─→ shard A  (partial pipeline execution)
            ├─→ shard B  (partial pipeline execution)
            └─→ shard C  (partial pipeline execution)
                     ↓
         mongos merges: $group, $sort, $limit, $lookup across shards
```

Stages that **stay on each shard** (pushed down — cheap):
- `$match` on the shard key or a shard-key prefix
- `$project`, `$addFields`, `$unwind` (document-level, no cross-shard data)
- `$group` with a shard-key `_id` (each shard groups its own data independently)

Stages that **must merge on mongos** (pulled up — expensive):
- `$group` on anything other than the full shard key — each shard emits partial
  accumulators, mongos re-aggregates them (two-pass `$group`)
- `$sort` without a limit — all shards stream sorted data to the merger
- `$lookup` — the foreign collection may live on different shards; mongos must
  coordinate fetch per document
- `$graphLookup` — recursive traversal fans out to every shard at every depth level
- `$out` / `$merge` — the merger shard writes the final result; other shards are idle

In practice, a query that runs in 20 ms on a single replica set can take 200–500 ms on
a 3-shard cluster if mongos is the merger — not because the data is larger, but because
of the coordination overhead and the extra network round-trips between shards.

**The compound problem: sharding + `$lookup`**

`$lookup` on a sharded cluster is a code smell. If the foreign collection is also
sharded, mongos must issue a correlated sub-query *per document* across shard boundaries.
This is functionally a distributed nested-loop join with 2PC coordination on each
iteration — latency compounds with document count:

```
# On a single replica set: one query plan, one engine, fast
# On a sharded cluster: per-document cross-shard lookup, coordination overhead × N docs
db.orders.aggregate([
    {"$lookup": {
        "from":         "products",    # sharded on product_id
        "localField":   "product_id",
        "foreignField": "_id",
        "as":           "product",
    }},
])
```

**How to mitigate:**
- **Include the shard key in `$match`** — routes to a single shard, no scatter-gather.
- **Denormalize the hot fields** — embed the fields you'd `$lookup` directly in the
  document so the join never happens. This is the MongoDB document model working as intended.
- **Co-locate related collections on the same shard** using zone sharding — `$lookup`
  between two collections on the same shard stays local, no cross-shard coordination.
- **Pre-aggregate into a summary collection** with `$merge` so dashboards read one
  already-computed document rather than running a distributed aggregation on every request.
- **Avoid sharding until you actually need it** — a well-indexed replica set handles
  hundreds of millions of documents. Sharding adds operational complexity and turns every
  aggregation into a distributed systems problem.

**Q: What is the 60-second transaction timeout and why does it catch people off guard?**
A: MongoDB hard-aborts any transaction that has been open for longer than
`transactionLifetimeLimitSeconds` (default: **60 seconds**). The abort is server-side and
unconditional — the driver receives no warning, the transaction just dies mid-flight with:

```
MongoServerError: Transaction ... has been aborted.
pymongo.errors.OperationFailure: Transaction ... has been aborted due to timeout
```

**Why it catches people off guard:**

1. **It is a server setting, not a driver setting.** You cannot raise it from the client.
   There is no `session.setTimeout(120)` call. The only way to change it is `mongod.conf`
   or a `setParameter` admin command:

   ```javascript
   // mongosh — raises the limit to 120 s on a running server (not persisted on restart):
   db.adminCommand({ setParameter: 1, transactionLifetimeLimitSeconds: 120 })
   ```

   ```yaml
   # mongod.conf — persisted:
   setParameter:
     transactionLifetimeLimitSeconds: 120
   ```

   On Atlas you cannot change this at all — 60 s is the hard ceiling.

2. **It fires even under zero contention.** A transaction doing a slow aggregation,
   waiting on an external service call, or stalled by cache pressure eviction will be
   killed at exactly 60 s regardless of whether any other writer exists.

3. **It interacts badly with the retry storm.** Under cache pressure, transactions stall
   (eviction blocks page reads). A stalled transaction approaches the 60 s limit. When it
   hits the limit and is aborted, the driver retries — but the retry starts a fresh
   60 s clock on an already-pressured system. Each retry is nearly guaranteed to hit the
   timeout too, creating a retry loop where every attempt burns a full 60 s before failing:

   ```
   Attempt 1: stalls at ~58 s → timeout abort at 60 s → retry
   Attempt 2: stalls at ~58 s → timeout abort at 60 s → retry
   Attempt 3: ...
   # wall-clock time consumed: 3 × 60 s = 3 minutes, zero progress
   ```

4. **The background cleanup job runs on the same 60 s cadence.** MongoDB's transaction
   reaper runs every `transactionLifetimeLimitSeconds` seconds. If you lower the timeout
   to 10 s, the reaper also runs every 10 s, adding its own periodic CPU spike.

5. **It applies to the entire transaction wall-clock duration**, not just individual
   operation time. A transaction that does five fast operations but sleeps between them
   (waiting for user input, an API call, or a queue message) will be killed just as
   surely as a slow query.

**What the timeout is actually for:**

It exists to prevent "zombie" transactions — sessions that opened a transaction and then
crashed or were abandoned — from holding snapshot versions and dirty pages in the cache
indefinitely. Without it, a single dead client could block WiredTiger cache reclamation
forever. The 60 s default is a reasonable ceiling for OLTP transactions, not for long-
running analytical or batch operations.

**Design rules given the constraint:**

| Pattern | Verdict |
|---|---|
| Open transaction → call external API → commit | Code smell — external call can stall; never hold a transaction across I/O you don't control |
| Open transaction → user thinks about it → commit | Code smell — interactive latency easily exceeds 60 s |
| Open transaction → large aggregation → commit | Code smell — run the aggregation first, outside the transaction, then open and commit |
| Open transaction → 3–5 fast document reads/writes → commit | Correct — keep transactions short and data-local |
| Batch job inside a transaction | Code smell — batch over many documents; use `with_transaction` per small batch, not one giant transaction |

**Practical limit to design for:** treat 5–10 seconds as your real budget, not 60.
Leave the remaining headroom for slow networks, GC pauses, and cache stalls.
If your transaction logic cannot complete in 5–10 s under normal load, redesign it —
raising `transactionLifetimeLimitSeconds` only delays the problem.

**Q: Why do I see endless transaction retries in MongoDB even with zero write contention?**
A: The most common non-obvious cause is **WiredTiger storage engine cache pressure**, not
lock contention. This shows up frequently in Kubernetes where pod memory limits are set
too close to what WiredTiger actually needs.

**What happens internally:**

WiredTiger uses MVCC (Multi-Version Concurrency Control). Every in-progress transaction
holds dirty pages in the cache. When the cache fills, WiredTiger must evict pages to make
room. If a transaction's dirty pages cannot be evicted without rolling back the transaction,
WiredTiger does exactly that — it **internally aborts the transaction and surfaces a
`WriteConflict` error**, even if no other writer touched those documents.

The MongoDB driver sees `WriteConflict`, classifies it as `TransientTransactionError`, and
retries automatically. But since the underlying cause is memory pressure rather than a
conflicting write, the pressure doesn't resolve between retries — every retry hits the same
wall, producing an endless retry loop that looks like contention but isn't.

```
Transaction starts
  └─→ dirty pages accumulate in WiredTiger cache
  └─→ cache hits high-water mark (default: 80% of cache size)
  └─→ WiredTiger eviction thread can't clear fast enough
  └─→ WiredTiger aborts transaction → WriteConflict
  └─→ driver retries (TransientTransactionError)
  └─→ same pressure → same abort → same retry → loop
```

**Why Kubernetes makes this worse:**

WiredTiger defaults to using `50% of (total RAM − 1 GB)` or 256 MB, whichever is larger.
It reads `/proc/meminfo` for total RAM — which in a K8s pod reports the **node's** total
RAM, not the pod's memory limit. So WiredTiger believes it has, say, 32 GB available and
sizes its cache accordingly, while the pod limit is 4 GB. The cache grows until the cgroup
OOM killer fires, or until cache pressure triggers the retry storm described above.

```yaml
# Pod memory limit:     4Gi
# WiredTiger sees:      32Gi node RAM
# WiredTiger cache:     ~15.5 GB (50% of 31 GB)  ← will exceed the pod limit
```

**How to diagnose:**

```javascript
// In mongosh — check cache and eviction stats:
db.serverStatus().wiredTiger.cache

// Key fields to watch:
// "bytes currently in the cache"                        ← current usage
// "maximum bytes configured"                            ← cache ceiling
// "tracked dirty bytes in the cache"                   ← dirty pages held by txns
// "pages evicted because they exceeded the in-memory maximum"  ← non-zero = pressure
// "transaction rollback due to cache overflow"          ← the exact counter you want

db.serverStatus().wiredTiger.transaction
// "transaction rollback due to cache overflow"  > 0 confirms this is the cause
```

```bash
# From outside the pod — watch memory approaching the limit:
kubectl top pod <mongo-pod> --containers
# If memory usage is consistently > 80% of the limit, you're in the danger zone.
```

**Fixes:**

1. **Set `wiredTigerCacheSizeGB` explicitly** — the most important fix. Leave at least
   1–2 GB headroom for the OS page cache, connection threads, and index builds:

   ```yaml
   # mongod.conf
   storage:
     wiredTiger:
       engineConfig:
         cacheSizeGB: 2.5   # pod limit is 4Gi; leave 1.5 GB for OS + threads
   ```

   Or as a flag: `--wiredTigerCacheSizeGB 2.5`

2. **Set K8s memory request = limit** — prevents the scheduler from placing the pod on a
   node that cannot actually back the limit, and prevents burstable-class throttling:

   ```yaml
   resources:
     requests:
       memory: "4Gi"
     limits:
       memory: "4Gi"
   ```

3. **Use `allowDiskUse: True` for heavy aggregations** — lets WiredTiger spill intermediate
   results to disk instead of holding them in the cache during large `$group` / `$sort`:

   ```python
   db.orders.aggregate(pipeline, allowDiskUse=True)
   ```

4. **Increase the pod memory limit** — if the workload genuinely needs more cache.
   Rule of thumb: working set (hot data + indexes) should fit comfortably in the cache.

5. **Add a read replica and route analytical queries there** — keeps the primary's cache
   free for transactional workloads; heavy aggregations run on the replica without
   competing for primary cache pages.

**The `TransientTransactionError` retry budget:**

`with_transaction` retries indefinitely on `TransientTransactionError`. If you're using
manual `start_transaction` / `commit_transaction`, add a retry cap:

```python
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    try:
        with session.start_transaction():
            do_work(session)
            session.commit_transaction()
        break
    except (WriteConflict, OperationFailure) as e:
        if "TransientTransactionError" in e.details.get("errorLabels", []):
            if attempt == MAX_RETRIES - 1:
                raise   # don't loop forever — surface the real problem
            continue
        raise
```

A cap turns an infinite loop into a visible failure, which surfaces the cache pressure
in your error logs rather than hiding it behind silent retries.

**The cascading feedback loop:**

Cache pressure does not just cause individual transaction aborts — it creates a
self-reinforcing death spiral:

```
Cache pressure
  → longer transaction duration (eviction stalls block page reads)
  → longer snapshot retention (MVCC must keep older versions alive for in-flight txns)
  → more dirty pages accumulate (long-lived txns hold more in-cache state)
  → deeper cache pressure
  → more aborts → more retries
  → retries re-enter the system, adding new transactions on top of already-stalled ones
  → system load increases → latency increases → transactions run even longer
  → back to the top, worse than before
```

Each leg amplifies the next. A cluster that was stable under normal load can tip into
this loop from a single large aggregation or a temporary memory spike, and will not
self-recover as long as the retry storm keeps adding new load. The only exit is to
reduce pressure from outside: kill long-running operations (`db.killOp()`), drop the
`allowDiskUse` aggregations onto a replica, or reduce the cache ceiling so WiredTiger
starts rejecting new allocations earlier (before the stall cascades).

This is why treating `TransientTransactionError` as always safe to retry silently is
dangerous — under cache pressure, each retry is fuel on the fire.

**Q: Given extra RAM, should I expand the WiredTiger cache or put an application cache (Redis) in front of MongoDB?**
A: Give it to WiredTiger first as the default. MongoDB has no query result cache — WiredTiger's
page cache is the only layer between your queries and disk I/O. Expanding it directly reduces
eviction pressure, snapshot retention overhead, and the transaction abort cascades that follow.
An application cache does nothing for writes and won't stop the abort storm.

**Give the RAM to WiredTiger when:**
- Your working set is being evicted — `db.serverStatus().wiredTiger.cache` shows a high
  `"bytes read into cache"` rate relative to `"bytes currently in cache"` (pages are being
  evicted and re-read constantly)
- You are seeing transaction aborts or retry storms under low contention (cache pressure
  cascade — see above)
- Query patterns are diverse — a Redis cache would have a low hit rate and help little

**Put a cache in front (Redis) when:**
- You have a small, identifiable hot key set read at very high fan-out: trending content,
  leaderboards, session tokens, expensive pre-aggregated results
- Sub-millisecond read latency is required — Redis serves from memory without touching
  MongoDB's connection pool or query engine at all
- MongoDB connection pool or CPU is the bottleneck, not storage throughput
- You can cache computed aggregation results that would otherwise require a full pipeline
  on every request

**The diagnostic tell:**

```javascript
const cache = db.serverStatus().wiredTiger.cache;
const readInRate   = cache["bytes read into cache"];
const currentBytes = cache["bytes currently in cache"];
const maxBytes     = cache["maximum bytes configured"];

// High read-in relative to cache size → eviction happening → give RAM to WiredTiger
// Cache is stable but connections/CPU are high → bottleneck is query throughput → cache in front
```

**The antipattern:** putting Redis in front of MongoDB to paper over WiredTiger cache
pressure. It hides the read problem but the abort storms continue on writes — you have
now added infrastructure complexity without fixing the root cause.

**Rule of thumb:** WiredTiger first until the working set fits comfortably (cache
utilisation < 80% at peak). If read throughput is still the bottleneck after that,
add Redis in front for the hot key subset.

**Q: When is `asyncio.gather` better than `$facet`, and vice versa?**
A: `$facet` runs multiple sub-pipelines in a single round-trip against the **same collection**
and is always preferable for that case — one query plan, one network call, results arrive
together. `asyncio.gather` is the right tool when your concurrent queries hit **different
collections or databases** where `$facet` cannot reach:

```python
# $facet — one round-trip, same collection (better):
db.orders.aggregate([{"$facet": {
    "by_status": [{"$group": {"_id": "$status", "n": {"$sum": 1}}}],
    "total_revenue": [{"$group": {"_id": None, "rev": {"$sum": "$amount"}}}],
}}])

# asyncio.gather — concurrent queries across different collections (better):
counts, revenue, active_users = await asyncio.gather(
    db.orders.count_documents({"status": "pending"}),
    db.orders.aggregate(revenue_pipeline).to_list(None),
    db.users.count_documents({"active": True}),   # different collection
)
```

---

## 19. Read-After-Write Consistency

### Why it is not automatic in MongoDB

By default, MongoDB acknowledges a write when the **primary** has applied it (`w=1`).
Secondaries replicate asynchronously — there is no bound on how far behind they can be.
If you write to the primary and then read from a secondary (or even re-read from the primary
after a failover), you may see stale or missing data.

### The three knobs

| Setting | Where | What it does |
|---|---|---|
| `causal_consistency=True` | `client.start_session()` | Tracks `operationTime`; every op in the session sends `afterClusterTime` so the server waits until it has applied all prior ops before responding |
| `WriteConcern(w="majority")` | Collection / operation | Write ACKed only after majority of replica set members confirm it — safe from rollback |
| `ReadConcern("majority")` | Collection / operation | Returns only majority-committed data — safe from rollback |

All three must be used together to get the guarantee, because:
- `w=1` + causal session → the secondary may not have the write yet even though `afterClusterTime` is set (secondary hasn't applied it)
- `w="majority"` without a session → a *different* session's read has no ordering guarantee
- `w="majority"` + session without `ReadConcern("majority")` → the node may return an older snapshot

### Code pattern

```python
from pymongo import WriteConcern, ReadPreference
from pymongo.read_concern import ReadConcern

with client.start_session(causal_consistency=True) as session:
    # Write — durable on majority
    result = collection.with_options(
        write_concern=WriteConcern(w="majority"),
    ).insert_one({"title": "post"}, session=session)

    # Read — guaranteed to see the write above, even on a secondary
    doc = collection.with_options(
        read_concern=ReadConcern("majority"),
    ).find_one({"_id": result.inserted_id}, session=session)
    assert doc is not None
```

### Motor (async) pattern

```python
async with await client.start_session(causal_consistency=True) as session:
    result = await collection.with_options(
        write_concern=WriteConcern(w="majority"),
    ).insert_one(doc, session=session)

    found = await collection.with_options(
        read_concern=ReadConcern("majority"),
    ).find_one({"_id": result.inserted_id}, session=session)
```

### Write concern reference

| `w=` | Meaning | Latency | Use when |
|---|---|---|---|
| `0` | Fire and forget | Lowest | Metrics, log ingestion |
| `1` (default) | Primary ACK | Low | Most writes where replica lag is acceptable |
| `"majority"` | Majority ACK | Medium | Financial, inventory, anything causal reads follow |
| `N` (int) | N members ACK | High | Rarely needed; use majority instead |

`j=True` additionally waits for the journal flush to disk before ACKing — survives a primary crash between ACK and fsync.
`wtimeout=ms` caps how long to wait; raises `WriteConcernError` on timeout (write may or may not have committed).

### Read concern reference

| Level | Sees | Safe from rollback | Notes |
|---|---|---|---|
| `"local"` (default) | Latest on this node | No | Fast; may return data that gets rolled back on failover |
| `"majority"` | Majority-committed data | Yes | Needed for causal consistency guarantee |
| `"linearizable"` | Most recent majority-committed write | Yes | Strongest; blocks until majority confirms; high latency |
| `"snapshot"` | Consistent point-in-time within a transaction | Yes | Multi-document transactions only |

### Compared to PostgreSQL

| | MongoDB | PostgreSQL |
|---|---|---|
| **Default guarantee** | None — eventual consistency on secondaries | Automatic on the primary (READ COMMITTED) |
| **Configuration needed** | `causal_consistency` + `w=majority` + `rc=majority` | None (primary reads) |
| **Read replica** | Safe with causal session + rc=majority | Needs `synchronous_commit=remote_apply` or pg_wal_replay_wait |
| **Cross-session guarantee** | Requires causal session | READ COMMITTED sees all committed data automatically |

---
