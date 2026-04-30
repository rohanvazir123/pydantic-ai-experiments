"""
MongoDB Reference: Blog Platform Domain
========================================
Domain: users, posts, comments
Each function demonstrates one concept.
No top-level side-effects — nothing runs on import.

Install:  pip install pymongo motor mongoengine
Run:      python basic_mongo.py  (calls demo_all())
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import asyncio
from datetime import datetime, timezone, timedelta

from bson import ObjectId
from pymongo import (
    MongoClient,
    ASCENDING,
    DESCENDING,
    TEXT,
    GEOSPHERE,
    InsertOne,
    UpdateOne,
    UpdateMany,
    DeleteOne,
    ReplaceOne,
)
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo.read_preferences import ReadPreference
from pymongo.operations import BulkWrite
from pymongo.return_document import ReturnDocument

# ---------------------------------------------------------------------------
# 1. Connection & Client Options
# ---------------------------------------------------------------------------

def demo_connection():
    """MongoClient with common options."""
    # Basic connection
    client = MongoClient("mongodb://localhost:27017")

    # With explicit options
    client_opts = MongoClient(
        host="localhost",
        port=27017,
        serverSelectionTimeoutMS=3000,   # give up finding a server after 3 s
        connectTimeoutMS=2000,           # socket connect timeout
        socketTimeoutMS=10000,           # socket read/write timeout
        maxPoolSize=50,                  # max connections in pool
        authSource="admin",              # DB that holds credentials
        username="blog_user",
        password="secret",
        replicaSet="rs0",                # omit for standalone
        tls=False,
    )

    # URI-based — same options as query params
    client_uri = MongoClient(
        "mongodb://blog_user:secret@host1:27017,host2:27017/"
        "blog?replicaSet=rs0&authSource=admin"
        "&serverSelectionTimeoutMS=3000"
    )

    db = client["blog"]
    print("databases:", client.list_database_names())
    client.close()


# ---------------------------------------------------------------------------
# 2. PyMongo CRUD
# ---------------------------------------------------------------------------

def demo_insert(db):
    """insert_one and insert_many."""
    users = db["users"]
    posts = db["posts"]
    comments = db["comments"]

    # Insert one user
    result = users.insert_one({
        "username": "alice",
        "email": "alice@example.com",
        "age": 30,
        "tags": ["python", "mongodb"],
        "active": True,
        "created_at": datetime.now(timezone.utc),
        "location": {"type": "Point", "coordinates": [-73.97, 40.77]},
    })
    alice_id = result.inserted_id
    print("inserted user:", alice_id)

    # Insert many posts
    result = posts.insert_many([
        {
            "author_id": alice_id,
            "title": "Intro to MongoDB",
            "body": "MongoDB is a document database...",
            "tags": ["mongodb", "nosql"],
            "likes": 42,
            "published": True,
            "published_at": datetime.now(timezone.utc),
        },
        {
            "author_id": alice_id,
            "title": "Python Tips",
            "body": "Here are some Python tips...",
            "tags": ["python", "tips"],
            "likes": 15,
            "published": False,
            "published_at": None,
        },
    ])
    print("inserted posts:", result.inserted_ids)

    # Insert comments
    post_id = result.inserted_ids[0]
    comments.insert_many([
        {"post_id": post_id, "author": "bob", "text": "Great article!", "score": 5},
        {"post_id": post_id, "author": "carol", "text": "Very helpful.", "score": 4},
    ])
    return alice_id


def demo_find(db):
    """find_one / find with query operators, projection, sort, limit, skip."""
    users = db["users"]
    posts = db["posts"]

    # find_one — returns a single dict or None
    user = users.find_one({"username": "alice"})

    # Comparison operators
    posts.find_one({"likes": {"$gt": 10}})              # greater than
    posts.find_one({"likes": {"$gte": 15}})             # >=
    posts.find_one({"likes": {"$lt": 100}})             # <
    posts.find_one({"likes": {"$lte": 42}})             # <=
    posts.find_one({"likes": {"$eq": 42}})              # ==
    posts.find_one({"published": {"$ne": True}})        # !=

    # $in / $nin
    users.find({"tags": {"$in": ["python", "rust"]}})
    users.find({"tags": {"$nin": ["php"]}})

    # $exists — field presence
    posts.find({"published_at": {"$exists": True}})

    # $regex — pattern match
    users.find({"email": {"$regex": r"@example\.com$", "$options": "i"}})

    # $all — array contains all values
    users.find({"tags": {"$all": ["python", "mongodb"]}})

    # $elemMatch — element in array matches multiple conditions
    comments = db["comments"]
    comments.find({"score": {"$elemMatch": {"$gte": 4, "$lte": 5}}})  # scalar demo
    # More typical usage: array of sub-documents
    # posts.find({"reviews": {"$elemMatch": {"rating": {"$gte": 4}, "verified": True}}})

    # Projection — include or exclude fields (not mixed except _id)
    published = list(posts.find(
        {"published": True},
        {"title": 1, "likes": 1, "_id": 0},   # 1=include, 0=exclude
    ))

    # Sort, limit, skip
    top_posts = list(posts.find().sort("likes", DESCENDING).limit(5))
    page2 = list(posts.find().sort("published_at", DESCENDING).skip(10).limit(10))

    print("published posts:", published)
    print("top posts:", top_posts)


# ---------------------------------------------------------------------------
# 3. Update Operators
# ---------------------------------------------------------------------------

def demo_update_operators(db):
    """Demonstrate all common update operators."""
    users = db["users"]
    posts = db["posts"]

    # $set / $unset
    users.update_one(
        {"username": "alice"},
        {"$set": {"bio": "Python & MongoDB enthusiast"}, "$unset": {"legacy_field": ""}}
    )

    # $inc — atomic increment/decrement
    posts.update_one({"title": "Intro to MongoDB"}, {"$inc": {"likes": 1}})

    # $push — append to array
    posts.update_one(
        {"title": "Intro to MongoDB"},
        {"$push": {"tags": "tutorial"}}
    )

    # $pull — remove matching values from array
    posts.update_one(
        {"title": "Intro to MongoDB"},
        {"$pull": {"tags": "nosql"}}
    )

    # $addToSet — push only if not already present
    users.update_one(
        {"username": "alice"},
        {"$addToSet": {"tags": "backend"}}
    )

    # $pop — remove first (-1) or last (1) element
    users.update_one({"username": "alice"}, {"$pop": {"tags": 1}})  # remove last

    # $rename — rename a field
    users.update_one({"username": "alice"}, {"$rename": {"bio": "about"}})

    # $currentDate — set field to current date
    users.update_one(
        {"username": "alice"},
        {"$currentDate": {"updated_at": True}}  # True = Date; {"$type": "timestamp"} for Timestamp
    )

    print("update operators applied")


# ---------------------------------------------------------------------------
# 4. replace_one / find_one_and_update / find_one_and_delete
# ---------------------------------------------------------------------------

def demo_advanced_writes(db):
    """replace_one, find_one_and_update, find_one_and_delete."""
    users = db["users"]

    # replace_one — replaces the entire document (keeps _id)
    users.replace_one(
        {"username": "alice"},
        {
            "username": "alice",
            "email": "alice@example.com",
            "age": 31,
            "active": True,
            "replaced_at": datetime.now(timezone.utc),
        }
    )

    # find_one_and_update — atomic read-then-write, returns document
    updated = users.find_one_and_update(
        {"username": "alice"},
        {"$inc": {"age": 1}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,   # return the NEW version
        upsert=False,
    )
    print("updated doc:", updated)

    # find_one_and_delete — remove and return in one shot
    deleted = users.find_one_and_delete(
        {"username": "ghost"},
        sort=[("created_at", ASCENDING)],
    )
    print("deleted doc:", deleted)


# ---------------------------------------------------------------------------
# 5. Bulk Write
# ---------------------------------------------------------------------------

def demo_bulk_write(db):
    """BulkWrite with mixed operations; ordered=False continues past errors."""
    users = db["users"]

    operations = [
        InsertOne({"username": "dave", "email": "dave@example.com", "active": True}),
        InsertOne({"username": "eve", "email": "eve@example.com", "active": False}),
        UpdateOne({"username": "alice"}, {"$set": {"verified": True}}),
        UpdateMany({"active": False}, {"$set": {"deactivated_at": datetime.now(timezone.utc)}}),
        ReplaceOne(
            {"username": "dave"},
            {"username": "dave", "email": "dave2@example.com", "active": True},
            upsert=True,
        ),
        DeleteOne({"username": "ghost_user"}),
    ]

    try:
        result = users.bulk_write(operations, ordered=False)  # don't stop on first error
        print(
            f"bulk: inserted={result.inserted_count}, "
            f"modified={result.modified_count}, "
            f"deleted={result.deleted_count}"
        )
    except BulkWriteError as e:
        print("bulk write errors:", e.details)


# ---------------------------------------------------------------------------
# 6. Indexes
# ---------------------------------------------------------------------------

def demo_indexes(db):
    """Create and inspect indexes of every common type."""
    users = db["users"]
    posts = db["posts"]

    # Single-field index
    users.create_index("email", unique=True, name="idx_email_unique")

    # Compound index (order matters for prefix queries)
    posts.create_index(
        [("author_id", ASCENDING), ("published_at", DESCENDING)],
        name="idx_author_date",
    )

    # Text index — supports $text search across multiple fields
    posts.create_index(
        [("title", TEXT), ("body", TEXT)],
        weights={"title": 10, "body": 1},   # title matches score higher
        default_language="english",
        name="idx_text_search",
    )

    # TTL index — MongoDB automatically deletes docs after expireAfterSeconds
    db["sessions"].create_index(
        "expires_at",
        expireAfterSeconds=0,   # 0 = delete at the exact datetime stored in the field
        name="idx_sessions_ttl",
    )

    # Sparse index — only indexes docs where the field exists
    users.create_index("phone", sparse=True, name="idx_phone_sparse")

    # Partial index — only index documents matching a filter expression
    posts.create_index(
        "likes",
        partialFilterExpression={"published": True},   # only index published posts
        name="idx_likes_published",
    )

    # 2dsphere index — required for geospatial queries ($near, $geoWithin)
    users.create_index([("location", GEOSPHERE)], name="idx_location_geo")

    # background parameter (pre-4.2; ignored in 4.2+ where all builds are non-blocking)
    users.create_index("age", background=True, name="idx_age")

    # List indexes
    for idx in users.list_indexes():
        print(idx)

    # index_information — dict keyed by index name
    info = posts.index_information()
    print(info.keys())

    # hint() — force the query planner to use a specific index
    result = list(posts.find({"author_id": ObjectId()}).hint("idx_author_date"))

    # explain() — inspect query plan
    plan = posts.find({"published": True}).explain()
    print("winning plan:", plan["queryPlanner"]["winningPlan"])


# ---------------------------------------------------------------------------
# 7. Schema Validation
# ---------------------------------------------------------------------------

def demo_schema_validation(db):
    """Add a $jsonSchema validator to the users collection via collMod."""
    db.command(
        "collMod",
        "users",
        validator={
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["username", "email", "active"],
                "properties": {
                    "username": {
                        "bsonType": "string",
                        "minLength": 3,
                        "maxLength": 50,
                        "description": "must be a string, 3–50 chars, required",
                    },
                    "email": {
                        "bsonType": "string",
                        "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                        "description": "must match email pattern",
                    },
                    "age": {
                        "bsonType": "int",
                        "minimum": 0,
                        "maximum": 150,
                    },
                    "active": {"bsonType": "bool"},
                },
                "additionalProperties": True,   # allow extra fields
            }
        },
        validationLevel="moderate",     # only validate inserts and updates that match existing docs
        validationAction="error",       # "warn" to log instead of reject
    )
    print("schema validator applied")


# ---------------------------------------------------------------------------
# 8. Transactions
# ---------------------------------------------------------------------------

def demo_transactions(db):
    """
    Transactions require a replica set (or sharded cluster).
    Pattern 1: manual start/commit/abort.
    Pattern 2: with_transaction() callback — retries on transient errors.
    """
    client = db.client

    # --- Pattern 1: manual ---
    with client.start_session() as session:
        session.start_transaction(
            read_concern={"level": "snapshot"},
            write_concern={"w": "majority"},
        )
        try:
            db["users"].update_one(
                {"username": "alice"},
                {"$inc": {"post_count": 1}},
                session=session,
            )
            db["posts"].insert_one(
                {"title": "Transactional Post", "author_id": ObjectId(), "likes": 0},
                session=session,
            )
            session.commit_transaction()
            print("transaction committed")
        except Exception as e:
            session.abort_transaction()
            print("transaction aborted:", e)

    # --- Pattern 2: with_transaction() — preferred ---
    def transfer_callback(session):
        db["users"].update_one(
            {"username": "alice"}, {"$inc": {"credits": -10}}, session=session
        )
        db["users"].update_one(
            {"username": "bob"}, {"$inc": {"credits": 10}}, session=session
        )

    with client.start_session() as session:
        session.with_transaction(transfer_callback)
        print("with_transaction completed")


# ---------------------------------------------------------------------------
# 9. Aggregation Basics
# ---------------------------------------------------------------------------

def demo_aggregation_basics(db):
    """Core aggregation stages."""
    posts = db["posts"]

    pipeline = [
        # $match — filter early to reduce documents
        {"$match": {"published": True}},

        # $addFields — add computed fields without losing existing ones
        {"$addFields": {"year": {"$year": "$published_at"}}},

        # $group — aggregate by key
        {"$group": {
            "_id": "$year",
            "total_posts": {"$sum": 1},
            "total_likes": {"$sum": "$likes"},
            "avg_likes": {"$avg": "$likes"},
            "tags": {"$push": "$tags"},          # accumulate arrays
            "max_likes": {"$max": "$likes"},
        }},

        # $project — reshape output
        {"$project": {
            "year": "$_id",
            "total_posts": 1,
            "avg_likes": {"$round": ["$avg_likes", 1]},
            "_id": 0,
        }},

        # $sort / $limit / $skip
        {"$sort": {"total_posts": -1}},
        {"$skip": 0},
        {"$limit": 10},
    ]

    # $unwind — deconstruct array field into individual documents
    unwind_pipeline = [
        {"$match": {"published": True}},
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]

    # $replaceRoot — promote a nested document to the top level
    replace_pipeline = [
        {"$match": {}},
        {"$replaceRoot": {"newRoot": {"$mergeObjects": [{"_id": "$_id"}, "$meta"]}}},
    ]

    # $count — count matching documents (as a stage)
    count_pipeline = [{"$match": {"published": True}}, {"$count": "published_count"}]

    # $sample — random sample of N documents
    sample_pipeline = [{"$sample": {"size": 3}}]

    for doc in posts.aggregate(pipeline):
        print(doc)
    print("tag counts:", list(posts.aggregate(unwind_pipeline)))
    print("count:", list(posts.aggregate(count_pipeline)))
    print("sample:", list(posts.aggregate(sample_pipeline)))


# ---------------------------------------------------------------------------
# 10. Aggregation $lookup (joins)
# ---------------------------------------------------------------------------

def demo_lookup(db):
    """Simple $lookup and pipeline $lookup with conditions."""
    posts = db["posts"]

    # Simple join — like SQL LEFT JOIN
    simple_lookup = [
        {"$match": {"published": True}},
        {"$lookup": {
            "from": "users",            # foreign collection
            "localField": "author_id",  # field in posts
            "foreignField": "_id",      # field in users
            "as": "author",             # output array field name
        }},
        # $unwind because author is always a single doc; preserveNullAndEmpty for optional
        {"$unwind": {"path": "$author", "preserveNullAndEmptyArrays": True}},
        {"$project": {"title": 1, "author.username": 1, "author.email": 1}},
    ]

    # Pipeline $lookup — more powerful: join with filter conditions
    pipeline_lookup = [
        {"$match": {"published": True}},
        {"$lookup": {
            "from": "comments",
            "let": {"post_id": "$_id"},          # expose local var to pipeline
            "pipeline": [
                {"$match": {"$expr": {"$eq": ["$post_id", "$$post_id"]}}},
                {"$match": {"score": {"$gte": 4}}},  # only high-score comments
                {"$sort": {"score": -1}},
                {"$limit": 5},
                {"$project": {"author": 1, "text": 1, "score": 1}},
            ],
            "as": "top_comments",
        }},
        {"$addFields": {"comment_count": {"$size": "$top_comments"}}},
    ]

    print("simple lookup:", list(posts.aggregate(simple_lookup)))
    print("pipeline lookup:", list(posts.aggregate(pipeline_lookup)))


# ---------------------------------------------------------------------------
# 11. Aggregation $facet
# ---------------------------------------------------------------------------

def demo_facet(db):
    """$facet runs multiple sub-pipelines on the same input in one pass."""
    posts = db["posts"]

    pipeline = [
        {"$match": {"published": True}},
        {"$facet": {
            # Sub-pipeline 1: total count
            "total": [
                {"$count": "count"},
            ],
            # Sub-pipeline 2: posts per year
            "by_year": [
                {"$group": {"_id": {"$year": "$published_at"}, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}},
            ],
            # Sub-pipeline 3: likes distribution buckets
            "likes_buckets": [
                {"$bucket": {
                    "groupBy": "$likes",
                    "boundaries": [0, 10, 50, 200, 1000],
                    "default": "1000+",
                    "output": {"count": {"$sum": 1}, "titles": {"$push": "$title"}},
                }},
            ],
            # Sub-pipeline 4: top 3 posts by likes
            "top_posts": [
                {"$sort": {"likes": -1}},
                {"$limit": 3},
                {"$project": {"title": 1, "likes": 1}},
            ],
        }},
    ]

    result = list(posts.aggregate(pipeline))
    print("facet result:", result)


# ---------------------------------------------------------------------------
# 12. Aggregation $graphLookup
# ---------------------------------------------------------------------------

def demo_graph_lookup(db):
    """$graphLookup for recursive relationships — org hierarchy."""
    employees = db["employees"]

    # Seed data
    employees.drop()
    employees.insert_many([
        {"_id": 1, "name": "CEO",      "reports_to": None},
        {"_id": 2, "name": "CTO",      "reports_to": 1},
        {"_id": 3, "name": "CFO",      "reports_to": 1},
        {"_id": 4, "name": "Eng Lead", "reports_to": 2},
        {"_id": 5, "name": "Alice",    "reports_to": 4},
        {"_id": 6, "name": "Bob",      "reports_to": 4},
    ])

    # Find all reports under the CTO, up to 5 levels deep
    pipeline = [
        {"$match": {"name": "CTO"}},
        {"$graphLookup": {
            "from": "employees",
            "startWith": "$_id",            # start traversal from this value
            "connectFromField": "_id",       # follow _id in the current result
            "connectToField": "reports_to", # to find docs where reports_to matches
            "as": "all_reports",
            "maxDepth": 5,
            "depthField": "depth",          # annotate each result with its depth
            "restrictSearchWithMatch": {    # only traverse active employees
                "active": {"$ne": False}
            },
        }},
        {"$project": {"name": 1, "all_reports.name": 1, "all_reports.depth": 1}},
    ]

    for doc in employees.aggregate(pipeline):
        print(doc)


# ---------------------------------------------------------------------------
# 13. Aggregation $setWindowFields (MongoDB 5.0+)
# ---------------------------------------------------------------------------

def demo_set_window_fields(db):
    """Running totals, rank, and lag/lead equivalent using $setWindowFields."""
    posts = db["posts"]

    pipeline = [
        {"$match": {"published": True}},
        {"$setWindowFields": {
            "partitionBy": None,                       # single partition = all docs
            "sortBy": {"published_at": 1},             # required for ordered windows
            "output": {
                # Running total of likes up to the current doc
                "running_likes": {
                    "$sum": "$likes",
                    "window": {"documents": ["unbounded", "current"]},
                },
                # Rank (1-based, dense rank)
                "rank_by_likes": {
                    "$rank": {},  # no window needed for $rank / $denseRank
                },
                # Preceding doc's likes (lag equivalent)
                "prev_likes": {
                    "$shift": {"output": "$likes", "by": -1, "default": None},
                },
                # Following doc's likes (lead equivalent)
                "next_likes": {
                    "$shift": {"output": "$likes", "by": 1, "default": None},
                },
                # 3-doc moving average
                "moving_avg_likes": {
                    "$avg": "$likes",
                    "window": {"documents": [-1, 1]},   # current ± 1 doc
                },
            },
        }},
        {"$project": {
            "title": 1, "likes": 1,
            "running_likes": 1, "rank_by_likes": 1,
            "prev_likes": 1, "next_likes": 1, "moving_avg_likes": 1,
        }},
    ]

    for doc in posts.aggregate(pipeline):
        print(doc)


# ---------------------------------------------------------------------------
# 14. Aggregation $merge
# ---------------------------------------------------------------------------

def demo_merge(db):
    """$merge writes aggregation results into another collection."""
    posts = db["posts"]

    pipeline = [
        {"$match": {"published": True}},
        {"$group": {
            "_id": {"$year": "$published_at"},
            "total_posts": {"$sum": 1},
            "total_likes": {"$sum": "$likes"},
        }},
        {"$merge": {
            "into": "posts_yearly_stats",   # target collection (same DB)
            "on": "_id",                    # merge key — must be unique in target
            "whenMatched": "merge",         # merge fields if doc exists
            "whenNotMatched": "insert",     # insert if new
        }},
    ]

    posts.aggregate(pipeline)
    print("merged into posts_yearly_stats")


# ---------------------------------------------------------------------------
# 15. Text Search
# ---------------------------------------------------------------------------

def demo_text_search(db):
    """Full-text search using a text index."""
    posts = db["posts"]

    # Ensure text index exists (idempotent)
    try:
        posts.create_index(
            [("title", TEXT), ("body", TEXT)],
            weights={"title": 10, "body": 1},
            name="idx_text_search",
        )
    except Exception:
        pass  # index already exists

    # $text search — returns docs where any indexed field matches
    results = list(posts.find(
        {"$text": {"$search": "mongodb nosql",  # OR by default
                   "$caseSensitive": False,
                   "$diacriticSensitive": False}},
        {"score": {"$meta": "textScore"}, "title": 1},  # project relevance score
    ).sort([("score", {"$meta": "textScore"})]))         # sort by relevance

    print("text search results:", results)

    # Phrase search — wrap in quotes
    phrase = list(posts.find({"$text": {"$search": '"document database"'}}))

    # Exclude a term — prefix with -
    excluded = list(posts.find({"$text": {"$search": "mongodb -nosql"}}))


# ---------------------------------------------------------------------------
# 16. Change Streams
# ---------------------------------------------------------------------------

def demo_change_streams_sync(db):
    """
    Synchronous change stream — watch a collection for changes.
    In production, run this in a background thread or process.
    """
    posts = db["posts"]

    # Watch with full document on update (not just the diff)
    with posts.watch(
        pipeline=[{"$match": {"operationType": {"$in": ["insert", "update", "delete"]}}}],
        full_document="updateLookup",   # fetch full doc on update events
    ) as stream:
        resume_token = None
        for change in stream:
            op = change["operationType"]
            doc_key = change["documentKey"]
            resume_token = change["_id"]   # save to resume after restart

            if op == "insert":
                print("inserted:", change["fullDocument"])
            elif op == "update":
                print("updated:", change.get("fullDocument"))
            elif op == "delete":
                print("deleted key:", doc_key)

            # In a real app: persist resume_token, then break after processing
            break  # break immediately in this demo

    # Resume from a saved token (after process restart)
    if resume_token:
        with posts.watch(resume_after=resume_token) as stream:
            for change in stream:
                print("resumed change:", change)
                break


async def demo_change_streams_async():
    """Async change stream using Motor."""
    from motor.motor_asyncio import AsyncIOMotorClient

    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["blog"]
    posts = db["posts"]

    async with posts.watch(full_document="updateLookup") as stream:
        async for change in stream:
            print("async change:", change["operationType"])
            break  # demo: process one event then exit


# ---------------------------------------------------------------------------
# 17. Replica Sets & Read Preferences
# ---------------------------------------------------------------------------

def demo_replica_sets():
    """Read preferences and write concerns for replica sets."""
    # Connect to replica set
    client = MongoClient(
        "mongodb://host1:27017,host2:27017,host3:27017/?replicaSet=rs0"
    )

    # Read preference: route reads to secondaries (reduces primary load)
    db_secondary = client.get_database(
        "blog",
        read_preference=ReadPreference.SECONDARY,
    )

    # NEAREST — lowest network latency, primary or secondary
    db_nearest = client.get_database(
        "blog",
        read_preference=ReadPreference.NEAREST,
    )

    # Write concern: w=majority waits until majority of replica set acknowledges
    from pymongo import WriteConcern
    db_durable = client.get_database(
        "blog",
        write_concern=WriteConcern(w="majority", j=True, wtimeout=5000),
        # j=True: wait for journal flush (durable on disk)
        # wtimeout: give up after 5 s (raises WriteConcernError)
    )

    # w=0 — fire and forget (no acknowledgement, fastest but no error detection)
    db_fast = client.get_database(
        "blog",
        write_concern=WriteConcern(w=0),
    )

    # Per-operation override
    db = client["blog"]
    db["posts"].insert_one(
        {"title": "Urgent post"},
        session=None,
    )
    # Override write concern inline (PyMongo 4+)
    db.get_collection(
        "posts",
        write_concern=WriteConcern(w="majority", j=True),
    ).insert_one({"title": "Durable post"})

    client.close()


# ---------------------------------------------------------------------------
# 18. Motor (async PyMongo)
# ---------------------------------------------------------------------------

async def demo_motor():
    """Async CRUD, aggregation, and change stream with Motor."""
    from motor.motor_asyncio import AsyncIOMotorClient

    client = AsyncIOMotorClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=3000,
    )
    db = client["blog"]
    users = db["users"]
    posts = db["posts"]

    # Async insert
    result = await users.insert_one({
        "username": "motor_user",
        "email": "motor@example.com",
        "active": True,
    })
    uid = result.inserted_id

    # Async find_one
    user = await users.find_one({"_id": uid})
    print("motor user:", user)

    # Async cursor iteration
    async for post in posts.find({"published": True}).sort("likes", -1).limit(5):
        print("  post:", post.get("title"))

    # Async update
    await users.update_one({"_id": uid}, {"$set": {"verified": True}})

    # Async aggregation
    pipeline = [
        {"$match": {"published": True}},
        {"$group": {"_id": None, "total_likes": {"$sum": "$likes"}}},
    ]
    async for doc in posts.aggregate(pipeline):
        print("motor agg:", doc)

    # Async delete
    await users.delete_one({"_id": uid})

    # Async change stream — watch for one event, trigger it concurrently.
    # The writer coroutine runs alongside the stream so the stream doesn't hang.
    # Requires a replica set; change streams are not supported on standalones.
    async def _write_trigger():
        # Small sleep lets the stream cursor open before the write fires.
        await asyncio.sleep(0.1)
        await posts.insert_one({"title": "motor change stream trigger",
                                "published": True, "likes": 0})

    async with posts.watch(full_document="updateLookup") as stream:
        writer = asyncio.create_task(_write_trigger())
        async for change in stream:
            print("motor change:", change["operationType"],
                  change.get("fullDocument", {}).get("title", ""))
            break
        await writer  # ensure the task is cleaned up

    client.close()


# ---------------------------------------------------------------------------
# 19. MongoEngine ODM
# ---------------------------------------------------------------------------

def demo_mongoengine():
    """MongoEngine document definitions, queries, signals, and aggregation."""
    import mongoengine as me
    from mongoengine import signals

    me.connect("blog", host="localhost", port=27017)

    # --- Document definitions ---

    class Address(me.EmbeddedDocument):
        """Embedded document — stored inside the parent, no separate collection."""
        street = me.StringField(max_length=200)
        city = me.StringField(max_length=100)
        country = me.StringField(max_length=100, default="US")

    class User(me.Document):
        username = me.StringField(required=True, max_length=50, unique=True)
        email = me.EmailField(required=True, unique=True)
        age = me.IntField(min_value=0, max_value=150)
        active = me.BooleanField(default=True)
        address = me.EmbeddedDocumentField(Address)   # single embedded doc
        tags = me.ListField(me.StringField(max_length=50))  # list of strings
        meta_info = me.DictField()                    # arbitrary key-value
        created_at = me.DateTimeField(default=datetime.now)

        meta = {
            "collection": "users",
            "indexes": ["email", ("username", "active")],
        }

        def __str__(self):
            return self.username

    class Post(me.Document):
        title = me.StringField(required=True, max_length=200)
        body = me.StringField()
        author = me.ReferenceField(User, reverse_delete_rule=me.CASCADE)
        tags = me.ListField(me.StringField())
        likes = me.IntField(default=0)
        published = me.BooleanField(default=False)
        published_at = me.DateTimeField()

        meta = {"collection": "posts"}

    class Comment(me.Document):
        post = me.ReferenceField(Post, reverse_delete_rule=me.CASCADE)
        author = me.ReferenceField(User)
        text = me.StringField(required=True)
        score = me.IntField(min_value=1, max_value=5)
        created_at = me.DateTimeField(default=datetime.now)

        meta = {"collection": "comments"}

    # --- Signals ---

    def on_post_save(sender, document, created, **kwargs):
        """Called after every Post save."""
        if created:
            print(f"New post created: {document.title}")

    signals.post_save.connect(on_post_save, sender=Post)

    # --- CRUD ---

    # Create & save
    user = User(username="meuser", email="me@example.com", age=28)
    user.address = Address(street="123 Main St", city="NY")
    user.tags = ["python", "mongodb"]
    user.save()

    post = Post(title="MongoEngine Guide", author=user, published=True)
    post.save()

    # Queryset methods
    active_users = User.objects(active=True)
    alice = User.objects(username="meuser").first()
    older = User.objects(age__gte=25)                    # gte => __gte
    tagged = User.objects(tags__in=["python"])            # array contains
    User.objects(username__icontains="me")               # case-insensitive contains
    Post.objects(author=user, published=True)            # reference comparison
    Post.objects.order_by("-likes").limit(10)            # sort + limit
    Post.objects(published=True).count()                 # count
    Post.objects(published=True).only("title", "likes") # projection
    Post.objects(published=False).update(set__published=True)  # bulk update

    # Aggregation via __raw__
    pipeline = [
        {"$match": {"published": True}},
        {"$group": {"_id": None, "total_likes": {"$sum": "$likes"}}},
    ]
    result = list(Post.objects.aggregate(*pipeline))
    print("mongoengine agg:", result)

    # Disconnect (cleanup)
    me.disconnect()


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 20. The 16 MB BSON document limit — how to hit it and how to escape
# ---------------------------------------------------------------------------
def demo_16mb_limit(db) -> None:
    """
    MongoDB enforces a hard 16 MB limit on every BSON document.
    This includes all nested subdocuments and arrays.

    Why it exists: BSON uses a 32-bit signed integer for document length,
    capping the theoretical max at ~2 GB. MongoDB deliberately chose 16 MB
    as a practical ceiling to keep working sets in memory and discourage
    designs that embed unbounded data. The limit applies to:
      - Individual documents in any collection
      - Pipeline intermediate documents ($group accumulator output)
      - Command responses (find, aggregate result documents)
      - Oplog entries (so transactions are also capped)

    PostgreSQL comparison:
      - JSONB columns: no per-column size limit beyond the 1 GB row limit
        (itself rarely hit; TOAST stores values > 2 KB out-of-line)
      - TEXT / BYTEA: 1 GB per field, transparent overflow to disk
      - No equivalent gotcha: you can store an arbitrarily deep JSON tree
        and Postgres will TOAST it without blowing up

    Common ways to hit the 16 MB wall:
      1. Growing arrays — $push without pruning (audit logs, chat messages,
         time-series events, comment threads)
      2. Storing binary blobs inline (images, PDFs, video thumbnails)
      3. User-supplied nested JSON with no depth/size validation
      4. Aggregation pipelines that accumulate into a single document
         ($group + $push on a large collection → explodes at 16 MB)

    Error you'll see:
      pymongo.errors.DocumentTooLarge: BSON document too large (16793600 > 16777216)
      OR during aggregation:
      pymongo.errors.OperationFailure: document too large

    Fixes:
      A. Bucketing pattern — split the growing array across many documents
      B. GridFS — for binary data > 16 MB
      C. Reference pattern — stop embedding; store the growing part in its
         own collection (the normalisation you avoided for performance)
      D. Capped arrays — $push + $slice to evict old entries
      E. $merge into a summary collection so aggregation doesn't accumulate
    """
    print("\n[16mb] Demonstrating the 16 MB trap …")

    db.event_log.drop()
    db.event_buckets.drop()

    # ── Anti-pattern: unbounded $push into one document ──────────────────
    # Imagine this runs for every user event (clicks, page views, API calls).
    # Works fine for months, then one busy user's doc hits 16 MB and the
    # next $push raises DocumentTooLarge.
    user_id = "user-42"
    db.event_log.insert_one({"_id": user_id, "events": []})

    # Simulate a few events (safe — real blowup needs ~200k+ events)
    small_event = {"ts": datetime.now(timezone.utc).isoformat(),
                   "action": "click", "page": "/home", "metadata": "x" * 50}
    db.event_log.update_one(
        {"_id": user_id},
        {"$push": {"events": {"$each": [small_event] * 100}}},
    )
    doc_size = len(str(db.event_log.find_one({"_id": user_id})))
    print(f"  Anti-pattern doc size after 100 events: ~{doc_size} bytes "
          f"(grows linearly, hits 16 MB at ~{16_777_216 // (doc_size // 100):,} events)")

    # ── Fix A: Capped array ($push + $slice) ─────────────────────────────
    # Keep only the last N events in the document. Oldest entries are dropped.
    # Use when you only need a recent window (e.g., last 500 actions for UX).
    KEEP_LAST = 500
    db.event_log.update_one(
        {"_id": user_id},
        {"$push": {"events": {
            "$each":  [small_event],
            "$slice": -KEEP_LAST,    # negative = keep the last N
        }}},
    )
    print(f"  Fix A (capped $slice): array capped at {KEEP_LAST} entries")

    # ── Fix B: Bucketing pattern ──────────────────────────────────────────
    # Each bucket document holds at most BUCKET_SIZE events.
    # A new bucket is created when the current one is full.
    # Query: find all buckets for a user, unwind events, filter by ts range.
    # This is the pattern MongoDB Atlas Data Lake / time-series collections
    # use internally.
    BUCKET_SIZE = 200

    def append_event_bucketed(user_id: str, event: dict) -> None:
        result = db.event_buckets.update_one(
            {"user_id": user_id, "count": {"$lt": BUCKET_SIZE}},
            {"$push":  {"events": event},
             "$inc":   {"count": 1},
             "$setOnInsert": {
                 "user_id":    user_id,
                 "bucket_seq": db.event_buckets.count_documents(
                     {"user_id": user_id}),
             }},
            upsert=True,
        )
        return result

    for i in range(5):
        append_event_bucketed(user_id, {**small_event, "seq": i})

    bucket_count = db.event_buckets.count_documents({"user_id": user_id})
    print(f"  Fix B (bucketing): {bucket_count} bucket document(s), "
          f"max {BUCKET_SIZE} events each → no single doc > ~{BUCKET_SIZE * 200} bytes")

    # ── Fix C: GridFS (binary data) ───────────────────────────────────────
    # GridFS splits files into 255 KB chunks stored in fs.chunks, with
    # metadata in fs.files. No 16 MB limit per file.
    # PyMongo usage (requires gridfs package):
    print("\n  Fix C (GridFS for binary blobs):")
    print("""
    import gridfs
    fs = gridfs.GridFS(db)

    # Store a large file (any size):
    with open("report.pdf", "rb") as f:
        file_id = fs.put(f, filename="report.pdf", content_type="application/pdf",
                         user_id="user-42")

    # Retrieve:
    grid_out = fs.get(file_id)
    data = grid_out.read()

    # Query metadata (no content loaded):
    meta = db["fs.files"].find_one({"_id": file_id})
    print(meta["length"])   # bytes

    # Never store PDFs / images inline in a document — always use GridFS
    # or an object store (S3, GCS). A single 20 MB PDF will crash the insert.
    """)

    # ── Fix D: Aggregation blowup — use $merge to escape ─────────────────
    # pipeline with $group + $push accumulates ALL matching docs into one
    # group document. If the group has > ~100k docs with any payload, it
    # hits 16 MB before the pipeline finishes.
    print("  Fix D (aggregation blowup with $group + $push):")
    print("""
    # BAD — accumulates all events into one result document:
    db.raw_events.aggregate([
        {"$match": {"user_id": "user-42"}},
        {"$group": {
            "_id": "$user_id",
            "all_events": {"$push": "$$ROOT"},   # ← blows up at 16 MB
        }},
    ])

    # GOOD — stream results into a collection with $merge; no 16 MB cap:
    db.raw_events.aggregate([
        {"$match": {"user_id": "user-42"}},
        {"$group": {
            "_id": {"user_id": "$user_id", "day": {"$dateToString":
                {"format": "%Y-%m-%d", "date": "$ts"}}},
            "event_count": {"$sum": 1},
            "action_types": {"$addToSet": "$action"},
        }},
        {"$merge": {
            "into": "user_daily_summary",
            "on":   "_id",
            "whenMatched": "merge",
            "whenNotMatched": "insert",
        }},
    ])
    # Each result document is a small summary — well under 16 MB.
    """)

    # ── Size check utility ────────────────────────────────────────────────
    # Use this to monitor document growth before it becomes an incident.
    print("  Monitoring document sizes (run periodically):")
    print("""
    import bson

    def check_large_docs(collection, warn_mb=8, limit=10):
        \"\"\"Find documents approaching the 16 MB limit.\"\"\"
        results = []
        for doc in collection.find({}, limit=1000):
            size = len(bson.encode(doc))
            if size > warn_mb * 1024 * 1024:
                results.append({"_id": doc["_id"], "size_mb": size / 1024 / 1024})
        return sorted(results, key=lambda x: -x["size_mb"])[:limit]

    # Or via the aggregation $bsonSize operator (MongoDB 4.4+):
    db.event_log.aggregate([
        {"$project": {"size": {"$bsonSize": "$$ROOT"}}},
        {"$match":   {"size": {"$gt": 8 * 1024 * 1024}}},   # > 8 MB
        {"$sort":    {"size": -1}},
        {"$limit":   20},
    ])
    """)

    print("\n  [16mb] Summary:")
    print("""
    | Pattern              | Max doc size | Best for                          |
    |----------------------|-------------|-----------------------------------|
    | Inline array (naive) | 16 MB hard  | Small, bounded arrays (<1k items) |
    | Capped $slice        | Controlled  | Recent-window use cases           |
    | Bucketing            | Unlimited*  | Time-series, unbounded events     |
    | GridFS               | Unlimited*  | Binary blobs > 16 MB              |
    | Reference (normalize)| Unlimited*  | Audits, chat, anything unbounded  |
    | $merge aggregation   | Unlimited*  | Large aggregation outputs         |
    * total data unlimited; each individual document still ≤ 16 MB
    """)


# ---------------------------------------------------------------------------
# 21. Normalized query performance trap
#     Demonstrates why multi-$lookup / $graphLookup pipelines scan entire
#     collections and why cursor skip-based pagination compounds the cost.
# ---------------------------------------------------------------------------
def demo_normalized_query_perf_trap(db) -> None:
    """
    Anti-pattern: treating MongoDB like a relational DB.

    Schema (highly normalized, SQL-style):
        authors   { _id, name, country }
        tags      { _id, label }
        posts     { _id, author_id, tag_ids[], title, body }
        comments  { _id, post_id, author_id, body }
        likes     { _id, post_id, user_id }

    Query goal: "For each post in category 'tech', get author name, all
    comment authors, like count — paginated, page 3 of 25-per-page."

    This mirrors a JOIN-heavy SQL query. In MongoDB it causes:
      1. Full collection scan on posts (no index on tag_ids unless
         multikey index exists — and $lookup ignores the caller's index).
      2. Per-document N correlated sub-pipelines for comments $lookup.
      3. $graphLookup on comment authors (recursive org-chart style) —
         one round-trip per level per document.
      4. SKIP(50) forces the server to materialise and discard 50 docs
         before returning page 3.
    """
    print("\n[perf-trap] Setting up normalized collections …")

    # ── seed data ─────────────────────────────────────────────────────────
    db.pt_authors.drop()
    db.pt_tags.drop()
    db.pt_posts.drop()
    db.pt_comments.drop()
    db.pt_likes.drop()

    author_ids = db.pt_authors.insert_many([
        {"name": "Alice",   "country": "US"},
        {"name": "Bob",     "country": "UK"},
        {"name": "Charlie", "country": "AU"},
    ]).inserted_ids

    tag_ids = db.pt_tags.insert_many([
        {"label": "tech"},
        {"label": "science"},
    ]).inserted_ids

    post_ids = db.pt_posts.insert_many([
        {"author_id": author_ids[0], "tag_ids": [tag_ids[0]], "title": "AI trends",    "body": "…"},
        {"author_id": author_ids[1], "tag_ids": [tag_ids[0]], "title": "Cloud costs",   "body": "…"},
        {"author_id": author_ids[2], "tag_ids": [tag_ids[1]], "title": "CRISPR update", "body": "…"},
    ]).inserted_ids

    db.pt_comments.insert_many([
        {"post_id": post_ids[0], "author_id": author_ids[1], "body": "Great post!"},
        {"post_id": post_ids[0], "author_id": author_ids[2], "body": "Agreed."},
        {"post_id": post_ids[1], "author_id": author_ids[0], "body": "Costs are wild."},
    ])

    db.pt_likes.insert_many([
        {"post_id": post_ids[0], "user_id": author_ids[1]},
        {"post_id": post_ids[0], "user_id": author_ids[2]},
        {"post_id": post_ids[1], "user_id": author_ids[0]},
    ])

    # ── the slow pipeline ─────────────────────────────────────────────────
    # PERF NOTE: every $lookup here is an O(n) nested-loop scan unless
    # `from` collection has an index on the join field — and even then
    # the planner may choose a collection scan if selectivity is low.
    PAGE = 0          # page index (0-based)
    PAGE_SIZE = 2

    pipeline = [
        # Stage 1 — filter posts by tag. Requires multikey index on tag_ids
        # for index scan; without it: COLLSCAN on pt_posts.
        {"$match": {"tag_ids": {"$in": tag_ids[:1]}}},  # 'tech' tag

        # Stage 2 — join author. One correlated lookup per matched post.
        {"$lookup": {
            "from":         "pt_authors",
            "localField":   "author_id",
            "foreignField": "_id",
            "as":           "author_doc",
        }},
        {"$unwind": "$author_doc"},  # flatten single-element array

        # Stage 3 — join comments. Each post → full scan of pt_comments
        # unless an index exists on pt_comments.post_id.
        {"$lookup": {
            "from":         "pt_comments",
            "localField":   "_id",
            "foreignField": "post_id",
            "as":           "comments",
        }},

        # Stage 4 — for each comment, join commenter profile via graphLookup.
        # graphLookup is breadth-first; here depth 0 just resolves one level,
        # but in an org-chart scenario it fans out exponentially.
        {"$lookup": {
            "from":              "pt_authors",
            "startWith":         "$comments.author_id",
            "connectFromField":  "author_id",
            "connectToField":    "_id",
            "as":                "comment_authors",
            "maxDepth":          0,          # just the direct parent level
            "depthField":        "depth",
        }},

        # Stage 5 — like count: group likes per post
        {"$lookup": {
            "from":         "pt_likes",
            "localField":   "_id",
            "foreignField": "post_id",
            "as":           "likes",
        }},

        # Stage 6 — reshape
        {"$group": {
            "_id":            "$_id",
            "title":          {"$first": "$title"},
            "author_name":    {"$first": "$author_doc.name"},
            "comment_count":  {"$sum": {"$size": "$comments"}},
            "like_count":     {"$sum": {"$size": "$likes"}},
            "commenters":     {"$first": "$comment_authors.name"},
        }},

        # Stage 7 — PAGINATION TRAP:
        # $skip + $limit works, but $skip(N) forces MongoDB to pull N
        # documents through ALL prior stages and discard them.
        # Cost grows linearly with page number: page 100 of 25 = 2 500
        # wasted pipeline executions. Use range-based pagination instead.
        {"$sort":  {"_id": ASCENDING}},   # stable sort required for skip/limit
        {"$skip":  PAGE * PAGE_SIZE},      # ← O(PAGE * PAGE_SIZE) wasted work
        {"$limit": PAGE_SIZE},
    ]

    results = list(db.pt_posts.aggregate(pipeline))
    for r in results:
        print(f"  {r['title']} | author={r['author_name']} "
              f"| comments={r['comment_count']} | likes={r['like_count']} "
              f"| commenters={r['commenters']}")

    # ── explain (shows COLLSCAN / stage costs) ────────────────────────────
    # Run this manually in mongosh / Compass to see executionStats.
    # db.pt_posts.aggregate(pipeline, explain=True) or:
    # db.command("aggregate", "pt_posts", pipeline=pipeline, explain=True)
    print("\n  [explain hint] Run:")
    print("    db.command('aggregate','pt_posts',pipeline=pipeline,explain=True)")
    print("  Look for 'COLLSCAN' on pt_authors/pt_comments and high")
    print("  'nReturned' vs 'totalDocsExamined' ratios.")

    # ── better alternative: embed hot read data ───────────────────────────
    # Instead of 5 lookups, denormalize into post document:
    # {
    #   _id, title, body,
    #   author: { _id, name },        ← embedded snapshot
    #   tag_labels: ["tech"],          ← duplicated from tags
    #   comment_count: 12,             ← counter (inc on insert)
    #   like_count: 34,                ← counter (inc on like)
    # }
    # Single IXSCAN on { "tag_labels": 1, _id: 1 } covers filter + sort.
    # Cursor pagination: { _id: { $gt: last_seen_id } } replaces $skip.
    print("\n  [better] Embed author snapshot + counters; cursor-paginate on _id.")
    print("  Query: db.posts.find({'tag_labels':'tech', '_id':{'$gt': last_id}})")
    print("         .sort({'_id':1}).limit(25)")
    print("  Cost: single IXSCAN — O(1) regardless of page number.")


# ---------------------------------------------------------------------------
# 21. Write scale and sharding strategies (toy/mock — requires mongos)
# ---------------------------------------------------------------------------
def demo_write_sharding(client) -> None:
    """
    Sharding routes documents to shards by shard key. Commands below show the
    mongosh / PyMongo admin API; they need a sharded cluster (mongos + config
    servers). In a standalone they raise OperationFailure — run as a reference.

    Three strategies:
      A. Hashed shard key   — even write distribution, no range scans
      B. Ranged shard key   — range scans efficient, hotspot risk on monotonic keys
      C. Zone sharding      — pin data to geographic regions / tiers
    """
    admin = client.admin

    # ── A. Hashed sharding ────────────────────────────────────────────────
    # Best for: high-cardinality keys with no range query need (e.g. user_id).
    # Even write distribution: mongos hashes the key → random shard assignment.
    # Downside: range queries on shard key fan out to ALL shards (scatter-gather).
    print("\n[sharding] A. Hashed shard key (user_id)")
    print("""
    # mongosh equivalent:
    sh.enableSharding("myapp")
    sh.shardCollection("myapp.events", { user_id: "hashed" })

    # PyMongo:
    # admin.command("enableSharding", "myapp")
    # admin.command("shardCollection", "myapp.events",
    #               key={"user_id": "hashed"})
    #
    # Result: documents spread evenly. A write for user_id=42 hashes to
    # shard B; user_id=43 may land on shard A. No sequential hotspot.
    """)

    # ── B. Ranged sharding ────────────────────────────────────────────────
    # Best for: time-series or ordered keys where you need efficient range scans.
    # Risk: monotonically increasing keys (ObjectId, timestamp) always write
    # to the last (max-key) chunk → single shard hotspot until chunk splits+migrates.
    # Mitigation: prefix with a random bucket (bucket + timestamp compound key).
    print("[sharding] B. Ranged shard key (ts — with hotspot warning)")
    print("""
    # sh.shardCollection("myapp.metrics", { ts: 1 })  ← hotspot!
    #
    # Safer compound: prepend a low-cardinality "bucket" field so writes
    # distribute across N buckets while retaining per-bucket range scans.
    # sh.shardCollection("myapp.metrics", { bucket: 1, ts: 1 })
    #
    # bucket = hash(device_id) % NUM_SHARDS   (computed at write time)
    #
    # Range query on one device:
    #   db.metrics.find({ bucket: b, ts: { $gte: t0, $lte: t1 } })
    #   → targets single shard (ixscan on {bucket,ts} compound index)
    """)

    # ── C. Zone sharding ──────────────────────────────────────────────────
    # Pin a range of shard key values to a named zone → zone lives on a
    # specific shard (or set of shards). Common for data residency (GDPR).
    print("[sharding] C. Zone sharding (geographic data residency)")
    print("""
    # 1. Tag shards with zone names (mongosh):
    #    sh.addShardToZone("shard0", "EU")
    #    sh.addShardToZone("shard1", "US")
    #
    # 2. Define key ranges that map to zones:
    #    sh.updateZoneKeyRange("myapp.users",
    #        { region: "EU" }, { region: "EU~" }, "EU")
    #    sh.updateZoneKeyRange("myapp.users",
    #        { region: "US" }, { region: "US~" }, "US")
    #
    # 3. Shard collection on the zone key:
    #    sh.shardCollection("myapp.users", { region: 1, _id: 1 })
    #
    # Result: all documents with region="EU" live only on shard0 (EU data
    # centre). region="US" documents live only on shard1.
    # Writes from EU clients routed by mongos → shard0 only (no fan-out).
    """)

    # ── Shard key selection rules ─────────────────────────────────────────
    print("[sharding] Shard key selection rules:")
    print("""
    | Property          | Goal                                          |
    |-------------------|-----------------------------------------------|
    | High cardinality  | Enough distinct values to fill many chunks    |
    | Low frequency     | No single value dominates (jumbo chunks)      |
    | Non-monotonic     | Avoid single-shard write hotspot              |
    | Query coverage    | Key appears in most queries → targeted shards |
    | Immutable         | Shard key cannot be changed after insert      |

    Anti-patterns:
      • { _id: 1 }        — ObjectId is monotonic → hotspot
      • { status: 1 }     — low cardinality (few statuses) → jumbo chunks
      • { email: 1 }      — good cardinality but rarely in range queries
    """)


# ---------------------------------------------------------------------------
# 22. Distributed transaction coordinator (cross-shard ACID)
# ---------------------------------------------------------------------------
def demo_distributed_transactions(client) -> None:
    """
    MongoDB uses a two-phase commit (2PC) coordinator built into mongos
    (or the primary of a replica set for single-RS transactions).

    Cross-shard transaction flow:
      1. Application opens a session + starts a transaction.
      2. Each write is sent to the relevant shard(s) with the session lsid.
      3. mongos elects itself coordinator; sends prepareTransaction to each shard.
      4. Each shard WAL-logs the prepare record; votes commit/abort.
      5. Coordinator writes commit record to config server (durable decision).
      6. Coordinator sends commitTransaction to all shards.
      7. Shards apply and release locks.

    If coordinator crashes after step 5, any other mongos can recover the
    decision from config server and complete the commit.
    If coordinator crashes before step 5, the transaction is aborted.
    """
    # For a single-RS / standalone, multi-document transactions still use
    # the same session API (just no cross-shard coordination needed).
    print("\n[dist-tx] Cross-shard transaction example (single-RS shown):")

    db = client["txdemo"]
    db.accounts.drop()
    db.audit_log.drop()
    db.accounts.insert_many([
        {"_id": "alice", "balance": 1000, "shard_hint": "shard0"},
        {"_id": "bob",   "balance":  500, "shard_hint": "shard1"},
    ])

    # ── happy path ────────────────────────────────────────────────────────
    def transfer(session, from_id: str, to_id: str, amount: float) -> None:
        """Atomic debit + credit + audit across two 'shards' (simulated)."""
        accounts = db.get_collection("accounts")
        audit    = db.get_collection("audit_log")

        result = accounts.find_one_and_update(
            {"_id": from_id, "balance": {"$gte": amount}},
            {"$inc": {"balance": -amount}},
            session=session,
            return_document=True,
        )
        if result is None:
            raise ValueError(f"Insufficient funds or unknown account: {from_id}")

        accounts.update_one(
            {"_id": to_id},
            {"$inc": {"balance": amount}},
            session=session,
        )
        audit.insert_one(
            {"from": from_id, "to": to_id, "amount": amount,
             "ts": datetime.now(timezone.utc)},
            session=session,
        )

    with client.start_session() as session:
        # with_transaction handles retry on transient errors (network blip,
        # write conflict) automatically — preferred over manual start/commit.
        session.with_transaction(
            lambda s: transfer(s, "alice", "bob", 200),
        )

    alice = db.accounts.find_one({"_id": "alice"})
    bob   = db.accounts.find_one({"_id": "bob"})
    print(f"  After transfer: alice={alice['balance']}  bob={bob['balance']}")
    assert alice["balance"] == 800
    assert bob["balance"]   == 700

    # ── coordinator failure scenarios ─────────────────────────────────────
    print("\n  [dist-tx] Coordinator failure matrix:")
    print("""
    Phase at crash          | Outcome
    ------------------------|--------------------------------------------------
    Before prepareTransaction sent   | Transaction aborts (no prepare record).
    After prepare, before commit     | Coordinator reads config server on restart;
                                     | decision not yet durable → aborts.
    After commit record written      | Any mongos can finalize from config server
                                     | → transaction commits despite coordinator crash.
    Shard unreachable at prepare     | Shard votes abort; coordinator aborts all.
    Network partition during commit  | Pending shards retry on reconnect using the
                                     | durable commit record from config server.
    """)

    # ── write concern for cross-shard transactions ─────────────────────────
    print("  [dist-tx] Recommended write concern for distributed transactions:")
    print("""
    client = MongoClient(
        uri,
        w="majority",          # all writes (including txn commits) wait for
                               # a majority of replica set members to ack
        journal=True,          # flush to WAL before ack — survives crash
        wtimeout=5000,         # ms — avoid hanging forever on node failure
    )
    # Inside a transaction the session inherits the client-level write concern.
    # You can override per-operation only OUTSIDE transactions.
    """)


# ---------------------------------------------------------------------------
# 23. MongoDB locking patterns — pessimistic and optimistic
# ---------------------------------------------------------------------------
def demo_locking_patterns(db) -> None:
    """
    MongoDB's lock hierarchy (internal, not user-visible):
      Global → Database → Collection → Document (WiredTiger row lock)

    Intent locks (IS, IX, S, X) are taken at the higher levels; actual
    data lock is at the document level. This means two concurrent writes
    to *different* documents in the same collection do NOT block each other.

    User-space locking patterns:
      A. Pessimistic — serialize access to a resource before operating.
         Implemented via findOneAndUpdate on a status field (compare-and-set).
         Multi-document pessimistic locks require transactions.
      B. Optimistic — assume no conflict; detect and retry on collision.
         Implemented via a version counter (__v) in the document.
    """
    print("\n[locking] Setup …")
    db.inventory.drop()
    db.tasks.drop()

    db.inventory.insert_many([
        {"_id": "SKU-001", "stock": 10, "locked": False},
        {"_id": "SKU-002", "stock": 5,  "locked": False},
    ])
    db.tasks.insert_many([
        {"_id": "T1", "status": "pending", "owner": None},
        {"_id": "T2", "status": "pending", "owner": None},
    ])

    # ── A1. Pessimistic — atomic test-and-set (single document) ───────────
    # findOneAndUpdate is a single server-side atomic operation.
    # No external lock needed for a single document.
    # Pattern: set locked=True only if locked=False (CAS).
    print("\n[locking] A1. Pessimistic single-doc lock (findOneAndUpdate CAS):")

    def acquire_lock(collection, doc_id: str, owner: str) -> bool:
        result = collection.find_one_and_update(
            {"_id": doc_id, "locked": False},   # guard: only unowned
            {"$set": {"locked": True, "owner": owner,
                       "locked_at": datetime.now(timezone.utc)}},
            return_document=True,
        )
        return result is not None  # None means already locked

    def release_lock(collection, doc_id: str, owner: str) -> None:
        collection.update_one(
            {"_id": doc_id, "owner": owner},    # only the owner can release
            {"$set": {"locked": False, "owner": None, "locked_at": None}},
        )

    acquired = acquire_lock(db.inventory, "SKU-001", "worker-1")
    print(f"  worker-1 acquired: {acquired}")                     # True
    again    = acquire_lock(db.inventory, "SKU-001", "worker-2")
    print(f"  worker-2 acquired (should fail): {again}")          # False
    release_lock(db.inventory, "SKU-001", "worker-1")
    reacquired = acquire_lock(db.inventory, "SKU-001", "worker-2")
    print(f"  worker-2 acquired after release: {reacquired}")     # True
    release_lock(db.inventory, "SKU-001", "worker-2")

    # ── A2. Pessimistic — multi-doc lock via transaction ──────────────────
    # For operations that must atomically lock SEVERAL documents, wrap in a
    # session transaction. All locked docs are held until commit/abort.
    # Requires replica set (transactions not supported on standalone).
    print("\n[locking] A2. Pessimistic multi-doc lock (transaction):")
    print("""
    with client.start_session() as session:
        with session.start_transaction():
            # Lock both SKUs atomically before decrementing stock.
            for sku in ["SKU-001", "SKU-002"]:
                result = db.inventory.find_one_and_update(
                    {"_id": sku, "locked": False},
                    {"$set": {"locked": True, "owner": "order-99"}},
                    session=session,
                )
                if result is None:
                    session.abort_transaction()  # one already locked → rollback
                    raise RuntimeError(f"{sku} unavailable")

            # Both docs are write-locked for the duration of this transaction.
            db.inventory.update_many(
                {"_id": {"$in": ["SKU-001", "SKU-002"]}},
                {"$inc": {"stock": -1}},
                session=session,
            )
        # commit releases all locks
    """)

    # ── A3. Task queue — SKIP LOCKED equivalent ───────────────────────────
    # Workers compete for pending tasks. findOneAndUpdate atomically claims
    # a task, preventing double-assignment (pessimistic, single-doc).
    print("[locking] A3. Task queue (atomic claim — pessimistic):")

    def claim_task(worker_id: str):
        return db.tasks.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running", "owner": worker_id,
                       "started_at": datetime.now(timezone.utc)}},
            sort=[("_id", ASCENDING)],   # FIFO
            return_document=True,
        )

    t = claim_task("worker-A")
    print(f"  worker-A claimed: {t['_id'] if t else None}")
    t2 = claim_task("worker-B")
    print(f"  worker-B claimed: {t2['_id'] if t2 else None}")

    # ── B. Optimistic — version field (__v) ───────────────────────────────
    # No lock is held. Each document carries a version counter.
    # Writer reads current version, does work, then updates ONLY IF version
    # hasn't changed. If another writer incremented version first → retry.
    print("\n[locking] B. Optimistic concurrency (version counter __v):")

    db.products.drop()
    db.products.insert_one({"_id": "P1", "price": 9.99, "__v": 0})

    def optimistic_update_price(product_id: str, new_price: float,
                                max_retries: int = 5) -> bool:
        for attempt in range(max_retries):
            doc = db.products.find_one({"_id": product_id})
            if doc is None:
                return False
            current_version = doc["__v"]

            # Simulate work between read and write (in real code this is
            # where business logic / validation happens).

            result = db.products.update_one(
                # Guard: version must still match what we read.
                {"_id": product_id, "__v": current_version},
                {"$set":  {"price": new_price},
                 "$inc":  {"__v": 1}},         # atomically bump version
            )
            if result.modified_count == 1:
                print(f"  Updated price to {new_price} "
                      f"(was v{current_version}, now v{current_version+1})")
                return True
            # modified_count == 0 means a concurrent writer already changed __v
            print(f"  Attempt {attempt+1}: version conflict — retrying …")
        print("  Max retries exceeded; giving up.")
        return False

    optimistic_update_price("P1", 12.99)
    optimistic_update_price("P1", 11.49)

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n  [locking] Pessimistic vs Optimistic summary:")
    print("""
    | Aspect           | Pessimistic (lock field / txn)   | Optimistic (__v CAS) |
    |------------------|----------------------------------|----------------------|
    | Best for         | High contention, long operations | Low contention, fast ops |
    | Write throughput | Lower (serialized)               | Higher (parallel reads) |
    | Conflict cost    | No wasted work (blocked upfront) | Retry on conflict       |
    | Deadlock risk    | Yes (multi-doc transactions)     | None                    |
    | Requires replica | Only for multi-doc txn           | No                      |
    | Stale lock risk  | Yes (crash before release)       | No (no lock held)       |

    Stale lock mitigation (pessimistic):
      Add locked_at + TTL: a background job (or $currentDate) unlocks docs
      where locked=True AND locked_at < NOW() - timeout.
      Or set a TTL index on locked_at that resets the field automatically.
    """)


# ---------------------------------------------------------------------------
# Updated demo_all — includes new sections 20-23
# ---------------------------------------------------------------------------
def demo_all() -> None:
    """Run all demos against a local MongoDB instance."""
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.drop_database("blog_demo")
    db = client["blog_demo"]

    print("\n--- 1. Connection ---")
    demo_connection()

    print("\n--- 2. Insert ---")
    demo_insert(db)

    print("\n--- 2b. Find ---")
    demo_find(db)

    print("\n--- 3. Update Operators ---")
    demo_update_operators(db)

    print("\n--- 4. Advanced Writes ---")
    demo_advanced_writes(db)

    print("\n--- 5. Bulk Write ---")
    demo_bulk_write(db)

    print("\n--- 6. Indexes ---")
    demo_indexes(db)

    print("\n--- 7. Schema Validation ---")
    demo_schema_validation(db)

    # 8 requires replica set — skip in standalone demo
    # demo_transactions(db)

    print("\n--- 9. Aggregation Basics ---")
    demo_aggregation_basics(db)

    print("\n--- 10. $lookup ---")
    demo_lookup(db)

    print("\n--- 11. $facet ---")
    demo_facet(db)

    print("\n--- 12. $graphLookup ---")
    demo_graph_lookup(db)

    # 13 requires MongoDB 5.0+
    print("\n--- 13. $setWindowFields ---")
    demo_set_window_fields(db)

    print("\n--- 14. $merge ---")
    demo_merge(db)

    print("\n--- 15. Text Search ---")
    demo_text_search(db)

    # 16 (sync) would block — omit in batch demo
    # demo_change_streams_sync(db)

    # 17 requires replica set
    # demo_replica_sets()

    print("\n--- 18. Motor (async) ---")
    asyncio.run(demo_motor())

    print("\n--- 19. MongoEngine ---")
    demo_mongoengine()

    print("\n--- 20. The 16 MB BSON Document Limit ---")
    demo_16mb_limit(db)

    print("\n--- 21. Normalized Query Performance Trap ---")
    demo_normalized_query_perf_trap(db)

    print("\n--- 22. Write Sharding Strategies ---")
    demo_write_sharding(client)

    print("\n--- 23. Distributed Transactions ---")
    demo_distributed_transactions(client)

    print("\n--- 24. Locking Patterns ---")
    demo_locking_patterns(db)

    client.close()
    print("\nAll demos complete.")


if __name__ == "__main__":
    demo_all()
