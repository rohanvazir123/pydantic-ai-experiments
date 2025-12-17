# MongoDB Patterns & Best Practices

Reference guide for MongoDB-specific implementation patterns for the RAG agent.

## Collection Design

### Two-Collection Pattern (REQUIRED)

```python
# documents collection - source of truth
{
  "_id": ObjectId("..."),
  "title": str,
  "source": str,
  "content": str,  # Full document text
  "metadata": dict,
  "created_at": datetime
}

# chunks collection - searchable units
{
  "_id": ObjectId("..."),
  "document_id": ObjectId("..."),  # FK to documents
  "content": str,  # Chunk text with context
  "embedding": list[float],  # 1536-dim for text-embedding-3-small
  "chunk_index": int,
  "metadata": dict,
  "token_count": int
}
```

**Why this pattern?**
- **Source attribution**: Always trace chunks back to original documents
- **Metadata joins**: Use `$lookup` to enrich search results
- **Document management**: Update/delete source without touching chunks
- **Chunk reprocessing**: Regenerate chunks while preserving documents

## Aggregation Pipeline Patterns

### Semantic Search (Vector Only)

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": embedding,  # list[float], not string!
            "path": "embedding",
            "numCandidates": 100,  # Search space (10x limit is good default)
            "limit": match_count
        }
    },
    {
        "$lookup": {
            "from": "documents",
            "localField": "document_id",
            "foreignField": "_id",
            "as": "document_info"
        }
    },
    {
        "$unwind": "$document_info"
    },
    {
        "$project": {
            "chunk_id": "$_id",
            "document_id": 1,
            "content": 1,
            "similarity": {"$meta": "vectorSearchScore"},
            "metadata": 1,
            "document_title": "$document_info.title",
            "document_source": "$document_info.source"
        }
    }
]

# Execute
results = await collection.aggregate(pipeline).to_list(length=match_count)
```

**Key Points:**
- `queryVector` must be a Python list, not a string
- `numCandidates` controls search space (larger = better recall, slower)
- Always use `$lookup` to join with documents collection
- Extract score with `{"$meta": "vectorSearchScore"}`

### Hybrid Search with $rankFusion (PREFERRED)

```python
pipeline = [
    {
        "$rankFusion": {
            "input": {
                "pipelines": {
                    "semantic": [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "queryVector": embedding,
                                "path": "embedding",
                                "numCandidates": 100,
                                "limit": match_count * 2  # Over-fetch for fusion
                            }
                        },
                        {
                            "$project": {
                                "content": 1,
                                "document_id": 1,
                                "metadata": 1,
                                "chunk_index": 1,
                                "vector_score": {"$meta": "vectorSearchScore"}
                            }
                        }
                    ],
                    "fulltext": [
                        {
                            "$search": {
                                "index": "text_index",
                                "text": {
                                    "query": query,
                                    "path": "content",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 3
                                    }
                                }
                            }
                        },
                        {
                            "$limit": match_count * 2  # Over-fetch for fusion
                        },
                        {
                            "$project": {
                                "content": 1,
                                "document_id": 1,
                                "metadata": 1,
                                "chunk_index": 1,
                                "text_score": {"$meta": "searchScore"}
                            }
                        }
                    ]
                }
            }
        }
    },
    {
        "$limit": match_count  # Final result count
    },
    {
        "$lookup": {
            "from": "documents",
            "localField": "document_id",
            "foreignField": "_id",
            "as": "document_info"
        }
    },
    {
        "$unwind": "$document_info"
    },
    {
        "$project": {
            "chunk_id": "$_id",
            "document_id": 1,
            "content": 1,
            "combined_score": {"$meta": "rankFusionScore"},
            "vector_score": 1,  # Optional: for debugging
            "text_score": 1,    # Optional: for debugging
            "metadata": 1,
            "document_title": "$document_info.title",
            "document_source": "$document_info.source"
        }
    }
]

# Execute
results = await collection.aggregate(pipeline).to_list(length=match_count)
```

**Key Points:**
- `$rankFusion` automatically de-duplicates results
- Uses Reciprocal Rank Fusion (RRF) algorithm for score combination
- Over-fetch in sub-pipelines (2x limit) for better fusion results
- Fuzzy matching catches typos and variations
- Extract combined score with `{"$meta": "rankFusionScore"}`

## Connection Management

### Initialization

```python
from motor.motor_asyncio import AsyncIOMotorClient

class MongoDBClient:
    """MongoDB connection manager."""

    def __init__(self, uri: str, database: str):
        self.uri = uri
        self.database_name = database
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        """Establish MongoDB connection."""
        self.client = AsyncIOMotorClient(self.uri)
        self.db = self.client[self.database_name]

        # Verify connection
        await self.client.admin.command('ping')

    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
```

### Usage in Dependencies

```python
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

class AgentDependencies:
    """Dependencies for the RAG agent."""

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def initialize(self):
        """Initialize connections."""
        settings = load_settings()

        # MongoDB connection
        self.mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.mongo_client[settings.mongodb_database]

        # Verify connection
        try:
            await self.mongo_client.admin.command('ping')
        except Exception as e:
            logger.exception("mongodb_connection_failed", uri=settings.mongodb_uri)
            raise

    async def cleanup(self):
        """Clean up connections."""
        if self.mongo_client:
            self.mongo_client.close()
```

## Index Management

### Vector Search Index

**IMPORTANT**: Must be created in Atlas UI. Cannot be created programmatically.

**Atlas UI:** Database → Search and Vector Search → Create Search Index → Pick "Vector Search"

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

**Similarity options:** `cosine`, `euclidean`, `dotProduct`
**Dimensions:** 1536 for text-embedding-3-small, 3072 for text-embedding-3-large

### Full-Text Search Index

**Atlas UI:** Database → Search and Vector Search → Create Search Index → Pick "Atlas Search"

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

**Analyzer options:** `lucene.standard` (default), `lucene.english` (stemming), `lucene.keyword` (exact)

## Error Handling

### Connection Errors

```python
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

try:
    await client.admin.command('ping')
except ConnectionFailure:
    logger.exception("mongodb_connection_failed")
    raise ConnectionError("Could not connect to MongoDB Atlas")
except ServerSelectionTimeoutError:
    logger.exception("mongodb_timeout")
    raise ConnectionError("MongoDB server selection timeout")
```

### Operation Errors

```python
from pymongo.errors import OperationFailure

try:
    results = await collection.aggregate(pipeline).to_list(length=limit)
except OperationFailure as e:
    if e.code == 291:
        # Index not found
        logger.error("mongodb_index_missing", index="vector_index")
        raise ValueError(
            "Vector search index 'vector_index' not found. "
            "Create it in Atlas UI before running searches."
        )
    elif e.code == 2:
        # Bad value (e.g., wrong embedding dimensions)
        logger.error("mongodb_bad_value", error=str(e))
        raise ValueError(f"Invalid query parameters: {e}")
    else:
        logger.exception("mongodb_operation_failed", code=e.code)
        raise
```

### Query Timeout

```python
from pymongo.errors import ExecutionTimeout

try:
    results = await collection.aggregate(pipeline).to_list(length=limit)
except ExecutionTimeout:
    logger.warning("mongodb_query_timeout", pipeline=pipeline)
    # Reduce numCandidates or match_count and retry
    raise TimeoutError("Query took too long, try reducing result count")
```

## Performance Optimization

### numCandidates Tuning

```python
# Rule of thumb: numCandidates = 10 * limit
# Smaller = faster but might miss relevant results
# Larger = slower but better recall

# Fast search (good enough for most cases)
numCandidates = match_count * 10  # e.g., 50 for limit=5

# High-quality search (better recall)
numCandidates = match_count * 20  # e.g., 100 for limit=5

# Maximum quality (slower)
numCandidates = match_count * 50  # e.g., 250 for limit=5
```

### Projection Optimization

```python
# ❌ BAD - Returns entire document
pipeline = [
    {"$vectorSearch": {...}},
    {"$lookup": {...}}
]

# ✅ GOOD - Returns only needed fields
pipeline = [
    {"$vectorSearch": {...}},
    {"$lookup": {...}},
    {"$project": {
        "content": 1,
        "document_title": "$document_info.title",
        "document_source": "$document_info.source",
        "similarity": {"$meta": "vectorSearchScore"}
        # Exclude large fields like full document content
    }}
]
```

### Batch Operations

```python
# ✅ Batch insert for chunks
chunk_dicts = [chunk.dict() for chunk in chunks]
result = await db.chunks.insert_many(chunk_dicts, ordered=False)

# ordered=False allows partial success if some inserts fail
```

## Common Operations

### Insert Document and Chunks

```python
from bson import ObjectId
from datetime import datetime

async def save_document_with_chunks(
    db,
    title: str,
    source: str,
    content: str,
    chunks: list[DocumentChunk],
    metadata: dict
):
    """Save document and its chunks to MongoDB."""

    # Insert document
    document = {
        "title": title,
        "source": source,
        "content": content,
        "metadata": metadata,
        "created_at": datetime.utcnow()
    }
    doc_result = await db.documents.insert_one(document)
    document_id = doc_result.inserted_id

    # Prepare chunk documents
    chunk_dicts = []
    for chunk in chunks:
        chunk_dict = {
            "document_id": document_id,
            "content": chunk.content,
            "embedding": chunk.embedding,  # Python list
            "chunk_index": chunk.index,
            "metadata": chunk.metadata,
            "token_count": chunk.token_count
        }
        chunk_dicts.append(chunk_dict)

    # Batch insert chunks
    if chunk_dicts:
        await db.chunks.insert_many(chunk_dicts, ordered=False)

    return str(document_id)
```

### Find Document by ID

```python
from bson import ObjectId

async def get_document(db, document_id: str):
    """Retrieve document by ID."""
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    return doc
```

### Delete Document and Chunks

```python
async def delete_document(db, document_id: str):
    """Delete document and all its chunks."""
    doc_id = ObjectId(document_id)

    # Delete chunks first
    await db.chunks.delete_many({"document_id": doc_id})

    # Delete document
    await db.documents.delete_one({"_id": doc_id})
```

### Count Documents

```python
# Count total documents
doc_count = await db.documents.count_documents({})

# Count total chunks
chunk_count = await db.chunks.count_documents({})

# Count chunks for specific document
chunk_count = await db.chunks.count_documents({"document_id": ObjectId(doc_id)})
```
