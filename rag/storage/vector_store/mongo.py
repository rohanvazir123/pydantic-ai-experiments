from pymongo import MongoClient
from rag.storage.vector_store.base import VectorStore, RetrievedChunk
from rag.config.settings import settings


class MongoHybridStore:
    def __init__(self):
        self.client = MongoClient(settings.mongo_uri)
        self.collection = self.client[settings.mongo_db][settings.mongo_collection]

    def add(self, chunks, embeddings):
        docs = []
        for chunk, emb in zip(chunks, embeddings):
            docs.append(
                {
                    "_id": chunk.id,
                    "text": chunk.text,
                    "embedding": emb,
                    "metadata": chunk.metadata,
                }
            )
        if docs:
            self.collection.insert_many(docs, ordered=False)

    def query(self, embedding, query_text, k):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": k,
                }
            },
            {
                "$search": {
                    "index": "text_index",
                    "text": {
                        "query": query_text,
                        "path": "text",
                        "fuzzy": {},
                    },
                }
            },
            {"$addFields": {"score": {"$add": ["$vectorScore", "$searchScore"]}}},
            {"$sort": {"score": -1}},
            {"$limit": k},
        ]

        results = self.collection.aggregate(pipeline)
        return [
            RetrievedChunk(
                id=r["_id"],
                text=r["text"],
                metadata=r.get("metadata", {}),
                score=r["score"],
            )
            for r in results
        ]
