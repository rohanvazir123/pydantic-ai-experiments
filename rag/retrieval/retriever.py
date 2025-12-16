from rag.config.settings import settings
from rag.ingestion.embedder import LocalEmbedder
from rag.storage.vector_store.base import VectorStore


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.embedder = LocalEmbedder()

    def retrieve(self, query: str):
        embedding = self.embedder.embed([query])[0]
        return self.store.query(embedding, query, settings.top_k)
