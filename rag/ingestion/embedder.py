import requests
from typing import List
from rag.config.settings import settings


class LocalEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{settings.llm_base_url}/api/embeddings",
            json={
                "model": settings.embedding_model,
                "prompt": texts,
            },
            timeout=30,
        )
        response.raise_for_status()
        return [e["embedding"] for e in response.json()["data"]]
