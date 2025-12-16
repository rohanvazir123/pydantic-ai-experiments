from pydantic import BaseModel
from typing import Dict, Any


class IngestedDocument(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
