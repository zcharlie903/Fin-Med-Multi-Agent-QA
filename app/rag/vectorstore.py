from __future__ import annotations
from typing import Optional, List, Dict, Any
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.rag.embeddings import get_embedder
from app.config import settings

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

def get_vectorstore(collection: Optional[str] = None) -> QdrantVectorStore:
    client = get_qdrant_client()
    embedder = get_embedder()
    collection_name = collection or settings.default_collection
    vs = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedder)
    return vs

def get_retriever(collection: Optional[str] = None, k: int = 6):
    return get_vectorstore(collection).as_retriever(search_kwargs={"k": k})
