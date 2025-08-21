from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from app.rag.vectorstore import get_retriever

class RetrieveInput(BaseModel):
    query: str = Field(..., description="natural language question")
    k: int = Field(6, description="top-k docs to retrieve")

def make_retrieval_tool(collection: str | None = None):
    retriever = get_retriever(collection=collection, k=6)
    def _retrieve(query: str, k: int = 6):
        retriever.search_kwargs = {"k": k}
        docs = retriever.get_relevant_documents(query)
        # return compact strings with source markers
        lines = []
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "unknown")
            lines.append(f"[S{i}] {d.page_content}\n(source: {source})")
        return "\n\n".join(lines)
    return StructuredTool.from_function(
        name="retrieve_docs",
        description="Retrieve top-k semantically relevant chunks for the query from the vector store.",
        func=_retrieve,
        args_schema=RetrieveInput,
    )
