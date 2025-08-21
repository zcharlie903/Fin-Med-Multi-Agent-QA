from __future__ import annotations
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from app.config import settings

def _char_budget() -> int:
    # crude approximation; adjust as needed for your model
    return 6000

def compress_with_llm(llm: ChatOpenAI, texts: List[str]) -> str:
    joined = "\n\n".join(texts)
    prompt = (
        "You are a helpful assistant. Compress the following notes into concise bullet points "
        "while preserving factual content and source markers like [S1], [S2].\n\n"
        f"{joined}"
    )
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

def build_hierarchical_context(query: str, retrieved: List[Document], llm: ChatOpenAI | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    llm = llm or ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
    # attach source markers
    notes = []
    provenance = []
    for i, d in enumerate(retrieved, start=1):
        marker = f"[S{i}]"
        notes.append(f"{marker} {d.page_content}")
        provenance.append({"marker": marker, "source": d.metadata.get("source"), "domain": d.metadata.get("domain")})

    context = "\n\n".join(notes)
    if len(context) > _char_budget():
        context = compress_with_llm(llm, notes)

    return context, provenance
