from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings
from app.rag.ingestion import ingest
from app.rag.vectorstore import get_retriever
from app.agents.reasoning_agent import run_reasoner
from app.agents.validator_agent import validate_and_reflect
from app.rag.context import build_hierarchical_context
from app.rag.memory import get_memory
from langchain_openai import ChatOpenAI

app = FastAPI(title="Multi-Agent Medical & Finance QA Backend")

class IngestReq(BaseModel):
    paths: List[str]
    domain: str = Field("medical", pattern="^(medical|finance)$")
    collection: Optional[str] = None

class AskReq(BaseModel):
    question: str
    domain: str = Field("medical", pattern="^(medical|finance)$")
    session_id: str = "default"
    collection: Optional[str] = None

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": settings.openai_model}

@app.post("/ingest")
def ingest_api(req: IngestReq):
    try:
        ingest(req.paths, domain=req.domain, collection=req.collection)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_api(req: AskReq):
    try:
        # Memory context
        llm = ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
        mem = get_memory(session_id=req.session_id, llm=llm)

        # Reasoner first pass
        res = run_reasoner(req.question, domain=req.domain, collection=req.collection or settings.default_collection)
        draft = res["draft"]
        provenance = res["provenance"]

        # Build context again (fresh retrieval) for validator
        retriever = get_retriever(collection=req.collection or settings.default_collection, k=6)
        retrieved = retriever.get_relevant_documents(req.question)
        context, _ = build_hierarchical_context(req.question, retrieved, llm=llm)

        # Validate & possibly reflect once
        verdict = validate_and_reflect(req.question, context, draft)
        if verdict.get("status") == "REVISE":
            # One simple self-reflection loop: append validator notes and re-ask reasoner
            revision_prompt = f"""The validator requested a revision:
{verdict.get('notes','')}

Question: {req.question}
Context:
{context}

Please revise with correct citations."""
            res2 = run_reasoner(revision_prompt, domain=req.domain, collection=req.collection or settings.default_collection)
            draft = res2["draft"]
            verdict = validate_and_reflect(req.question, context, draft)

        final_answer = verdict.get("final_answer") or draft
        # Save memory
        mem.save_context({"question": req.question}, {"answer": final_answer})

        return {"answer": final_answer, "provenance": provenance, "verdict": verdict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
