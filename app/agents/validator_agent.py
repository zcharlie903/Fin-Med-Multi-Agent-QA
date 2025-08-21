from __future__ import annotations
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from app.config import settings

VALIDATOR_SYSTEM = (
    "You are a strict validator. Compare the DRAFT answer against the CONTEXT.
"
    "If the draft contains claims not supported by sources, request a revision.
"
    "Otherwise, output 'OK' and then a final answer with citations [S#]."
)

def validate_and_reflect(question: str, context: str, draft: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.0, api_key=settings.openai_api_key)
    prompt = f"""{VALIDATOR_SYSTEM}
QUESTION:
{question}

CONTEXT (citations available as [S1], [S2], ...):
{context}

DRAFT:
{draft}

Return a JSON object with keys:
- status: one of ["OK", "REVISE"]
- notes: short notes on validity or required changes
- final_answer: if status == "OK", provide the final grounded answer with citations.
"""
    resp = llm.invoke(prompt)
    # Best-effort JSON parse
    import json
    content = resp.content if hasattr(resp, "content") else str(resp)
    try:
        obj = json.loads(content)
    except Exception:
        obj = {"status": "REVISE", "notes": "Validator could not parse output; request revision.", "final_answer": ""}
    return obj
