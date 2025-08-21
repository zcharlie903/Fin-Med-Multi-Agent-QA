from __future__ import annotations
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from app.agents.tools import make_retrieval_tool
from app.rag.context import build_hierarchical_context
from app.rag.vectorstore import get_retriever
from app.config import settings

SYSTEM_INSTRUCTIONS = (
    "You are a careful domain QA assistant. Answer concisely with clear bullet points.
"
    "Use retrieved evidence; add inline source markers like [S1], [S2] next to claims.
"
    "If a question asks for medical or financial advice, add a disclaimer and suggest consulting a professional."
)

def run_reasoner(question: str, domain: str = "medical", collection: str | None = None) -> Dict[str, Any]:
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
    tool = make_retrieval_tool(collection=collection)
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM_INSTRUCTIONS},
    )
    # 1) retrieve for context window building (also used by tool during reasoning)
    retriever = get_retriever(collection=collection, k=6)
    retrieved = retriever.get_relevant_documents(question)
    context, provenance = build_hierarchical_context(question, retrieved, llm=llm)
    prompt = f"Question: {question}

Context:
{context}

Answer:"
    answer = agent.run(prompt)
    return {"draft": answer, "provenance": provenance}
