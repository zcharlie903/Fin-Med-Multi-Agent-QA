from __future__ import annotations
from typing import Optional
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from app.config import settings

def get_memory(session_id: str, llm: Optional[ChatOpenAI] = None, k_token_budget: int = 2000):
    history = RedisChatMessageHistory(url=settings.redis_url, session_id=session_id)
    llm = llm or ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
    mem = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=history,
        max_token_limit=k_token_budget,
        return_messages=True,
        memory_key="history",
        input_key="question",
        output_key="answer",
    )
    return mem
