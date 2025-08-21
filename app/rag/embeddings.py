from langchain_openai import OpenAIEmbeddings
from app.config import settings

def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-large", api_key=settings.openai_api_key)
