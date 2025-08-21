from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    default_collection: str = os.getenv("DEFAULT_COLLECTION", "medfin_docs")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

settings = Settings()
