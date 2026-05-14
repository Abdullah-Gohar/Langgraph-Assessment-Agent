"""Centralized configuration loaded from environment variables.

Keeping all knobs in one place makes it easy to tune the system without
hunting through the code. Pydantic-settings reads from .env automatically
and validates types.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Retrieval
    top_k_lo_match: int = 8
    top_k_chunk_retrieval: int = 4

    # Persistence
    sqlite_checkpoint_path: str = "./data/checkpoints.sqlite"
    embeddings_cache_path: str = "./data/embeddings_cache.json"

    # Data files
    lo_xlsx_path: str = "./data/LO.xlsx"
    chunks_json_path: str = "./data/chunks.json"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
