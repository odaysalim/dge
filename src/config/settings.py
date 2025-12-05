"""
Configuration settings for the Agentic RAG system.
Supports both OpenAI and Ollama providers.
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "evaluation"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres_password@localhost:5432/rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_db")

# LLM Provider Configuration
LLM_PROVIDER: Literal["openai", "ollama"] = os.getenv("LLM_PROVIDER", "ollama")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "131072"))

# Embedding Configuration
# 3072 for text-embedding-3-large (OpenAI), 1024 for mxbai-embed-large (Ollama), 768 for nomic
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024" if LLM_PROVIDER == "ollama" else "3072"))

# RAG Configuration
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

# Arize Phoenix Configuration
PHOENIX_HOST = os.getenv("PHOENIX_HOST", "localhost")
PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    f"http://{PHOENIX_HOST}:{PHOENIX_PORT}"
)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Vector Store Configuration
VECTOR_STORE_TABLE_PREFIX = "rag_embeddings"

# Context Generation Configuration (for Anthropic-style contextual RAG)
CONTEXT_GENERATION_ENABLED = os.getenv("CONTEXT_GENERATION_ENABLED", "true").lower() == "true"
CONTEXT_PROMPT_TEMPLATE = """You are analyzing a document. Your task is to provide brief context for a specific chunk.

<document>
{WHOLE_DOCUMENT}
</document>

<chunk>
{CHUNK_CONTENT}
</chunk>

Provide a brief context (1-2 sentences) explaining:
1. Which section/topic this chunk relates to
2. How it connects to the overall document
3. Its relationship to other sections

Respond with only the context, nothing else."""

# Conversation Memory Configuration
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "2000"))
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))  # Number of messages to keep

# Validation
def validate_config():
    """Validate critical configuration settings"""
    errors = []

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")

    if LLM_PROVIDER == "ollama" and not OLLAMA_BASE_URL:
        errors.append("OLLAMA_BASE_URL is required when LLM_PROVIDER is 'ollama'")

    if not DATABASE_URL:
        errors.append("DATABASE_URL is required")

    if EMBED_DIM not in [768, 1024, 1536, 3072]:
        errors.append(f"EMBED_DIM must be 768, 1024, 1536, or 3072, got {EMBED_DIM}")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    return True

# Run validation on import
validate_config()

# Export configuration as a dictionary for easy access
CONFIG = {
    "llm_provider": LLM_PROVIDER,
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
    },
    "ollama": {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "temperature": OLLAMA_TEMPERATURE,
        "max_tokens": OLLAMA_MAX_TOKENS,
    },
    "database": {
        "url": DATABASE_URL,
        "host": DB_HOST,
        "port": DB_PORT,
        "name": DB_NAME,
        "user": DB_USER,
    },
    "rag": {
        "embed_dim": EMBED_DIM,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "sparse_top_k": SPARSE_TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hybrid_search": ENABLE_HYBRID_SEARCH,
    },
    "phoenix": {
        "host": PHOENIX_HOST,
        "port": PHOENIX_PORT,
        "endpoint": PHOENIX_COLLECTOR_ENDPOINT,
    },
    "memory": {
        "enabled": MEMORY_ENABLED,
        "max_tokens": MEMORY_MAX_TOKENS,
        "window_size": MEMORY_WINDOW_SIZE,
    },
    "context_generation": {
        "enabled": CONTEXT_GENERATION_ENABLED,
        "prompt_template": CONTEXT_PROMPT_TEMPLATE,
    }
}
