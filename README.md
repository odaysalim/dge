# Agentic RAG System (Open Source Edition)

A production-ready Retrieval-Augmented Generation system powered by **multi-agent AI orchestration** using 100% open-source components. Three specialized AI agents work together to understand queries, retrieve relevant documents, and synthesize accurate answers with source citations.

---

## Key Features

- **Fully Open Source**: Ollama + Llama 3.2 + mxbai-embed-large (no API keys required)
- **Multi-Agent Orchestration**: CrewAI-powered workflow with 3 specialized agents
- **Contextual RAG**: Anthropic-style chunk enrichment for better retrieval
- **Hybrid Search**: Semantic + BM25 keyword search
- **Docling Integration**: Superior PDF/document processing
- **RAGAs Evaluation**: Built-in evaluation framework
- **Phoenix Observability**: Complete tracing and prompt management
- **OpenWebUI**: Beautiful chat interface

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [The Three Agents](#the-three-agents)
- [Quick Start](#quick-start)
- [Document Ingestion](#document-ingestion)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │    OpenWebUI    │    │   REST Client   │    │       CLI       │       │
│  │  localhost:3000 │    │    (curl/etc)   │    │    main.py      │       │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘       │
└───────────┼──────────────────────┼──────────────────────┼────────────────┘
            └──────────────────────┼──────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER (api.py)                              │
│                         localhost:8000                                    │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  /v1/chat/completions  │  /v1/models  │  /health  │  /memory/* │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                           │
│  Features:                                                                │
│  • OpenAI-compatible API          • Conversation memory                   │
│  • Greeting detection             • Session management                    │
│  • OpenWebUI integration          • Error handling                        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    CREWAI ORCHESTRATION (crew.py)                         │
│                                                                           │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│   │    Query     │      │   Document   │      │   Insight    │           │
│   │    Router    │ ───▶ │  Researcher  │ ───▶ │ Synthesizer  │           │
│   │              │      │              │      │              │           │
│   │ Routing Tool │      │Retrieval Tool│      │  (No tools)  │           │
│   └──────────────┘      └──────────────┘      └──────────────┘           │
│                                                                           │
│   Process: Sequential    │    Verbose: Off    │    Tracing: Phoenix      │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│    OLLAMA (LLM)      │  │  POSTGRESQL+PGVECTOR │  │   ARIZE PHOENIX      │
│   localhost:11434    │  │   localhost:5432     │  │   localhost:6006     │
│                      │  │                      │  │                      │
│  Models:             │  │  • Vector embeddings │  │  • Request tracing   │
│  • llama3.2 (chat)   │  │    (1024d mxbai)     │  │  • Prompt management │
│  • mxbai-embed-large │  │  • Hybrid search     │  │  • Token tracking    │
│    (embeddings)      │  │  • Document metadata │  │  • Latency metrics   │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

---

## How It Works

This system uses **CrewAI** to orchestrate three specialized AI agents that work sequentially to answer questions from your documents. The agentic approach:

1. **Intelligently routes** queries to the most relevant document set
2. **Retrieves context** using hybrid search (semantic + keyword)
3. **Synthesizes answers** with proper source citations

### Contextual RAG (Anthropic-style)

Each document chunk is enriched with contextual information during ingestion:

```
Original Chunk:               Enriched Chunk:
┌─────────────────────┐      ┌─────────────────────────────────────────┐
│ "The employee must  │      │ Context: This chunk is from Article 71 │
│  submit a leave     │  ──▶ │ of the HR Bylaws, discussing the leave │
│  request 7 days     │      │ request process for annual leave.      │
│  in advance..."     │      │                                        │
└─────────────────────┘      │ "The employee must submit a leave      │
                             │  request 7 days in advance..."         │
                             └─────────────────────────────────────────┘
```

This contextual enrichment significantly improves retrieval quality.

---

## The Three Agents

### Agent 1: Query Router

**Role:** Analyze queries and determine which documents to search

Routes queries to:
| Category | Documents | Example Queries |
|----------|-----------|-----------------|
| `hr` | HR Bylaws | "What is the annual leave policy?" |
| `infosec` | Information Security | "What are the password requirements?" |
| `procurement_ariba` | SAP Ariba Manual | "How do I create a sourcing project?" |
| `procurement_business` | Business Process Manual | "What is the procurement workflow?" |
| `procurement_standards` | Abu Dhabi Standards | "What are the government regulations?" |

### Agent 2: Document Researcher

**Role:** Retrieve relevant document chunks using hybrid search

- Receives routing decision from Query Router
- Performs semantic + keyword search (pgvector + BM25)
- Returns top-k relevant chunks with metadata

### Agent 3: Insight Synthesizer

**Role:** Create comprehensive answers with source citations

- Analyzes retrieved chunks
- Formulates coherent answers
- Includes proper source citations (document, page, section)

---

## Quick Start

### Prerequisites

- Docker Desktop with GPU support (optional but recommended)
- 16GB+ RAM recommended
- ~10GB disk space for models

### 1. Clone and Configure

```bash
git clone <repository-url>
cd dge

# Create environment file
cp .env.docker.example .env.docker
```

### 2. Start Services

```bash
docker compose up -d
```

This starts all services:
| Service | Port | Description |
|---------|------|-------------|
| Ollama | 11434 | Local LLM hosting |
| PostgreSQL | 5432 | Vector database with pgvector |
| Arize Phoenix | 6006 | Observability dashboard |
| RAG API | 8000 | FastAPI server |
| OpenWebUI | 3000 | Chat interface |

### 3. Wait for Models to Download

The first startup will pull the models (~4GB):
```bash
# Monitor Ollama model download
docker logs -f ollama_pull

# Check if models are ready
curl http://localhost:11434/api/tags
```

### 4. Ingest Documents

Place PDF files in `data/raw/` then run:
```bash
# From within the container
docker exec -it crewai_rag_api python -m src.data_ingestion.ingest_docling

# Or with specific options
docker exec -it crewai_rag_api python -m src.data_ingestion.ingest_docling \
    --data-dir /app/data/raw \
    --no-context  # Skip contextual generation (faster)
```

### 5. Test

**Via curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag",
    "messages": [{"role": "user", "content": "What is the annual leave policy?"}]
  }'
```

**Via OpenWebUI:**
1. Open http://localhost:3000
2. Select `agentic-rag` model
3. Start chatting!

---

## Document Ingestion

### Using Docling

The system uses Docling for superior PDF processing:

```bash
# Basic ingestion
python -m src.data_ingestion.ingest_docling

# With options
python -m src.data_ingestion.ingest_docling \
    --data-dir ./data/raw \
    --table-name rag_embeddings_custom \
    --no-context  # Disable contextual generation
```

### Document Categories

Place documents in `data/raw/` and name them appropriately:
- `*HR*` or `*Bylaw*` → HR category
- `*Security*` or `*InfoSec*` → Information Security
- `*Ariba*` → Procurement (Ariba)
- `*Business*Process*` → Procurement (Business)
- `*Standard*` → Procurement (Standards)

---

## API Reference

### Chat Completions (OpenAI-compatible)

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "agentic-rag",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ],
  "session_id": "optional-session-id"
}
```

**Models:**
- `agentic-rag` - Standard queries
- `agentic-rag-memory` - With conversation memory

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/config` | GET | Current configuration |
| `/memory/sessions` | GET | List active sessions |
| `/memory/{id}` | GET | Get session history |
| `/memory/{id}` | DELETE | Clear session |

---

## Evaluation

The system includes RAGAs-based evaluation:

```bash
# Create sample test set
python -m src.evaluation.ragas_eval --create-sample

# Run evaluation
python -m src.evaluation.ragas_eval --test-file data/evaluation/sample_test_set.json
```

### Metrics

- **Faithfulness**: Are answers grounded in the context?
- **Answer Relevancy**: Do answers address the question?
- **Context Precision**: Is the retrieved context relevant?
- **Context Recall**: Is all necessary information retrieved?

---

## Configuration

### Environment Variables

```bash
# LLM Provider (ollama or openai)
LLM_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Database (auto-configured in Docker)
DATABASE_URL=postgresql://postgres:postgres_password@postgres:5432/rag_db

# Embedding Configuration
EMBED_DIM=1024  # 1024 for mxbai-embed-large

# RAG Settings
RETRIEVAL_TOP_K=5
CHUNK_SIZE=512
ENABLE_HYBRID_SEARCH=true

# Contextual RAG
CONTEXT_GENERATION_ENABLED=true
```

---

## Project Structure

```
dge/
├── api.py                    # FastAPI server - main entry point
├── main.py                   # CLI entry point for testing
├── diagnose_db.py            # Database diagnostic utility
│
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration management
│   │
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── ingest_docling.py # Docling-based ingestion with contextual RAG
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── ragas_eval.py     # RAGAs evaluation framework
│   │
│   └── rag_system/
│       ├── agents.py         # Three AI agent definitions
│       ├── crew.py           # CrewAI workflow orchestration
│       ├── memory.py         # Conversation memory system
│       ├── tools.py          # Routing and retrieval tools
│       └── phoenix_prompts.py # Prompt management
│
├── docker-compose.yml        # Docker services configuration
├── Dockerfile                # API container definition
├── docker-entrypoint.sh      # Container startup script
├── init-db.sql               # Database initialization
│
├── requirements.txt          # Python dependencies (local)
├── requirements-docker.txt   # Python dependencies (Docker)
│
├── .env.example              # Environment template (local)
├── .env.docker.example       # Environment template (Docker)
│
└── data/
    ├── raw/                  # Place PDF documents here
    ├── processed/            # Processed documents
    ├── evaluation/           # Evaluation test sets and results
    └── prompts/              # Stored prompts (auto-generated)
```

---

## Troubleshooting

### Container Issues

```bash
# View logs
docker logs crewai_rag_api --tail 50
docker logs ollama --tail 50

# Rebuild API
docker compose up -d --no-deps --build rag-api

# Restart all services
docker compose restart
```

### Ollama Model Issues

```bash
# Check available models
curl http://localhost:11434/api/tags

# Manually pull a model
docker exec -it ollama ollama pull llama3.2

# Check Ollama status
docker exec -it ollama ollama list
```

### Database Issues

```bash
# Run diagnostics
python diagnose_db.py

# Check embeddings table
docker compose exec postgres psql -U postgres -d rag_db \
  -c "SELECT COUNT(*) FROM rag_embeddings_ollama;"
```

### GPU Issues

If running without GPU, remove the GPU reservation from `docker-compose.yml`:
```yaml
# Remove or comment out this section in the ollama service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## Performance Tips

1. **Use GPU**: Ollama performs significantly better with GPU support
2. **Adjust chunk size**: Larger chunks = fewer calls, smaller = better precision
3. **Disable context generation**: Use `--no-context` for faster ingestion
4. **Tune retrieval**: Adjust `RETRIEVAL_TOP_K` based on your needs

---

## License

MIT
