# Agentic RAG System

An intelligent RAG (Retrieval-Augmented Generation) system with multi-agent workflows, powered by CrewAI and OpenAI.

## Features

- **Multi-Agent Workflow** - Query Router, Document Researcher, and Insight Synthesizer agents
- **OpenAI Integration** - GPT-4o-mini for generation, text-embedding-3-large for embeddings
- **Vector Search** - PostgreSQL + pgvector for semantic retrieval
- **Hybrid Search** - Combined semantic + keyword (BM25) search
- **OpenWebUI Integration** - Chat interface at http://localhost:3000
- **Observability** - Arize Phoenix tracing at http://localhost:6006

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenWebUI (port 3000)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server (port 8000)                  │
│  • OpenAI-compatible API (/v1/chat/completions)              │
│  • Conversation memory                                       │
│  • Greeting/chitchat detection                               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  CrewAI Agent Orchestration                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │Query Router │→ │Document Researcher│→ │Insight Synth.  │  │
│  └─────────────┘  └──────────────────┘  └────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  PostgreSQL + pgvector   │   │         OpenAI API         │
│  • Vector embeddings     │   │  • gpt-4o-mini             │
│  • Hybrid search         │   │  • text-embedding-3-large  │
└──────────────────────────┘   └────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker Desktop
- OpenAI API Key

### 1. Configure Environment

```bash
# Copy environment template
cp .env.docker.example .env.docker

# Edit and add your OpenAI API key
nano .env.docker
```

### 2. Start All Services

```bash
docker compose up -d
```

This starts:
- PostgreSQL with pgvector (port 5432)
- Arize Phoenix (port 6006)
- RAG API (port 8000)
- OpenWebUI (port 3000)

### 3. Test the System

```bash
# Test API directly
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "agentic-rag", "messages": [{"role": "user", "content": "Hello"}]}'

# Or open OpenWebUI
open http://localhost:3000
```

## Project Structure

```
dge/
├── api.py                  # FastAPI server (main entry point)
├── main.py                 # CLI entry point
├── docker-compose.yml      # Docker services configuration
├── Dockerfile              # API container build
├── docker-entrypoint.sh    # Container startup script
├── diagnose_db.py          # Database diagnostic tool
├── src/
│   ├── config/
│   │   └── settings.py     # Configuration management
│   └── rag_system/
│       ├── agents.py       # CrewAI agent definitions
│       ├── crew.py         # RAG workflow orchestration
│       ├── memory.py       # Conversation memory
│       └── tools.py        # Retrieval and routing tools
└── data/                   # Document storage
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/memory/sessions` | GET | List active sessions |
| `/memory/{session_id}` | GET | Get session history |
| `/memory/{session_id}` | DELETE | Clear session |
| `/config` | GET | Current configuration |

## Available Models

- **agentic-rag** - Standard RAG queries
- **agentic-rag-memory** - RAG with conversation memory

## Environment Variables

Key variables in `.env.docker`:

```bash
OPENAI_API_KEY=sk-...           # Required
DATABASE_URL=postgresql://...    # Auto-configured in Docker
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBED_DIM=3072
```

## Troubleshooting

### Check container logs
```bash
docker logs crewai_rag_api --tail 50
```

### Diagnose database
```bash
python diagnose_db.py
```

### Rebuild API container
```bash
docker compose up -d --no-deps --build rag-api
```

## License

MIT
