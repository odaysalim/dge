# Agentic RAG System

A production-ready Retrieval-Augmented Generation system powered by **multi-agent AI orchestration**. Three specialized AI agents work together to understand queries, retrieve relevant documents, and synthesize accurate answers with source citations.

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Three Agents](#the-three-agents)
- [Workflow Steps](#workflow-steps)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## How It Works

This system uses **CrewAI** to orchestrate three specialized AI agents that work sequentially to answer questions from your documents. Unlike simple RAG systems that just retrieve and respond, this agentic approach:

1. **Intelligently routes** queries to the most relevant document set
2. **Retrieves context** using hybrid search (semantic + keyword)
3. **Synthesizes answers** with proper source citations

```
User Question
     │
     ▼
┌─────────────────┐
│  Query Router   │  "What document category does this belong to?"
│     Agent       │
└────────┬────────┘
         │ Routing Decision (JSON)
         ▼
┌─────────────────┐
│   Document      │  "Let me search the right documents..."
│   Researcher    │
└────────┬────────┘
         │ Retrieved Chunks
         ▼
┌─────────────────┐
│    Insight      │  "Based on these documents, here's your answer..."
│   Synthesizer   │
└────────┬────────┘
         │
         ▼
   Final Answer
   + Sources
```

---

## The Three Agents

### Agent 1: Query Router

**Role:** Analyze queries and determine which documents to search

The Query Router examines each question and decides which document category is most relevant. This prevents searching through irrelevant documents and improves response accuracy.

**Document Categories:**
| Category | Documents | Example Queries |
|----------|-----------|-----------------|
| `hr` | HR Bylaws | "What is the annual leave policy?" |
| `infosec` | Information Security | "What are the password requirements?" |
| `procurement_ariba` | SAP Ariba Manual | "How do I create a sourcing project?" |
| `procurement_business` | Business Process Manual | "What is the procurement workflow?" |
| `procurement_standards` | Abu Dhabi Standards | "What are the government regulations?" |

**How it works:**
```python
# Keyword scoring determines the route
hr_keywords = ["leave", "salary", "employee", "benefits", ...]
infosec_keywords = ["security", "password", "encryption", ...]
procurement_keywords = ["purchase", "vendor", "contract", ...]

# Highest score wins
route = "hr" if hr_score > other_scores else ...
```

---

### Agent 2: Document Researcher

**Role:** Retrieve relevant document chunks using the routing decision

The Document Researcher uses the routing decision to search only relevant documents. It performs **hybrid search** combining:

- **Semantic search** (vector similarity using embeddings)
- **Keyword search** (BM25 for exact matches)

**Retrieved chunk format:**
```
---
**Source:** Procurement Manual (Business Process).PDF
**Page:** 45
**Section Context:** Purchase Requisition Process

The End-User must create a Purchase Requisition (PR) to request
goods, services, or projects...
---
```

---

### Agent 3: Insight Synthesizer

**Role:** Create comprehensive answers with source citations

The Insight Synthesizer takes the retrieved chunks and formulates a professional response that:

- Directly answers the user's question
- Uses only information from the documents (no hallucination)
- Includes proper source citations
- Formats the response appropriately (bullets, paragraphs, etc.)

**Output format:**
```
To submit a Purchase Requisition, follow these steps:

1. Navigate to the Requisition module in SAP Ariba
2. Click "Create Requisition" and select the appropriate category
3. Fill in the required fields including description and quantity
4. Submit for approval based on the Delegation of Authority

**Sources:**
- Procurement Manual (Business Process).PDF (Page 45)
- Procurement Manual (Ariba Aligned).PDF (Page 132)
```

---

## Workflow Steps

### Step 1: Query Reception
```
User: "How do I submit a purchase requisition?"
```
The API receives the query via the `/v1/chat/completions` endpoint (OpenAI-compatible).

### Step 2: Pre-processing Checks
Before invoking the agent crew, the system checks for:
- **Greetings/chitchat** → Returns friendly response without RAG
- **OpenWebUI system requests** → Handles title generation, etc.

### Step 3: Query Routing
```json
{
  "route": "procurement_business",
  "justification": "Query relates to procurement processes (matched 2 keywords)",
  "query": "How do I submit a purchase requisition?"
}
```

### Step 4: Document Retrieval
The Document Researcher:
1. Receives the routing decision
2. Filters documents to `procurement_business` category
3. Performs hybrid search (semantic + keyword)
4. Returns top-k relevant chunks with metadata

### Step 5: Answer Synthesis
The Insight Synthesizer:
1. Analyzes all retrieved chunks
2. Extracts relevant information
3. Formulates a coherent answer
4. Adds source citations

### Step 6: Response Delivery
```json
{
  "model": "agentic-rag",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "To submit a Purchase Requisition..."
    }
  }]
}
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │    OpenWebUI    │    │   REST Client   │    │       CLI       │   │
│  │  localhost:3000 │    │    (curl/etc)   │    │    main.py      │   │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘   │
└───────────┼──────────────────────┼──────────────────────┼────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER (api.py)                          │
│                         localhost:8000                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  /v1/chat/completions  │  /v1/models  │  /health  │  /memory/* │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Features:                                                            │
│  • OpenAI-compatible API          • Conversation memory               │
│  • Greeting detection             • Session management                │
│  • OpenWebUI integration          • Error handling                    │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    CREWAI ORCHESTRATION (crew.py)                     │
│                                                                       │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│   │    Query     │      │   Document   │      │   Insight    │       │
│   │    Router    │ ───▶ │  Researcher  │ ───▶ │ Synthesizer  │       │
│   │              │      │              │      │              │       │
│   │ Routing Tool │      │Retrieval Tool│      │  (No tools)  │       │
│   └──────────────┘      └──────────────┘      └──────────────┘       │
│                                                                       │
│   Process: Sequential    │    Verbose: Off    │    Tracing: Off      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│     POSTGRESQL + PGVECTOR    │   │           OPENAI API             │
│        localhost:5432        │   │                                  │
│                              │   │   Model: gpt-4o-mini             │
│  • Vector embeddings (3072d) │   │   Embeddings: text-embedding-    │
│  • Hybrid search enabled     │   │               3-large            │
│  • Document metadata         │   │   Temperature: 0.1               │
│  • Contextual chunks         │   │                                  │
└──────────────────────────────┘   └──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ARIZE PHOENIX (Observability)                      │
│                         localhost:6006                                │
│                                                                       │
│  • Request tracing            • Latency monitoring                    │
│  • Token usage tracking       • Agent step visualization              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Docker Desktop
- OpenAI API Key

### 1. Clone and Configure

```bash
git clone <repository-url>
cd dge

# Create environment file
cp .env.docker.example .env.docker

# Add your OpenAI API key
nano .env.docker
# Set: OPENAI_API_KEY=sk-your-key-here
```

### 2. Start Services

```bash
docker compose up -d
```

This starts:
| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL | 5432 | Vector database with pgvector |
| Arize Phoenix | 6006 | Observability dashboard |
| RAG API | 8000 | FastAPI server |
| OpenWebUI | 3000 | Chat interface |

### 3. Verify

```bash
# Check all containers are running
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check available models
curl http://localhost:8000/v1/models
```

### 4. Test

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

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Database (auto-configured in Docker)
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/rag_db

# RAG Settings
EMBED_DIM=3072
RETRIEVAL_TOP_K=5
CHUNK_SIZE=512
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
│   └── rag_system/
│       ├── agents.py         # Three AI agent definitions
│       ├── crew.py           # CrewAI workflow orchestration
│       ├── memory.py         # Conversation memory system
│       └── tools.py          # Routing and retrieval tools
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
└── data/                     # Document storage
    └── raw/                  # Place PDF documents here
```

---

## Troubleshooting

### Container Issues

```bash
# View logs
docker logs crewai_rag_api --tail 50

# Rebuild API
docker compose up -d --no-deps --build rag-api

# Restart all services
docker compose restart
```

### Database Issues

```bash
# Run diagnostics
python diagnose_db.py

# Check database connection
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT COUNT(*) FROM rag_embeddings_openai;"
```

### API Issues

```bash
# Check if API is responding
curl http://localhost:8000/health

# Check configuration
curl http://localhost:8000/config
```

---

## License

MIT
