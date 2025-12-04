# 🚀 Agentic RAG System - Complete Setup Guide

This guide will help you set up the complete Agentic RAG system on your MacBook, block by block.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Testing](#testing)
5. [Troubleshooting](#troubleshooting)
6. [Architecture Overview](#architecture-overview)

---

## Prerequisites

### Required Software

- **macOS** (tested on macOS 12+)
- **Docker Desktop** (for PostgreSQL, Phoenix, and OpenWebUI)
- **Python 3.9+**
- **Git**

### API Keys

- **OpenAI API Key** - Get from [platform.openai.com](https://platform.openai.com)

### Installation

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.11

# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop
```

---

## 🎯 Quick Start (15 minutes)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd dge

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your favorite editor
```

**Update `.env` with:**

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/rag_db
DB_USER=postgres
DB_PASSWORD=postgres_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Embedding Configuration
EMBED_DIM=3072

# Arize Phoenix Configuration
PHOENIX_HOST=localhost
PHOENIX_PORT=6006

# RAG Configuration
RETRIEVAL_TOP_K=5
SPARSE_TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Step 3: Start Infrastructure (PostgreSQL + Phoenix)

```bash
# Start PostgreSQL and Arize Phoenix
docker-compose up -d postgres phoenix

# Wait for services to be ready (about 30 seconds)
docker-compose ps

# Verify PostgreSQL is ready
docker-compose logs postgres | grep "ready to accept connections"

# Verify Phoenix is ready
curl http://localhost:6006/healthz
```

### Step 4: Ingest Documents

```bash
# Place your PDF documents in data/raw/
mkdir -p data/raw
# Copy your PDFs to data/raw/

# Optional: Convert PDFs to Markdown using Docling
python src/data_ingestion/ingestion_docling.py

# Ingest documents with contextual embeddings
python src/data_ingestion/ingest_openai.py
```

**Expected output:**
```
✅ Configuration validated
✅ PostgreSQL connected
✅ pgvector extension is installed
✅ Using OpenAI model: gpt-4o-mini
✅ Using OpenAI embeddings: text-embedding-3-large (dim=3072)
✅ Loaded X documents
✅ Generated Y contextualized chunks
✅ INGESTION COMPLETE
```

### Step 5: Test the RAG Pipeline

```bash
# Test via command line
python main.py "What is the main topic of the documents?"
```

### Step 6: Start API Server

```bash
# Start the FastAPI server
python api.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║             🚀 Agentic RAG API Starting                      ║
╠══════════════════════════════════════════════════════════════╣
║  Host:                                             0.0.0.0   ║
║  Port:                                                8000   ║
║  LLM Provider:                                       openai  ║
║  Phoenix Tracing:                    http://localhost:6006  ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 7: Start OpenWebUI (Optional)

```bash
# Start OpenWebUI in a separate terminal
docker-compose up -d open-webui

# Access OpenWebUI
open http://localhost:3000
```

---

## 📚 Detailed Setup

### Block 1: Environment Setup

#### 1.1 Python Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### 1.2 Environment Variables

Create `.env` file with all necessary configurations:

- **OPENAI_API_KEY**: Your OpenAI API key
- **DATABASE_URL**: PostgreSQL connection string
- **LLM_PROVIDER**: Set to `openai`
- **OPENAI_MODEL**: Choose from `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **OPENAI_EMBEDDING_MODEL**: `text-embedding-3-large` (recommended) or `text-embedding-3-small`
- **EMBED_DIM**: `3072` for text-embedding-3-large, `1536` for text-embedding-3-small

### Block 2: Infrastructure Setup

#### 2.1 Start PostgreSQL with pgvector

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Verify it's running
docker-compose logs postgres

# Test connection
docker exec -it rag_postgres psql -U postgres -d rag_db -c "SELECT version();"
docker exec -it rag_postgres psql -U postgres -d rag_db -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

#### 2.2 Start Arize Phoenix

```bash
# Start Phoenix
docker-compose up -d phoenix

# Verify it's running
curl http://localhost:6006/healthz

# Access Phoenix UI
open http://localhost:6006
```

### Block 3: Document Ingestion

#### 3.1 Prepare Documents

```bash
# Create directories
mkdir -p data/raw
mkdir -p data/processed/md

# Copy your PDF documents to data/raw/
cp /path/to/your/documents/*.pdf data/raw/
```

#### 3.2 Convert PDFs to Markdown (Optional but Recommended)

```bash
# Run Docling converter
python src/data_ingestion/ingestion_docling.py

# This will create markdown files in data/processed/md/
ls -la data/processed/md/
```

#### 3.3 Ingest with Contextual Embeddings

```bash
# Run the OpenAI ingestion script
python src/data_ingestion/ingest_openai.py
```

**What this does:**
1. Loads documents from `data/raw/` or `data/processed/md/`
2. Splits documents into chunks (512 tokens with 50 token overlap)
3. Generates contextual information for each chunk using OpenAI
4. Creates embeddings using `text-embedding-3-large`
5. Stores everything in PostgreSQL with pgvector

**Monitor progress:**
- Check the console for progress updates
- Look for the completion message with statistics
- A table name will be saved to `.env` automatically

### Block 4: RAG System Testing

#### 4.1 Test via Command Line

```bash
# Activate virtual environment
source venv/bin/activate

# Run a test query
python main.py "Summarize the main points of the documents"
```

#### 4.2 Test the API

```bash
# Start the API server
python api.py

# In another terminal, test the API
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag",
    "messages": [
      {"role": "user", "content": "What are the main topics covered?"}
    ]
  }'
```

#### 4.3 Test Conversation Memory

```bash
# Test with session memory
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag-memory",
    "messages": [
      {"role": "user", "content": "What is procurement?"}
    ],
    "session_id": "test-session-123"
  }'

# Follow-up question with same session
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag-memory",
    "messages": [
      {"role": "user", "content": "Can you elaborate on that?"}
    ],
    "session_id": "test-session-123"
  }'

# View session memory
curl http://localhost:8000/memory/test-session-123
```

### Block 5: OpenWebUI Integration

#### 5.1 Start OpenWebUI

```bash
# Start OpenWebUI container
docker-compose up -d open-webui

# Access the interface
open http://localhost:3000
```

#### 5.2 Configure OpenWebUI

1. Open http://localhost:3000
2. Sign up (create a local account)
3. Go to **Settings** → **Connections**
4. Click on **OpenAI API**
5. Set:
   - **API Base URL**: `http://rag-api:8000/v1`
   - **API Key**: (leave empty or type "dummy")
6. Click **Verify Connection**
7. You should see the models: `agentic-rag` and `agentic-rag-memory`

#### 5.3 Test in OpenWebUI

1. Start a new chat
2. Select **agentic-rag** model from dropdown
3. Ask a question about your documents
4. View the response with sources

### Block 6: Arize Phoenix Observability

#### 6.1 Access Phoenix Dashboard

```bash
# Open Phoenix UI
open http://localhost:6006
```

#### 6.2 View Traces

1. Navigate to **Traces** tab
2. You'll see all API calls, CrewAI executions, and LlamaIndex retrievals
3. Click on any trace to see detailed information:
   - Input/output
   - Token usage
   - Latency
   - Agent steps
   - Retrieval results

#### 6.3 Monitor Performance

- **Latency**: Average response time
- **Token Usage**: Tokens consumed per request
- **Error Rate**: Failed requests
- **Retrieval Quality**: Relevance scores

### Block 7: RAGAs Evaluation

#### 7.1 Create Evaluation Dataset

Create `data/evaluation/eval_dataset.jsonl`:

```jsonl
{"question": "What is the procurement policy?", "ground_truth": "The procurement policy outlines..."}
{"question": "What are the approval requirements?", "ground_truth": "Approval requirements include..."}
```

#### 7.2 Run Evaluation

```bash
# Ensure RAGAS is configured for OpenAI
# Update src/evaluation/run_ragas_eval.py if needed

# Run evaluation
python src/evaluation/run_ragas_eval.py
```

**Metrics evaluated:**
- **Faithfulness**: How accurate is the answer based on context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Recall**: Did retrieval find all relevant information?
- **Context Precision**: How precise is the retrieved context?

---

## 🧪 Testing Checklist

- [ ] PostgreSQL with pgvector is running
- [ ] Arize Phoenix is running and accessible
- [ ] Documents are ingested successfully
- [ ] Command-line RAG query works
- [ ] API server starts without errors
- [ ] API health check returns healthy
- [ ] API models endpoint lists models
- [ ] Chat completion request works
- [ ] Conversation memory persists across requests
- [ ] OpenWebUI connects to API
- [ ] OpenWebUI shows available models
- [ ] Queries in OpenWebUI return responses with sources
- [ ] Phoenix dashboard shows traces

---

## 🐛 Troubleshooting

### PostgreSQL Issues

**Problem**: `psycopg2.OperationalError: could not connect to server`

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

**Problem**: `pgvector extension not found`

```bash
# Connect to PostgreSQL and enable extension
docker exec -it rag_postgres psql -U postgres -d rag_db
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
\dx  -- List extensions
```

### OpenAI API Issues

**Problem**: `AuthenticationError: Incorrect API key provided`

- Check that your `.env` file has the correct `OPENAI_API_KEY`
- Verify the key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**Problem**: `RateLimitError: Rate limit exceeded`

- You've hit OpenAI's rate limit
- Wait a few minutes or upgrade your OpenAI plan
- Consider using `gpt-3.5-turbo` instead of `gpt-4` for faster/cheaper requests

### Ingestion Issues

**Problem**: `No documents found in data/raw/`

```bash
# Verify files exist
ls -la data/raw/

# Check file permissions
chmod -R 755 data/
```

**Problem**: `Context generation too slow`

- Use smaller chunks: Set `CHUNK_SIZE=256` in `.env`
- Use fewer documents initially to test
- Consider disabling context generation: `CONTEXT_GENERATION_ENABLED=false`

### API Issues

**Problem**: `Module not found errors`

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: `Port 8000 already in use`

```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or change the port in .env
API_PORT=8001
```

### OpenWebUI Issues

**Problem**: Models not showing in OpenWebUI

1. Check API is running: `curl http://localhost:8000/v1/models`
2. In OpenWebUI settings, use container-to-container networking:
   - If running OpenWebUI via Docker: `http://rag-api:8000/v1`
   - If running OpenWebUI locally: `http://localhost:8000/v1`
3. Restart OpenWebUI: `docker-compose restart open-webui`

### Phoenix Issues

**Problem**: Phoenix not showing traces

- Ensure `PHOENIX_COLLECTOR_ENDPOINT` is set correctly in `.env`
- Restart the API server
- Check Phoenix logs: `docker-compose logs phoenix`

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│         OpenWebUI (http://localhost:3000)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server                              │
│         (http://localhost:8000)                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • OpenAI-compatible API                             │   │
│  │  • Conversation Memory                                │   │
│  │  • Session Management                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              CrewAI Agent Orchestration                      │
│  ┌────────────────────────┐  ┌──────────────────────────┐   │
│  │ Document Researcher    │→ │ Insight Synthesizer      │   │
│  │ - Vector retrieval     │  │ - Response generation    │   │
│  │ - Hybrid search        │  │ - Citation handling      │   │
│  └────────────────────────┘  └──────────────────────────┘   │
└────────────┬───────────────────────────────┬────────────────┘
             │                               │
             ▼                               ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  PGVector Database       │   │  OpenAI API                │
│  - Embeddings storage    │   │  - gpt-4o-mini             │
│  - Hybrid search         │   │  - text-embedding-3-large  │
│  - Metadata & context    │   │  - Context generation      │
└──────────────────────────┘   └────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                Arize Phoenix Observability                   │
│         (http://localhost:6006)                              │
│  • Request tracing                                           │
│  • Token usage monitoring                                    │
│  • Latency tracking                                          │
│  • Agent step visualization                                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:

1. **OpenWebUI** - User-friendly chat interface
2. **FastAPI Server** - API gateway with memory management
3. **CrewAI Agents** - Multi-agent workflow orchestration
4. **PGVector** - Vector database with hybrid search
5. **OpenAI** - LLM and embeddings provider
6. **Arize Phoenix** - Observability and tracing

---

## 🎯 Next Steps

### Switching to Ollama (Later)

When you're ready to switch from OpenAI to Ollama:

1. **Install Ollama**:
   ```bash
   brew install ollama
   ollama serve
   ```

2. **Pull models**:
   ```bash
   ollama pull gemma3:4b
   ollama pull nomic-embed-text:v1.5
   ```

3. **Update `.env`**:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gemma3:4b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5
   EMBED_DIM=768
   ```

4. **Re-ingest documents** (embeddings are different):
   ```bash
   python src/data_ingestion/ingest_contextual_rag.py
   ```

5. **Restart API**:
   ```bash
   python api.py
   ```

### Advanced Features

- **Add re-ranking**: Implement LLM-based re-ranking for better results
- **Multi-modal**: Add support for images and tables
- **Fine-tuning**: Fine-tune embeddings on your specific domain
- **Caching**: Add Redis for caching frequently asked questions
- **Authentication**: Add user authentication to the API
- **Rate limiting**: Implement rate limiting for production

---

## 📞 Support

For issues or questions:

1. Check the logs: `docker-compose logs`
2. Review Phoenix traces: http://localhost:6006
3. Check environment variables: `.env` file
4. Verify all services are running: `docker-compose ps`

---

## ✅ Success Criteria

You'll know the system is working correctly when:

- ✅ All Docker containers are running
- ✅ Documents are ingested without errors
- ✅ API returns responses with relevant information
- ✅ Sources are properly cited in responses
- ✅ Conversation memory maintains context across turns
- ✅ Phoenix dashboard shows all traces
- ✅ OpenWebUI connects and lists models
- ✅ Queries return accurate, contextual answers

---

**Congratulations! You now have a fully functional Agentic RAG system! 🎉**
