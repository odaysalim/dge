# Fix: Empty Database After Ingestion

## Problem Summary

You successfully ran the ingestion script and it showed:
- ✅ 848 documents processed
- ✅ 971 chunks created

But when you queried the Docker PostgreSQL database:
- ❌ Table exists but has **0 rows**

## Root Cause

The ingestion script connected to a **different PostgreSQL instance** than your Docker container. This typically happens when:

1. **System PostgreSQL is installed** on your Mac and running on port 5432
2. **Docker PostgreSQL** is also trying to use port 5432
3. The ingestion script connected to the system PostgreSQL instead of Docker

## How to Fix

### Step 1: Check Which PostgreSQL Instance is Running

Run this command to see what's on port 5432:

```bash
lsof -i :5432
```

If you see both system PostgreSQL (`postgres`) and Docker PostgreSQL, that's the issue.

### Step 2: Stop System PostgreSQL (if running)

If you have system PostgreSQL installed via Homebrew:

```bash
brew services stop postgresql@16
# or
brew services stop postgresql
```

### Step 3: Verify Docker PostgreSQL is Running

```bash
docker compose ps
```

You should see the `postgres` container running. If not:

```bash
docker compose down
docker compose up -d postgres phoenix
```

Wait for it to be ready:

```bash
docker compose exec postgres pg_isready -U postgres
```

### Step 4: Run the Diagnostic Script

```bash
python diagnose_db.py
```

This will:
- ✅ Check if Docker containers are running
- ✅ Test connection to Docker PostgreSQL
- ✅ List all RAG tables and their row counts
- ✅ Identify configuration issues

### Step 5: Verify/Create .env File

Make sure you have a `.env` file in the project root:

```bash
# If .env doesn't exist, create it:
cp .env.example .env
```

Then edit `.env` and ensure these settings:

```bash
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key-here

# Database - Must point to Docker PostgreSQL
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/rag_db

# LLM Provider
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBED_DIM=3072
```

**IMPORTANT:** The `DATABASE_URL` must be exactly:
```
postgresql://postgres:postgres_password@localhost:5432/rag_db
```

### Step 6: Clear Old Data and Re-run Ingestion

If there's conflicting data, let's start fresh:

```bash
# Connect to Docker PostgreSQL
docker compose exec postgres psql -U postgres -d rag_db

# Inside psql, drop old tables (if any):
DROP TABLE IF EXISTS rag_embeddings_openai_20251204;
DROP TABLE IF EXISTS rag_embeddings_openai_20251205;
DROP TABLE IF EXISTS data_rag_embeddings_openai_20251204;
DROP TABLE IF EXISTS data_rag_embeddings_openai_20251205;

# Exit psql
\q
```

### Step 7: Re-run Ingestion

Make sure you have documents in `data/raw/`:

```bash
ls -la data/raw/
# Should show your PDF files
```

Then run ingestion:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Run ingestion
python src/data_ingestion/ingest_openai.py
```

Watch for:
- ✅ "PostgreSQL connected" message
- ✅ "Loaded X documents"
- ✅ "Generated X contextualized chunks"
- ✅ "INGESTION COMPLETE"

The script will create a table like `rag_embeddings_openai_20251204` and automatically add `VECTOR_TABLE_NAME` to your `.env` file.

### Step 8: Verify Data Persisted

Check the Docker PostgreSQL has your data:

```bash
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT tablename, schemaname FROM pg_tables WHERE tablename LIKE 'rag%';"
```

Then count the rows:

```bash
# Replace with your actual table name from above
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT COUNT(*) FROM rag_embeddings_openai_20251204;"
```

You should see 900+ rows (one for each chunk).

### Step 9: Test the RAG System

Now test your RAG system:

```bash
python main.py "What are the main topics covered in these documents?"
```

You should see:
1. Document Researcher agent retrieving chunks
2. Retrieved documents with sources
3. Insight Synthesizer creating the final answer

## Alternative: Use Docker Internal Network

If port conflicts persist, you can modify `docker-compose.yml` to use an internal network and update your connection accordingly.

## Verification Checklist

- [ ] System PostgreSQL stopped (if installed)
- [ ] Docker PostgreSQL running on port 5432
- [ ] `.env` file exists with correct DATABASE_URL
- [ ] `.env` has valid OPENAI_API_KEY
- [ ] Documents exist in `data/raw/`
- [ ] Ingestion completed successfully
- [ ] Table exists in Docker PostgreSQL with 900+ rows
- [ ] `VECTOR_TABLE_NAME` added to `.env`
- [ ] RAG system test returns relevant documents

## Still Having Issues?

Run the diagnostic script for detailed troubleshooting:

```bash
python diagnose_db.py
```

Check the ingestion log file created in your project directory:
```bash
ls -la openai_contextual_rag_*.log
cat openai_contextual_rag_*.log
```
