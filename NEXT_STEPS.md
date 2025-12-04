# Next Steps: Fix Database Connection & Test RAG System

## 🎯 Current Situation

Your ingestion appeared successful (971 chunks created) but the Docker PostgreSQL table has **0 rows**. This happened because the ingestion script connected to a different PostgreSQL instance (likely system PostgreSQL) instead of your Docker container.

## ✅ What I Just Did

I've created three tools to help diagnose and fix this issue:

### 1. **diagnose_db.py** - Interactive Diagnostic Tool
- Checks if Docker containers are running
- Tests connection to Docker PostgreSQL
- Lists all RAG tables and their row counts
- Identifies configuration problems
- Provides specific fix recommendations

### 2. **fix_database.sh** - Automated Fix Script
- Stops system PostgreSQL (if running on port 5432)
- Starts Docker PostgreSQL and Phoenix containers
- Validates your `.env` configuration
- Tests the database connection
- Provides next steps

### 3. **FIX_EMPTY_DATABASE.md** - Comprehensive Guide
- Detailed explanation of the problem
- Step-by-step troubleshooting checklist
- Manual fix instructions if scripts don't work
- Common issues and solutions

## 🚀 What You Need to Do Next

### On Your MacBook (in VS Code Terminal)

**1. Pull the latest changes:**
```bash
git pull origin claude/agentic-rag-system-01BjzSNHdSGWaCmFTUGdQnCs
```

**2. Run the automated fix script:**
```bash
./fix_database.sh
```

This script will:
- Check for port conflicts
- Stop system PostgreSQL if needed
- Start Docker services
- Validate your `.env` file
- Test the connection

**3. If the script asks you to set OPENAI_API_KEY:**
```bash
nano .env
# Add your OpenAI API key, then save (Ctrl+O, Enter, Ctrl+X)
```

**4. Verify your `.env` has the correct DATABASE_URL:**

The `.env` file **must** have this exact line:
```bash
DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/rag_db
```

**5. Make sure you have PDF documents:**
```bash
ls -la data/raw/
```

You should see your PDF files. If not:
```bash
cp /path/to/your/pdfs/*.pdf data/raw/
```

**6. Re-run the ingestion:**
```bash
# Activate your virtual environment
source venv/bin/activate

# Run ingestion
python src/data_ingestion/ingest_openai.py
```

Watch for these success indicators:
- ✅ "PostgreSQL connected"
- ✅ "Loaded X documents"
- ✅ "Generated X contextualized chunks"
- ✅ "INGESTION COMPLETE"

**7. Verify the data is in Docker PostgreSQL:**
```bash
python diagnose_db.py
```

This should show:
- Docker containers running
- Connection to Docker PostgreSQL successful
- Table `rag_embeddings_openai_20251204` with 900+ rows

**8. Test the RAG system:**
```bash
python main.py "What are the main topics covered in these documents?"
```

You should see:
1. **Document Researcher** retrieving relevant chunks
2. Retrieved documents with sources and context
3. **Insight Synthesizer** creating a comprehensive answer

## 🔍 Troubleshooting

### If you still see "No relevant documents found"

Run the diagnostic tool:
```bash
python diagnose_db.py
```

This will tell you exactly what's wrong and how to fix it.

### If ingestion fails

Check the log file created in your project directory:
```bash
ls -la openai_contextual_rag_*.log
cat openai_contextual_rag_*.log
```

### If you get a port 5432 conflict

You likely have system PostgreSQL running. Stop it:
```bash
brew services stop postgresql
# or
brew services stop postgresql@16
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Port 5432 already in use | `brew services stop postgresql` |
| Cannot connect to database | `docker compose up -d postgres` |
| Table has 0 rows | Re-run ingestion after fixing DATABASE_URL |
| OPENAI_API_KEY error | Edit `.env` and set your API key |
| No documents found | Add PDFs to `data/raw/` directory |

## 📊 Expected Results After Fix

Once you complete these steps, you should have:

- ✅ Docker PostgreSQL running on port 5432
- ✅ System PostgreSQL stopped (if it was running)
- ✅ `.env` file with correct DATABASE_URL
- ✅ Table `rag_embeddings_openai_20251204` with 900+ rows
- ✅ RAG system returning relevant documents with sources
- ✅ Multi-agent workflow: Researcher → Synthesizer

## 📁 Files to Check

The key files for troubleshooting:

```
dge/
├── .env                              # Your configuration (check DATABASE_URL)
├── diagnose_db.py                    # Run this to diagnose issues
├── fix_database.sh                   # Run this to auto-fix common issues
├── FIX_EMPTY_DATABASE.md            # Detailed troubleshooting guide
├── data/raw/                         # Your PDF documents should be here
├── src/data_ingestion/ingest_openai.py  # The ingestion script
└── main.py                           # Test script for RAG system
```

## 🎉 Once Everything Works

After your RAG system is working, we'll move on to:

1. **Start the FastAPI server** - `python api.py`
2. **Test API endpoints** - Use REST client in VS Code
3. **Set up OpenWebUI** - `docker compose up -d open-webui`
4. **Test the full system** - Chat interface with your documents
5. **Run RAGAs evaluation** - Measure retrieval quality

## ❓ Need Help?

If you run into any issues:

1. Run `python diagnose_db.py` for automated diagnostics
2. Read `FIX_EMPTY_DATABASE.md` for detailed troubleshooting
3. Check the ingestion log files: `openai_contextual_rag_*.log`
4. Share the error messages and diagnostic output

---

**Ready to fix this?** Start with:
```bash
git pull origin claude/agentic-rag-system-01BjzSNHdSGWaCmFTUGdQnCs
./fix_database.sh
```
