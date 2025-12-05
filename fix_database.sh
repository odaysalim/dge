#!/bin/bash

# Fix Database Connection Script
# Ensures ingestion connects to Docker PostgreSQL, not system PostgreSQL

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Fix Database Connection Issue                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check for system PostgreSQL
echo -e "${YELLOW}[1] Checking for PostgreSQL on port 5432...${NC}"
if lsof -i :5432 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Something is running on port 5432${NC}"
    lsof -i :5432
    echo ""

    # Check if it's system PostgreSQL
    if brew services list | grep postgresql | grep started > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  System PostgreSQL is running${NC}"
        read -p "Stop system PostgreSQL? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            brew services stop postgresql || brew services stop postgresql@16
            echo -e "${GREEN}✅ System PostgreSQL stopped${NC}"
        fi
    fi
else
    echo -e "${GREEN}✅ Port 5432 is available${NC}"
fi

# Step 2: Start Docker services
echo ""
echo -e "${YELLOW}[2] Starting Docker services...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "Please start Docker Desktop and run this script again"
    exit 1
fi

echo "Starting PostgreSQL and Phoenix containers..."
docker compose up -d postgres phoenix

echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Wait for PostgreSQL
max_attempts=30
attempt=0
until docker compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}❌ PostgreSQL failed to start${NC}"
        exit 1
    fi
    echo "  Waiting... ($attempt/$max_attempts)"
    sleep 2
done

echo -e "${GREEN}✅ PostgreSQL is ready${NC}"

# Step 3: Check .env file
echo ""
echo -e "${YELLOW}[3] Checking .env configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env and set your OPENAI_API_KEY${NC}"
    echo ""
    echo "Run this command:"
    echo "  nano .env"
    echo ""
    exit 0
fi

# Check if DATABASE_URL is correct
if grep -q "DATABASE_URL=postgresql://postgres:postgres_password@localhost:5432/rag_db" .env; then
    echo -e "${GREEN}✅ DATABASE_URL is correct${NC}"
else
    echo -e "${YELLOW}⚠️  DATABASE_URL might be incorrect${NC}"
    echo "Expected: postgresql://postgres:postgres_password@localhost:5432/rag_db"
    echo "Current:"
    grep DATABASE_URL .env || echo "  Not set"
fi

# Check if OpenAI API key is set
if grep -q "OPENAI_API_KEY=sk-" .env; then
    echo -e "${GREEN}✅ OPENAI_API_KEY is set${NC}"
else
    echo -e "${RED}❌ OPENAI_API_KEY not set in .env${NC}"
    echo ""
    echo "Please edit .env and set your OpenAI API key:"
    echo "  nano .env"
    echo ""
    exit 1
fi

# Step 4: Test connection
echo ""
echo -e "${YELLOW}[4] Testing Docker PostgreSQL connection...${NC}"

if docker compose exec -T postgres psql -U postgres -d rag_db -c "SELECT version();" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Successfully connected to Docker PostgreSQL${NC}"

    # List RAG tables
    echo ""
    echo "Existing RAG tables:"
    docker compose exec -T postgres psql -U postgres -d rag_db -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE '%rag%' ORDER BY tablename;" 2>/dev/null || true
else
    echo -e "${RED}❌ Cannot connect to Docker PostgreSQL${NC}"
    exit 1
fi

# Step 5: Ready to ingest
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  ✅ Setup Complete!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Your environment is now configured correctly."
echo ""
echo "Next steps:"
echo ""
echo "1. Make sure you have PDFs in data/raw/:"
echo "   ls -la data/raw/"
echo ""
echo "2. Run the ingestion script:"
echo "   source venv/bin/activate"
echo "   python src/data_ingestion/ingest_openai.py"
echo ""
echo "3. After ingestion completes, verify the data:"
echo "   python diagnose_db.py"
echo ""
echo "4. Test the RAG system:"
echo "   python main.py \"What are the main topics covered in these documents?\""
echo ""
