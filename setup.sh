#!/bin/bash

# Agentic RAG System - Automated Setup Script
# This script automates the setup process for macOS

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Agentic RAG System - Automated Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This script is designed for macOS${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Install it with: brew install python@3.11"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Step 1: Create virtual environment
echo ""
echo -e "${YELLOW}🐍 Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Step 2: Install dependencies
echo ""
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 3: Setup environment file
echo ""
echo -e "${YELLOW}⚙️  Setting up environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env and add your OPENAI_API_KEY${NC}"
    echo "   Then run this script again"
    echo ""
    echo "   nano .env"
    echo ""
    exit 0
else
    # Check if OpenAI API key is set
    if grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env || grep -q "OPENAI_API_KEY=$" .env; then
        echo -e "${RED}❌ OPENAI_API_KEY not set in .env file${NC}"
        echo "   Please edit .env and add your OpenAI API key:"
        echo "   nano .env"
        exit 1
    fi
    echo -e "${GREEN}✅ Environment file configured${NC}"
fi

# Step 4: Create necessary directories
echo ""
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed/md
mkdir -p data/processed/json
mkdir -p data/evaluation
mkdir -p data/memory
echo -e "${GREEN}✅ Directories created${NC}"

# Step 5: Start Docker services
echo ""
echo -e "${YELLOW}🐳 Starting Docker services (PostgreSQL + Phoenix)...${NC}"
docker-compose up -d postgres phoenix

# Wait for services to be ready
echo "   Waiting for PostgreSQL to be ready..."
sleep 10

# Check PostgreSQL
until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo "   PostgreSQL is starting..."
    sleep 2
done
echo -e "${GREEN}✅ PostgreSQL is ready${NC}"

# Check Phoenix
echo "   Waiting for Phoenix to be ready..."
until curl -sf http://localhost:6006/healthz > /dev/null 2>&1; do
    echo "   Phoenix is starting..."
    sleep 2
done
echo -e "${GREEN}✅ Arize Phoenix is ready${NC}"

# Step 6: Instructions for document ingestion
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Setup Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "1. Place your PDF documents in the data/raw/ directory:"
echo "   cp /path/to/your/documents/*.pdf data/raw/"
echo ""
echo "2. (Optional) Convert PDFs to Markdown using Docling:"
echo "   python src/data_ingestion/ingestion_docling.py"
echo ""
echo "3. Ingest documents with contextual embeddings:"
echo "   python src/data_ingestion/ingest_openai.py"
echo ""
echo "4. Start the API server:"
echo "   python api.py"
echo ""
echo "5. (Optional) Start OpenWebUI:"
echo "   docker-compose up -d open-webui"
echo "   open http://localhost:3000"
echo ""
echo -e "${YELLOW}📊 Services:${NC}"
echo "   • PostgreSQL:      localhost:5432"
echo "   • Arize Phoenix:   http://localhost:6006"
echo "   • API Server:      http://localhost:8000 (when started)"
echo "   • OpenWebUI:       http://localhost:3000 (when started)"
echo ""
echo -e "${YELLOW}📚 For detailed instructions, see:${NC} SETUP_GUIDE.md"
echo ""
