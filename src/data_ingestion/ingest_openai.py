#!/usr/bin/env python3
"""
Enhanced PGVector Indexing with Contextual RAG using OpenAI
Implements Anthropic's Contextual Retrieval approach with OpenAI embeddings and LLM
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document, TextNode
import asyncio
import nest_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import CONFIG, validate_config

# Allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'openai_contextual_rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MD_DIR = str(project_root / "data" / "processed" / "md")
RAW_DIR = str(project_root / "data" / "raw")

# Use settings from config
DATABASE_URL = CONFIG['database']['url']
TABLE_NAME = f"rag_embeddings_openai_{datetime.now().strftime('%Y%m%d')}"
EMBED_DIM = CONFIG['rag']['embed_dim']
CHUNK_SIZE = CONFIG['rag']['chunk_size']
CHUNK_OVERLAP = CONFIG['rag']['chunk_overlap']

# OpenAI Configuration
OPENAI_API_KEY = CONFIG['openai']['api_key']
OPENAI_MODEL = CONFIG['openai']['model']
OPENAI_EMBEDDING_MODEL = CONFIG['openai']['embedding_model']

# Context generation prompt (Anthropic-style)
CONTEXT_PROMPT_TEMPLATE = CONFIG['context_generation']['prompt_template']


def check_database_connection():
    """Test database connectivity and check for pgvector extension"""
    try:
        logger.info(f"Testing connection to database")
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ PostgreSQL connected: {version}")

            # Check pgvector extension
            result = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
            has_pgvector = result.fetchone()[0]

            if has_pgvector:
                logger.info("✅ pgvector extension is installed")
            else:
                logger.error("❌ pgvector extension not found")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def generate_contextual_info(llm: OpenAI, chunk_text: str, whole_document: str) -> str:
    """
    Generate contextual information for a chunk using OpenAI.
    This implements Anthropic's contextual retrieval approach.

    Args:
        llm: OpenAI LLM instance
        chunk_text: The text chunk to generate context for
        whole_document: The full document text for context

    Returns:
        Generated context string (1-2 sentences)
    """
    try:
        # Truncate whole document if too long (keep first 8000 chars to stay within context limits)
        truncated_doc = whole_document[:8000] if len(whole_document) > 8000 else whole_document

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            WHOLE_DOCUMENT=truncated_doc,
            CHUNK_CONTENT=chunk_text
        )

        response = llm.complete(prompt)
        context = response.text.strip()

        logger.debug(f"Generated context: {context[:100]}...")
        return context

    except Exception as e:
        logger.warning(f"Context generation failed: {e}")
        return ""


async def process_document_with_context(
    document: Document,
    llm: OpenAI,
    text_splitter: TokenTextSplitter
) -> List[TextNode]:
    """
    Process a document into chunks with contextual information.

    Args:
        document: LlamaIndex Document object
        llm: OpenAI LLM instance for context generation
        text_splitter: Text splitter for chunking

    Returns:
        List of TextNode objects with contextual metadata
    """
    try:
        logger.info(f"Processing document: {document.metadata.get('file_name', 'Unknown')}")

        # Get the whole document text
        whole_document = document.get_content()

        # Split into chunks
        chunks = text_splitter.split_text(whole_document)
        logger.info(f"Split into {len(chunks)} chunks")

        # Create nodes with contextual information
        nodes = []
        for i, chunk_text in enumerate(chunks):
            # Generate contextual information for this chunk
            if CONFIG['context_generation']['enabled']:
                context = generate_contextual_info(llm, chunk_text, whole_document)
            else:
                context = ""

            # Create a TextNode with the chunk and its context
            node = TextNode(
                text=chunk_text,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'context': context,  # Store the generated context
                }
            )
            nodes.append(node)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")

        logger.info(f"✅ Completed processing document with {len(nodes)} contextualized chunks")
        return nodes

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return []


def ingest_documents():
    """
    Main ingestion function with OpenAI contextual RAG.
    """
    try:
        # Validate configuration
        validate_config()
        logger.info("✅ Configuration validated")

        # Check database connection
        if not check_database_connection():
            logger.error("❌ Cannot proceed without database connection")
            return

        # Initialize OpenAI models
        logger.info("Initializing OpenAI models...")
        llm = OpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=CONFIG['openai']['temperature'],
            max_tokens=CONFIG['openai']['max_tokens']
        )

        embed_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        logger.info(f"✅ Using OpenAI model: {OPENAI_MODEL}")
        logger.info(f"✅ Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL} (dim={EMBED_DIM})")

        # Load documents
        # Try markdown directory first, then raw directory
        if os.path.exists(MD_DIR) and os.listdir(MD_DIR):
            logger.info(f"Loading documents from: {MD_DIR}")
            documents = SimpleDirectoryReader(
                input_dir=MD_DIR,
                filename_as_id=True,
                recursive=True
            ).load_data()
        elif os.path.exists(RAW_DIR) and os.listdir(RAW_DIR):
            logger.info(f"Loading documents from: {RAW_DIR}")
            documents = SimpleDirectoryReader(
                input_dir=RAW_DIR,
                filename_as_id=True,
                recursive=True
            ).load_data()
        else:
            logger.error(f"❌ No documents found in {MD_DIR} or {RAW_DIR}")
            return

        logger.info(f"✅ Loaded {len(documents)} documents")

        # Initialize text splitter
        text_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Process documents with contextual information
        logger.info("Generating contextual information for chunks...")
        all_nodes = []
        for doc in documents:
            nodes = asyncio.run(process_document_with_context(doc, llm, text_splitter))
            all_nodes.extend(nodes)

        logger.info(f"✅ Generated {len(all_nodes)} contextualized chunks from {len(documents)} documents")

        # Initialize vector store
        logger.info(f"Initializing PGVector store with table: {TABLE_NAME}")
        db_url_parts = urlparse(DATABASE_URL)

        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=TABLE_NAME,
            embed_dim=EMBED_DIM,
            hybrid_search=CONFIG['rag']['hybrid_search'],
            text_search_config="english"
        )

        # Create index and store embeddings
        logger.info("Creating embeddings and storing in vector database...")
        index = VectorStoreIndex(
            nodes=all_nodes,
            vector_store=vector_store,
            embed_model=embed_model,
            show_progress=True
        )

        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║                  ✅ INGESTION COMPLETE                       ║
╠══════════════════════════════════════════════════════════════╣
║  Documents processed: {len(documents):>40} ║
║  Total chunks created: {len(all_nodes):>39} ║
║  Table name: {TABLE_NAME:>47} ║
║  Embedding model: {OPENAI_EMBEDDING_MODEL:>44} ║
║  Embedding dimension: {EMBED_DIM:>40} ║
║  Context generation: {'ENABLED' if CONFIG['context_generation']['enabled'] else 'DISABLED':>41} ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Save table name to environment file for retrieval
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'a') as f:
                f.write(f"\n# Auto-generated by ingestion script\nVECTOR_TABLE_NAME={TABLE_NAME}\n")
            logger.info(f"✅ Saved table name to {env_file}")

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Starting OpenAI Contextual RAG Ingestion Pipeline")
    logger.info("="*80)

    ingest_documents()
