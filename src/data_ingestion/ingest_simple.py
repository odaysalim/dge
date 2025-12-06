"""
Simple Document Ingestion Pipeline using LlamaIndex PDF reader.
Works reliably in Docker without extra dependencies.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.settings import (
    LLM_PROVIDER, DATABASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_DIM,
    RAW_DATA_DIR
)


def get_table_name() -> str:
    """Get the vector table name from environment."""
    return os.getenv("VECTOR_TABLE_NAME", f"rag_embeddings_{LLM_PROVIDER}")


def categorize_document(file_path: Path) -> str:
    """Categorize document based on filename."""
    filename_lower = file_path.name.lower()

    if "hr" in filename_lower or "bylaw" in filename_lower:
        return "hr"
    elif "infosec" in filename_lower or "security" in filename_lower:
        return "infosec"
    elif "ariba" in filename_lower:
        return "procurement_ariba"
    elif "business" in filename_lower and "process" in filename_lower:
        return "procurement_business"
    elif "standard" in filename_lower or "abu dhabi" in filename_lower:
        return "procurement_standards"
    elif "procurement" in filename_lower:
        return "procurement_all"
    else:
        return "general"


def ingest_documents(data_dir: Optional[str] = None, table_name: Optional[str] = None) -> int:
    """
    Ingest PDF documents using simple LlamaIndex reader.

    Args:
        data_dir: Directory containing PDF files
        table_name: PostgreSQL table name

    Returns:
        Number of nodes ingested
    """
    data_path = Path(data_dir) if data_dir else RAW_DATA_DIR
    table = table_name or get_table_name()

    logger.info("=" * 60)
    logger.info("Simple Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_path}")
    logger.info(f"Table name: {table}")
    logger.info(f"LLM Provider: {LLM_PROVIDER}")
    logger.info(f"Embedding model: {OLLAMA_EMBEDDING_MODEL}")
    logger.info(f"Embed dimensions: {EMBED_DIM}")
    logger.info("=" * 60)

    # Setup embedding model
    logger.info("Setting up Ollama embedding model...")
    embed_model = OllamaEmbedding(
        model_name=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0
    )
    Settings.embed_model = embed_model

    # Setup LLM (for potential future use)
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=300.0
    )
    Settings.llm = llm

    # Setup vector store
    logger.info("Connecting to PostgreSQL vector store...")
    db_url_parts = urlparse(DATABASE_URL)

    vector_store = PGVectorStore.from_params(
        host=db_url_parts.hostname,
        port=db_url_parts.port or 5432,
        database=db_url_parts.path.lstrip('/'),
        user=db_url_parts.username,
        password=db_url_parts.password,
        table_name=table,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config="english"
    )
    logger.info(f"Connected to {db_url_parts.hostname}:{db_url_parts.port}")

    # Find PDF files
    pdf_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_path}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Node parser
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Process each file
    all_nodes = []

    for pdf_file in pdf_files:
        logger.info(f"\nProcessing: {pdf_file.name}")

        try:
            # Read PDF using SimpleDirectoryReader (works with most PDFs)
            reader = SimpleDirectoryReader(
                input_files=[str(pdf_file)],
                filename_as_id=True
            )
            documents = reader.load_data()

            if not documents:
                logger.warning(f"  No content extracted from {pdf_file.name}")
                continue

            logger.info(f"  Extracted {len(documents)} document(s)")

            # Get category
            category = categorize_document(pdf_file)
            logger.info(f"  Category: {category}")

            # Parse into chunks
            file_nodes = []
            for doc in documents:
                chunks = node_parser.get_nodes_from_documents([doc])

                for i, chunk in enumerate(chunks):
                    node = TextNode(
                        text=chunk.get_content(),
                        metadata={
                            "file_name": pdf_file.name,
                            "file_path": str(pdf_file),
                            "category": category,
                            "chunk_index": i,
                            "source_file": pdf_file.name,
                        },
                        excluded_embed_metadata_keys=["file_path", "chunk_index"],
                        excluded_llm_metadata_keys=["file_path", "chunk_index"],
                    )
                    file_nodes.append(node)

            logger.info(f"  Created {len(file_nodes)} chunks")
            all_nodes.extend(file_nodes)

        except Exception as e:
            logger.error(f"  Error processing {pdf_file.name}: {e}")
            continue

    if not all_nodes:
        logger.warning("No nodes created from any documents")
        return 0

    # Create index and store embeddings
    logger.info(f"\nCreating vector index with {len(all_nodes)} nodes...")
    logger.info("This may take a while (embedding with Ollama)...")

    # Process in batches to show progress
    batch_size = 10
    total_batches = (len(all_nodes) + batch_size - 1) // batch_size

    for i in range(0, len(all_nodes), batch_size):
        batch = all_nodes[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} nodes)...")

        if i == 0:
            # First batch creates the index
            index = VectorStoreIndex(
                nodes=batch,
                vector_store=vector_store,
                embed_model=embed_model,
                show_progress=True
            )
        else:
            # Subsequent batches add to the index
            index.insert_nodes(batch)

    logger.info("=" * 60)
    logger.info(f"SUCCESS! Ingested {len(all_nodes)} nodes into {table}")
    logger.info("=" * 60)

    return len(all_nodes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents with Ollama embeddings")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with PDFs")
    parser.add_argument("--table-name", type=str, default=None, help="Vector table name")

    args = parser.parse_args()

    count = ingest_documents(
        data_dir=args.data_dir,
        table_name=args.table_name
    )

    if count == 0:
        logger.error("No documents were ingested!")
        exit(1)
