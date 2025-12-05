"""
Document Ingestion Pipeline using Docling and Anthropic-style Contextual RAG.

This module provides:
- PDF document processing using Docling
- Contextual chunk generation (Anthropic-style)
- Vector embeddings using Ollama (mxbai-embed-large) or OpenAI
- Storage in PostgreSQL with pgvector
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.readers.docling import DoclingReader

# Conditional imports for LLM and embeddings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.settings import (
    CONFIG, LLM_PROVIDER, DATABASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_DIM,
    RAW_DATA_DIR, CONTEXT_GENERATION_ENABLED, CONTEXT_PROMPT_TEMPLATE
)


class DocumentIngester:
    """
    Document ingestion pipeline with Docling and contextual RAG.

    Features:
    - Docling for high-quality PDF parsing
    - Anthropic-style contextual chunk enrichment
    - Hybrid search support (semantic + BM25)
    - PostgreSQL + pgvector storage
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        table_name: Optional[str] = None,
        generate_context: bool = True
    ):
        """
        Initialize the document ingester.

        Args:
            data_dir: Directory containing PDF files to ingest
            table_name: PostgreSQL table name for embeddings
            generate_context: Whether to generate contextual summaries for chunks
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.table_name = table_name or self._get_table_name()
        self.generate_context = generate_context and CONTEXT_GENERATION_ENABLED

        # Initialize LLM and embedding model
        self._setup_models()

        # Initialize vector store
        self._setup_vector_store()

        # Initialize document reader
        self.reader = DoclingReader()

        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        logger.info(f"DocumentIngester initialized:")
        logger.info(f"  - Data directory: {self.data_dir}")
        logger.info(f"  - Table name: {self.table_name}")
        logger.info(f"  - LLM Provider: {LLM_PROVIDER}")
        logger.info(f"  - Context generation: {self.generate_context}")

    def _get_table_name(self) -> str:
        """Get the appropriate table name based on the LLM provider."""
        env_table = os.getenv("VECTOR_TABLE_NAME")
        if env_table:
            return env_table
        return f"rag_embeddings_{LLM_PROVIDER}"

    def _setup_models(self):
        """Configure LLM and embedding models based on provider."""
        if LLM_PROVIDER == "ollama":
            # Ollama configuration
            self.embed_model = OllamaEmbedding(
                model_name=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=120.0
            )
            self.llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=300.0,
                temperature=0.1
            )
            logger.info(f"Using Ollama - LLM: {OLLAMA_MODEL}, Embeddings: {OLLAMA_EMBEDDING_MODEL}")
        else:
            # OpenAI configuration
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI packages not installed. Install with: pip install llama-index-llms-openai llama-index-embeddings-openai")

            self.embed_model = OpenAIEmbedding(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY
            )
            self.llm = OpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.1
            )
            logger.info(f"Using OpenAI - LLM: {OPENAI_MODEL}, Embeddings: {OPENAI_EMBEDDING_MODEL}")

        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

    def _setup_vector_store(self):
        """Initialize PostgreSQL vector store connection."""
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set")

        db_url_parts = urlparse(DATABASE_URL)

        self.vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port or 5432,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=self.table_name,
            embed_dim=EMBED_DIM,
            hybrid_search=True,
            text_search_config="english"
        )

        logger.info(f"Vector store connected to {db_url_parts.hostname}:{db_url_parts.port}")

    def _categorize_document(self, file_path: Path) -> str:
        """
        Categorize a document based on its filename.

        Returns a category string used for metadata filtering.
        """
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

    def _generate_chunk_context(self, chunk_text: str, document_text: str, file_name: str) -> str:
        """
        Generate contextual summary for a chunk using the LLM.

        This implements Anthropic's contextual RAG approach where each chunk
        is enriched with context about its role in the overall document.
        """
        if not self.generate_context:
            return ""

        # Limit document text to avoid token limits
        max_doc_chars = 8000
        doc_excerpt = document_text[:max_doc_chars]
        if len(document_text) > max_doc_chars:
            doc_excerpt += "\n... [truncated for context generation]"

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            WHOLE_DOCUMENT=doc_excerpt,
            CHUNK_CONTENT=chunk_text
        )

        try:
            response = self.llm.complete(prompt)
            context = str(response).strip()
            logger.debug(f"Generated context for chunk from {file_name}: {context[:100]}...")
            return context
        except Exception as e:
            logger.warning(f"Failed to generate context for chunk: {e}")
            return ""

    def process_document(self, file_path: Path) -> List[TextNode]:
        """
        Process a single document and return enriched nodes.

        Args:
            file_path: Path to the PDF document

        Returns:
            List of TextNode objects with metadata and optional context
        """
        logger.info(f"Processing document: {file_path.name}")

        # Read document with Docling
        try:
            documents = self.reader.load_data(file_path=str(file_path))
        except Exception as e:
            logger.error(f"Failed to read document {file_path}: {e}")
            return []

        if not documents:
            logger.warning(f"No content extracted from {file_path}")
            return []

        # Combine all document text for context generation
        full_text = "\n\n".join([doc.text for doc in documents])

        # Parse into chunks
        all_nodes = []
        category = self._categorize_document(file_path)

        for doc_idx, document in enumerate(documents):
            # Get page number if available
            page_num = document.metadata.get("page_number", doc_idx + 1)

            # Split into chunks
            chunks = self.node_parser.get_nodes_from_documents([document])

            for chunk_idx, chunk in enumerate(chunks):
                # Generate contextual summary if enabled
                context = ""
                if self.generate_context:
                    context = self._generate_chunk_context(
                        chunk.get_content(),
                        full_text,
                        file_path.name
                    )

                # Create enriched node with metadata
                node = TextNode(
                    text=chunk.get_content(),
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "category": category,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "context": context,
                        "source_file": file_path.name,
                    },
                    excluded_embed_metadata_keys=["file_path", "chunk_index"],
                    excluded_llm_metadata_keys=["file_path", "chunk_index"],
                )
                all_nodes.append(node)

        logger.info(f"Created {len(all_nodes)} nodes from {file_path.name}")
        return all_nodes

    def ingest_directory(self, file_patterns: List[str] = None) -> int:
        """
        Ingest all PDF documents from the data directory.

        Args:
            file_patterns: Optional list of filename patterns to match

        Returns:
            Number of nodes successfully ingested
        """
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.PDF"]

        # Find all matching files
        pdf_files = []
        for pattern in file_patterns:
            pdf_files.extend(self.data_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process all documents
        all_nodes = []
        for pdf_file in pdf_files:
            nodes = self.process_document(pdf_file)
            all_nodes.extend(nodes)

        if not all_nodes:
            logger.warning("No nodes created from any documents")
            return 0

        # Create index and store in vector store
        logger.info(f"Creating vector index with {len(all_nodes)} nodes...")

        index = VectorStoreIndex(
            nodes=all_nodes,
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            show_progress=True
        )

        logger.info(f"Successfully ingested {len(all_nodes)} nodes into {self.table_name}")
        return len(all_nodes)

    def clear_table(self):
        """Clear all data from the vector store table."""
        # Note: This is a placeholder - actual implementation depends on pgvector
        logger.warning("Table clearing not implemented - manually clear if needed")


def ingest_documents(
    data_dir: Optional[str] = None,
    table_name: Optional[str] = None,
    generate_context: bool = True
) -> int:
    """
    Convenience function to ingest documents.

    Args:
        data_dir: Directory containing PDF files
        table_name: PostgreSQL table name for embeddings
        generate_context: Whether to generate contextual summaries

    Returns:
        Number of nodes ingested
    """
    data_path = Path(data_dir) if data_dir else None
    ingester = DocumentIngester(
        data_dir=data_path,
        table_name=table_name,
        generate_context=generate_context
    )
    return ingester.ingest_directory()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing PDF files (default: data/raw)"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="PostgreSQL table name (default: based on LLM provider)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable contextual chunk generation"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai"],
        default=None,
        help="Override LLM provider"
    )

    args = parser.parse_args()

    # Override provider if specified
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    logger.info("=" * 60)
    logger.info("Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"LLM Provider: {LLM_PROVIDER}")
    logger.info(f"Context Generation: {not args.no_context}")
    logger.info("=" * 60)

    count = ingest_documents(
        data_dir=args.data_dir,
        table_name=args.table_name,
        generate_context=not args.no_context
    )

    logger.info("=" * 60)
    logger.info(f"Ingestion complete! Total nodes: {count}")
    logger.info("=" * 60)
