"""
Custom tools for the Agentic RAG system.
Supports both OpenAI and Ollama embeddings.
"""

import os
from urllib.parse import urlparse
from typing import Dict, Union, Any

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import tool

from ..config.settings import CONFIG, LLM_PROVIDER, DATABASE_URL


def get_embedding_model():
    """
    Get the configured embedding model based on the provider setting.
    Returns either OpenAI or Ollama embedding model.
    """
    if LLM_PROVIDER == "openai":
        return OpenAIEmbedding(
            model=CONFIG['openai']['embedding_model'],
            api_key=CONFIG['openai']['api_key'],
            timeout=120.0
        )
    else:
        return OllamaEmbedding(
            model_name=CONFIG['ollama']['embedding_model'],
            base_url=CONFIG['ollama']['base_url'],
            request_timeout=120.0
        )


@tool("Document Retrieval Tool")
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """
    Retrieves relevant context from a collection of documents using vector similarity search.
    Use this tool to search for information in indexed documents.

    Args:
        query: The search query string

    Returns:
        Formatted context with document chunks and source citations
    """
    try:
        # Debug: Log what we actually receive
        print(f"DEBUG: Tool received query parameter: {repr(query)}")

        # Handle CrewAI's parameter passing - extract actual query from different formats
        search_query = None

        if isinstance(query, str):
            search_query = query
        elif isinstance(query, dict):
            # CrewAI passes: {"description": "actual query", "type": "str"}
            if "description" in query:
                search_query = query["description"]
            elif "query" in query:
                search_query = query["query"]
            else:
                # Fallback: convert dict to string
                search_query = str(query)
        else:
            search_query = str(query)

        # Validate we have a proper query string
        if not search_query or not isinstance(search_query, str):
            return "Error: No valid search query provided."

        # Check if we got a placeholder description instead of real query
        if search_query in ["The search query to find relevant documents", ""]:
            return "Error: Tool received schema placeholder instead of actual query."

        search_query = search_query.strip()
        print(f"DEBUG: Extracted search query: {repr(search_query)}")

        if not DATABASE_URL:
            return "Error: DATABASE_URL environment variable not set."

        db_url_parts = urlparse(DATABASE_URL)

        # Debug print parsed connection details (password masked)
        print(f"DEBUG: Parsed Postgres connection details - "
              f"host: {db_url_parts.hostname}, "
              f"port: {db_url_parts.port}, "
              f"database: {db_url_parts.path.lstrip('/')}, "
              f"user: {db_url_parts.username}")

        # Initialize the vector store with connection details and hybrid search enabled
        # Table name will be dynamically determined based on available tables
        table_name = os.getenv("VECTOR_TABLE_NAME", "rag_embeddings_openai")

        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=table_name,
            embed_dim=CONFIG['rag']['embed_dim'],
            hybrid_search=CONFIG['rag']['hybrid_search'],
            text_search_config="english",
        )
        print(f"DEBUG: Using vector table: {table_name}")

        # Get the configured embedding model
        embed_model = get_embedding_model()
        print(f"DEBUG: Using {LLM_PROVIDER} embeddings")

        # Create a LlamaIndex VectorStoreIndex object from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Create a query engine with hybrid search mode
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid" if CONFIG['rag']['hybrid_search'] else "default",
            similarity_top_k=CONFIG['rag']['retrieval_top_k'],
            sparse_top_k=CONFIG['rag']['sparse_top_k'] if CONFIG['rag']['hybrid_search'] else None
        )

        # Query using the configured search mode
        response = query_engine.query(search_query)
        retrieved_nodes = response.source_nodes

        if not retrieved_nodes:
            return "No relevant documents found for this query."

        # Format the retrieved context with source metadata and contextual information
        formatted_chunks = []
        sources_seen = set()

        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()

            # Extract source file information from metadata
            source_info = "Unknown source"
            context_info = ""
            page_info = ""

            if hasattr(node, 'metadata') and node.metadata:
                # File information
                file_name = node.metadata.get('source_file', node.metadata.get('file_name', 'Unknown file'))
                file_path = node.metadata.get('file_path', '')
                if file_path:
                    source_info = f"Source: {os.path.basename(file_path)}"
                    sources_seen.add(os.path.basename(file_path))
                else:
                    source_info = f"Source: {file_name}"
                    sources_seen.add(file_name)

                # Contextual information (if available from contextual RAG)
                context = node.metadata.get('context', '')
                if context:
                    context_info = f"\nContext: {context}"

                # Page number information (if available)
                page_num = node.metadata.get('page_number', '')
                if page_num:
                    page_info = f" (Page {page_num})"

            formatted_chunk = f"**Document Chunk {i}**\n{source_info}{page_info}{context_info}\n\nContent:\n{content}"
            formatted_chunks.append(formatted_chunk)

        context = "\n\n" + "="*50 + "\n\n".join(formatted_chunks)

        # Add a summary of sources at the end
        if sources_seen:
            context += "\n\n" + "="*50 + "\n\n**Sources Used:**\n" + "\n".join(f"- {source}" for source in sorted(sources_seen))

        return context

    except Exception as e:
        error_msg = f"Error retrieving documents: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg
