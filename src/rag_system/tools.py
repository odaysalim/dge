"""
Custom tools for the Agentic RAG system.
Supports both OpenAI and Ollama embeddings.
"""

import os
from urllib.parse import urlparse
from typing import Dict, Union, Any, Optional, List
import json

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
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


@tool("Query Router Tool")
def query_router_tool(query: Union[str, Dict[str, Any]]) -> str:
    """
    Analyzes a user query to determine which document set should be searched.

    This tool specializes in routing queries to the appropriate procurement manual:
    - SAP Ariba Manual: For queries about system usage, Ariba-specific features
    - Business Process Manual: For general procurement policies and workflows
    - Both: For ambiguous queries or queries that benefit from both perspectives

    Args:
        query: The user's query string

    Returns:
        JSON string with routing decision and justification:
        {"route": "ariba|business_process|both", "justification": "..."}
    """
    # Extract query string
    if isinstance(query, dict):
        search_query = query.get("description", query.get("query", str(query)))
    else:
        search_query = str(query).strip()

    print(f"DEBUG [Router]: Analyzing query: {repr(search_query)}")

    # Convert query to lowercase for keyword matching
    query_lower = search_query.lower()

    # Keywords that strongly indicate Ariba system usage
    ariba_keywords = [
        "ariba", "sap", "system", "module", "workspace", "project owner",
        "sourcing project", "contract workspace", "how do i", "how to",
        "create in", "submit in", "approve in", "publish", "user interface"
    ]

    # Keywords that indicate general business process
    business_keywords = [
        "policy", "policies", "procedure", "process flow", "roles",
        "responsibilities", "approval authority", "pdoa", "what is the process",
        "general", "overview", "framework", "governance"
    ]

    # Count keyword matches
    ariba_score = sum(1 for kw in ariba_keywords if kw in query_lower)
    business_score = sum(1 for kw in business_keywords if kw in query_lower)

    # Decision logic
    if ariba_score > business_score and ariba_score >= 2:
        route = "ariba"
        justification = (
            f"Query contains Ariba/system-specific keywords (score: {ariba_score}). "
            "Routing to SAP Ariba Aligned Manual for system implementation details."
        )
    elif business_score > ariba_score and business_score >= 2:
        route = "business_process"
        justification = (
            f"Query contains business process keywords (score: {business_score}). "
            "Routing to Business Process Manual for policy and workflow information."
        )
    elif ariba_score == 0 and business_score == 0:
        # No specific keywords - default to both
        route = "both"
        justification = (
            "Query doesn't contain specific indicators for either manual. "
            "Searching both manuals to ensure comprehensive coverage."
        )
    else:
        # Ambiguous or could benefit from both
        route = "both"
        justification = (
            f"Query has mixed indicators (Ariba: {ariba_score}, Business: {business_score}). "
            "Searching both manuals for complete perspective."
        )

    result = {
        "route": route,
        "justification": justification,
        "query": search_query
    }

    print(f"DEBUG [Router]: Decision = {route}, Justification = {justification}")

    return json.dumps(result, indent=2)


@tool("Document Retrieval Tool")
def document_retrieval_tool(query: Union[str, Dict[str, Any]], routing_info: Optional[str] = None) -> str:
    """
    Retrieves relevant context from a collection of documents using vector similarity search.
    Use this tool to search for information in indexed documents.

    Args:
        query: The search query string
        routing_info: Optional JSON string from Query Router Tool containing routing decision.
                     Format: {"route": "ariba|business_process|both", "justification": "..."}
                     If not provided, searches all documents.

    Returns:
        Formatted context with document chunks and source citations
    """
    try:
        # Debug: Log what we actually receive
        print(f"DEBUG: Tool received query parameter: {repr(query)}")
        print(f"DEBUG: Tool received routing_info: {repr(routing_info)}")

        # Handle CrewAI's parameter passing - extract actual query from different formats
        search_query = None
        route_decision = None

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

        # Parse routing information if provided
        if routing_info:
            try:
                if isinstance(routing_info, str):
                    route_decision = json.loads(routing_info)
                elif isinstance(routing_info, dict):
                    route_decision = routing_info
                print(f"DEBUG: Parsed routing decision: {route_decision}")
            except json.JSONDecodeError:
                print(f"DEBUG: Failed to parse routing_info, searching all documents")
                route_decision = None

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

        # Apply metadata filtering based on routing decision
        filters = None
        if route_decision and route_decision.get("route") in ["ariba", "business_process"]:
            route = route_decision["route"]
            print(f"DEBUG: Applying document filter for route: {route}")

            # Define document name patterns for each route
            if route == "ariba":
                # Search only SAP Ariba Aligned Manual
                doc_pattern = "Procurement Manual (Ariba Aligned)"
                filter_msg = "Searching only: SAP Ariba Aligned Manual"
            else:  # business_process
                # Search only Business Process Manual
                doc_pattern = "Procurement Manual (Business Process)"
                filter_msg = "Searching only: Business Process Manual"

            print(f"DEBUG: {filter_msg}")

            # Note: LlamaIndex's MetadataFilters with ExactMatchFilter doesn't work well with
            # PGVector hybrid search. Instead, we'll filter the results post-retrieval.
            # This is a workaround until better filtering support is available.
            document_filter = doc_pattern
        else:
            document_filter = None
            print(f"DEBUG: No document filtering applied - searching all documents")

        # Create a query engine with hybrid search mode
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid" if CONFIG['rag']['hybrid_search'] else "default",
            similarity_top_k=CONFIG['rag']['retrieval_top_k'] * 2,  # Retrieve more for filtering
            sparse_top_k=CONFIG['rag']['sparse_top_k'] * 2 if CONFIG['rag']['hybrid_search'] else None
        )

        # Query using the configured search mode
        response = query_engine.query(search_query)
        all_nodes = response.source_nodes

        # Apply post-retrieval filtering if needed
        if document_filter:
            retrieved_nodes = []
            for node in all_nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    file_name = node.metadata.get('file_name', '')
                    file_path = node.metadata.get('file_path', '')

                    # Check if this node matches the document filter
                    if document_filter in file_name or document_filter in file_path:
                        retrieved_nodes.append(node)

            # Limit to top_k after filtering
            retrieved_nodes = retrieved_nodes[:CONFIG['rag']['retrieval_top_k']]
            print(f"DEBUG: Filtered {len(all_nodes)} nodes to {len(retrieved_nodes)} matching '{document_filter}'")
        else:
            retrieved_nodes = all_nodes[:CONFIG['rag']['retrieval_top_k']]

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
