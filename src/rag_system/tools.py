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

    Routes queries to the appropriate document category:
    - hr: HR Bylaws (leave, benefits, employee policies)
    - infosec: Information Security policies
    - procurement_ariba: SAP Ariba system usage and technical implementation
    - procurement_business: Procurement business processes and workflows
    - procurement_standards: Abu Dhabi Government procurement standards/regulations
    - procurement_all: Ambiguous procurement queries (search all procurement docs)
    - all: Non-specific queries (search everything)

    Args:
        query: The user's query string

    Returns:
        JSON string with routing decision and justification
    """
    # Extract query string
    if isinstance(query, dict):
        search_query = query.get("description", query.get("query", str(query)))
    else:
        search_query = str(query).strip()

    print(f"DEBUG [Router]: Analyzing query: {repr(search_query)}")

    query_lower = search_query.lower()

    # ===========================================
    # STEP 0: Check for greetings/chitchat (skip retrieval)
    # ===========================================

    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "greetings", "what's up", "whats up", "sup", "yo"
    ]

    chitchat_patterns = [
        "how are you", "how r u", "how do you do", "nice to meet",
        "thank you", "thanks", "bye", "goodbye", "see you", "take care",
        "who are you", "what are you", "what can you do", "help me"
    ]

    # Check if it's just a greeting (short message with greeting word)
    words = query_lower.split()
    is_greeting = (
        len(words) <= 3 and any(g in query_lower for g in greetings)
    ) or any(pattern in query_lower for pattern in chitchat_patterns)

    if is_greeting:
        result = {
            "route": "general",
            "justification": "This is a greeting or general chitchat - no document retrieval needed.",
            "query": search_query
        }
        print(f"DEBUG [Router]: Detected greeting/chitchat. Skipping document retrieval.")
        return json.dumps(result, indent=2)

    # ===========================================
    # STEP 1: Identify the topic/domain
    # ===========================================

    # HR keywords - employee-related policies
    hr_keywords = [
        "leave", "annual leave", "sick leave", "maternity", "paternity",
        "vacation", "holiday", "employee", "staff", "salary", "compensation",
        "benefits", "probation", "termination", "resignation", "promotion",
        "grade", "position", "hr", "human resources", "bylaws", "attendance",
        "working hours", "overtime", "training", "performance", "appraisal"
    ]

    # Information Security keywords
    infosec_keywords = [
        "security", "password", "access control", "data protection", "cybersecurity",
        "information security", "infosec", "encryption", "firewall", "malware",
        "phishing", "incident", "breach", "confidential", "classification",
        "backup", "disaster recovery", "authentication", "authorization"
    ]

    # General procurement keywords (indicates procurement domain)
    procurement_keywords = [
        "procurement", "purchase", "purchasing", "vendor", "supplier", "tender",
        "bid", "bidding", "contract", "rfp", "rfq", "quotation", "sourcing",
        "spend", "requisition", "purchase order", "po", "invoice", "payment terms",
        "award", "evaluation", "prequalification", "category management"
    ]

    # Calculate topic scores
    hr_score = sum(1 for kw in hr_keywords if kw in query_lower)
    infosec_score = sum(1 for kw in infosec_keywords if kw in query_lower)
    procurement_score = sum(1 for kw in procurement_keywords if kw in query_lower)

    print(f"DEBUG [Router]: Topic scores - HR: {hr_score}, InfoSec: {infosec_score}, Procurement: {procurement_score}")

    # Determine primary topic
    if hr_score > 0 and hr_score >= infosec_score and hr_score >= procurement_score:
        route = "hr"
        justification = f"Query relates to HR/employee matters (matched {hr_score} HR keywords). Routing to HR Bylaws."

    elif infosec_score > 0 and infosec_score >= hr_score and infosec_score >= procurement_score:
        route = "infosec"
        justification = f"Query relates to information security (matched {infosec_score} security keywords). Routing to Information Security document."

    elif procurement_score > 0 or _is_procurement_context(query_lower):
        # ===========================================
        # STEP 2: For procurement, determine sub-route
        # ===========================================
        route, justification = _route_procurement_query(query_lower, procurement_score)

    else:
        # No clear topic - search all documents
        route = "all"
        justification = "Query doesn't match a specific document category. Searching all documents."

    result = {
        "route": route,
        "justification": justification,
        "query": search_query
    }

    print(f"DEBUG [Router]: Decision = {route}, Justification = {justification}")

    return json.dumps(result, indent=2)


def _is_procurement_context(query_lower: str) -> bool:
    """Check if query is likely procurement-related even without explicit keywords."""
    procurement_context_hints = [
        "ariba", "sap", "sourcing project", "contract workspace",
        "supplier management", "spend analysis", "catalog"
    ]
    return any(hint in query_lower for hint in procurement_context_hints)


def _route_procurement_query(query_lower: str, procurement_score: int) -> tuple:
    """
    Sub-route procurement queries to the appropriate procurement document.

    Returns:
        tuple: (route, justification)
    """
    # Ariba system-specific keywords (technical/system usage)
    ariba_keywords = [
        "ariba", "sap ariba", "workspace", "sourcing project", "contract workspace",
        "project owner", "team member", "ariba network", "catalog", "punch-out",
        "user interface", "ui", "screen", "button", "click", "navigate", "menu",
        "module", "system", "login", "dashboard"
    ]

    # Business process keywords (policies/workflows)
    business_process_keywords = [
        "process", "workflow", "procedure", "step", "approval", "authority",
        "pdoa", "delegation", "roles", "responsibilities", "policy", "policies",
        "threshold", "limit", "category", "classification", "lifecycle"
    ]

    # Abu Dhabi Government standards keywords
    standards_keywords = [
        "abu dhabi", "government", "regulation", "standard", "compliance",
        "law", "legal", "mandatory", "requirement", "guideline", "framework",
        "public sector", "federal", "local"
    ]

    ariba_score = sum(1 for kw in ariba_keywords if kw in query_lower)
    business_score = sum(1 for kw in business_process_keywords if kw in query_lower)
    standards_score = sum(1 for kw in standards_keywords if kw in query_lower)

    print(f"DEBUG [Router]: Procurement sub-scores - Ariba: {ariba_score}, Business: {business_score}, Standards: {standards_score}")

    # Decision logic for procurement sub-routing
    if ariba_score > business_score and ariba_score > standards_score and ariba_score >= 1:
        return (
            "procurement_ariba",
            f"Procurement query with Ariba/system focus (score: {ariba_score}). "
            "Routing to Procurement Manual (Ariba Aligned)."
        )

    elif standards_score > ariba_score and standards_score > business_score and standards_score >= 1:
        return (
            "procurement_standards",
            f"Procurement query about government standards (score: {standards_score}). "
            "Routing to Abu Dhabi Procurement Standards."
        )

    elif business_score > ariba_score and business_score > standards_score and business_score >= 1:
        return (
            "procurement_business",
            f"Procurement query about business processes (score: {business_score}). "
            "Routing to Procurement Manual (Business Process)."
        )

    elif procurement_score >= 1:
        # General procurement query - search all procurement docs
        return (
            "procurement_all",
            f"General procurement query (score: {procurement_score}). "
            "Searching all procurement documents for comprehensive coverage."
        )

    else:
        # Fallback - likely procurement context but unclear
        return (
            "procurement_all",
            "Query appears procurement-related but no specific sub-category identified. "
            "Searching all procurement documents."
        )


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
        # Document patterns mapping for each route
        ROUTE_TO_DOCUMENTS = {
            "hr": ["HR Bylaws"],
            "infosec": ["Information Security", "Inforamation Security"],  # Handle potential typo
            "procurement_ariba": ["Procurement Manual (Ariba Aligned)"],
            "procurement_business": ["Procurement Manual (Business Process)"],
            "procurement_standards": ["Abu Dhabi Procurement Standards"],
            "procurement_all": [
                "Procurement Manual (Ariba Aligned)",
                "Procurement Manual (Business Process)",
                "Abu Dhabi Procurement Standards"
            ],
            "all": None,  # No filtering - search everything
        }

        document_patterns = None
        if route_decision:
            route = route_decision.get("route", "all")
            document_patterns = ROUTE_TO_DOCUMENTS.get(route)

            if document_patterns:
                print(f"DEBUG: Applying document filter for route '{route}': {document_patterns}")
            else:
                print(f"DEBUG: Route '{route}' - searching all documents")
        else:
            print(f"DEBUG: No routing info - searching all documents")

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
        if document_patterns:
            retrieved_nodes = []
            for node in all_nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    file_name = node.metadata.get('file_name', '')
                    file_path = node.metadata.get('file_path', '')

                    # Check if this node matches any of the document patterns
                    matches = any(
                        pattern in file_name or pattern in file_path
                        for pattern in document_patterns
                    )
                    if matches:
                        retrieved_nodes.append(node)

            # Limit to top_k after filtering
            retrieved_nodes = retrieved_nodes[:CONFIG['rag']['retrieval_top_k']]
            print(f"DEBUG: Filtered {len(all_nodes)} nodes to {len(retrieved_nodes)} matching patterns: {document_patterns}")
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
            file_name = "Unknown source"
            page_num = ""
            context_info = ""

            if hasattr(node, 'metadata') and node.metadata:
                # File information
                file_name = node.metadata.get('source_file', node.metadata.get('file_name', 'Unknown file'))
                file_path = node.metadata.get('file_path', '')
                if file_path:
                    file_name = os.path.basename(file_path)
                sources_seen.add(file_name)

                # Page number information (if available)
                page_num = node.metadata.get('page_number', node.metadata.get('page_label', ''))

                # Contextual information (if available from contextual RAG)
                context = node.metadata.get('context', '')
                if context:
                    context_info = f"**Section Context:** {context}\n"

            # Format chunk with clear, extractable source info (no "Document Chunk X")
            page_line = f"**Page:** {page_num}\n" if page_num else ""
            formatted_chunk = f"---\n**Source:** {file_name}\n{page_line}{context_info}\n{content}\n---"
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
