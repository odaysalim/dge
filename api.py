"""
FastAPI server for the Agentic RAG system.
Provides OpenAI-compatible API with conversation memory and observability.
"""

import os

# CRITICAL: Disable CrewAI telemetry BEFORE any CrewAI imports
# This prevents interactive prompts that block API responses
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import logging
from typing import List, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Arize Phoenix Tracing Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

phoenix_host = os.getenv("PHOENIX_HOST", "localhost")
phoenix_port = int(os.getenv("PHOENIX_PORT", "6006"))
phoenix_endpoint = f"http://{phoenix_host}:{phoenix_port}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name="agentic-rag",
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=True  # Auto-instrument CrewAI, LlamaIndex, and OpenAI
    )
    logging.info(f"✅ Arize Phoenix tracing initialized at {phoenix_endpoint}")
except ImportError as e:
    logging.warning(f"⚠️  Phoenix module not found: {e}")
except Exception as e:
    logging.warning(f"⚠️  Could not initialize Arize Phoenix tracing: {e}")

# Import after Phoenix setup to ensure instrumentation
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.rag_system.crew import create_rag_crew
from src.rag_system.memory import create_session_memory, get_global_memory
from src.config.settings import CONFIG, LLM_PROVIDER


def is_greeting_or_chitchat(message: str) -> bool:
    """Check if a message is a greeting or chitchat that doesn't need RAG."""
    msg_lower = message.lower().strip()
    words = msg_lower.split()

    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "greetings", "what's up", "whats up", "sup", "yo"
    ]

    chitchat_patterns = [
        "how are you", "how r u", "how do you do", "nice to meet",
        "thank you", "thanks", "bye", "goodbye", "see you", "take care",
        "who are you", "what are you", "what can you do"
    ]

    # Short greeting (1-3 words with greeting word)
    if len(words) <= 3 and any(g in msg_lower for g in greetings):
        return True

    # Chitchat patterns
    if any(pattern in msg_lower for pattern in chitchat_patterns):
        return True

    return False


def is_openwebui_system_request(message: str) -> tuple[bool, str]:
    """
    Detect OpenWebUI system requests (title generation, follow-ups, etc.)
    Returns (is_system_request, response_type)
    """
    msg_lower = message.lower()

    # Title generation request
    if "generate a concise" in msg_lower and "title" in msg_lower:
        return True, "title"

    # Follow-up suggestions request
    if "follow-up" in msg_lower and "questions" in msg_lower:
        return True, "followups"

    # Tags generation
    if "generate tags" in msg_lower or "extract tags" in msg_lower:
        return True, "tags"

    return False, ""


def get_openwebui_system_response(response_type: str, message: str) -> str:
    """Generate appropriate response for OpenWebUI system requests."""
    import json

    if response_type == "title":
        # Extract topic from chat history if present
        if "<chat_history>" in message:
            # Simple title based on context
            return json.dumps({"title": "📄 Document Query"})
        return json.dumps({"title": "💬 New Chat"})

    if response_type == "followups":
        return json.dumps({
            "follow_ups": [
                "What is the annual leave policy?",
                "How do I submit a purchase requisition?",
                "What are the password requirements?"
            ]
        })

    if response_type == "tags":
        return json.dumps({"tags": ["documents", "policies", "rag"]})

    return ""


def get_greeting_response(message: str) -> str:
    """Generate a friendly response for greetings without using RAG."""
    msg_lower = message.lower().strip()

    if any(g in msg_lower for g in ["hi", "hello", "hey", "howdy", "greetings"]):
        return "Hello! I'm your document assistant. I can help you find information about HR policies, procurement procedures, and information security guidelines. What would you like to know?"

    if "how are you" in msg_lower:
        return "I'm doing well, thank you for asking! I'm ready to help you find information in your documents. What can I look up for you?"

    if any(g in msg_lower for g in ["thank", "thanks"]):
        return "You're welcome! Let me know if you have any other questions."

    if any(g in msg_lower for g in ["bye", "goodbye", "see you"]):
        return "Goodbye! Feel free to come back anytime you need help with document queries."

    if "who are you" in msg_lower or "what are you" in msg_lower:
        return "I'm an AI document assistant powered by a RAG (Retrieval-Augmented Generation) system. I can search through HR policies, procurement manuals, and security guidelines to answer your questions with accurate source citations."

    if "what can you do" in msg_lower:
        return "I can help you find information from:\n- **HR Bylaws**: Leave policies, benefits, employee regulations\n- **Procurement Manuals**: Business processes, SAP Ariba system, Abu Dhabi standards\n- **Information Security**: Security policies and guidelines\n\nJust ask me a question about any of these topics!"

    return "Hello! How can I help you today? Feel free to ask me about HR policies, procurement procedures, or information security guidelines."

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="OpenAI-compatible API for Agentic RAG with conversation memory",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session memory storage
session_memories = {}


class Message(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model to use")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agentic RAG API",
        "version": "2.0.0",
        "status": "running",
        "llm_provider": LLM_PROVIDER,
        "features": [
            "OpenAI-compatible API",
            "Conversation memory",
            "Arize Phoenix observability",
            "Contextual retrieval",
            "Multi-agent orchestration"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    Required for OpenWebUI integration.
    Includes common model aliases so OpenWebUI can use them for title generation etc.
    """
    provider_info = f"({LLM_PROVIDER})" if LLM_PROVIDER else ""
    max_tokens = CONFIG['openai']['max_tokens'] if LLM_PROVIDER == 'openai' else CONFIG['ollama']['max_tokens']

    # Base models
    models = [
        {
            "id": "agentic-rag",
            "object": "model",
            "created": 1677652288,
            "owned_by": f"agentic-rag-{LLM_PROVIDER}",
            "permission": [],
            "root": "agentic-rag",
            "parent": None,
            "max_tokens": max_tokens,
            "description": f"Agentic RAG with CrewAI {provider_info}"
        },
        {
            "id": "agentic-rag-memory",
            "object": "model",
            "created": 1677652288,
            "owned_by": f"agentic-rag-{LLM_PROVIDER}",
            "permission": [],
            "root": "agentic-rag-memory",
            "parent": None,
            "max_tokens": max_tokens,
            "description": f"Agentic RAG with conversation memory {provider_info}"
        }
    ]

    # Add common model aliases for OpenWebUI compatibility
    # OpenWebUI uses these for title generation, tags, etc.
    common_aliases = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"]
    for alias in common_aliases:
        models.append({
            "id": alias,
            "object": "model",
            "created": 1677652288,
            "owned_by": "agentic-rag",
            "permission": [],
            "root": alias,
            "parent": None,
            "max_tokens": max_tokens,
            "description": f"Alias for Agentic RAG {provider_info}"
        })

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports conversation memory when session_id is provided.
    """
    try:
        # Extract the last user message as the query
        user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None
        )

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        logging.info(f"Received query: {user_message[:100]}...")

        # Check for OpenWebUI system requests (title generation, follow-ups, etc.)
        is_system_req, req_type = is_openwebui_system_request(user_message)
        if is_system_req:
            logging.info(f"Detected OpenWebUI system request: {req_type}")
            answer = get_openwebui_system_response(req_type, user_message)

            import time
            return {
                "id": f"chatcmpl-{uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(answer.split()),
                    "total_tokens": len(user_message.split()) + len(answer.split())
                }
            }

        # Check for greetings/chitchat - respond directly without RAG
        if is_greeting_or_chitchat(user_message):
            logging.info("Detected greeting/chitchat - responding directly")
            answer = get_greeting_response(user_message)

            import time
            return {
                "id": f"chatcmpl-{uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(answer.split()),
                    "total_tokens": len(user_message.split()) + len(answer.split())
                }
            }

        # Handle conversation memory
        use_memory = "memory" in request.model.lower() or request.session_id
        memory = None

        if use_memory:
            session_id = request.session_id or str(uuid4())
            if session_id not in session_memories:
                session_memories[session_id] = create_session_memory(session_id, persist=True)
            memory = session_memories[session_id]

            # Add user message to memory
            memory.add_message("user", user_message)

            # Get conversation context
            conversation_context = memory.get_recent_context(n_turns=3)
            if conversation_context:
                logging.info(f"Using conversation memory (session: {session_id[:8]}...)")
                # Prepend context to query
                enhanced_query = f"{conversation_context}\n\nCurrent question: {user_message}"
            else:
                enhanced_query = user_message
        else:
            enhanced_query = user_message

        # Create and run the RAG crew
        logging.info("Creating RAG crew...")
        rag_crew = create_rag_crew(enhanced_query)

        logging.info("Executing crew...")
        result = rag_crew.kickoff()

        # Convert result to string
        answer = str(result)

        # Store assistant response in memory
        if memory:
            memory.add_message("assistant", answer)

        logging.info(f"Generated response ({len(answer)} chars)")

        # Format OpenAI-compatible response
        import time
        response = {
            "id": f"chatcmpl-{uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(answer.split()),
                "total_tokens": len(user_message.split()) + len(answer.split())
            }
        }

        return response

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/sessions")
async def list_sessions():
    """List all active conversation sessions"""
    return {
        "sessions": list(session_memories.keys()),
        "total": len(session_memories)
    }


@app.get("/memory/{session_id}")
async def get_session_memory(session_id: str):
    """Get conversation history for a session"""
    if session_id not in session_memories:
        raise HTTPException(status_code=404, detail="Session not found")

    memory = session_memories[session_id]
    return {
        "session_id": session_id,
        "summary": memory.get_summary(),
        "messages": memory.messages
    }


@app.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear conversation history for a session"""
    if session_id not in session_memories:
        raise HTTPException(status_code=404, detail="Session not found")

    memory = session_memories[session_id]
    memory.clear()
    del session_memories[session_id]

    return {"message": f"Session {session_id} cleared"}


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "llm_provider": LLM_PROVIDER,
        "openai_model": CONFIG['openai']['model'] if LLM_PROVIDER == 'openai' else None,
        "ollama_model": CONFIG['ollama']['model'] if LLM_PROVIDER == 'ollama' else None,
        "embedding_dim": CONFIG['rag']['embed_dim'],
        "retrieval_top_k": CONFIG['rag']['retrieval_top_k'],
        "conversation_memory_enabled": CONFIG['memory']['enabled'],
        "context_generation_enabled": CONFIG['context_generation']['enabled'],
        "phoenix_endpoint": phoenix_endpoint
    }


if __name__ == "__main__":
    port = CONFIG.get('api', {}).get('port', 8000)
    host = CONFIG.get('api', {}).get('host', '0.0.0.0')

    logging.info(f"""
╔══════════════════════════════════════════════════════════════╗
║             🚀 Agentic RAG API Starting                      ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {host:>55} ║
║  Port: {port:>55} ║
║  LLM Provider: {LLM_PROVIDER:>47} ║
║  Phoenix Tracing: {phoenix_endpoint:>44} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=host, port=port)
