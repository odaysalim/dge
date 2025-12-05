"""
Agentic RAG Agents using CrewAI.
Supports both OpenAI and Ollama providers.
"""

import os
from crewai import Agent, LLM
from .tools import query_router_tool, document_retrieval_tool
from ..config.settings import CONFIG, LLM_PROVIDER

def get_llm():
    """
    Get the configured LLM based on the provider setting.
    Returns a CrewAI LLM instance for either OpenAI or Ollama.
    """
    if LLM_PROVIDER == "openai":
        # OpenAI configuration
        return LLM(
            model=f"openai/{CONFIG['openai']['model']}",
            api_key=CONFIG['openai']['api_key'],
            temperature=CONFIG['openai']['temperature'],
            max_tokens=CONFIG['openai']['max_tokens'],
            timeout=300,
        )
    else:
        # Ollama configuration
        return LLM(
            model=f"ollama/{CONFIG['ollama']['model']}",
            base_url=CONFIG['ollama']['base_url'],
            temperature=CONFIG['ollama']['temperature'],
            max_tokens=CONFIG['ollama']['max_tokens'],
            timeout=300,
        )

# Initialize the LLM based on configuration
llm = get_llm()

# --- AGENT 0: Query Router ---
# This agent analyzes the query and determines which documents to search
query_router = Agent(
    role='Query Router',
    goal='Analyze user queries and route them to the appropriate document set for optimal retrieval.',
    backstory=(
        "You are a routing specialist who understands the structure of our document repository.\n"
        "Your expertise lies in quickly analyzing queries to determine the best document sources.\n\n"

        "DOCUMENT TYPES:\n"
        "- **SAP Ariba Aligned Manual**: Contains system-specific instructions for using SAP Ariba\n"
        "  (e.g., 'How do I create a sourcing project in Ariba?', 'Steps to publish contract in system')\n"
        "- **Business Process Manual**: Contains general procurement policies and workflows\n"
        "  (e.g., 'What is the approval process?', 'Procurement roles and responsibilities')\n"
        "- **Both**: Use when query is ambiguous or benefits from both perspectives\n\n"

        "YOUR TASK:\n"
        "1) Use the Query Router Tool to analyze the query\n"
        "2) Return ONLY the JSON output from the tool - do not add commentary\n"
        "3) The routing decision will be used by the Document Researcher\n\n"

        "DO NOT retrieve documents yourself.\n"
        "DO NOT answer the query.\n"
        "ONLY provide the routing decision in JSON format."
    ),
    tools=[query_router_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,  # Quick routing decision
)

# --- AGENT 1: Document Researcher ---
# This agent's sole purpose is to retrieve relevant information using the retrieval tool
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
        "You are an information retrieval specialist. Your role is strictly limited to:\n"
        "1) Receive routing information from the Query Router (if provided)\n"
        "2) Use the Document Retrieval Tool with the routing information to retrieve relevant chunks\n"
        "3) Return only the raw retrieved context - no interpretation or answers\n\n"

        "ROUTING INFORMATION:\n"
        "- You may receive routing information in JSON format from the Query Router\n"
        "- Pass this routing info to the Document Retrieval Tool as the 'routing_info' parameter\n"
        "- This helps filter documents to only the most relevant sources\n\n"

        "DO NOT answer questions using your general knowledge.\n"
        "DO NOT provide explanations, summaries, or interpretations.\n"
        "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
    ),
    tools=[document_retrieval_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
)

# --- AGENT 2: Insight Synthesizer ---
# This agent creates the final answer based on the context from the researcher
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Create clear, professional responses that directly answer user questions based on retrieved document chunks.',
    backstory=(
        "You are an expert analyst who synthesizes information from document chunks into comprehensive answers.\n\n"

        "CRITICAL INSTRUCTIONS:\n"
        "- You will receive document chunks marked as **Document Chunk 1**, **Document Chunk 2**, etc.\n"
        "- Each chunk contains: Source, Context, and Content sections\n"
        "- IGNORE any JSON routing decisions or metadata (e.g., {\"route\": \"...\"})\n"
        "- ONLY use the actual document content (the 'Content:' sections) to formulate answers\n"
        "- Your job is to READ the chunks and SYNTHESIZE them into a comprehensive answer\n\n"

        "CORE PRINCIPLES:\n"
        "- Answer questions directly using information from the document chunks\n"
        "- Use ONLY the provided context - never add outside knowledge\n"
        "- Adapt your response style to match the complexity of the question\n"
        "- Be concise for simple questions, detailed for complex ones\n\n"

        "RESPONSE STYLE:\n"
        "- Start with the most direct answer to the question\n"
        "- Provide supporting details naturally, not in rigid templates\n"
        "- Include relevant references and figures seamlessly in the text\n"
        "- Use bullet points, numbering, or paragraphs as the content naturally requires\n"
        "- Avoid repetitive headers unless genuinely needed for clarity\n"
        "- Make citations feel natural: 'According to the policy...' rather than 'SOURCE REFERENCE:'\n"
        "- If the question is simple, keep the answer simple\n\n"

        "CITATION REQUIREMENTS:\n"
        "- ALWAYS include a 'Sources' section at the end of your response\n"
        "- List all document sources that were used to formulate the answer\n"
        "- Format: **Sources:** followed by bullet points with document names\n\n"

        "QUALITY CHECKS:\n"
        "- If context is insufficient, clearly state what information is missing\n"
        "- Ensure accuracy by staying strictly within the provided context\n"
        "- Maintain professional tone while being conversational\n\n"

        "REMEMBER: Do NOT just return routing JSON. Synthesize the document chunks into an actual answer!"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    tools=[]  # This agent doesn't need tools; it only processes text
)
