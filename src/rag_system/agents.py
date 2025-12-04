"""
Agentic RAG Agents using CrewAI.
Supports both OpenAI and Ollama providers.
"""

import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool
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

# --- AGENT 1: Document Researcher ---
# This agent's sole purpose is to retrieve relevant information using the retrieval tool
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
        "You are an information retrieval specialist. Your role is strictly limited to:\n"
        "1) Analyze the user's query to understand intent\n"
        "2) Retrieve relevant text chunks using the Document Retrieval Tool\n"
        "3) Return only the raw retrieved context - no interpretation or answers\n\n"
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
    goal='Create clear, professional responses that directly answer user questions based on the provided context.',
    backstory=(
        "You are an expert analyst who specializes in creating natural, professional responses.\n"
        "You receive context from a document researcher and must craft responses that feel conversational yet authoritative.\n\n"

        "CORE PRINCIPLES:\n"
        "- Answer questions directly and naturally, like a knowledgeable colleague would\n"
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
        "- Maintain professional tone while being conversational"
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    tools=[]  # This agent doesn't need tools; it only processes text
)
