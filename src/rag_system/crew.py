import os
# Ensure telemetry is disabled before CrewAI runs
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Crew, Process, Task
from .agents import query_router, document_researcher, insight_synthesizer

def create_rag_crew(query: str):
    """
    Creates and configures a three-agent RAG crew to process a query.
    - The Query Router analyzes the query and determines which documents to search.
    - The Document Researcher finds relevant information using the routing decision.
    - The Insight Synthesizer formulates the final answer based on the retrieved context.
    """

    # Task 1: Route the query to appropriate documents
    routing_task = Task(
        description=f"Analyze this query and determine which document set(s) should be searched: '{query}'.",
        expected_output="JSON object with routing decision containing 'route' and 'justification' fields.",
        agent=query_router
    )

    # Task 2: Retrieve documents using routing information
    # This task receives the routing decision from Task 1
    research_task = Task(
        description=(
            f"You received a routing decision from the Query Router. Now you MUST:\n"
            f"1. Use the Document Retrieval Tool to search for: '{query}'\n"
            f"2. Pass the routing decision JSON from the previous task as the 'routing_info' parameter\n"
            f"3. Return ONLY the document chunks retrieved by the tool\n\n"
            f"DO NOT return the routing JSON. DO NOT skip calling the tool. "
            f"Your job is to retrieve document chunks using the tool."
        ),
        expected_output="A block of text containing chunks of the most relevant document sections and their source file names.",
        agent=document_researcher,
        context=[routing_task]  # Uses output from routing_task
    )

    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        description=f"Analyze the provided document context from the Document Researcher and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="""A professional, well-structured response that directly answers the user's question. Format the response naturally and appropriately based on the content:

Guidelines for response formatting:
- Start with a clear, direct answer to the question
- Provide supporting details, explanations, or calculations only when relevant
- Include specific references to policy articles, sections, or documents when citing sources
- Use natural language flow rather than rigid templates
- Adapt the structure to fit the content (simple answers for simple questions, detailed breakdowns for complex ones)
- Use proper formatting (bullet points, numbering, or paragraphs) as appropriate for the content
- Ensure professional tone and clarity
- Include precise figures, timeframes, and regulatory references where applicable

The response should feel conversational yet authoritative, avoiding repetitive headers unless the content genuinely requires structured breakdown.""",
        agent=insight_synthesizer,
        context=[research_task] # This ensures it uses the output from the research_task
    )

    # Create the crew with a sequential process
    # The workflow: Router → Document Researcher → Insight Synthesizer
    rag_crew = Crew(
        agents=[query_router, document_researcher, insight_synthesizer],
        tasks=[routing_task, research_task, synthesis_task],
        process=Process.sequential,  # The tasks will be executed one after the other
        verbose=False  # Disable verbose to prevent interactive prompts
    )

    return rag_crew
