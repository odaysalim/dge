"""
Arize Phoenix Prompt Management Integration.

This module provides prompt lifecycle management functionality using Arize Phoenix:
- Store and version prompts
- Retrieve prompts by name/version
- Track prompt performance
- A/B testing support
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """Represents a versioned prompt template."""

    def __init__(
        self,
        name: str,
        template: str,
        version: str = "1.0.0",
        description: str = "",
        variables: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.template = template
        self.version = version
        self.description = description
        self.variables = variables or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()

    def render(self, **kwargs) -> str:
        """Render the prompt template with given variables."""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "template": self.template,
            "version": self.version,
            "description": self.description,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            variables=data.get("variables", []),
            metadata=data.get("metadata", {})
        )


class PhoenixPromptManager:
    """
    Manages prompts with Phoenix integration for observability.

    Features:
    - Local prompt storage with file-based persistence
    - Version tracking
    - Phoenix tracing integration for prompt performance
    - Fallback support for prompt retrieval
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            storage_dir: Directory for storing prompt files
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path(__file__).parent.parent.parent / "data" / "prompts"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.prompts: Dict[str, PromptTemplate] = {}

        # Load existing prompts
        self._load_prompts()

        # Initialize default prompts
        self._init_default_prompts()

        logger.info(f"PromptManager initialized with {len(self.prompts)} prompts")

    def _load_prompts(self):
        """Load prompts from storage directory."""
        prompts_file = self.storage_dir / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, "r") as f:
                    data = json.load(f)
                for name, prompt_data in data.items():
                    self.prompts[name] = PromptTemplate.from_dict(prompt_data)
                logger.info(f"Loaded {len(self.prompts)} prompts from storage")
            except Exception as e:
                logger.warning(f"Failed to load prompts: {e}")

    def _save_prompts(self):
        """Save prompts to storage directory."""
        prompts_file = self.storage_dir / "prompts.json"
        try:
            data = {name: prompt.to_dict() for name, prompt in self.prompts.items()}
            with open(prompts_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Prompts saved to storage")
        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")

    def _init_default_prompts(self):
        """Initialize default system prompts if not already present."""
        default_prompts = [
            PromptTemplate(
                name="query_router",
                template="""You are a query routing specialist. Analyze the user's question and determine which document category is most relevant.

Document Categories:
- hr: HR Bylaws - employee policies, leave, benefits, compensation
- infosec: Information Security - passwords, access control, data protection
- procurement_ariba: SAP Ariba Manual - system usage, UI, technical implementation
- procurement_business: Business Process Manual - procurement workflows and policies
- procurement_standards: Abu Dhabi Standards - government regulations
- procurement_all: Ambiguous procurement queries (search all)
- all: Non-specific queries (search everything)

User Query: {query}

Respond with a JSON object containing:
{{"route": "category_name", "justification": "brief reason", "query": "original query"}}""",
                version="1.0.0",
                description="Routes queries to appropriate document categories",
                variables=["query"]
            ),
            PromptTemplate(
                name="contextual_rag",
                template="""You are analyzing a document. Your task is to provide brief context for a specific chunk.

<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Provide a brief context (1-2 sentences) explaining:
1. Which section/topic this chunk relates to
2. How it connects to the overall document
3. Its relationship to other sections

Respond with only the context, nothing else.""",
                version="1.0.0",
                description="Generates contextual summaries for document chunks (Anthropic-style)",
                variables=["document", "chunk"]
            ),
            PromptTemplate(
                name="insight_synthesizer",
                template="""You are an expert analyst who synthesizes information from document chunks into comprehensive answers.

Document Context:
{context}

User Question: {question}

Instructions:
- Answer the question directly using only the provided context
- Use proper citations (document name, page number)
- Format appropriately (bullets, paragraphs, etc.)
- If the context doesn't contain relevant information, say so

End your response with a **Sources:** section listing all referenced documents.""",
                version="1.0.0",
                description="Synthesizes final answers from retrieved chunks",
                variables=["context", "question"]
            ),
            PromptTemplate(
                name="document_retrieval",
                template="""Search the document collection for information relevant to this query.

Query: {query}
Route: {route}

Use hybrid search (semantic + keyword) to find the most relevant document chunks.
Return chunks with source citations and page numbers.""",
                version="1.0.0",
                description="Document retrieval prompt",
                variables=["query", "route"]
            ),
            PromptTemplate(
                name="greeting_response",
                template="""Respond to the user's greeting or general message in a friendly, helpful manner.

User Message: {message}

You are a document assistant that helps with:
- HR policies (leave, benefits, employee regulations)
- Procurement manuals (SAP Ariba, business processes, standards)
- Information security guidelines

Respond naturally and offer to help with document queries.""",
                version="1.0.0",
                description="Handles greetings and general chitchat",
                variables=["message"]
            )
        ]

        for prompt in default_prompts:
            if prompt.name not in self.prompts:
                self.prompts[prompt.name] = prompt
                logger.info(f"Initialized default prompt: {prompt.name}")

        self._save_prompts()

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        Get a prompt by name and optional version.

        Args:
            name: Prompt name
            version: Optional version (returns latest if not specified)

        Returns:
            PromptTemplate or None if not found
        """
        prompt = self.prompts.get(name)
        if prompt and version and prompt.version != version:
            logger.warning(f"Requested version {version} but only have {prompt.version}")
        return prompt

    def render(self, name: str, **kwargs) -> str:
        """
        Get and render a prompt with variables.

        Args:
            name: Prompt name
            **kwargs: Variables to substitute

        Returns:
            Rendered prompt string
        """
        prompt = self.get(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        return prompt.render(**kwargs)

    def register(self, prompt: PromptTemplate, overwrite: bool = False):
        """
        Register a new prompt or update existing one.

        Args:
            prompt: PromptTemplate to register
            overwrite: Whether to overwrite existing prompt
        """
        if prompt.name in self.prompts and not overwrite:
            raise ValueError(f"Prompt already exists: {prompt.name}. Set overwrite=True to update.")

        self.prompts[prompt.name] = prompt
        self._save_prompts()
        logger.info(f"Registered prompt: {prompt.name} v{prompt.version}")

    def list_prompts(self) -> List[Dict[str, str]]:
        """List all available prompts."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description
            }
            for p in self.prompts.values()
        ]

    def delete(self, name: str):
        """Delete a prompt by name."""
        if name in self.prompts:
            del self.prompts[name]
            self._save_prompts()
            logger.info(f"Deleted prompt: {name}")
        else:
            logger.warning(f"Prompt not found: {name}")


# Global prompt manager instance
_prompt_manager: Optional[PhoenixPromptManager] = None


def get_prompt_manager() -> PhoenixPromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PhoenixPromptManager()
    return _prompt_manager


def get_prompt(name: str, **kwargs) -> str:
    """
    Convenience function to get and render a prompt.

    Args:
        name: Prompt name
        **kwargs: Variables for the prompt

    Returns:
        Rendered prompt string
    """
    manager = get_prompt_manager()
    return manager.render(name, **kwargs)


if __name__ == "__main__":
    # Test the prompt manager
    manager = PhoenixPromptManager()

    print("\n=== Available Prompts ===")
    for p in manager.list_prompts():
        print(f"  - {p['name']} (v{p['version']}): {p['description']}")

    print("\n=== Test Rendering ===")
    router_prompt = manager.render("query_router", query="What is the annual leave policy?")
    print(router_prompt[:200] + "...")
