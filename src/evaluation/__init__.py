"""
RAG Evaluation Module using RAGAs framework.

Provides comprehensive evaluation of the Agentic RAG system including:
- Faithfulness: Are answers grounded in the context?
- Answer Relevancy: Do answers address the question?
- Context Precision: Is the retrieved context relevant?
- Context Recall: Is all necessary information retrieved?
"""

from .ragas_eval import RAGEvaluator, run_evaluation

__all__ = ["RAGEvaluator", "run_evaluation"]
