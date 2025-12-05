"""
Data Ingestion Module for Agentic RAG System.
Uses Docling for document processing and Anthropic-style contextual RAG.
"""

from .ingest_docling import DocumentIngester, ingest_documents

__all__ = ["DocumentIngester", "ingest_documents"]
