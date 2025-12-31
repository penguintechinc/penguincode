"""Documentation RAG system for PenguinCode.

Provides automatic detection of project languages and libraries,
fetching of relevant documentation, and context injection for
improved code assistance.

Key features:
- Only indexes docs for languages/libraries actually used
- TTL-based cache expiration
- Automatic cleanup of unused library docs
- Token-aware context injection
"""

from .models import (
    DocChunk,
    DocSearchResult,
    Language,
    Library,
    ProjectContext,
)
from .detector import ProjectDetector
from .sources import (
    DocSource,
    LANGUAGE_DOCS,
    LIBRARY_DOCS,
    get_doc_source,
    get_language_doc_source,
    get_priority_docs_for_project,
)
from .fetcher import DocumentationFetcher, CacheEntry
from .indexer import DocumentationIndexer
from .injector import ContextInjector

__all__ = [
    # Models
    "DocChunk",
    "DocSearchResult",
    "Language",
    "Library",
    "ProjectContext",
    # Detection
    "ProjectDetector",
    # Sources
    "DocSource",
    "LANGUAGE_DOCS",
    "LIBRARY_DOCS",
    "get_doc_source",
    "get_language_doc_source",
    "get_priority_docs_for_project",
    # Fetching
    "DocumentationFetcher",
    "CacheEntry",
    # Indexing
    "DocumentationIndexer",
    # Injection
    "ContextInjector",
]
