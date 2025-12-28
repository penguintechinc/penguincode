"""Documentation source definitions.

Maps libraries to their official documentation URLs.
Only libraries detected in the project will have docs fetched.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .models import Language


@dataclass
class DocSource:
    """Documentation source configuration."""
    base_url: str
    doc_format: str = "html"  # html, markdown, rst
    sitemap_url: Optional[str] = None
    api_docs_path: Optional[str] = None  # Path to API reference
    guide_path: Optional[str] = None  # Path to guides/tutorials


# Core language documentation
LANGUAGE_DOCS: Dict[Language, DocSource] = {
    Language.PYTHON: DocSource(
        base_url="https://docs.python.org/3/",
        sitemap_url="https://docs.python.org/3/sitemap.xml",
        api_docs_path="library/",
    ),
    Language.JAVASCRIPT: DocSource(
        base_url="https://developer.mozilla.org/en-US/docs/Web/JavaScript/",
        api_docs_path="Reference/",
    ),
    Language.TYPESCRIPT: DocSource(
        base_url="https://www.typescriptlang.org/docs/",
        api_docs_path="handbook/",
    ),
    Language.GO: DocSource(
        base_url="https://go.dev/doc/",
        api_docs_path="effective_go",
    ),
    Language.RUST: DocSource(
        base_url="https://doc.rust-lang.org/",
        api_docs_path="std/",
        guide_path="book/",
    ),
}

# Popular library documentation sources
# Only libraries actually used in the project will be indexed
LIBRARY_DOCS: Dict[str, DocSource] = {
    # Python libraries
    "fastapi": DocSource(
        base_url="https://fastapi.tiangolo.com/",
        api_docs_path="reference/",
        guide_path="tutorial/",
    ),
    "django": DocSource(
        base_url="https://docs.djangoproject.com/en/stable/",
        api_docs_path="ref/",
        guide_path="topics/",
    ),
    "flask": DocSource(
        base_url="https://flask.palletsprojects.com/en/latest/",
        api_docs_path="api/",
    ),
    "sqlalchemy": DocSource(
        base_url="https://docs.sqlalchemy.org/en/20/",
        api_docs_path="core/",
    ),
    "pydantic": DocSource(
        base_url="https://docs.pydantic.dev/latest/",
        api_docs_path="api/",
        guide_path="concepts/",
    ),
    "requests": DocSource(
        base_url="https://requests.readthedocs.io/en/latest/",
        api_docs_path="api/",
    ),
    "aiohttp": DocSource(
        base_url="https://docs.aiohttp.org/en/stable/",
        api_docs_path="client_reference.html",
    ),
    "pytest": DocSource(
        base_url="https://docs.pytest.org/en/stable/",
        api_docs_path="reference/",
        guide_path="how-to/",
    ),
    "numpy": DocSource(
        base_url="https://numpy.org/doc/stable/",
        api_docs_path="reference/",
        guide_path="user/",
    ),
    "pandas": DocSource(
        base_url="https://pandas.pydata.org/docs/",
        api_docs_path="reference/",
        guide_path="user_guide/",
    ),
    "httpx": DocSource(
        base_url="https://www.python-httpx.org/",
        api_docs_path="api/",
    ),
    "typer": DocSource(
        base_url="https://typer.tiangolo.com/",
        guide_path="tutorial/",
    ),
    "rich": DocSource(
        base_url="https://rich.readthedocs.io/en/stable/",
        api_docs_path="reference/",
    ),
    "click": DocSource(
        base_url="https://click.palletsprojects.com/en/stable/",
        api_docs_path="api/",
    ),
    "celery": DocSource(
        base_url="https://docs.celeryq.dev/en/stable/",
        api_docs_path="reference/",
    ),
    "redis": DocSource(
        base_url="https://redis-py.readthedocs.io/en/stable/",
    ),
    "boto3": DocSource(
        base_url="https://boto3.amazonaws.com/v1/documentation/api/latest/",
        api_docs_path="reference/services/",
    ),

    # JavaScript/TypeScript libraries
    "react": DocSource(
        base_url="https://react.dev/",
        api_docs_path="reference/",
        guide_path="learn/",
    ),
    "vue": DocSource(
        base_url="https://vuejs.org/",
        api_docs_path="api/",
        guide_path="guide/",
    ),
    "next": DocSource(
        base_url="https://nextjs.org/docs/",
        api_docs_path="app/api-reference/",
    ),
    "express": DocSource(
        base_url="https://expressjs.com/",
        api_docs_path="4x/api.html",
    ),
    "axios": DocSource(
        base_url="https://axios-http.com/docs/",
        api_docs_path="api_intro",
    ),
    "prisma": DocSource(
        base_url="https://www.prisma.io/docs/",
        api_docs_path="reference/",
    ),
    "zod": DocSource(
        base_url="https://zod.dev/",
    ),
    "tailwindcss": DocSource(
        base_url="https://tailwindcss.com/docs/",
    ),
    "vite": DocSource(
        base_url="https://vitejs.dev/",
        api_docs_path="config/",
        guide_path="guide/",
    ),
    "vitest": DocSource(
        base_url="https://vitest.dev/",
        api_docs_path="api/",
        guide_path="guide/",
    ),

    # Go libraries
    "gin": DocSource(
        base_url="https://gin-gonic.com/docs/",
    ),
    "echo": DocSource(
        base_url="https://echo.labstack.com/docs/",
    ),
    "fiber": DocSource(
        base_url="https://docs.gofiber.io/",
        api_docs_path="api/",
    ),
    "gorm": DocSource(
        base_url="https://gorm.io/docs/",
    ),

    # Rust libraries
    "tokio": DocSource(
        base_url="https://tokio.rs/tokio/",
        guide_path="tutorial/",
    ),
    "serde": DocSource(
        base_url="https://serde.rs/",
    ),
    "actix-web": DocSource(
        base_url="https://actix.rs/docs/",
    ),
    "diesel": DocSource(
        base_url="https://diesel.rs/guides/",
    ),
    "reqwest": DocSource(
        base_url="https://docs.rs/reqwest/latest/reqwest/",
    ),
}


def get_doc_source(library_name: str) -> Optional[DocSource]:
    """Get documentation source for a library."""
    # Normalize library name (lowercase, handle common variations)
    normalized = library_name.lower().replace('-', '_').replace('.', '_')

    # Direct lookup
    if normalized in LIBRARY_DOCS:
        return LIBRARY_DOCS[normalized]

    # Try without common prefixes
    for prefix in ['python_', 'py_', 'node_', 'go_', 'rust_']:
        if normalized.startswith(prefix):
            without_prefix = normalized[len(prefix):]
            if without_prefix in LIBRARY_DOCS:
                return LIBRARY_DOCS[without_prefix]

    return None


def get_language_doc_source(language: Language) -> Optional[DocSource]:
    """Get documentation source for a language."""
    return LANGUAGE_DOCS.get(language)


def get_priority_docs_for_project(
    detected_libraries: list,
    priority_list: list,
    max_count: int = 20
) -> list:
    """
    Get prioritized list of libraries to index.

    Args:
        detected_libraries: Libraries detected in the project
        priority_list: Priority library names from config
        max_count: Maximum number of libraries to return

    Returns:
        List of library names to index, prioritized
    """
    detected_names = {lib.name.lower() for lib in detected_libraries}
    priority_set = {name.lower() for name in priority_list}

    # Priority libraries that are detected
    priority_detected = [
        lib for lib in detected_libraries
        if lib.name.lower() in priority_set
    ]

    # Non-priority libraries that are detected
    other_detected = [
        lib for lib in detected_libraries
        if lib.name.lower() not in priority_set
    ]

    # Combine, respecting max count
    result = priority_detected + other_detected
    return result[:max_count]
