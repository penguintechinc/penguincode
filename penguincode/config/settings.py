"""Configuration settings for PenguinCode."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class OllamaConfig:
    """Ollama API configuration."""

    api_url: str = "http://localhost:11434"
    timeout: int = 120


@dataclass
class ModelsConfig:
    """Global model role configuration."""

    planning: str = "deepseek-coder:6.7b"
    orchestration: str = "llama3.2:3b"
    research: str = "llama3.2:3b"
    # Execution models - use lightweight for simple tasks, full for complex
    execution: str = "qwen2.5-coder:7b"  # Complex execution (refactoring, multi-file)
    execution_lite: str = "qwen2.5-coder:1.5b"  # Lightweight execution (simple edits)
    # Exploration models
    exploration: str = "llama3.2:3b"  # Standard exploration
    exploration_lite: str = "llama3.2:1b"  # Quick file reads, simple searches


@dataclass
class AgentConfig:
    """Individual agent configuration."""

    model: str
    description: str


@dataclass
class DefaultsConfig:
    """Default generation parameters."""

    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 8192


@dataclass
class SecurityConfig:
    """Security settings."""

    level: int = 2  # 1=always prompt, 2=prompt for destructive, 3=no prompts


@dataclass
class HistoryConfig:
    """Session history configuration."""

    enabled: bool = True
    location: str = "per-project"
    max_sessions: int = 50


@dataclass
class DuckDuckGoEngineConfig:
    """DuckDuckGo search engine configuration."""

    safesearch: str = "moderate"
    region: str = "wt-wt"


@dataclass
class FireplexityEngineConfig:
    """Fireplexity search engine configuration."""

    firecrawl_api_key: str = ""


@dataclass
class SciraAIEngineConfig:
    """SciraAI search engine configuration."""

    api_key: str = ""
    endpoint: str = "https://api.scira.ai"


@dataclass
class SearXNGEngineConfig:
    """SearXNG search engine configuration."""

    url: str = "https://searx.be"
    categories: list[str] = field(default_factory=lambda: ["general"])


@dataclass
class GoogleEngineConfig:
    """Google Custom Search engine configuration."""

    api_key: str = ""
    cx_id: str = ""


@dataclass
class EnginesConfig:
    """All search engine configurations."""

    duckduckgo: DuckDuckGoEngineConfig = field(default_factory=DuckDuckGoEngineConfig)
    fireplexity: FireplexityEngineConfig = field(default_factory=FireplexityEngineConfig)
    sciraai: SciraAIEngineConfig = field(default_factory=SciraAIEngineConfig)
    searxng: SearXNGEngineConfig = field(default_factory=SearXNGEngineConfig)
    google: GoogleEngineConfig = field(default_factory=GoogleEngineConfig)


@dataclass
class ResearchConfig:
    """Research and web search configuration."""

    engine: str = "duckduckgo"  # duckduckgo | fireplexity | sciraai | searxng | google
    use_mcp: bool = False
    max_results: int = 5
    engines: EnginesConfig = field(default_factory=EnginesConfig)


@dataclass
class ChromaStoreConfig:
    """Chroma vector store configuration."""

    path: str = "./.penguincode/memory"
    collection: str = "penguincode_memory"


@dataclass
class QdrantStoreConfig:
    """Qdrant vector store configuration."""

    url: str = "http://localhost:6333"
    collection: str = "penguincode_memory"


@dataclass
class PGVectorStoreConfig:
    """PostgreSQL PGVector store configuration."""

    connection_string: str = ""
    table: str = "penguincode_memory"


@dataclass
class MemoryStoresConfig:
    """Memory vector store configurations."""

    chroma: ChromaStoreConfig = field(default_factory=ChromaStoreConfig)
    qdrant: QdrantStoreConfig = field(default_factory=QdrantStoreConfig)
    pgvector: PGVectorStoreConfig = field(default_factory=PGVectorStoreConfig)


@dataclass
class MemoryConfig:
    """mem0 memory layer configuration."""

    enabled: bool = True
    vector_store: str = "chroma"  # chroma | qdrant | pgvector
    embedding_model: str = "nomic-embed-text"
    stores: MemoryStoresConfig = field(default_factory=MemoryStoresConfig)


@dataclass
class RegulatorsConfig:
    """GPU rate limiting and agent concurrency configuration."""

    auto_detect: bool = True
    gpu_type: str = "auto"
    gpu_model: str = ""
    vram_mb: int = 8192
    max_concurrent_requests: int = 2
    max_models_loaded: int = 1
    request_queue_size: int = 10
    min_request_interval_ms: int = 100
    cooldown_after_error_ms: int = 1000
    # Agent concurrency settings
    max_concurrent_agents: int = 5  # Max agents running in parallel
    agent_timeout_seconds: int = 300  # Timeout for individual agent tasks


@dataclass
class UsageAPIConfig:
    """Hosted Ollama usage API configuration."""

    enabled: bool = False
    endpoint: str = "https://ollama.example.com/api/usage"
    jwt_token: str = ""
    refresh_interval: int = 300
    show_warnings_at: int = 80


@dataclass
class DocsRagConfig:
    """Documentation RAG configuration.

    Controls automatic indexing of documentation for detected
    languages and libraries. Only indexes docs for libraries
    actually used in the project to avoid bloat.
    """

    enabled: bool = True
    cache_dir: str = "./.penguincode/docs"
    collection: str = "penguincode_docs"
    # Limits to prevent bloat
    max_pages_per_library: int = 50
    max_libraries_to_index: int = 20  # Only index top N libraries
    cache_max_age_days: int = 7
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Context injection limits
    max_context_tokens: int = 2000
    max_chunks_per_query: int = 5
    # Behavior settings
    auto_detect_on_start: bool = True
    auto_detect_on_request: bool = True  # Detect languages from request content
    auto_index_on_detect: bool = False  # Require explicit /docs index
    auto_index_on_request: bool = True  # Index on-demand when docs needed
    # Manual language configuration (dict of language -> bool)
    languages_manual: dict = field(default_factory=lambda: {
        "python": False,
        "javascript": False,
        "typescript": False,
        "go": False,
        "rust": False,
        "hcl": False,      # Terraform/OpenTofu
        "ansible": False,
    })
    # User-specified libraries to always index (e.g., ["fastapi", "pytest"])
    libraries_manual: list = field(default_factory=list)
    # Library priority - index these first if detected
    priority_libraries: list = field(default_factory=lambda: [
        # Python
        "fastapi", "django", "flask", "sqlalchemy", "pydantic",
        "requests", "aiohttp", "pytest", "numpy", "pandas",
        # JavaScript/TypeScript
        "react", "vue", "next", "express", "axios", "prisma",
        # Go
        "gin", "echo", "fiber", "gorm",
        # Rust
        "tokio", "serde", "actix-web", "diesel",
    ])


@dataclass
class Settings:
    """Main settings configuration."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    regulators: RegulatorsConfig = field(default_factory=RegulatorsConfig)
    usage_api: UsageAPIConfig = field(default_factory=UsageAPIConfig)
    docs_rag: DocsRagConfig = field(default_factory=DocsRagConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from YAML file with environment variable expansion."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Expand environment variables
        data = cls._expand_env_vars(data)

        # Parse nested configurations
        return cls(
            ollama=OllamaConfig(**data.get("ollama", {})),
            models=ModelsConfig(**data.get("models", {})),
            agents={
                name: AgentConfig(**config) for name, config in data.get("agents", {}).items()
            },
            defaults=DefaultsConfig(**data.get("defaults", {})),
            security=SecurityConfig(**data.get("security", {})),
            history=HistoryConfig(**data.get("history", {})),
            research=cls._parse_research_config(data.get("research", {})),
            memory=cls._parse_memory_config(data.get("memory", {})),
            regulators=RegulatorsConfig(**data.get("regulators", {})),
            usage_api=UsageAPIConfig(**data.get("usage_api", {})),
            docs_rag=cls._parse_docs_rag_config(data.get("docs_rag", {})),
        )

    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(data, dict):
            return {k: Settings._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Settings._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.environ.get(env_var, "")
        return data

    @staticmethod
    def _parse_research_config(data: Dict[str, Any]) -> ResearchConfig:
        """Parse research configuration with nested engines."""
        engines_data = data.get("engines", {})
        engines = EnginesConfig(
            duckduckgo=DuckDuckGoEngineConfig(**engines_data.get("duckduckgo", {})),
            fireplexity=FireplexityEngineConfig(**engines_data.get("fireplexity", {})),
            sciraai=SciraAIEngineConfig(**engines_data.get("sciraai", {})),
            searxng=SearXNGEngineConfig(**engines_data.get("searxng", {})),
            google=GoogleEngineConfig(**engines_data.get("google", {})),
        )
        return ResearchConfig(
            engine=data.get("engine", "duckduckgo"),
            use_mcp=data.get("use_mcp", False),
            max_results=data.get("max_results", 5),
            engines=engines,
        )

    @staticmethod
    def _parse_memory_config(data: Dict[str, Any]) -> MemoryConfig:
        """Parse memory configuration with nested stores."""
        stores_data = data.get("stores", {})
        stores = MemoryStoresConfig(
            chroma=ChromaStoreConfig(**stores_data.get("chroma", {})),
            qdrant=QdrantStoreConfig(**stores_data.get("qdrant", {})),
            pgvector=PGVectorStoreConfig(**stores_data.get("pgvector", {})),
        )
        return MemoryConfig(
            enabled=data.get("enabled", True),
            vector_store=data.get("vector_store", "chroma"),
            embedding_model=data.get("embedding_model", "nomic-embed-text"),
            stores=stores,
        )

    @staticmethod
    def _parse_docs_rag_config(data: Dict[str, Any]) -> DocsRagConfig:
        """Parse documentation RAG configuration."""
        # Parse languages_manual dict
        default_langs = DocsRagConfig().languages_manual
        languages_manual = data.get("languages_manual", default_langs)
        if not isinstance(languages_manual, dict):
            languages_manual = default_langs

        return DocsRagConfig(
            enabled=data.get("enabled", True),
            cache_dir=data.get("cache_dir", "./.penguincode/docs"),
            collection=data.get("collection", "penguincode_docs"),
            max_pages_per_library=data.get("max_pages_per_library", 50),
            max_libraries_to_index=data.get("max_libraries_to_index", 20),
            cache_max_age_days=data.get("cache_max_age_days", 7),
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 200),
            max_context_tokens=data.get("max_context_tokens", 2000),
            max_chunks_per_query=data.get("max_chunks_per_query", 5),
            auto_detect_on_start=data.get("auto_detect_on_start", True),
            auto_detect_on_request=data.get("auto_detect_on_request", True),
            auto_index_on_detect=data.get("auto_index_on_detect", False),
            auto_index_on_request=data.get("auto_index_on_request", True),
            languages_manual=languages_manual,
            libraries_manual=data.get("libraries_manual", []),
            priority_libraries=data.get("priority_libraries", DocsRagConfig().priority_libraries),
        )


def get_research_engine(settings: Settings) -> str:
    """Get the configured research engine name."""
    return settings.research.engine


def get_memory_config(settings: Settings) -> MemoryConfig:
    """Get the memory configuration."""
    return settings.memory


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load settings from configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return Settings.from_yaml(config_path)
