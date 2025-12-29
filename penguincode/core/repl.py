"""Interactive REPL loop for PenguinCode chat."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from rich.prompt import Prompt
from rich.table import Table

from penguincode.config.settings import Settings, load_settings
from penguincode.ollama import OllamaClient
from penguincode.ui import console, print_error, print_info, print_success

from .session import Session, SessionManager

# Lazy imports to avoid circular dependency
if TYPE_CHECKING:
    from penguincode.agents import ChatAgent, ExecutorAgent, ExplorerAgent


class REPLSession:
    """Interactive REPL session with agentic chat loop."""

    def __init__(self, project_dir: str = ".", config_path: str = "config.yaml"):
        """
        Initialize REPL session.

        Args:
            project_dir: Project directory
            config_path: Path to config.yaml
        """
        self.project_dir = Path(project_dir).resolve()
        self.config_path = config_path

        # Load settings
        try:
            self.settings = load_settings(config_path)
        except FileNotFoundError:
            print_error(f"Config file not found: {config_path}")
            print_info("Using default configuration")
            self.settings = Settings()

        # Initialize session manager
        self.session_manager = SessionManager(str(self.project_dir))
        self.session = self.session_manager.create_session()

        # Ollama client (will be initialized in async context)
        self.ollama_client: Optional[OllamaClient] = None

        # Chat agent (main orchestrator) and specialized agents
        self.chat_agent: Optional[ChatAgent] = None
        self.agents = {}

        # Docs RAG components (initialized if enabled)
        self.project_context = None
        self.docs_fetcher = None
        self.docs_indexer = None
        self.context_injector = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Lazy import agents to avoid circular import
        from penguincode.agents import ChatAgent, ExecutorAgent, ExplorerAgent

        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            base_url=self.settings.ollama.api_url,
            timeout=self.settings.ollama.timeout,
        )
        await self.ollama_client.__aenter__()

        # Initialize chat agent (main orchestrator)
        self.chat_agent = ChatAgent(
            ollama_client=self.ollama_client,
            settings=self.settings,
            project_dir=str(self.project_dir),
        )

        # Keep direct agent references for manual commands (/explore, /execute)
        explorer_model = self.settings.models.orchestration
        executor_model = self.settings.models.execution

        self.agents["executor"] = ExecutorAgent(
            ollama_client=self.ollama_client,
            working_dir=str(self.project_dir),
            model=executor_model,
        )
        self.agents["explorer"] = ExplorerAgent(
            ollama_client=self.ollama_client,
            working_dir=str(self.project_dir),
            model=explorer_model,
        )

        # Initialize docs RAG if enabled
        if self.settings.docs_rag.enabled:
            await self._init_docs_rag()

        return self

    async def _init_docs_rag(self) -> None:
        """Initialize documentation RAG system."""
        try:
            from penguincode.docs_rag import (
                ProjectDetector,
                DocumentationFetcher,
                DocumentationIndexer,
                ContextInjector,
                Language,
                ProjectContext,
            )

            # Start with manual languages from config
            manual_languages = []
            for lang_name, enabled in self.settings.docs_rag.languages_manual.items():
                if enabled:
                    try:
                        manual_languages.append(Language(lang_name.lower()))
                    except ValueError:
                        print_error(f"Unknown language in config: {lang_name}")

            # Auto-detect project languages if enabled
            if self.settings.docs_rag.auto_detect_on_start:
                detector = ProjectDetector(str(self.project_dir))
                self.project_context = detector.detect()

                # Merge manual languages with detected ones
                for lang in manual_languages:
                    if lang not in self.project_context.languages:
                        self.project_context.languages.append(lang)

                if self.project_context.languages:
                    langs = ", ".join(self.project_context.language_names)
                    libs_count = len(self.project_context.libraries)
                    print_info(f"Detected: {langs} ({libs_count} libraries)")
            else:
                # Use only manual languages
                self.project_context = ProjectContext(languages=manual_languages)

            # Initialize fetcher and indexer
            self.docs_fetcher = DocumentationFetcher(
                cache_dir=self.settings.docs_rag.cache_dir,
                max_pages_per_library=self.settings.docs_rag.max_pages_per_library,
                cache_max_age_days=self.settings.docs_rag.cache_max_age_days,
            )

            self.docs_indexer = DocumentationIndexer(
                collection_name=self.settings.docs_rag.collection,
                embedding_model=self.settings.memory.embedding_model,
                chunk_size=self.settings.docs_rag.chunk_size,
                chunk_overlap=self.settings.docs_rag.chunk_overlap,
                ollama_base_url=self.settings.ollama.api_url,
            )

            self.context_injector = ContextInjector(
                indexer=self.docs_indexer,
                max_context_tokens=self.settings.docs_rag.max_context_tokens,
                max_chunks=self.settings.docs_rag.max_chunks_per_query,
            )

            # Cleanup expired cache entries
            expired = self.docs_fetcher.expunge_expired()
            if expired > 0:
                print_info(f"Cleaned up {expired} expired doc cache entries")

            # Cleanup unused library docs
            if self.project_context:
                removed = self.docs_fetcher.cleanup_unused_libraries(
                    self.project_context.libraries
                )
                if removed:
                    print_info(f"Removed docs for unused libraries: {', '.join(removed.keys())}")

            # Auto-index on detect if enabled
            if self.settings.docs_rag.auto_index_on_detect and self.project_context:
                await self._auto_index_languages()

        except ImportError as e:
            print_info(f"Docs RAG not available: {e}")
        except Exception as e:
            print_error(f"Docs RAG init failed: {e}")

    async def _auto_index_languages(self) -> None:
        """Auto-index documentation for detected/configured languages."""
        if not self.project_context or not self.docs_fetcher or not self.docs_indexer:
            return

        from penguincode.docs_rag import get_language_doc_source

        indexed_count = 0
        for lang in self.project_context.languages:
            # Check if already indexed (fresh)
            if self.docs_indexer.is_language_indexed(lang.value):
                continue

            # Get doc source for language
            doc_source = get_language_doc_source(lang)
            if not doc_source:
                continue

            console.print(f"[dim]Indexing {lang.value} documentation...[/dim]")

            try:
                # Fetch language docs
                docs = await self.docs_fetcher.fetch_language_docs(lang)
                if docs:
                    chunks = await self.docs_indexer.index_language(lang, docs)
                    indexed_count += chunks
                    console.print(f"[dim]  Indexed {chunks} chunks for {lang.value}[/dim]")
            except Exception as e:
                console.print(f"[dim]  Failed to index {lang.value}: {e}[/dim]")

        if indexed_count > 0:
            print_info(f"Auto-indexed {indexed_count} documentation chunks")

    async def _ensure_language_indexed(self, language: str) -> bool:
        """Ensure a language's documentation is indexed (on-demand).

        Args:
            language: Language name (e.g., "python", "javascript")

        Returns:
            True if indexed successfully or already indexed
        """
        if not self.docs_fetcher or not self.docs_indexer:
            return False

        from penguincode.docs_rag import Language, get_language_doc_source

        # Check if already indexed
        if self.docs_indexer.is_language_indexed(language):
            return True

        # Get Language enum
        try:
            lang_enum = Language(language.lower())
        except ValueError:
            return False

        # Get doc source
        doc_source = get_language_doc_source(lang_enum)
        if not doc_source:
            return False

        console.print(f"[dim]Indexing {language} documentation on-demand...[/dim]")

        try:
            docs = await self.docs_fetcher.fetch_language_docs(lang_enum)
            if docs:
                chunks = await self.docs_indexer.index_language(lang_enum, docs)
                console.print(f"[dim]  Indexed {chunks} chunks[/dim]")
                return True
        except Exception as e:
            console.print(f"[dim]  Failed: {e}[/dim]")

        return False

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Save session
        self.session_manager.save_session(self.session)

        # Close Ollama client
        if self.ollama_client:
            await self.ollama_client.__aexit__(exc_type, exc_val, exc_tb)

    async def handle_command(self, command: str) -> bool:
        """
        Handle REPL commands.

        Args:
            command: Command string

        Returns:
            True to continue REPL, False to exit
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self.show_help()
        elif cmd == "/exit" or cmd == "/quit":
            return False
        elif cmd == "/clear":
            console.clear()
            if self.chat_agent:
                self.chat_agent.reset_conversation()
            print_info("Screen and conversation cleared")
        elif cmd == "/history":
            self.show_history()
        elif cmd == "/agents":
            self.show_agents()
        elif cmd == "/read":
            await self.handle_read(args)
        elif cmd == "/explore":
            await self.handle_explore(args)
        elif cmd == "/execute":
            await self.handle_execute(args)
        elif cmd == "/reset":
            if self.chat_agent:
                self.chat_agent.reset_conversation()
            print_info("Conversation reset")
        elif cmd == "/docs":
            await self.handle_docs_command(args)
        else:
            print_error(f"Unknown command: {cmd}")
            print_info("Type /help for available commands")

        return True

    def show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold cyan]PenguinCode Commands:[/bold cyan]

[yellow]General:[/yellow]
  /help              Show this help message
  /exit, /quit       Exit PenguinCode
  /clear             Clear screen and reset conversation
  /reset             Reset conversation history
  /history           Show conversation history
  /agents            List available agents

[yellow]Agent Commands:[/yellow]
  /explore <query>   Explore codebase (read-only)
  /execute <task>    Execute code changes
  /read <path>       Read a file

[yellow]Documentation RAG:[/yellow]
  /docs status       Show detection and index status
  /docs detect       Re-run project detection
  /docs index [lib]  Index documentation (all or specific library)
  /docs search <q>   Search indexed documentation
  /docs clear [lib]  Clear index (all or specific library)
  /docs cleanup      Remove docs for unused libraries

[yellow]Chat:[/yellow]
  Just type your message to chat with the orchestrator.
  The orchestrator will automatically delegate to the right agent.

[yellow]Examples:[/yellow]
  > Find all Python files         (uses explorer)
  > What does main.py do?         (uses explorer)
  > Create a new file hello.py    (uses executor)
  > Fix the bug in auth.py        (uses executor)
  > Run the tests                 (uses executor)
"""
        console.print(help_text)

    def show_history(self) -> None:
        """Show conversation history."""
        if not self.session.messages:
            print_info("No messages in this session")
            return

        console.print("\n[bold cyan]Session History:[/bold cyan]\n")
        for msg in self.session.messages:
            role_color = "green" if msg.role == "user" else "blue"
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            console.print(f"[{role_color}]{msg.role}:[/{role_color}] {content}\n")

    def show_agents(self) -> None:
        """Show available agents."""
        console.print("\n[bold cyan]Available Agents:[/bold cyan]\n")
        for name, agent in self.agents.items():
            console.print(
                f"  [green]{name}[/green]: {agent.config.description} "
                f"[dim](model: {agent.config.model})[/dim]"
            )
        console.print()

    async def handle_read(self, path: str) -> None:
        """Handle /read command."""
        if not path:
            print_error("Usage: /read <path>")
            return

        explorer = self.agents["explorer"]
        result = await explorer.execute_tool("read", path=path)

        if result.success:
            console.print(result.data)
        else:
            print_error(result.error or "Failed to read file")

    async def handle_explore(self, query: str) -> None:
        """Handle /explore command."""
        if not query:
            print_error("Usage: /explore <query>")
            return

        console.print(f"\n[cyan]Exploring:[/cyan] {query}\n")

        explorer = self.agents["explorer"]
        result = await explorer.run(query)

        if result.success:
            console.print(result.output)
        else:
            print_error(result.error or "Exploration failed")

    async def handle_execute(self, task: str) -> None:
        """Handle /execute command."""
        if not task:
            print_error("Usage: /execute <task>")
            return

        console.print(f"\n[cyan]Executing:[/cyan] {task}\n")

        executor = self.agents["executor"]
        result = await executor.run(task)

        if result.success:
            console.print(result.output)
            print_success("Task completed")
        else:
            print_error(result.error or "Execution failed")

    async def handle_docs_command(self, args: str) -> None:
        """Handle /docs subcommands."""
        if not self.settings.docs_rag.enabled:
            print_error("Docs RAG is disabled in config")
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "status"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "status":
            await self._docs_status()
        elif subcmd == "detect":
            await self._docs_detect()
        elif subcmd == "index":
            await self._docs_index(subargs)
        elif subcmd == "search":
            await self._docs_search(subargs)
        elif subcmd == "clear":
            await self._docs_clear(subargs)
        elif subcmd == "cleanup":
            await self._docs_cleanup()
        else:
            print_error(f"Unknown docs command: {subcmd}")
            print_info("Use: /docs status|detect|index|search|clear|cleanup")

    async def _docs_status(self) -> None:
        """Show docs RAG status."""
        console.print("\n[bold cyan]Documentation RAG Status[/bold cyan]\n")

        # Project detection
        if self.project_context:
            console.print("[yellow]Detected Languages:[/yellow]")
            for lang in self.project_context.languages:
                console.print(f"  - {lang.value}")

            console.print(f"\n[yellow]Detected Libraries ({len(self.project_context.libraries)}):[/yellow]")
            # Group by language
            by_lang = {}
            for lib in self.project_context.libraries[:20]:  # Show first 20
                lang = lib.language.value
                if lang not in by_lang:
                    by_lang[lang] = []
                by_lang[lang].append(lib.name)

            for lang, libs in by_lang.items():
                console.print(f"  [{lang}] {', '.join(libs[:10])}")
                if len(libs) > 10:
                    console.print(f"        ... and {len(libs) - 10} more")
        else:
            print_info("No project context (run /docs detect)")

        # Index status
        if self.docs_indexer:
            console.print("\n[yellow]Index Status:[/yellow]")
            status = self.docs_indexer.get_index_status()

            if status["libraries"]:
                table = Table(show_header=True)
                table.add_column("Library")
                table.add_column("Chunks")
                table.add_column("Indexed")
                table.add_column("Status")

                for lib, info in status["libraries"].items():
                    status_str = "[red]expired[/red]" if info["is_expired"] else "[green]valid[/green]"
                    table.add_row(
                        lib,
                        str(info["chunk_count"]),
                        info["indexed_at"][:10],
                        status_str,
                    )
                console.print(table)
            else:
                print_info("No libraries indexed")

            console.print(f"\nTotal chunks: {status['total_chunks']}")

        # Cache status
        if self.docs_fetcher:
            console.print("\n[yellow]Cache Status:[/yellow]")
            cache_stats = self.docs_fetcher.get_cache_stats()
            console.print(f"  Valid entries: {cache_stats['valid_entries']}")
            console.print(f"  Expired entries: {cache_stats['expired_entries']}")

        console.print()

    async def _docs_detect(self) -> None:
        """Re-run project detection."""
        from penguincode.docs_rag import ProjectDetector

        detector = ProjectDetector(str(self.project_dir))
        self.project_context = detector.detect()

        console.print("\n[bold cyan]Project Detection Results[/bold cyan]\n")

        if self.project_context.languages:
            console.print("[yellow]Languages:[/yellow]")
            for lang in self.project_context.languages:
                console.print(f"  - {lang.value}")

            console.print(f"\n[yellow]Libraries ({len(self.project_context.libraries)}):[/yellow]")
            for lib in self.project_context.libraries[:15]:
                version = f" ({lib.version})" if lib.version else ""
                console.print(f"  - {lib.name}{version} [{lib.language.value}]")

            if len(self.project_context.libraries) > 15:
                console.print(f"  ... and {len(self.project_context.libraries) - 15} more")
        else:
            print_info("No languages or libraries detected")

        console.print()

    async def _docs_index(self, library_name: str = "") -> None:
        """Index documentation for libraries."""
        if not self.project_context:
            print_error("Run /docs detect first")
            return

        from penguincode.docs_rag import get_priority_docs_for_project

        if library_name:
            # Index specific library
            lib = next(
                (l for l in self.project_context.libraries if l.name.lower() == library_name.lower()),
                None
            )
            if not lib:
                print_error(f"Library '{library_name}' not detected in project")
                return

            libs_to_index = [lib]
        else:
            # Index priority libraries
            libs_to_index = get_priority_docs_for_project(
                self.project_context.libraries,
                self.settings.docs_rag.priority_libraries,
                self.settings.docs_rag.max_libraries_to_index,
            )

        console.print(f"\n[cyan]Indexing {len(libs_to_index)} libraries...[/cyan]\n")

        total_chunks = 0
        for lib in libs_to_index:
            console.print(f"  Fetching {lib.name}...")

            # Fetch docs
            docs = await self.docs_fetcher.fetch_library_docs(lib)

            if docs:
                # Index docs
                chunks = await self.docs_indexer.index_library(lib, docs)
                total_chunks += chunks
                console.print(f"    Indexed {chunks} chunks")
            else:
                console.print(f"    [dim]No docs found[/dim]")

        print_success(f"Indexed {total_chunks} total chunks")

    async def _docs_search(self, query: str) -> None:
        """Search indexed documentation."""
        if not query:
            print_error("Usage: /docs search <query>")
            return

        if not self.docs_indexer:
            print_error("Docs indexer not initialized")
            return

        console.print(f"\n[cyan]Searching:[/cyan] {query}\n")

        # Filter to project libraries only
        library_names = self.project_context.library_names if self.project_context else None

        results = await self.docs_indexer.search(
            query=query,
            libraries=library_names,
            limit=5,
        )

        if results:
            for i, result in enumerate(results, 1):
                console.print(f"[bold]{i}. [{result.library}][/bold] (score: {result.relevance_score:.2f})")
                # Truncate long content
                content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                console.print(f"   {content}\n")
        else:
            print_info("No results found")

    async def _docs_clear(self, library_name: str = "") -> None:
        """Clear indexed documentation."""
        if not self.docs_indexer:
            print_error("Docs indexer not initialized")
            return

        if library_name:
            count = await self.docs_indexer.clear_library_index(library_name)
            print_success(f"Cleared {count} chunks for {library_name}")
        else:
            # Clear all
            status = self.docs_indexer.get_index_status()
            total = 0
            for lib in list(status["libraries"].keys()):
                count = await self.docs_indexer.clear_library_index(lib)
                total += count
            print_success(f"Cleared {total} total chunks")

    async def _docs_cleanup(self) -> None:
        """Remove docs for libraries no longer in project."""
        if not self.project_context:
            print_error("Run /docs detect first")
            return

        # Cleanup cache
        cache_removed = self.docs_fetcher.cleanup_unused_libraries(
            self.project_context.libraries
        )

        # Cleanup index
        index_removed = await self.docs_indexer.cleanup_unused(
            self.project_context.libraries,
            self.project_context.languages,
        )

        if cache_removed or index_removed:
            console.print("\n[cyan]Cleanup Results:[/cyan]")
            if cache_removed:
                for lib, count in cache_removed.items():
                    console.print(f"  Cache: removed {count} pages for {lib}")
            if index_removed:
                for lib, count in index_removed.items():
                    console.print(f"  Index: removed {count} chunks for {lib}")
            console.print()
        else:
            print_info("Nothing to clean up")

    def _detect_languages_in_message(self, message: str) -> list:
        """Detect programming languages mentioned in user message.

        Args:
            message: User's message

        Returns:
            List of detected language names
        """
        msg_lower = message.lower()
        detected = []

        # Language patterns to detect
        language_patterns = {
            "python": ["python", "py ", ".py", "pip ", "pytest", "django", "flask", "fastapi"],
            "javascript": ["javascript", "js ", ".js", "node", "npm ", "react", "vue", "express"],
            "typescript": ["typescript", "ts ", ".ts", ".tsx"],
            "go": [" go ", "golang", ".go", "go mod", "go build"],
            "rust": ["rust", ".rs", "cargo ", "rustc"],
            "hcl": ["terraform", "opentofu", "tofu ", ".tf", "hcl"],
            "ansible": ["ansible", "playbook", "ansible-playbook", ".yml playbook", ".yaml playbook"],
        }

        for lang, patterns in language_patterns.items():
            if any(p in msg_lower for p in patterns):
                detected.append(lang)

        return detected

    async def handle_chat(self, message: str) -> None:
        """
        Handle regular chat messages by sending to the chat agent.

        The chat agent decides whether to respond directly or spawn
        specialized agents for code/file operations.
        """
        # Save user message to session
        self.session.add_message("user", message)

        console.print()  # Add some spacing

        try:
            # On-demand language detection and indexing
            if self.settings.docs_rag.auto_detect_on_request and self.settings.docs_rag.auto_index_on_request:
                detected_langs = self._detect_languages_in_message(message)
                for lang in detected_langs:
                    await self._ensure_language_indexed(lang)

            # Inject documentation context if available
            if self.context_injector and self.project_context:
                should_inject = await self.context_injector.should_inject_context(
                    message, self.project_context
                )
                if should_inject:
                    context = await self.context_injector.get_relevant_context(
                        message, self.project_context
                    )
                    if context:
                        # Augment the chat agent's system prompt temporarily
                        original_prompt = self.chat_agent.system_prompt
                        self.chat_agent.system_prompt = self.context_injector.build_augmented_prompt(
                            original_prompt, context
                        )
                        console.print("[dim](using documentation context)[/dim]")

            # Use chat agent to process the message
            response = await self.chat_agent.process(message)

            # Restore original prompt if modified
            if hasattr(self, '_original_prompt'):
                self.chat_agent.system_prompt = self._original_prompt

            # Display the response
            console.print(f"\n[bold blue]Assistant:[/bold blue]")
            console.print(response)
            console.print()

            # Save assistant response to session
            if response:
                self.session.add_message("assistant", response)

        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]\n")
            console.print("[dim]Make sure Ollama is running: ollama serve[/dim]\n")

    async def run(self) -> None:
        """Run the REPL loop."""
        console.print("[bold cyan]PenguinCode Chat[/bold cyan]")
        console.print(f"Project: {self.project_dir}")
        console.print(f"Models: orchestration={self.settings.models.orchestration}, execution={self.settings.models.execution}")
        console.print("\nType [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit\n")

        # Track consecutive Ctrl+C presses using closure
        state = {"interrupt_count": 0, "should_exit": False}

        def signal_handler(signum, frame):
            """Handle SIGINT (Ctrl+C)."""
            state["interrupt_count"] += 1
            if state["interrupt_count"] >= 2:
                state["should_exit"] = True
                console.print("\n")
                print_info("Goodbye!")
                sys.exit(0)
            else:
                console.print("\n[yellow]Press Ctrl+C again to exit[/yellow]\n")
                # Re-display prompt
                console.print("[bold green]You[/bold green]: ", end="")

        # Set up signal handler
        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            while not state["should_exit"]:
                try:
                    # Get user input
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: Prompt.ask("[bold green]You[/bold green]")
                    )

                    # Reset interrupt count on successful input
                    state["interrupt_count"] = 0

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        should_continue = await self.handle_command(user_input)
                        if not should_continue:
                            break
                    else:
                        # Regular chat message - send to orchestrator
                        await self.handle_chat(user_input)

                except EOFError:
                    break
                except Exception as e:
                    if "interrupt" in str(e).lower():
                        # Treat as Ctrl+C
                        continue
                    print_error(f"Error: {str(e)}")
                    state["interrupt_count"] = 0
                    continue
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        print_info("Goodbye!")


async def start_repl(project_dir: str = ".", config_path: str = "config.yaml") -> None:
    """
    Start the REPL session.

    Args:
        project_dir: Project directory
        config_path: Path to config file
    """
    async with REPLSession(project_dir, config_path) as repl:
        await repl.run()
