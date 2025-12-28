"""Interactive REPL loop for PenguinCode chat."""

import asyncio
from pathlib import Path
from typing import Optional

from rich.prompt import Prompt

from penguincode.agents import ExecutorAgent, ExplorerAgent
from penguincode.config.settings import Settings, load_settings
from penguincode.ollama import OllamaClient
from penguincode.ui import console, print_error, print_info, print_success

from .session import Session, SessionManager


class REPLSession:
    """Interactive REPL session."""

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

        # Agents
        self.agents = {}

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            base_url=self.settings.ollama.api_url,
            timeout=self.settings.ollama.timeout,
        )
        await self.ollama_client.__aenter__()

        # Initialize agents
        self.agents["executor"] = ExecutorAgent(
            ollama_client=self.ollama_client, working_dir=str(self.project_dir)
        )
        self.agents["explorer"] = ExplorerAgent(
            ollama_client=self.ollama_client, working_dir=str(self.project_dir)
        )

        return self

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
  /clear             Clear the screen
  /history           Show conversation history
  /agents            List available agents

[yellow]Agent Commands:[/yellow]
  /explore <query>   Explore codebase (read-only)
  /execute <task>    Execute code changes
  /read <path>       Read a file

[yellow]Chat:[/yellow]
  Just type your message to chat with the orchestrator
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
            console.print(f"[{role_color}]{msg.role}:[/{role_color}] {msg.content}\n")

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

    async def handle_chat(self, message: str) -> None:
        """Handle regular chat messages by sending to Ollama."""
        from penguincode.ollama import Message

        # Build conversation history for context
        messages = [
            Message(
                role="system",
                content=(
                    f"You are PenguinCode, an AI coding assistant. "
                    f"You are helping with a project in: {self.project_dir}\n"
                    f"Be concise and helpful. If you need to explore files or run commands, "
                    f"tell the user to use /explore or /execute commands."
                ),
            )
        ]

        # Add recent conversation history (last 10 messages)
        for msg in self.session.messages[-10:]:
            messages.append(Message(role=msg.role, content=msg.content))

        # Stream the response
        console.print("\n[bold blue]Assistant[/bold blue]: ", end="")

        full_response = ""
        try:
            async for chunk in self.ollama_client.chat(
                model=self.settings.models.orchestration,
                messages=messages,
                stream=True,
            ):
                if chunk.message and chunk.message.content:
                    content = chunk.message.content
                    console.print(content, end="")
                    full_response += content

            console.print("\n")

            # Save assistant response to session
            if full_response:
                self.session.add_message("assistant", full_response)

        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]\n")
            console.print("[dim]Make sure Ollama is running: ollama serve[/dim]\n")

    async def run(self) -> None:
        """Run the REPL loop."""
        console.print("[bold cyan]PenguinCode Chat[/bold cyan]")
        console.print(f"Project: {self.project_dir}")
        console.print("\nType [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit\n")

        while True:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Prompt.ask("[bold green]You[/bold green]")
                )

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = await self.handle_command(user_input)
                    if not should_continue:
                        break
                else:
                    # Regular chat message - send to Ollama
                    self.session.add_message("user", user_input)
                    await self.handle_chat(user_input)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Use /exit to quit[/yellow]\n")
                continue
            except EOFError:
                break
            except Exception as e:
                print_error(f"Error: {str(e)}")
                continue

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
