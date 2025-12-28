"""Chat agent - the main orchestrating agent for PenguinCode.

This is the primary agent that users interact with. It:
1. Answers simple questions directly (greetings, general knowledge)
2. Delegates ALL code/file operations to specialized agents

Unlike Claude Code where the chat agent may do work itself,
PenguinCode's chat agent is purely an orchestrator - it NEVER
uses tools directly, always delegating to specialized agents.
"""

import json
import time
from typing import Dict, List, Optional

from penguincode.ollama import Message, OllamaClient
from penguincode.config.settings import Settings
from penguincode.ui import console


CHAT_SYSTEM_PROMPT = """You are PenguinCode, an AI coding assistant orchestrator.

Your role is to understand user requests and delegate work to specialized agents. You do NOT perform any file or code operations yourself - you always delegate.

**Your capabilities:**

1. **Direct responses** - For greetings, general questions, or explaining concepts, respond directly.

2. **spawn_explorer** - Delegate to the explorer agent for ANY codebase reading or searching:
   - Finding files
   - Reading code
   - Searching for patterns
   - Understanding how code works
   - Answering questions about the codebase

3. **spawn_executor** - Delegate to the executor agent for ANY code modifications:
   - Creating new files
   - Editing existing files
   - Running commands or tests
   - Fixing bugs
   - Implementing features

**Rules:**

1. NEVER try to read, write, or search files yourself - always delegate to an agent.

2. For questions about code or the codebase -> spawn_explorer

3. For any request to create, modify, or execute -> spawn_executor

4. For greetings or general questions -> respond directly without spawning agents

5. Always provide the agent with a clear, detailed task description.

6. After an agent completes its work, summarize the results for the user.

Project directory: {project_dir}
"""

# Tool definitions for spawning agents
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_explorer",
            "description": "Delegate to the explorer agent for reading files, searching code, or understanding the codebase. Use for ANY question about code or files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task for the explorer (e.g., 'Find all Python files that handle user authentication and explain how the auth flow works')"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_executor",
            "description": "Delegate to the executor agent for creating files, editing code, or running commands. Use for ANY request to modify or execute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task for the executor (e.g., 'Create a new file auth.py with a login function that validates username and password')"
                    }
                },
                "required": ["task"]
            }
        }
    },
]


class ChatAgent:
    """Main chat agent that orchestrates all interactions.

    This agent is the "job foreman" - it understands user requests
    and delegates all actual work to specialized agents.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        settings: Settings,
        project_dir: str,
    ):
        """
        Initialize chat agent.

        Args:
            ollama_client: Ollama client for LLM calls
            settings: Application settings
            project_dir: Project directory path
        """
        self.client = ollama_client
        self.settings = settings
        self.project_dir = project_dir
        self.model = settings.models.orchestration

        # Conversation history
        self.conversation_history: List[Message] = []

        # Lazy-loaded specialized agents
        self._explorer_agent = None
        self._executor_agent = None

        # System prompt
        self.system_prompt = CHAT_SYSTEM_PROMPT.format(project_dir=project_dir)

    def _get_explorer_agent(self):
        """Lazy-load explorer agent."""
        if self._explorer_agent is None:
            from .explorer import ExplorerAgent
            self._explorer_agent = ExplorerAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=self.settings.models.orchestration,
            )
        return self._explorer_agent

    def _get_executor_agent(self):
        """Lazy-load executor agent."""
        if self._executor_agent is None:
            from .executor import ExecutorAgent
            self._executor_agent = ExecutorAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=self.settings.models.execution,
            )
        return self._executor_agent

    async def _spawn_agent(self, agent_type: str, task: str) -> str:
        """
        Spawn a specialized agent to handle a task.

        Args:
            agent_type: "explorer" or "executor"
            task: Task description for the agent

        Returns:
            Agent result as string
        """
        if agent_type == "explorer":
            console.print(f"[cyan]> Spawning explorer agent...[/cyan]")
            agent = self._get_explorer_agent()
        elif agent_type == "executor":
            console.print(f"[cyan]> Spawning executor agent...[/cyan]")
            agent = self._get_executor_agent()
        else:
            return f"Unknown agent type: {agent_type}"

        try:
            result = await agent.run(task)
            if result.success:
                return result.output
            else:
                return f"Agent error: {result.error or 'Unknown error'}"
        except Exception as e:
            return f"Agent failed: {str(e)}"

    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls from response text."""
        tool_calls = []
        valid_tools = {"spawn_explorer", "spawn_executor"}

        try:
            if "{" in response_text and "}" in response_text:
                start = 0
                while True:
                    start = response_text.find("{", start)
                    if start == -1:
                        break

                    # Find matching closing brace
                    brace_count = 0
                    end = start
                    for i, char in enumerate(response_text[start:], start):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break

                    if end > start:
                        try:
                            json_str = response_text[start:end]
                            data = json.loads(json_str)

                            if "name" in data and data["name"] in valid_tools:
                                tool_calls.append(data)
                        except json.JSONDecodeError:
                            pass

                    start = end
        except Exception:
            pass

        return tool_calls

    async def process(self, user_message: str) -> str:
        """
        Process a user message.

        The chat agent will either:
        1. Respond directly for simple queries
        2. Spawn an agent for code/file operations

        Args:
            user_message: The user's message

        Returns:
            Final response string
        """
        # Build messages
        messages = [
            Message(role="system", content=self.system_prompt),
        ]

        # Add conversation history (last 10 messages for context)
        messages.extend(self.conversation_history[-10:])

        # Add current user message
        messages.append(Message(role="user", content=user_message))

        console.print("[dim]Thinking...[/dim]", end="\r")

        try:
            response_text = ""
            tool_calls = []

            # Call LLM with agent spawning tools
            async for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                tools=AGENT_TOOLS,
                stream=True,
            ):
                if chunk.message and chunk.message.content:
                    response_text += chunk.message.content

                # Check for structured tool calls
                if chunk.done and hasattr(chunk, "message"):
                    msg = chunk.message
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls.extend(msg.tool_calls)

            console.print("            ", end="\r")  # Clear "Thinking..."

            # Try parsing tool calls from text if none structured
            if not tool_calls:
                tool_calls = self._parse_tool_calls(response_text)

            # Check for agent keywords in response
            if not tool_calls:
                response_lower = response_text.lower()
                if "spawn_explorer" in response_lower:
                    # Extract task from response if possible
                    tool_calls = [{"name": "spawn_explorer", "arguments": {"task": user_message}}]
                elif "spawn_executor" in response_lower:
                    tool_calls = [{"name": "spawn_executor", "arguments": {"task": user_message}}]

            # Execute tool calls (spawn agents)
            if tool_calls:
                results = []
                for tc in tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"task": user_message}

                    task = args.get("task", user_message)

                    if name == "spawn_explorer":
                        result = await self._spawn_agent("explorer", task)
                        results.append(result)
                    elif name == "spawn_executor":
                        result = await self._spawn_agent("executor", task)
                        results.append(result)

                final_response = "\n\n".join(results) if results else response_text

                # Update conversation history
                self.conversation_history.append(Message(role="user", content=user_message))
                self.conversation_history.append(Message(role="assistant", content=final_response))

                return final_response

            # No tool calls - direct response
            self.conversation_history.append(Message(role="user", content=user_message))
            self.conversation_history.append(Message(role="assistant", content=response_text))

            return response_text

        except Exception as e:
            console.print("            ", end="\r")
            return f"Error: {str(e)}"

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
