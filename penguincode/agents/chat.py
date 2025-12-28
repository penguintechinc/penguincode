"""Chat agent - the main orchestrating agent for PenguinCode.

This is the primary agent that users interact with. It serves two roles:

1. **Knowledge Base** - Answers general questions directly without spawning agents
2. **Foreman** - Delegates work to specialized agents, reviews their output,
   and can dispatch follow-up agents to fix issues if needed

Unlike Claude Code where the chat agent may do work itself,
PenguinCode's chat agent NEVER uses tools directly - it only delegates.
"""

import json
from typing import Dict, List, Optional

from penguincode.ollama import Message, OllamaClient
from penguincode.config.settings import Settings
from penguincode.ui import console


CHAT_SYSTEM_PROMPT = """You are PenguinCode, an AI coding assistant.

You have two roles:

## Role 1: Knowledge Base
For general questions, greetings, or explaining concepts - respond directly without spawning agents.

## Role 2: Foreman (Job Supervisor)
For any code or file operations, you delegate to specialized agents and supervise their work:

**spawn_explorer** - For reading, searching, or understanding code
**spawn_executor** - For creating, editing, or running code

As foreman, you:
1. Delegate clear, specific tasks to agents
2. Review the agent's work when they report back
3. If the work is incomplete or has errors, spawn another agent to fix it
4. Provide a final summary to the user

**Rules:**
- NEVER read, write, or search files yourself - always delegate
- For questions about code -> spawn_explorer
- For requests to change/create/run code -> spawn_executor
- For greetings or general questions -> respond directly
- Review agent work critically - spawn follow-up agents if needed

Project directory: {project_dir}
"""

REVIEW_PROMPT = """You are reviewing work done by a specialized agent.

Original user request: {user_request}

Agent type: {agent_type}
Agent output:
---
{agent_output}
---

As the foreman, evaluate this work:

1. Did the agent complete the task successfully?
2. Are there any errors or issues that need fixing?
3. Is any follow-up work needed?

Respond with one of:
- If work is complete and good: Summarize the results for the user
- If work has issues: Call spawn_executor or spawn_explorer to fix the problem
- If more exploration is needed: Call spawn_explorer for additional information

Be concise but thorough in your assessment.
"""

# Tool definitions for spawning agents
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_explorer",
            "description": "Delegate to explorer agent for reading files, searching code, or understanding the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task for the explorer"
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
            "description": "Delegate to executor agent for creating files, editing code, or running commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task for the executor"
                    }
                },
                "required": ["task"]
            }
        }
    },
]


class ChatAgent:
    """Main chat agent - knowledge base and job foreman.

    This agent understands user requests and either:
    1. Answers directly (knowledge base role)
    2. Delegates to agents, reviews their work, and supervises fixes (foreman role)
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        settings: Settings,
        project_dir: str,
    ):
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

        # Max supervision iterations (prevent infinite loops)
        self.max_supervision_rounds = 3

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

    async def _spawn_agent(self, agent_type: str, task: str) -> tuple[bool, str]:
        """
        Spawn a specialized agent to handle a task.

        Returns:
            Tuple of (success, output)
        """
        if agent_type == "explorer":
            console.print(f"[cyan]> Spawning explorer agent...[/cyan]")
            agent = self._get_explorer_agent()
        elif agent_type == "executor":
            console.print(f"[cyan]> Spawning executor agent...[/cyan]")
            agent = self._get_executor_agent()
        else:
            return False, f"Unknown agent type: {agent_type}"

        try:
            result = await agent.run(task)
            return result.success, result.output if result.success else (result.error or "Unknown error")
        except Exception as e:
            return False, f"Agent failed: {str(e)}"

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

    async def _call_llm(self, messages: List[Message], use_tools: bool = True) -> tuple[str, List[Dict]]:
        """Call the LLM and return response text and tool calls."""
        response_text = ""
        tool_calls = []

        async for chunk in self.client.chat(
            model=self.model,
            messages=messages,
            tools=AGENT_TOOLS if use_tools else None,
            stream=True,
        ):
            if chunk.message and chunk.message.content:
                response_text += chunk.message.content

            if chunk.done and hasattr(chunk, "message"):
                msg = chunk.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)

        # Try parsing tool calls from text if none structured
        if not tool_calls:
            tool_calls = self._parse_tool_calls(response_text)

        # Check for agent keywords in response
        if not tool_calls:
            response_lower = response_text.lower()
            if "spawn_explorer" in response_lower:
                tool_calls = [{"name": "spawn_explorer", "arguments": {"task": ""}}]
            elif "spawn_executor" in response_lower:
                tool_calls = [{"name": "spawn_executor", "arguments": {"task": ""}}]

        return response_text, tool_calls

    async def _review_and_supervise(
        self,
        user_request: str,
        agent_type: str,
        agent_output: str,
        agent_success: bool,
        round_num: int,
    ) -> str:
        """
        Review agent work and decide if follow-up is needed.

        Returns final response for the user.
        """
        if round_num >= self.max_supervision_rounds:
            console.print("[yellow]> Max supervision rounds reached[/yellow]")
            return agent_output

        # Build review prompt
        review_content = REVIEW_PROMPT.format(
            user_request=user_request,
            agent_type=agent_type,
            agent_output=agent_output if agent_success else f"AGENT ERROR: {agent_output}",
        )

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=review_content),
        ]

        console.print("[dim]Reviewing work...[/dim]", end="\r")

        try:
            response_text, tool_calls = await self._call_llm(messages)
            console.print("                  ", end="\r")

            # Extract tool call info
            if tool_calls:
                tc = tool_calls[0]
                name = tc.get("name") or tc.get("function", {}).get("name")
                args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                task = args.get("task", "")
                if not task:
                    # Use context from review to form task
                    task = f"Follow up on: {user_request}"

                if name == "spawn_explorer":
                    console.print(f"[yellow]> Foreman requesting explorer follow-up[/yellow]")
                    success, output = await self._spawn_agent("explorer", task)
                    return await self._review_and_supervise(
                        user_request, "explorer", output, success, round_num + 1
                    )
                elif name == "spawn_executor":
                    console.print(f"[yellow]> Foreman requesting executor follow-up[/yellow]")
                    success, output = await self._spawn_agent("executor", task)
                    return await self._review_and_supervise(
                        user_request, "executor", output, success, round_num + 1
                    )

            # No follow-up needed - return the review summary or original output
            return response_text if response_text else agent_output

        except Exception as e:
            console.print("                  ", end="\r")
            return agent_output  # Fall back to original output on error

    async def process(self, user_message: str) -> str:
        """
        Process a user message.

        The chat agent will either:
        1. Respond directly (knowledge base role)
        2. Delegate to agents and supervise (foreman role)
        """
        messages = [
            Message(role="system", content=self.system_prompt),
        ]
        messages.extend(self.conversation_history[-10:])
        messages.append(Message(role="user", content=user_message))

        console.print("[dim]Thinking...[/dim]", end="\r")

        try:
            response_text, tool_calls = await self._call_llm(messages)
            console.print("            ", end="\r")

            # If tool calls, spawn agents and supervise
            if tool_calls:
                tc = tool_calls[0]
                name = tc.get("name") or tc.get("function", {}).get("name")
                args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                task = args.get("task", user_message)

                if name == "spawn_explorer":
                    success, output = await self._spawn_agent("explorer", task)
                    final_response = await self._review_and_supervise(
                        user_message, "explorer", output, success, round_num=1
                    )
                elif name == "spawn_executor":
                    success, output = await self._spawn_agent("executor", task)
                    final_response = await self._review_and_supervise(
                        user_message, "executor", output, success, round_num=1
                    )
                else:
                    final_response = response_text

                self.conversation_history.append(Message(role="user", content=user_message))
                self.conversation_history.append(Message(role="assistant", content=final_response))
                return final_response

            # No tool calls - direct response (knowledge base role)
            self.conversation_history.append(Message(role="user", content=user_message))
            self.conversation_history.append(Message(role="assistant", content=response_text))
            return response_text

        except Exception as e:
            console.print("            ", end="\r")
            return f"Error: {str(e)}"

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
