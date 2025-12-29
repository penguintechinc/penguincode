"""Chat agent - the main orchestrating agent for PenguinCode.

This is the primary agent that users interact with. It serves two roles:

1. **Knowledge Base** - Answers general questions directly without spawning agents
2. **Foreman** - Delegates work to specialized agents, reviews their output,
   and can dispatch follow-up agents to fix issues if needed

For complex tasks, it uses the PlannerAgent to break down work and can
execute multiple agents in parallel (up to max_concurrent_agents).
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

from penguincode.ollama import Message, OllamaClient
from penguincode.config.settings import Settings
from penguincode.ui import console


CHAT_SYSTEM_PROMPT = """You are PenguinCode, an AI coding assistant that routes tasks to specialized agents.

## YOUR ONLY JOB IS TO ROUTE REQUESTS

You MUST respond with a JSON tool call for ANY request involving:
- Files (create, write, read, edit, find, search)
- Code (write, run, test, build, install)
- Research (documentation, how-to, tutorials)

## TOOL CALL FORMAT (YOU MUST USE THIS)

For file/code operations:
{{"name": "spawn_executor", "arguments": {{"task": "the full user request"}}}}

For reading/searching:
{{"name": "spawn_explorer", "arguments": {{"task": "the full user request"}}}}

For research/docs:
{{"name": "spawn_researcher", "arguments": {{"task": "the full user request"}}}}

For complex multi-step:
{{"name": "spawn_planner", "arguments": {{"task": "the full user request"}}}}

## EXAMPLES

User: "Create a python script hello.py"
You: {{"name": "spawn_executor", "arguments": {{"task": "Create a python script hello.py"}}}}

User: "Write a file that counts 1 to 100"
You: {{"name": "spawn_executor", "arguments": {{"task": "Write a file that counts 1 to 100"}}}}

User: "Create an app"
You: {{"name": "spawn_executor", "arguments": {{"task": "Create an app"}}}}

User: "What's in config.yaml?"
You: {{"name": "spawn_explorer", "arguments": {{"task": "Read and show config.yaml"}}}}

User: "How do I use pandas?"
You: {{"name": "spawn_researcher", "arguments": {{"task": "How to use pandas library"}}}}

User: "Hello"
You: Hello! I'm PenguinCode. How can I help you with your code today?

## RULES

1. ANY request mentioning files, code, scripts, apps, programs → spawn_executor
2. ANY request to read, find, search, show → spawn_explorer
3. ANY request about how-to, documentation, tutorials → spawn_researcher
4. ONLY greetings and general chat get direct text responses
5. NEVER say "I will create..." - just output the JSON tool call

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
    {
        "type": "function",
        "function": {
            "name": "spawn_researcher",
            "description": "Delegate to researcher agent for web searches, documentation lookup, and information gathering. Use when user asks about external topics, documentation, or needs web research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The research task or question to investigate"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_planner",
            "description": "Delegate to planner agent to break down a complex task into steps. Use for multi-step tasks, refactoring, or features requiring design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The complex task to plan"
                    }
                },
                "required": ["task"]
            }
        }
    },
]


class AgentSemaphore:
    """Dynamic semaphore for controlling concurrent agent execution."""

    def __init__(self, max_concurrent: int = 5):
        self._max = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a slot for agent execution."""
        await self._semaphore.acquire()
        async with self._lock:
            self._active_count += 1

    def release(self):
        """Release a slot after agent completion."""
        self._semaphore.release()
        asyncio.create_task(self._decrement_count())

    async def _decrement_count(self):
        async with self._lock:
            self._active_count -= 1

    @property
    def active_agents(self) -> int:
        return self._active_count

    @property
    def available_slots(self) -> int:
        return self._max - self._active_count

    def adjust_max(self, new_max: int):
        """Dynamically adjust max concurrent agents (for resource regulation)."""
        # Note: This is a simplified version. Full implementation would
        # need to handle in-flight tasks carefully.
        self._max = max(1, new_max)


class ChatAgent:
    """Main chat agent - knowledge base and job foreman.

    This agent understands user requests and either:
    1. Answers directly (knowledge base role)
    2. Delegates to agents, reviews their work, and supervises fixes (foreman role)
    3. Uses planner for complex tasks, then executes the plan with parallel agents
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
        self._planner_agent = None
        self._researcher_agent = None

        # System prompt
        self.system_prompt = CHAT_SYSTEM_PROMPT.format(project_dir=project_dir)

        # Max supervision iterations (prevent infinite loops)
        self.max_supervision_rounds = 3

        # Agent concurrency control
        max_agents = settings.regulators.max_concurrent_agents
        self.agent_semaphore = AgentSemaphore(max_concurrent=max_agents)
        self.agent_timeout = settings.regulators.agent_timeout_seconds

    def _get_explorer_agent(self, lite: bool = False):
        """
        Get explorer agent, optionally using lightweight model.

        Args:
            lite: If True, use lightweight model for simple searches
        """
        # For lite mode, always create fresh with lite model
        if lite:
            from .explorer import ExplorerAgent
            model = getattr(self.settings.models, 'exploration_lite', self.settings.models.orchestration)
            return ExplorerAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=model,
            )

        # Standard explorer (cached)
        if self._explorer_agent is None:
            from .explorer import ExplorerAgent
            model = getattr(self.settings.models, 'exploration', self.settings.models.orchestration)
            self._explorer_agent = ExplorerAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=model,
            )
        return self._explorer_agent

    def _get_executor_agent(self, lite: bool = False):
        """
        Get executor agent, optionally using lightweight model.

        Args:
            lite: If True, use lightweight model for simple edits
        """
        # For lite mode, always create fresh with lite model
        if lite:
            from .executor import ExecutorAgent
            model = getattr(self.settings.models, 'execution_lite', self.settings.models.execution)
            console.print(f"[dim](using lite model: {model})[/dim]")
            return ExecutorAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=model,
            )

        # Standard executor (cached)
        if self._executor_agent is None:
            from .executor import ExecutorAgent
            self._executor_agent = ExecutorAgent(
                ollama_client=self.client,
                working_dir=self.project_dir,
                model=self.settings.models.execution,
            )
        return self._executor_agent

    def _get_planner_agent(self):
        """Lazy-load planner agent."""
        if self._planner_agent is None:
            from .planner import PlannerAgent
            self._planner_agent = PlannerAgent(
                ollama_client=self.client,
                model=self.settings.models.planning,
            )
        return self._planner_agent

    def _get_researcher_agent(self):
        """Lazy-load researcher agent."""
        if self._researcher_agent is None:
            from .researcher import ResearcherAgent
            # Get research model from settings, fallback to orchestration model
            model = getattr(self.settings.models, 'research', self.settings.models.orchestration)
            self._researcher_agent = ResearcherAgent(
                ollama_client=self.client,
                research_config=self.settings.research,
                working_dir=self.project_dir,
                model=model,
            )
        return self._researcher_agent

    def _estimate_complexity(self, task: str) -> str:
        """
        Estimate task complexity to decide which model tier to use.

        Returns: "simple", "moderate", or "complex"
        """
        task_lower = task.lower()

        # Simple tasks - single file, basic operations
        simple_patterns = [
            "read ", "show ", "display ", "print ", "cat ",
            "find file", "list files", "what is", "where is",
            "add comment", "fix typo", "rename variable",
            "simple", "quick", "just ",
        ]
        if any(p in task_lower for p in simple_patterns):
            return "simple"

        # Complex tasks - multi-file, refactoring, features
        complex_patterns = [
            "refactor", "restructure", "redesign", "architect",
            "implement feature", "add feature", "create system",
            "multiple files", "across the codebase", "all files",
            "migrate", "upgrade", "overhaul",
        ]
        if any(p in task_lower for p in complex_patterns):
            return "complex"

        # Moderate - default for most tasks
        return "moderate"

    async def _spawn_agent(
        self,
        agent_type: str,
        task: str,
        force_lite: bool = False,
        force_full: bool = False,
    ) -> Tuple[bool, str]:
        """
        Spawn a specialized agent to handle a task.

        Automatically selects lite or full model based on task complexity,
        unless force_lite or force_full is specified.

        Args:
            agent_type: "explorer", "executor", or "planner"
            task: Task description
            force_lite: Force use of lightweight model
            force_full: Force use of full model

        Returns:
            Tuple of (success, output)
        """
        # Determine model tier based on complexity
        complexity = self._estimate_complexity(task)
        use_lite = force_lite or (complexity == "simple" and not force_full)

        if agent_type == "explorer":
            tier = "lite" if use_lite else "standard"
            console.print(f"[cyan]> Spawning explorer agent ({tier})...[/cyan]")
            agent = self._get_explorer_agent(lite=use_lite)
        elif agent_type == "executor":
            tier = "lite" if use_lite else "full"
            console.print(f"[cyan]> Spawning executor agent ({tier})...[/cyan]")
            agent = self._get_executor_agent(lite=use_lite)
        elif agent_type == "planner":
            console.print(f"[cyan]> Spawning planner agent...[/cyan]")
            agent = self._get_planner_agent()
        elif agent_type == "researcher":
            console.print(f"[cyan]> Spawning researcher agent...[/cyan]")
            agent = self._get_researcher_agent()
        else:
            return False, f"Unknown agent type: {agent_type}"

        try:
            # Acquire semaphore slot
            await self.agent_semaphore.acquire()
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    agent.run(task),
                    timeout=self.agent_timeout
                )
                return result.success, result.output if result.success else (result.error or "Unknown error")
            finally:
                self.agent_semaphore.release()
        except asyncio.TimeoutError:
            return False, f"Agent timed out after {self.agent_timeout} seconds"
        except Exception as e:
            return False, f"Agent failed: {str(e)}"

    async def _spawn_agents_parallel(
        self,
        tasks: List[Tuple[str, str]]  # List of (agent_type, task)
    ) -> List[Tuple[bool, str]]:
        """
        Spawn multiple agents in parallel, respecting max_concurrent_agents.

        Args:
            tasks: List of (agent_type, task_description) tuples

        Returns:
            List of (success, output) tuples in same order as input
        """
        console.print(f"[cyan]> Spawning {len(tasks)} agents (max {self.agent_semaphore._max} concurrent)...[/cyan]")

        async def run_task(agent_type: str, task: str, index: int) -> Tuple[int, bool, str]:
            success, output = await self._spawn_agent(agent_type, task)
            return index, success, output

        # Create tasks
        coroutines = [
            run_task(agent_type, task, i)
            for i, (agent_type, task) in enumerate(tasks)
        ]

        # Run with concurrency control (semaphore is checked in _spawn_agent)
        results_unordered = await asyncio.gather(*coroutines, return_exceptions=True)

        # Sort back to original order and handle exceptions
        results = [None] * len(tasks)
        for result in results_unordered:
            if isinstance(result, Exception):
                # Find first empty slot for exception
                for i in range(len(results)):
                    if results[i] is None:
                        results[i] = (False, str(result))
                        break
            else:
                index, success, output = result
                results[index] = (success, output)

        return results

    async def _execute_plan(self, plan, user_request: str) -> str:
        """
        Execute a plan by running agents according to parallel groups.

        Args:
            plan: Plan object from PlannerAgent
            user_request: Original user request

        Returns:
            Combined results from all steps
        """
        from .planner import Plan

        console.print(f"\n[bold cyan]Executing plan ({len(plan.steps)} steps)...[/bold cyan]")

        step_results: Dict[int, Tuple[bool, str]] = {}
        all_outputs = []

        for group_num, group in enumerate(plan.parallel_groups, 1):
            # Get steps for this group
            group_steps = [s for s in plan.steps if s.step_num in group]

            if not group_steps:
                continue

            console.print(f"\n[cyan]> Group {group_num}: executing {len(group_steps)} step(s) in parallel[/cyan]")

            # Build tasks for parallel execution
            tasks = [(step.agent_type, step.description) for step in group_steps]

            # Execute in parallel
            results = await self._spawn_agents_parallel(tasks)

            # Store results
            for step, (success, output) in zip(group_steps, results):
                step_results[step.step_num] = (success, output)
                status = "[green]✓[/green]" if success else "[red]✗[/red]"
                console.print(f"  {status} Step {step.step_num}: {step.description[:50]}...")
                all_outputs.append(f"### Step {step.step_num}: {step.description}\n{output}")

        # Combine outputs
        combined = "\n\n".join(all_outputs)

        # Review the overall results
        return await self._review_plan_execution(user_request, plan, combined, step_results)

    async def _review_plan_execution(
        self,
        user_request: str,
        plan,
        combined_output: str,
        step_results: Dict[int, Tuple[bool, str]]
    ) -> str:
        """Review the results of plan execution."""
        failed_steps = [num for num, (success, _) in step_results.items() if not success]

        if failed_steps:
            console.print(f"[yellow]> {len(failed_steps)} step(s) failed, reviewing...[/yellow]")

        # Use foreman to review and potentially fix
        return await self._review_and_supervise(
            user_request,
            "plan_execution",
            combined_output,
            len(failed_steps) == 0,
            round_num=1
        )

    def _parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls from response text."""
        tool_calls = []
        valid_tools = {"spawn_explorer", "spawn_executor", "spawn_planner", "spawn_researcher"}

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

    async def _call_llm(self, messages: List[Message], use_tools: bool = True, timeout: float = 60.0) -> Tuple[str, List[Dict]]:
        """Call the LLM and return response text and tool calls."""
        response_text = ""
        tool_calls = []

        try:
            async with asyncio.timeout(timeout):
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
        except asyncio.TimeoutError:
            console.print("[yellow]LLM response timed out[/yellow]")
            return "", []
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return "", []

        # Try parsing tool calls from text if none structured
        if not tool_calls:
            tool_calls = self._parse_tool_calls(response_text)

        # Check for agent keywords in response - more robust detection
        if not tool_calls:
            response_lower = response_text.lower()

            # Check for explicit function mentions
            if "spawn_planner" in response_lower or "planner agent" in response_lower:
                tool_calls = [{"name": "spawn_planner", "arguments": {"task": ""}}]
            elif "spawn_researcher" in response_lower or "researcher agent" in response_lower:
                tool_calls = [{"name": "spawn_researcher", "arguments": {"task": ""}}]
            elif "spawn_explorer" in response_lower or "explorer agent" in response_lower:
                tool_calls = [{"name": "spawn_explorer", "arguments": {"task": ""}}]
            elif "spawn_executor" in response_lower or "executor agent" in response_lower:
                tool_calls = [{"name": "spawn_executor", "arguments": {"task": ""}}]
            # Detect action-oriented language that needs executor
            elif any(kw in response_lower for kw in [
                "create the file", "write the file", "create a file",
                "write to file", "creating file", "writing file",
                "let me create", "i'll create", "i will create",
                "let me write", "i'll write", "i will write",
                "execute", "run the command", "run command",
                "add the file", "make the file",
            ]):
                tool_calls = [{"name": "spawn_executor", "arguments": {"task": ""}}]
            # Detect read/search oriented language that needs explorer
            elif any(kw in response_lower for kw in [
                "let me search", "let me look", "let me find",
                "searching for", "looking for", "i'll search",
                "read the file", "check the file", "examine",
            ]):
                tool_calls = [{"name": "spawn_explorer", "arguments": {"task": ""}}]
            # Detect research/documentation lookup language
            elif any(kw in response_lower for kw in [
                "search the web", "web search", "look up documentation",
                "find documentation", "research this", "let me research",
                "i'll look up", "search online", "check the docs",
            ]):
                tool_calls = [{"name": "spawn_researcher", "arguments": {"task": ""}}]

        return response_text, tool_calls

    def _detect_user_intent(self, user_message: str) -> Optional[str]:
        """
        Detect user intent from their message to determine which agent to spawn.

        This is a fallback when the LLM doesn't properly call tools.

        Args:
            user_message: The user's message

        Returns:
            Agent name to spawn, or None if unclear
        """
        msg_lower = user_message.lower()

        # Research patterns FIRST - check before executor to avoid false positives
        # (e.g., "documentation for pytest" shouldn't trigger "pytest" -> executor)
        if any(kw in msg_lower for kw in [
            "how do i ", "how to ", "what is ", "explain ",
            "documentation", "docs for ", "tutorial ",
            "research ", "look up ",
        ]):
            return "spawn_researcher"

        # Complex task patterns -> planner
        if any(kw in msg_lower for kw in [
            "implement ", "build a ", "create a system",
            "refactor ", "redesign ", "architect ",
        ]):
            return "spawn_planner"

        # File creation/writing patterns -> executor
        # Check for "write/create ... file/script" pattern with anything in between
        import re
        if re.search(r'\b(create|write|make|add)\s+(?:a\s+)?(?:\w+\s+)?(file|script)\b', msg_lower):
            return "spawn_executor"
        # Check for file extension patterns like "testing.py", "hello.sh"
        if re.search(r'\b\w+\.(py|js|ts|sh|bash|rb|go|rs|java|c|cpp|h|txt|json|yaml|yml|md|html|css)\b', msg_lower):
            # Has a file extension mentioned - likely wants to create/edit
            if any(kw in msg_lower for kw in ["write", "create", "make", "add", "generate"]):
                return "spawn_executor"
        if any(kw in msg_lower for kw in [
            "save to file", "save file", "new file", "touch ", "echo ",
        ]):
            return "spawn_executor"

        # Code execution patterns -> executor
        if any(kw in msg_lower for kw in [
            "run ", "execute ", "install ", "build ", "compile ",
            "test ", "pytest", "npm ", "pip ", "cargo ",
        ]):
            return "spawn_executor"

        # File editing patterns -> executor
        if any(kw in msg_lower for kw in [
            "edit ", "modify ", "change ", "update ", "fix ",
            "add to ", "remove from ", "delete from ",
        ]):
            return "spawn_executor"

        # Reading/exploring patterns -> explorer
        if any(kw in msg_lower for kw in [
            "read ", "show ", "display ", "what's in ", "what is in ",
            "find ", "search ", "look for ", "where is ",
            "list ", "ls ", "cat ",
        ]):
            return "spawn_explorer"

        return None

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
        3. Use planner for complex tasks, then execute with parallel agents
        """
        messages = [
            Message(role="system", content=self.system_prompt),
        ]
        messages.extend(self.conversation_history[-10:])
        messages.append(Message(role="user", content=user_message))

        console.print("[dim]Routing request...[/dim]")

        try:
            response_text, tool_calls = await self._call_llm(messages)

            # Debug: show what we got back
            if response_text:
                console.print(f"[dim]LLM response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}[/dim]")

            # If no tool calls from LLM, try to detect intent from user message
            if not tool_calls:
                user_intent = self._detect_user_intent(user_message)
                if user_intent:
                    console.print(f"[dim](detected intent: {user_intent})[/dim]")
                    tool_calls = [{"name": user_intent, "arguments": {"task": user_message}}]

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

                if name == "spawn_planner":
                    # Get a plan first
                    success, plan_output = await self._spawn_agent("planner", task)

                    if success:
                        # Parse and execute the plan
                        planner = self._get_planner_agent()
                        plan = planner._parse_plan(plan_output)

                        if plan.steps:
                            console.print(f"\n[bold]Plan created ({plan.complexity} complexity, {len(plan.steps)} steps)[/bold]")
                            console.print(f"[dim]{plan.analysis}[/dim]\n")

                            final_response = await self._execute_plan(plan, user_message)
                        else:
                            final_response = f"Plan created but no executable steps found:\n{plan_output}"
                    else:
                        final_response = f"Planning failed: {plan_output}"

                elif name == "spawn_explorer":
                    success, output = await self._spawn_agent("explorer", task)
                    final_response = await self._review_and_supervise(
                        user_message, "explorer", output, success, round_num=1
                    )
                elif name == "spawn_executor":
                    success, output = await self._spawn_agent("executor", task)
                    final_response = await self._review_and_supervise(
                        user_message, "executor", output, success, round_num=1
                    )
                elif name == "spawn_researcher":
                    success, output = await self._spawn_agent("researcher", task)
                    final_response = await self._review_and_supervise(
                        user_message, "researcher", output, success, round_num=1
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

    def get_agent_status(self) -> Dict:
        """Get current agent concurrency status."""
        return {
            "active_agents": self.agent_semaphore.active_agents,
            "available_slots": self.agent_semaphore.available_slots,
            "max_concurrent": self.agent_semaphore._max,
        }
