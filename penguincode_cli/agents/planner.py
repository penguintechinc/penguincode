"""Planner agent - breaks down complex tasks into actionable plans.

The planner analyzes complex user requests and creates structured plans
that can be executed by other agents (explorer, executor).
"""

from dataclasses import dataclass
from typing import List, Optional

from penguincode_cli.ollama import Message, OllamaClient
from penguincode_cli.ui import console

from .base import AgentConfig, AgentResult, Permission


PLANNER_SYSTEM_PROMPT = """You are a planning agent for PenguinCode. Your job is to analyze complex requests and break them down into clear, actionable steps.

When given a task, create a structured plan with:

1. **Analysis**: Brief understanding of what needs to be done
2. **Steps**: Numbered list of specific, actionable steps
3. **Agent assignments**: For each step, specify which agent should handle it:
   - `explorer` - for reading, searching, understanding code
   - `executor` - for writing, editing, running commands
4. **Dependencies**: Note which steps depend on others (can run in parallel vs sequential)
5. **Estimated complexity**: simple | moderate | complex

Output your plan in this format:

```plan
ANALYSIS: <brief description of the task>

STEPS:
1. [explorer] <step description>
2. [executor] <step description>
3. [explorer|executor] <step description> (depends on: 1, 2)
...

PARALLEL_GROUPS:
- Group 1: steps 1, 2 (can run together)
- Group 2: step 3 (after group 1)
...

COMPLEXITY: <simple|moderate|complex>
```

Be thorough but concise. Each step should be specific enough for an agent to execute independently.
"""


@dataclass
class PlanStep:
    """A single step in a plan."""
    step_num: int
    agent_type: str  # "explorer" or "executor"
    description: str
    depends_on: List[int]  # Step numbers this depends on


@dataclass
class Plan:
    """A structured plan for executing a complex task."""
    analysis: str
    steps: List[PlanStep]
    parallel_groups: List[List[int]]  # Groups of step numbers that can run in parallel
    complexity: str  # simple, moderate, complex
    raw_output: str  # Original LLM output


class PlannerAgent:
    """Agent that creates execution plans for complex tasks."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "deepseek-coder:6.7b",
    ):
        self.client = ollama_client
        self.model = model
        self.config = AgentConfig(
            name="planner",
            model=model,
            description="Breaks down complex tasks into actionable plans",
            permissions=[],  # Planner doesn't need tools, just thinks
            system_prompt=PLANNER_SYSTEM_PROMPT,
            max_iterations=1,
        )

    async def create_plan(self, task: str, context: str = "") -> Plan:
        """
        Create a plan for a complex task.

        Args:
            task: The task to plan
            context: Optional context about the codebase or previous work

        Returns:
            Structured Plan object
        """
        console.print("[cyan]> Planning task...[/cyan]")

        messages = [
            Message(role="system", content=PLANNER_SYSTEM_PROMPT),
        ]

        if context:
            messages.append(Message(
                role="user",
                content=f"Context:\n{context}\n\nTask to plan:\n{task}"
            ))
        else:
            messages.append(Message(role="user", content=f"Task to plan:\n{task}"))

        # Get plan from LLM
        response_text = ""
        async for chunk in self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        ):
            if chunk.message and chunk.message.content:
                response_text += chunk.message.content

        # Parse the plan
        plan = self._parse_plan(response_text)
        return plan

    def _parse_plan(self, raw_output: str) -> Plan:
        """Parse LLM output into a structured Plan."""
        lines = raw_output.split("\n")

        analysis = ""
        steps: List[PlanStep] = []
        parallel_groups: List[List[int]] = []
        complexity = "moderate"

        current_section = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("ANALYSIS:"):
                current_section = "analysis"
                analysis = line_stripped[9:].strip()
            elif line_stripped.startswith("STEPS:"):
                current_section = "steps"
            elif line_stripped.startswith("PARALLEL_GROUPS:"):
                current_section = "parallel"
            elif line_stripped.startswith("COMPLEXITY:"):
                complexity = line_stripped[11:].strip().lower()
                if complexity not in ["simple", "moderate", "complex"]:
                    complexity = "moderate"
            elif current_section == "analysis" and line_stripped and not line_stripped.startswith(("STEPS", "PARALLEL", "COMPLEXITY")):
                analysis += " " + line_stripped
            elif current_section == "steps" and line_stripped:
                step = self._parse_step(line_stripped, len(steps) + 1)
                if step:
                    steps.append(step)
            elif current_section == "parallel" and line_stripped.startswith("- Group"):
                group = self._parse_parallel_group(line_stripped)
                if group:
                    parallel_groups.append(group)

        # If no parallel groups defined, create default sequential groups
        if not parallel_groups and steps:
            parallel_groups = [[s.step_num] for s in steps]

        return Plan(
            analysis=analysis.strip(),
            steps=steps,
            parallel_groups=parallel_groups,
            complexity=complexity,
            raw_output=raw_output,
        )

    def _parse_step(self, line: str, default_num: int) -> Optional[PlanStep]:
        """Parse a single step line."""
        # Expected format: "1. [explorer] description (depends on: 1, 2)"
        import re

        # Try to extract step number
        num_match = re.match(r"(\d+)\.", line)
        step_num = int(num_match.group(1)) if num_match else default_num

        # Extract agent type
        agent_match = re.search(r"\[(explorer|executor)\]", line.lower())
        agent_type = agent_match.group(1) if agent_match else "executor"

        # Extract dependencies
        depends_match = re.search(r"\(depends on:\s*([\d,\s]+)\)", line.lower())
        depends_on = []
        if depends_match:
            deps_str = depends_match.group(1)
            depends_on = [int(d.strip()) for d in deps_str.split(",") if d.strip().isdigit()]

        # Extract description (remove step number, agent type, and dependencies)
        description = line
        description = re.sub(r"^\d+\.\s*", "", description)
        description = re.sub(r"\[(explorer|executor)\]\s*", "", description, flags=re.IGNORECASE)
        description = re.sub(r"\(depends on:[^)]+\)", "", description, flags=re.IGNORECASE)
        description = description.strip()

        if not description:
            return None

        return PlanStep(
            step_num=step_num,
            agent_type=agent_type,
            description=description,
            depends_on=depends_on,
        )

    def _parse_parallel_group(self, line: str) -> List[int]:
        """Parse a parallel group line."""
        import re
        # Expected format: "- Group 1: steps 1, 2 (can run together)"
        nums = re.findall(r"\d+", line.split(":")[1] if ":" in line else line)
        return [int(n) for n in nums]

    async def run(self, task: str, **kwargs) -> AgentResult:
        """Run the planner on a task."""
        try:
            context = kwargs.get("context", "")
            plan = await self.create_plan(task, context)

            # Format plan as readable output
            output_lines = [
                f"## Plan Analysis\n{plan.analysis}\n",
                "## Steps",
            ]

            for step in plan.steps:
                deps = f" (after steps {', '.join(map(str, step.depends_on))})" if step.depends_on else ""
                output_lines.append(f"{step.step_num}. [{step.agent_type}] {step.description}{deps}")

            output_lines.append(f"\n## Execution Groups (parallel)")
            for i, group in enumerate(plan.parallel_groups, 1):
                output_lines.append(f"- Group {i}: steps {', '.join(map(str, group))}")

            output_lines.append(f"\n## Complexity: {plan.complexity}")

            return AgentResult(
                agent_name="planner",
                success=True,
                output="\n".join(output_lines),
            )
        except Exception as e:
            return AgentResult(
                agent_name="planner",
                success=False,
                output="",
                error=str(e),
            )
