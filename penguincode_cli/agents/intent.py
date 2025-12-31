"""User intent detection for the chat agent."""

import re
from typing import Optional


def detect_user_intent(user_message: str) -> Optional[str]:
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


def estimate_complexity(task: str) -> str:
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
