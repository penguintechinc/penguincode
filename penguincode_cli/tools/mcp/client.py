"""MCP (Model Context Protocol) client wrapper for search engines."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx


class MCPClient:
    """
    Client for communicating with MCP servers.

    MCP servers run as separate processes and expose tools via stdio or HTTP.
    """

    def __init__(self, server_command: str, server_args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP client.

        Args:
            server_command: Command to start MCP server (e.g., "npx", "uvx")
            server_args: Arguments for server command
            env: Environment variables for server
        """
        self.server_command = server_command
        self.server_args = server_args
        self.env = env or {}
        self.process = None

    async def start(self):
        """Start the MCP server process."""
        if self.process:
            return

        self.process = await asyncio.create_subprocess_exec(
            self.server_command,
            *self.server_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**asyncio.subprocess.os.environ, **self.env},
        )

    async def stop(self):
        """Stop the MCP server process."""
        if not self.process:
            return

        self.process.terminate()
        await self.process.wait()
        self.process = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            RuntimeError: If server is not started or call fails
        """
        if not self.process:
            raise RuntimeError("MCP server not started")

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        # Send request to server via stdin
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        # Read response from stdout
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP tool call error: {response['error']}")

        return response.get("result")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from MCP server.

        Returns:
            List of tool definitions
        """
        if not self.process:
            raise RuntimeError("MCP server not started")

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        }

        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP list tools error: {response['error']}")

        return response.get("result", {}).get("tools", [])


class HTTPMCPClient:
    """
    HTTP-based MCP client for servers that expose HTTP endpoints.

    Alternative to stdio-based MCP for servers running as HTTP services.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize HTTP MCP client.

        Args:
            base_url: Base URL of MCP server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool via HTTP.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"MCP HTTP error: {response.status_code} - {response.text}")

            result = response.json()
            return result.get("result")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools via HTTP.

        Returns:
            List of tool definitions
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/tools/list")

            if response.status_code != 200:
                raise RuntimeError(f"MCP HTTP error: {response.status_code} - {response.text}")

            result = response.json()
            return result.get("tools", [])
