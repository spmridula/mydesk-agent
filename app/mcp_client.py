"""Thin MCP client: discovers tools from the server and calls them over stdio."""
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app import config


def _params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(config.MCP_SERVER_PATH)],
        env={"TASKS_DB": str(config.TASKS_DB)},
    )


async def list_tools() -> list[dict]:
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": t.name, "description": t.description, "schema": t.inputSchema}
                for t in result.tools
            ]


async def call_tool(name: str, arguments: dict) -> str:
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return "\n".join(c.text for c in result.content if getattr(c, "text", None))
