import asyncio
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from .mcp_config import get_mcp_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainMCPClient:
    """LangChain MCP Client for SSE connections using official adapters."""

    def __init__(self, sse_url: str):
        self.sse_url = sse_url
        self.client = None
        self.tools = []
        self.agent = None
        self.session = None

    async def connect(self):
        """Connect to MCP server via LangChain adapter using proper configuration with headers."""
        logger.info(f"🔄 Starting MCP client connection process")

        try:
            # Get the proper MCP configuration with headers
            logger.info("📋 Loading MCP configuration...")
            mcp_config = get_mcp_config()

            if not mcp_config:
                logger.error("❌ No MCP servers configured. Please check environment variables.")
                raise RuntimeError("No MCP servers configured")

            logger.info(f"✅ Found {len(mcp_config)} MCP server(s): {list(mcp_config.keys())}")
            logger.info("📊 MCP Configuration Details:")
            for server_name, server_config in mcp_config.items():
                logger.info(f"  - {server_name}:")
                logger.info(f"    URL: {server_config.get('url', 'Not set')}")
                logger.info(f"    Transport: {server_config.get('transport', 'Not set')}")
                logger.info(f"    Headers: {list(server_config.get('headers', {}).keys())}")

            logger.info("🔌 Initializing MultiServerMCPClient...")
            self.client = MultiServerMCPClient(mcp_config)
            logger.info("✅ MultiServerMCPClient initialized successfully")

            logger.info("🔧 Getting MCP tools...")
            self.tools = await self.client.get_tools()
            logger.info(f"✅ Connected successfully. Loaded {len(self.tools)} tools")

            if not self.tools:
                logger.warning("⚠️ No tools loaded from MCP servers")

        except Exception as e:
            logger.error(f"❌ Connection failed at step: {type(e).__name__}")
            logger.error(f"❌ Error details: {e}")
            import traceback
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            raise

    async def close(self):
        """Close the MCP client connection"""
        if self.client:
            try:
                await self.client.close()
                logger.info("MCP client connection closed")
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")

    def get_tools(self) -> List[Any]:
        """Get available MCP tools"""
        return self.tools if self.tools else []

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a specific MCP tool"""
        if not self.client:
            raise RuntimeError("MCP client not connected")

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = await tool.acall(kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {e}")
                    raise

        raise ValueError(f"Tool {tool_name} not found")

    async def call_tool_direct(self, tool_name: str, arguments: dict = None):
        """Call a tool directly with enhanced error handling and logging."""
        if not self.tools:
            logger.error("❌ No tools available - MCP client may not be connected properly")
            raise RuntimeError("No tools available - MCP client may not be connected properly")

        if arguments is None:
            arguments = {}

        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            logger.error(f"❌ Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}")
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}")

        try:
            # Try with timeout to avoid hanging
            import asyncio
            result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=60.0)
            return result

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Tool '{tool_name}' call timed out after 60 seconds")
            raise Exception(f"Tool '{tool_name}' call timed out - the MCP server may be unresponsive")

        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' call failed: {type(e).__name__}: {e}")
            logger.error(f"   Arguments were: {json.dumps(arguments, default=str, indent=2)}")

            # If it's a TaskGroup error, suggest reconnection
            if "TaskGroup" in str(e) or "unhandled errors" in str(e):
                raise Exception(f"Connection error occurred during '{tool_name}' execution. Try reconnecting. Error: {e}")
            else:
                raise Exception(f"Tool '{tool_name}' execution failed: {e}")

    async def list_tools(self):
        """List available tools."""
        if not self.tools:
            return []

        tool_info = []
        for tool in self.tools:
            tool_info.append({
                'name': tool.name,
                'description': tool.description,
                'args': getattr(tool, 'args_schema', None)
            })
        return tool_info