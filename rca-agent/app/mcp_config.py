"""
MCP server configuration management for RCA Agent
Supports Jira, Postgres, and Redis integrations
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MCPConfiguration:
    """Centralized MCP server configuration management for RCA agent."""

    def __init__(self):
        self.config_cache: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self):
        """Load and validate MCP server configurations for RCA agent."""
        config = {}

        # Configure Jira MCP server if available
        if os.getenv("JIRA_MCP_SERVER"):
            config["jira-mcp"] = {
                'url': os.getenv("JIRA_MCP_SERVER"),
                'transport': 'sse',
                'headers': {
                    "X-JIRA-Url": os.getenv("JIRA_URL", "https://your-company.atlassian.net"),
                    "X-JIRA-Username": os.getenv("JIRA_USERNAME", "your-email@company.com"),
                    "X-JIRA-Api-Token": os.getenv("JIRA_API_TOKEN", "your-jira-api-token"),
                }
            }
            logger.info("Jira MCP server configured for RCA agent")

        # Configure Postgres MCP server if available
        if os.getenv("POSTGRES_MCP_SERVER"):
            config["postgres-mcp"] = {
                'url': os.getenv("POSTGRES_MCP_SERVER"),
                'transport': 'sse',
                'headers': {
                    "X-DB-Host": os.getenv("POSTGRES_HOST", "localhost"),
                    "X-DB-Port": os.getenv("POSTGRES_PORT", "5432"),
                    "X-DB-Name": os.getenv("POSTGRES_DB", "postgres"),
                    "X-DB-User": os.getenv("POSTGRES_USER", "postgres"),
                    "X-DB-Password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                }
            }
            logger.info("Postgres MCP server configured for RCA agent")


        if not config:
            logger.warning("No MCP servers configured - RCA agent will run with limited functionality")

        self.config_cache = config
        logger.info(f"RCA agent MCP configuration loaded with {len(config)} servers")

    def get_config(self) -> Dict[str, Any]:
        """Get the complete MCP server configuration."""
        if self.config_cache is None:
            self._load_config()
        return self.config_cache or {}

    def get_enabled_servers(self) -> list:
        """Get list of enabled MCP server names."""
        config = self.get_config()
        return list(config.keys())

    def reload_config(self):
        """Reload configuration from environment variables."""
        logger.info("Reloading MCP server configuration")
        self.config_cache = None
        self._load_config()


# Global configuration instance
mcp_config = MCPConfiguration()


def get_mcp_config() -> Dict[str, Any]:
    """Get MCP server configuration in the format expected by MultiServerMCPClient."""
    return mcp_config.get_config()


def get_mcp_health() -> Dict[str, Any]:
    """Get health status of MCP servers for RCA agent."""
    servers = {
        "jira-mcp": {
            "enabled": bool(os.getenv("JIRA_MCP_SERVER")),
            "url": os.getenv("JIRA_MCP_SERVER", "Not configured"),
            "transport": "sse",
            "status": "configured" if os.getenv("JIRA_MCP_SERVER") else "not configured",
            "credentials": "headers" if os.getenv("JIRA_MCP_SERVER") else "none"
        },
        "postgres-mcp": {
            "enabled": bool(os.getenv("POSTGRES_MCP_SERVER")),
            "url": os.getenv("POSTGRES_MCP_SERVER", "Not configured"),
            "transport": "sse",
            "status": "configured" if os.getenv("POSTGRES_MCP_SERVER") else "not configured",
            "credentials": "headers" if os.getenv("POSTGRES_MCP_SERVER") else "none"
        }
    }

    return {
        "servers": servers,
        "total_enabled": sum(1 for s in servers.values() if s["enabled"]),
        "total_configured": len(servers)
    }