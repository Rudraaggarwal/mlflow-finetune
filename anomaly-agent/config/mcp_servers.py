"""
MCP server configuration management for Grafana, Jira, and Postgres integrations.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


def get_mcp_config() -> Dict[str, Any]:
    """Get MCP server configuration in the format expected by MultiServerMCPClient."""
    # Return hardcoded configuration with transport for now
    import os
    config = {}
    
    # Only include servers that have URLs configured
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
    
    if os.getenv("GRAFANA_MCP_SERVER"):
        config["grafana-mcp"] = {
            'url': os.getenv("GRAFANA_MCP_SERVER"),
            'transport': 'sse',
            'headers': {
                "X-Grafana-URL": os.getenv("GRAFANA_URL", "https://your-grafana.com"),
                "X-Grafana-API-Key": os.getenv("GRAFANA_API_KEY", "your-grafana-api-key"),
            }
        }
    
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
    
    return config



def get_mcp_health() -> Dict[str, Any]:
    """Get health status of MCP servers."""
    import os
    servers = {
        "jira-mcp": {
            "enabled": bool(os.getenv("JIRA_MCP_SERVER")),
            "url": os.getenv("JIRA_MCP_SERVER", "Not configured"),
            "transport": "sse",
            "status": "configured" if os.getenv("JIRA_MCP_SERVER") else "not configured",
            "credentials": "headers" if os.getenv("JIRA_MCP_SERVER") else "none"
        },
        "grafana-mcp": {
            "enabled": bool(os.getenv("GRAFANA_MCP_SERVER")),
            "url": os.getenv("GRAFANA_MCP_SERVER", "Not configured"),
            "transport": "sse", 
            "status": "configured" if os.getenv("GRAFANA_MCP_SERVER") else "not configured",
            "credentials": "headers" if os.getenv("GRAFANA_MCP_SERVER") else "none"
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