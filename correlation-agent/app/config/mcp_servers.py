import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_mcp_config() -> Dict[str, Dict[str, Any]]:
    """
    Get MCP server configuration from environment variables in the format expected by MultiServerMCPClient.
    This matches the format used in the anomaly agent.

    Returns:
        Dict[str, Dict[str, Any]]: MCP server configuration with transport keys
    """
    config = {}

    # Check for Grafana MCP server configuration
    grafana_url = os.getenv("GRAFANA_MCP_SERVER")  # Changed to match anomaly agent env var
    grafana_api_key = os.getenv("GRAFANA_API_KEY")
    grafana_url_base = os.getenv("GRAFANA_URL")

    if grafana_url:
        config["grafana-mcp"] = {
            'url': grafana_url,
            'transport': 'sse',
            'headers': {
                "X-Grafana-URL": grafana_url_base or "https://your-grafana.com",
                "X-Grafana-API-Key": grafana_api_key or "your-grafana-api-key",
            }
        }
        logger.info(f"Added Grafana MCP server to configuration: {grafana_url}")

    # Check for Jira MCP server configuration
    jira_url = os.getenv("JIRA_MCP_SERVER")
    jira_username = os.getenv("JIRA_USERNAME")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_url_base = os.getenv("JIRA_URL")

    if jira_url:
        config["jira-mcp"] = {
            'url': jira_url,
            'transport': 'sse',
            'headers': {
                "X-JIRA-Url": jira_url_base or "https://your-company.atlassian.net",
                "X-JIRA-Username": jira_username or "your-email@company.com",
                "X-JIRA-Api-Token": jira_api_token or "your-jira-api-token",
            }
        }
        logger.info(f"Added Jira MCP server to configuration: {jira_url}")

    # Check for Postgres MCP server configuration
    postgres_url = os.getenv("POSTGRES_MCP_SERVER")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    if postgres_url:
        config["postgres-mcp"] = {
            'url': postgres_url,
            'transport': 'sse',
            'headers': {
                "X-DB-Host": postgres_host or "localhost",
                "X-DB-Port": postgres_port or "5432",
                "X-DB-Name": postgres_db or "postgres",
                "X-DB-User": postgres_user or "postgres",
                "X-DB-Password": postgres_password or "postgres",
            }
        }
        logger.info(f"Added Postgres MCP server to configuration: {postgres_url}")

    if not config:
        logger.warning("No MCP servers configured. Please set environment variables like GRAFANA_MCP_SERVER, JIRA_MCP_SERVER, POSTGRES_MCP_SERVER")
    else:
        logger.info(f"MCP configuration created with {len(config)} servers: {list(config.keys())}")

    return config

def get_mcp_health() -> Dict[str, str]:
    """
    Check health of configured MCP servers

    Returns:
        Dict[str, str]: Health status of each MCP server
    """
    health = {}
    config = get_mcp_config()

    for server_name in config.keys():
        # Basic health check - just verify configuration exists
        health[server_name] = "configured"

    return health