"""
Grafana Helper for Correlation Agent.
Handles Grafana alert UID extraction and alert info fetching.
"""

import logging
from typing import Dict, Any, Optional
from langfuse import get_client

logger = logging.getLogger(__name__)
langfuse = get_client()


class GrafanaHelper:
    """Helper for Grafana operations."""

    def __init__(self, llm=None, mcp_client=None, langfuse_handler=None):
        """Initialize Grafana helper."""
        self.llm = llm
        self.mcp_client = mcp_client
        self.langfuse_handler = langfuse_handler

    async def extract_grafana_uid_with_llm(self, alert_payload: dict) -> str:
        """Use LLM to extract Grafana alert UID from complex alert payload."""
        with langfuse.start_as_current_span(name="llm-uid-extraction") as span:
            try:
                logger.info("Using LLM to extract Grafana alert UID")

                span.update(
                    input={"payload_size": len(str(alert_payload)) if alert_payload else 0},
                    metadata={"component": "llm_uid_extraction"}
                )

                extraction_prompt = f"""You are an expert at parsing Grafana alert payloads.

Extract the Grafana alert UID (fingerprint) from this alert payload. The UID is typically a long hexadecimal string.

Alert Payload:
{alert_payload}

Return ONLY the UID string, nothing else."""

                response = await self.llm.ainvoke(
                    [{"role": "user", "content": extraction_prompt}],
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent", "grafana_uid_extraction"]
                        }
                    }
                )

                extracted_uid = response.content.strip()

                span.update(
                    output={
                        "uid_extracted": bool(extracted_uid),
                        "uid_length": len(extracted_uid) if extracted_uid else 0
                    },
                    metadata={"status": "success"}
                )

                logger.info(f"Extracted Grafana UID: {extracted_uid[:20]}...")
                return extracted_uid

            except Exception as e:
                logger.error(f"Failed to extract Grafana UID with LLM: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return ""

    async def fetch_grafana_alert_info(self, alert_uid: str) -> dict:
        """Fetch detailed alert rule information from Grafana via MCP."""
        with langfuse.start_as_current_span(name="fetch-grafana-alert-info") as span:
            try:
                logger.info(f"Fetching Grafana alert info for UID: {alert_uid}")

                span.update(
                    input={"alert_uid": alert_uid},
                    metadata={"component": "grafana_alert_fetch"}
                )

                if not self.mcp_client:
                    logger.warning("No MCP client available for Grafana operations")
                    return {}

                # Check if Grafana tools are available
                available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
                tool_names = [tool.name for tool in available_tools] if available_tools else []

                has_grafana_tool = any('grafana' in tool_name.lower() for tool_name in tool_names)

                if not has_grafana_tool:
                    logger.warning("Grafana tools not available in MCP client")
                    return {}

                # Call Grafana MCP tool to get alert rule info
                result = await self.mcp_client.call_tool_direct(
                    "grafana_get_alert_rule",
                    {"uid": alert_uid}
                )

                if result and isinstance(result, dict):
                    logger.info("Successfully fetched Grafana alert info")
                    span.update(
                        output={"alert_info_fetched": True, "keys": list(result.keys())},
                        metadata={"status": "success"}
                    )
                    return result
                else:
                    logger.warning("No alert info returned from Grafana")
                    return {}

            except Exception as e:
                logger.error(f"Failed to fetch Grafana alert info: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return {}
