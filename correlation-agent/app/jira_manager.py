"""
JIRA Manager for Correlation Agent.
Handles all JIRA comment operations with retry logic.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from langfuse import get_client

from .langfuse_prompts import get_correlation_prompt
from .logging_utils import get_timestamp
from .utils import sanitize_unicode

logger = logging.getLogger(__name__)
langfuse = get_client()


class JiraManager:
    """Manages JIRA ticket comment operations."""

    def __init__(self, mcp_client=None, llm=None, langfuse_handler=None):
        """Initialize JIRA manager with MCP client and LLM."""
        self.mcp_client = mcp_client
        self.llm = llm
        self.langfuse_handler = langfuse_handler

    def _check_jira_tool_available(self) -> bool:
        """Check if JIRA add comment tool is available."""
        if not self.mcp_client:
            logger.warning("No MCP client available for JIRA operations")
            return False

        available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
        tool_names = [tool.name for tool in available_tools] if available_tools else []

        has_jira_tool = any('jira_add_comment' in tool_name for tool_name in tool_names)

        if not has_jira_tool:
            logger.warning("jira_add_comment tool not available")
            return False

        return True

    async def _generate_jira_comment(
        self,
        analysis_type: str,
        analysis_content: str,
        alert_name: str,
        severity: str
    ) -> str:
        """Generate JIRA comment using LLM."""
        with langfuse.start_as_current_span(name="llm-jira-comment-generation") as span:
            try:
                # Prepare variables
                jira_variables = {
                    "analysis_type": analysis_type,
                    "alert_name": alert_name,
                    "severity": severity,
                    "analysis_content": analysis_content,
                    "title": analysis_type.title()
                }

                # Get JIRA formatter prompt from Langfuse
                jira_formatter_prompt = get_correlation_prompt("jira-formatter", jira_variables)

                user_prompt = f"**Task:** Create focused markdown comment showing ONLY the {analysis_type} analysis results based on the provided content and context."

                # Generate comment using LLM
                response = await self.llm.ainvoke(
                    [
                        {"role": "system", "content": jira_formatter_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                )

                # Format with footer
                markdown_comment = f"""{response.content}

---
*{analysis_type.title()} analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"""

                # Sanitize Unicode
                sanitized_comment = sanitize_unicode(markdown_comment)

                span.update(
                    output={
                        "comment_generated": True,
                        "comment_length": len(sanitized_comment)
                    },
                    metadata={"status": "success"}
                )

                return sanitized_comment

            except Exception as e:
                logger.error(f"Failed to generate JIRA comment: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                # Return fallback comment
                return self._create_fallback_comment(analysis_type, analysis_content)

    def _create_fallback_comment(self, analysis_type: str, analysis_content: str) -> str:
        """Create fallback JIRA comment if LLM generation fails."""
        return f"""## {analysis_type.title()} Analysis

{analysis_content[:1000]}

---
*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"""

    async def add_analysis_comment(
        self,
        state: Dict[str, Any],
        analysis_type: str,
        analysis_content: str
    ) -> bool:
        """Add JIRA comment for specific analysis step with retry logic."""
        with langfuse.start_as_current_span(name="add-jira-comment-analysis") as span:
            try:
                # Get JIRA ticket ID from alert payload
                alert_payload = state.get("alert_payload", {})
                jira_ticket_id = alert_payload.get("jira_ticket_id")

                if not jira_ticket_id:
                    logger.warning(f"No JIRA ticket ID found for {analysis_type} analysis comment")
                    return False

                # Check if JIRA tool is available
                if not self._check_jira_tool_available():
                    return False

                logger.info(f"Adding {analysis_type} comment to JIRA ticket: {jira_ticket_id}")

                # Generate comment
                comment = await self._generate_jira_comment(
                    analysis_type,
                    analysis_content,
                    state.get("alertname", "Unknown Alert"),
                    state.get("severity", "Unknown")
                )

                # Prepare parameters
                jira_params = {
                    "issue_key": jira_ticket_id,
                    "comment": comment
                }

                # Add comment with retry logic
                max_retries = 3
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        with langfuse.start_as_current_span(name=f"jira-add-comment-attempt-{retry_count + 1}") as tool_span:
                            result = await self.mcp_client.call_tool_direct("jira_add_comment", jira_params)
                            success = True

                            tool_span.update(
                                input={
                                    "tool_name": "jira_add_comment",
                                    "issue_key": jira_ticket_id,
                                    "attempt": retry_count + 1
                                },
                                output={"execution_successful": True},
                                metadata={"tool_type": "mcp_jira_comment"}
                            )

                            logger.info(f"Successfully added {analysis_type} comment to JIRA (attempt {retry_count + 1})")

                    except Exception as retry_error:
                        retry_count += 1
                        logger.warning(f"JIRA comment attempt {retry_count} failed: {retry_error}")

                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        else:
                            logger.error(f"All {max_retries} JIRA comment attempts failed")
                            raise retry_error

                span.update(
                    output={
                        "comment_added": success,
                        "jira_ticket_id": jira_ticket_id,
                        "analysis_type": analysis_type,
                        "attempts_made": retry_count + (1 if success else 0)
                    },
                    metadata={"status": "success" if success else "failed"}
                )

                return success

            except Exception as e:
                logger.error(f"Failed to add {analysis_type} JIRA comment: {e}")
                span.update(
                    output={"error": str(e), "comment_added": False},
                    metadata={"status": "error"}
                )
                return False
