"""
Workflow Completion Handler for Correlation Agent.
Handles RCA analysis delegation and workflow completion.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from langfuse import get_client

from .logging_utils import get_timestamp
from .utils import get_completion_stats

logger = logging.getLogger(__name__)
langfuse = get_client()


class WorkflowCompletionHandler:
    """Handles RCA delegation and workflow completion."""

    def __init__(self):
        """Initialize workflow completion handler."""
        pass

    async def delegate_to_rca_agent(
        self,
        state: Dict[str, Any],
        current_trace_id: str = None,
        current_observation_id: str = None,
        global_session_id: str = None
    ) -> Dict[str, Any]:
        """Delegate to RCA agent for root cause analysis."""
        with langfuse.start_as_current_span(name="rca-analysis-delegation") as span:
            try:
                logger.info("Calling RCA agent for analysis")

                # Import RCA agent tool
                try:
                    from .tools.call_rca_agent import send_to_rca_agent
                except ImportError as import_error:
                    logger.error(f"Failed to import RCA agent tool: {import_error}")
                    state["rca_analysis"] = f"RCA Analysis failed: Could not import RCA agent tool - {str(import_error)}"
                    state["current_step"] = "rca_import_failed"
                    return state

                # Get correlation and metrics analysis
                log_correlation = state.get("log_correlation_result", "")
                metrics_correlation = state.get("metrics_correlation_result", "")

                if not log_correlation and not metrics_correlation:
                    logger.warning("No correlation analysis available for RCA")
                    state["rca_analysis"] = "RCA Analysis could not be completed: No correlation data available"
                    state["current_step"] = "rca_partial"
                    return state

                logger.info("Calling RCA agent with correlation data")

                # Extract incident_id from incident_key
                incident_key = state.get("incident_key", "unknown")
                incident_id = incident_key.split(":")[1] if ":" in incident_key and len(incident_key.split(":")) > 1 else incident_key

                # Prepare incident data
                incident_data = {
                    "incident_key": incident_key,
                    "alert_name": state.get("alertname", "Unknown Alert"),
                    "alertname": state.get("alertname", "Unknown Alert"),
                    "service": state.get("service", "Unknown Service"),
                    "severity": state.get("severity", "unknown"),
                    "description": state.get("description", "RCA analysis request"),
                    "instance": state.get("instance", state.get("service", "unknown")),
                    "timestamp": state.get("timestamp", ""),
                    "jira_ticket_id": state.get("alert_payload", {}).get("jira_ticket_id", "")
                }

                # Call RCA agent
                with langfuse.start_as_current_span(name="send-to-rca-agent") as rca_span:
                    rca_status = await send_to_rca_agent(
                        incident_id=incident_id,
                        incident_data=incident_data,
                        correlation_data=log_correlation,
                        metrics_analysis=metrics_correlation,
                        current_trace_id=current_trace_id,
                        current_observation_id=current_observation_id,
                        global_session_id=global_session_id
                    )

                    rca_span.update(
                        input={
                            "incident_id": incident_id,
                            "correlation_data_length": len(log_correlation),
                            "metrics_analysis_length": len(metrics_correlation)
                        },
                        output={
                            "rca_status": str(rca_status),
                            "delegation_successful": True
                        },
                        metadata={"tool_type": "rca_agent_delegation"}
                    )

                logger.info(f"RCA agent request sent: {rca_status}")
                state["rca_analysis"] = f"RCA analysis request sent to RCA agent. Status: {rca_status}"
                state["current_step"] = "rca_delegated"

                span.update(
                    output={
                        "rca_delegation_successful": True,
                        "rca_status": str(rca_status),
                        "incident_id": incident_id
                    },
                    metadata={"status": "success"}
                )

            except Exception as e:
                logger.error(f"Error sending request to RCA agent: {e}")
                state["rca_analysis"] = f"Failed to send request to RCA agent: {e}"
                state["current_step"] = "rca_failed"

                span.update(
                    output={"error": str(e), "rca_delegation_successful": False},
                    metadata={"status": "error"}
                )

        return state

    def complete_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Complete the workflow and gather statistics."""
        with langfuse.start_as_current_span(name="complete-correlation-workflow") as span:
            try:
                logger.info("Completing correlation workflow")

                # Gather completion statistics
                completion_stats = get_completion_stats(state)

                state["completed"] = True
                state["current_step"] = "workflow_complete"

                # Log final status
                logger.info(f"Correlation workflow completed for incident: {state['incident_key']}")
                logger.info(f"Redis storage: {'Success' if state.get('redis_stored') else 'Failed'}")
                logger.info(f"PostgreSQL storage: {'Success' if state.get('postgres_stored') else 'Failed'}")

                span.update(
                    output=completion_stats,
                    metadata={"status": "success"}
                )

            except Exception as e:
                logger.error(f"Error completing workflow: {e}")
                state["error"] = str(e)

                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )

        return state
