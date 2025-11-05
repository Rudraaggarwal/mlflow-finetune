
"""
Enhanced Correlation Agent with MCP tools and metrics analysis.
"""

import logging
import json
import redis
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import os
from dotenv import load_dotenv
import uuid
from sqlalchemy import text

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .llm_config import LLMConfig
from .models import CorrelationArray, CorrelatedLog
from .mcp_client import LangChainMCPClient
from .langfuse_prompts import get_correlation_prompt, get_client
from langfuse.langchain import CallbackHandler

# Langfuse imports for comprehensive observability
from langfuse import get_client, observe

# Import new utility modules
from .prompts import get_prompt, get_jira_template
from .logging_utils import get_logger, truncate_large_result, get_timestamp
from .validators import (
    is_successful_promql_result,
    is_successful_logql_result,
    get_result_count,
    has_prometheus_data,
    has_loki_data,
    count_total_metrics,
    count_total_logs,
    filter_successful_queries,
    is_valid_query_config
)

# Import modular handlers
from .storage_manager import StorageManager
from .jira_manager import JiraManager
from .workflow_completion import WorkflowCompletionHandler
from . import utils

# Initialize Langfuse client
load_dotenv()
langfuse = get_client()
logger = logging.getLogger(__name__)

class CorrelationAgentState(TypedDict):
    """State for the correlation workflow"""
    # Input data
    alert_payload: Dict[str, Any]
    alert_status: str
    alertname: str
    severity: str
    service: str
    description: str
    timestamp: str
    incident_key: str

    # Processing data
    service_dependencies: List[str]
    grafana_alert_info: Optional[Dict[str, Any]]

    # Log processing workflow
    generated_logql_queries: List[Dict[str, Any]]
    fetched_logs: Dict[str, Any]
    log_correlation_result: str
    structured_correlation: Optional[Dict[str, Any]]

    # Metrics processing workflow
    generated_promql_queries: List[Dict[str, Any]]
    fetched_metrics: Dict[str, Any]
    metrics_correlation_result: str
    structured_metrics: Optional[Dict[str, Any]]

    # RCA processing workflow
    rca_analysis: str
    structured_rca: Optional[Dict[str, Any]]

    # Correlation summary workflow
    correlation_summary: Optional[str]
    filtered_promql_queries: Optional[List[Dict[str, Any]]]

    # Control flow
    current_step: str
    error: Optional[str]
    completed: bool
    is_metric_based: bool
    logs_need_more_fetching: bool
    metrics_need_more_fetching: bool

    # Redis and Database integration
    redis_stored: bool
    postgres_stored: bool
    jira_updated: bool

class AgentState(TypedDict):
    """State shared between agents - Legacy for backward compatibility"""
    alert: Dict[str, Any]
    raw_logs: str
    correlated_logs: str
    prometheus_metrics: Optional[Dict[str, Any]]
    metrics_analysis: str
    rca_analysis: str
    remediation_steps: str
    current_step: str
    error: Optional[str]
    cpu_graph_data: Optional[Dict[str, Any]]
    affected_pods: Optional[List[Dict[str, Any]]]
    prometheus_datasource_uid: Optional[str]
    is_metric_based: bool
    structured_correlation: Optional[Dict[str, Any]]
    structured_rca: Optional[Dict[str, Any]]
    structured_remediation: Optional[Dict[str, Any]]
    incident_key: Optional[str]

current_trace_id=""
current_observation_id=""
_global_session_id = ""
class CorrelationAgent:
    """Enhanced correlation agent with LangGraph workflow for correlation and metrics analysis."""

    def __init__(self, db_url: str = None, redis_url: str = None, mcp_sse_url: str = None):
        """Initialize the correlation agent with all required components."""

        # Initialize LLM
        self.llm = LLMConfig.get_llm()

        # Initialize database
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if self.db_url:
            self.engine = create_engine(self.db_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        else:
            self.engine = None
            self.SessionLocal = None

        # Initialize Redis for incident information
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.redis_client = None
        if self.redis_url:
            self.redis_client = redis.from_url(self.redis_url)

        # Initialize MCP client
        self.mcp_client = None
        if mcp_sse_url:
            self.mcp_client = LangChainMCPClient(mcp_sse_url)

        # Initialize logs directory with error handling
        self.logs_dir = Path("logs")
        try:
            self.logs_dir.mkdir(exist_ok=True)
            logger.info(f"[{get_timestamp()}] Logs directory initialized: {self.logs_dir.absolute()}")
        except PermissionError:
            logger.warning(f"Permission denied creating logs directory {self.logs_dir.absolute()}, using temporary directory")
            import tempfile
            self.logs_dir = Path(tempfile.mkdtemp(prefix="correlation_logs_"))
        except Exception as e:
            logger.error(f"Failed to create logs directory: {e}, disabling log file writing")
            self.logs_dir = None

        # Initialize modular handlers
        self.storage_manager = StorageManager(mcp_client=self.mcp_client)
        self.jira_manager = JiraManager(mcp_client=self.mcp_client, llm=self.llm, langfuse_handler=None)
        self.workflow_completion = WorkflowCompletionHandler()

        # Build workflow graph
        self.graph = self._build_workflow_graph()
        self.langfuse_handler = CallbackHandler()
        self.jira_manager.langfuse_handler = self.langfuse_handler
        logger.info("CorrelationAgent initialized with LangGraph workflow and modular handlers")

    # @observe(name="workflow-graph-construction")
    def _build_workflow_graph(self):
        """Build the granular LangGraph workflow for correlation analysis with observability."""
        with langfuse.start_as_current_span(name="correlation-workflow-graph-build") as span:
            span.update(
                input={"agent_type": "correlation"},
                metadata={"component": "workflow-builder", "graph_type": "granular"}
            )

            try:
                workflow = StateGraph(CorrelationAgentState)

                # Add all nodes
                workflow.add_node("parse_alert", self._parse_alert_node)
                workflow.add_node("get_service_dependencies", self._get_service_dependencies_node)
                workflow.add_node("extract_grafana_info", self._extract_grafana_info_node)

                # Log processing subgraph
                workflow.add_node("generate_logql", self._generate_logql_node)
                workflow.add_node("fetch_logs", self._fetch_logs_node)
                workflow.add_node("analyze_log_correlation", self._analyze_log_correlation_node)

                # Metrics processing subgraph
                workflow.add_node("generate_promql", self._generate_promql_node)
                workflow.add_node("fetch_metrics", self._fetch_metrics_node)
                workflow.add_node("analyze_metrics_correlation", self._analyze_metrics_correlation_node)

                # Correlation summary node
                workflow.add_node("generate_correlation_summary", self._generate_correlation_summary_node)

                # RCA node
                workflow.add_node("rca_analysis", self._rca_analysis_node)

                # Storage and completion
                workflow.add_node("store_results", self._store_results_node)
                workflow.add_node("update_jira", self._update_jira_node)
                workflow.add_node("complete_workflow", self._complete_workflow_node)

                # Basic workflow flow with error handling
                workflow.add_conditional_edges(
                    "parse_alert",
                    self._check_for_errors,
                    {
                        "continue": "get_service_dependencies",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "get_service_dependencies",
                    self._check_for_errors,
                    {
                        "continue": "extract_grafana_info",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "extract_grafana_info",
                    self._check_for_errors,
                    {
                        "continue": "generate_logql",
                        "error": END
                    }
                )

                # Log processing workflow
                workflow.add_conditional_edges(
                    "generate_logql",
                    self._check_for_errors,
                    {
                        "continue": "fetch_logs",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "fetch_logs",
                    self._should_fetch_more_logs,
                    {
                        "fetch_more": "generate_logql",
                        "analyze": "analyze_log_correlation",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "analyze_log_correlation",
                    self._check_for_errors,
                    {
                        "continue": "generate_promql",
                        "error": END
                    }
                )

                # Metrics processing workflow
                workflow.add_conditional_edges(
                    "generate_promql",
                    self._check_for_errors,
                    {
                        "continue": "fetch_metrics",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "fetch_metrics",
                    self._should_fetch_more_metrics,
                    {
                        "fetch_more": "generate_promql",
                        "analyze": "analyze_metrics_correlation",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "analyze_metrics_correlation",
                    self._check_for_errors,
                    {
                        "continue": "generate_correlation_summary",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "generate_correlation_summary",
                    self._check_for_errors,
                    {
                        "continue": "store_results",
                        "error": END
                    }
                )

                # RCA and Remediation workflow
                workflow.add_conditional_edges(
                    "rca_analysis",
                    self._check_for_errors,
                    {
                        "continue": "update_jira",
                        "error": END
                    }
                )

                workflow.add_edge("store_results", "rca_analysis")
                workflow.add_edge("update_jira", "complete_workflow")
                workflow.add_edge("complete_workflow", END)

                # Set entry point
                workflow.set_entry_point("parse_alert")
                app = workflow.compile()
                print(app.get_graph().draw_ascii())

                span.update(
                    output={
                        "nodes_count": 12,
                        "conditional_edges": 8,
                        "graph_compiled": True
                    },
                    metadata={"status": "success"}
                )

                return app

            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                raise

    # @observe(name="mcp-client-initialization")
    async def initialize_mcp_client(self):
        """Initialize MCP client connection with observability."""
        with langfuse.start_as_current_span(name="mcp-client-connect") as span:
            logger.info("Starting MCP client initialization...")

            if self.mcp_client:
                try:
                    span.update(
                        input={"mcp_client_available": True},
                        metadata={"component": "mcp-initialization"}
                    )

                    logger.info("Attempting to connect MCP client...")
                    await self.mcp_client.connect()

                    span.update(
                        output={"connection_successful": True},
                        metadata={"status": "success"}
                    )

                    logger.info("MCP client connected successfully")

                except Exception as e:
                    span.update(
                        output={"error": str(e)},
                        metadata={"status": "error"}
                    )

                    logger.error(f"Failed to connect MCP client: {e}")
                    import traceback
                    logger.error(f"MCP connection stack trace: {traceback.format_exc()}")
                    self.mcp_client = None
            else:
                span.update(
                    output={"mcp_client_configured": False},
                    metadata={"status": "skipped"}
                )
                logger.warning("No MCP client configured - continuing without MCP tools")

    # @observe(name="error-check-routing")
    def _check_for_errors(self, state: CorrelationAgentState) -> str:
        """Check if there are errors that should stop the workflow with tracing."""
        with langfuse.start_as_current_span(name="error-validation") as span:
            has_error = bool(state.get("error"))
            decision = "error" if has_error else "continue"

            span.update(
                input={"has_error": has_error, "error": state.get("error")},
                output={"routing_decision": decision},
                metadata={"workflow_step": "error_check"}
            )

            if has_error:
                logger.error(f"Workflow stopping due to error: {state['error']}")

            return decision

    # @observe(name="proceed-to-storage-routing")
    def _should_proceed_to_storage(self, state: CorrelationAgentState) -> str:
        """Decide whether to proceed to storage or skip with tracing."""
        with langfuse.start_as_current_span(name="storage-routing-decision") as span:
            has_error = bool(state.get("error"))
            has_correlation = bool(state.get("log_correlation_result"))
            has_metrics = bool(state.get("metrics_correlation_result"))

            if has_error:
                decision = "error"
            elif has_correlation or has_metrics:
                decision = "continue"
            else:
                decision = "skip"

            span.update(
                input={
                    "has_error": has_error,
                    "has_correlation": has_correlation,
                    "has_metrics": has_metrics
                },
                output={"routing_decision": decision},
                metadata={"workflow_step": "storage_routing"}
            )

            return decision

    # @observe(name="logs-fetching-routing")
    def _should_fetch_more_logs(self, state: CorrelationAgentState) -> str:
        """Decide whether to fetch more logs or proceed to analysis with tracing."""
        with langfuse.start_as_current_span(name="logs-fetching-decision") as span:
            has_error = bool(state.get("error"))
            needs_more_logs = state.get("logs_need_more_fetching", False)

            if has_error:
                decision = "error"
            elif needs_more_logs:
                decision = "fetch_more"
            else:
                decision = "analyze"

            span.update(
                input={
                    "has_error": has_error,
                    "needs_more_logs": needs_more_logs
                },
                output={"routing_decision": decision},
                metadata={"workflow_step": "logs_routing"}
            )

            return decision

    # @observe(name="metrics-fetching-routing")
    def _should_fetch_more_metrics(self, state: CorrelationAgentState) -> str:
        """Decide whether to fetch more metrics or proceed to analysis with tracing."""
        with langfuse.start_as_current_span(name="metrics-fetching-decision") as span:
            has_error = bool(state.get("error"))
            needs_more_metrics = state.get("metrics_need_more_fetching", False)

            if has_error:
                decision = "error"
            elif needs_more_metrics:
                decision = "fetch_more"
            else:
                decision = "analyze"

            span.update(
                input={
                    "has_error": has_error,
                    "needs_more_metrics": needs_more_metrics
                },
                output={"routing_decision": decision},
                metadata={"workflow_step": "metrics_routing"}
            )

            return decision

    # @observe(name="alert-parsing")
    async def _parse_alert_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 1: Parse incoming alert and extract basic information with comprehensive tracing."""
        with langfuse.start_as_current_span(name="parse-alert") as span:
            logger.info("STEP 1: Parsing alert payload")

            try:
                alert_payload = state["alert_payload"]

                span.update(
                    input={
                        "payload_type": type(alert_payload).__name__,
                        "payload_keys": list(alert_payload.keys()) if isinstance(alert_payload, dict) else []
                    },
                    metadata={"step": "alert_parsing", "workflow_position": 1}
                )

                # Handle different alert payload formats
                if isinstance(alert_payload, str):
                    with langfuse.start_as_current_span(name="payload-parsing") as parse_span:
                        try:
                            alert_payload = json.loads(alert_payload)
                            state["alert_payload"] = alert_payload
                            parse_span.update(
                                output={"parsing_method": "json_parse"},
                                metadata={"status": "success"}
                            )
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse alert payload: {alert_payload}")
                            state["error"] = "Invalid alert payload format"
                            parse_span.update(
                                output={"error": "Invalid alert payload format"},
                                metadata={"status": "error"}
                            )
                            return state

                # Extract basic alert information
                state["alertname"] = alert_payload.get("alertname", "Unknown Alert")
                state["severity"] = alert_payload.get("severity", "medium")
                state["service"] = (
                    alert_payload.get("service") or
                    alert_payload.get("service_name") or
                    alert_payload.get("container_name") or
                    alert_payload.get("namespace") or
                    "unknown"
                )
                state["description"] = alert_payload.get("description", "No description available")
                state["timestamp"] = alert_payload.get("timestamp", datetime.now().isoformat())
                state["alert_status"] = alert_payload.get("status", "firing")

                state["current_step"] = "parse_alert_complete"

                span.update(
                    output={
                        "alertname": state["alertname"],
                        "severity": state["severity"],
                        "service": state["service"],
                        "step_completed": True
                    },
                    metadata={"status": "success", "workflow_position": 1}
                )

                logger.info(f"[{get_timestamp()}] Parsed alert: {state['alertname']} - {state['severity']}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error parsing alert: {error_msg}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 1}
                )

        return state

    # @observe(name="service-dependencies-retrieval")
    async def _get_service_dependencies_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 2: Get service dependencies using MCP tools with comprehensive tracing."""
        with langfuse.start_as_current_span(name="get-service-dependencies") as span:
            logger.info("STEP 2: Getting service dependencies")

            try:
                service_name = state["service"]

                span.update(
                    input={"service_name": service_name},
                    metadata={"step": "service_dependencies", "workflow_position": 2}
                )

                logger.info(f"[{get_timestamp()}] Getting dependencies for service: {service_name}")

                # Call the service dependencies tool with tracing
                with langfuse.start_as_current_span(name="service-dependencies-tool-call") as tool_span:
                    service_deps_tool = self._create_service_dependencies_tool()
                    dependencies_result = await service_deps_tool.ainvoke(input={"service_name": service_name}, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    })

                    tool_span.update(
                        input={"service_name": service_name},
                        output={"result_type": type(dependencies_result).__name__},
                        metadata={"tool_execution": "service_dependencies"}
                    )

                    logger.info(f"[{get_timestamp()}] Service dependencies tool result: {dependencies_result}")

                # Parse dependencies from JSON result
                if isinstance(dependencies_result, str):
                    with langfuse.start_as_current_span(name="dependencies-parsing") as parse_span:
                        try:
                            deps_data = json.loads(dependencies_result)

                            if "error" in deps_data:
                                logger.error(f"Service dependencies tool error: {deps_data['error']}")
                                state["service_dependencies"] = [service_name]  # Fallback to just the service itself
                                parse_span.update(
                                    output={"error": deps_data['error'], "fallback_used": True},
                                    metadata={"status": "fallback"}
                                )
                            else:
                                # Use all_services_to_check which includes the main service + all dependencies
                                all_services = deps_data.get("all_services_to_check", [service_name])
                                state["service_dependencies"] = all_services

                                # Log detailed dependency information
                                direct_deps = deps_data.get("direct_dependencies", [])
                                all_deps = deps_data.get("all_dependencies", [])
                                namespace = deps_data.get("namespace", "unknown")

                                parse_span.update(
                                    output={
                                        "all_services_count": len(all_services),
                                        "direct_deps_count": len(direct_deps),
                                        "namespace": namespace
                                    },
                                    metadata={"status": "success"}
                                )

                                logger.info(f"[{get_timestamp()}] Service: {service_name}")
                                logger.info(f"[{get_timestamp()}] Namespace: {namespace}")
                                logger.info(f"[{get_timestamp()}] Direct dependencies: {direct_deps}")
                                logger.info(f"[{get_timestamp()}] All recursive dependencies: {all_deps}")
                                logger.info(f"[{get_timestamp()}] Total services to check: {len(all_services)}")

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse service dependencies JSON: {e}")
                            state["service_dependencies"] = [service_name]
                            parse_span.update(
                                output={"error": str(e), "fallback_used": True},
                                metadata={"status": "fallback"}
                            )
                else:
                    logger.warning("Service dependencies result is not a string")
                    state["service_dependencies"] = [service_name]

                state["current_step"] = "service_dependencies_complete"

                span.update(
                    output={
                        "service_dependencies": state["service_dependencies"],
                        "dependencies_count": len(state["service_dependencies"])
                    },
                    metadata={"status": "success", "workflow_position": 2}
                )

                logger.info(f"[{get_timestamp()}] Final service dependencies list: {state['service_dependencies']}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error getting service dependencies: {error_msg}")
                state["service_dependencies"] = [state["service"]]  # Fallback
                logger.warning(f"Using fallback dependencies: {state['service_dependencies']}")

                span.update(
                    output={"error": error_msg, "fallback_dependencies": state["service_dependencies"]},
                    metadata={"status": "fallback", "workflow_position": 2}
                )

        return state

    # @observe(name="grafana-info-extraction")
    async def _extract_grafana_info_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 3: Extract Grafana alert information if available with comprehensive tracing."""
        with langfuse.start_as_current_span(name="extract-grafana-info") as span:
            logger.info("STEP 3: Extracting Grafana alert information")

            try:
                alert_payload = state["alert_payload"]

                span.update(
                    input={
                        "payload_type": type(alert_payload).__name__,
                        "payload_keys": list(alert_payload.keys()) if isinstance(alert_payload, dict) else []
                    },
                    metadata={"step": "grafana_extraction", "workflow_position": 3}
                )

                logger.info(f"[{get_timestamp()}] Alert payload type: {type(alert_payload)}")
                logger.info(f"[{get_timestamp()}] Alert payload keys: {list(alert_payload.keys()) if isinstance(alert_payload, dict) else 'Not a dict'}")

                # Use LLM to extract Grafana alert UID from alert payload with tracing
                with langfuse.start_as_current_span(name="grafana-uid-extraction") as uid_span:
                    grafana_uid = await self._extract_grafana_uid_with_llm(alert_payload)

                    uid_span.update(
                        input={"payload_provided": bool(alert_payload)},
                        output={"uid_found": bool(grafana_uid), "uid": grafana_uid},
                        metadata={"extraction_method": "llm"}
                    )

                if grafana_uid:
                    # Fetch detailed alert info using the fetch_alert_info tool
                    with langfuse.start_as_current_span(name="grafana-alert-info-fetch") as fetch_span:
                        grafana_info = await self._fetch_grafana_alert_info(grafana_uid)

                        fetch_span.update(
                            input={"grafana_uid": grafana_uid},
                            output={
                                "info_fetched": bool(grafana_info),
                                "info_keys": list(grafana_info.keys()) if isinstance(grafana_info, dict) else []
                            },
                            metadata={"fetch_method": "mcp_tool"}
                        )

                        if not grafana_info:
                            # If fetch_alert_info failed, use the original payload
                            grafana_info = alert_payload
                else:
                    # No UID found, use empty dict
                    grafana_info = {}

                try:
                    state["grafana_alert_info"] = grafana_info
                    state["current_step"] = "grafana_info_extracted"

                    span.update(
                        output={
                            "grafana_info_extracted": True,
                            "info_size": len(str(grafana_info)) if grafana_info else 0
                        },
                        metadata={"status": "success", "workflow_position": 3}
                    )

                    logger.info(f"[{get_timestamp()}] Successfully extracted Grafana info: {grafana_info}")

                except Exception as assign_error:
                    logger.error(f"Error assigning grafana_info to state: {assign_error}")
                    logger.error(f"Assign error type: {type(assign_error).__name__}")
                    import traceback
                    logger.error(f"Assign stack trace: {traceback.format_exc()}")
                    raise

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error extracting Grafana info: {error_msg}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                state["grafana_alert_info"] = {}
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 3}
                )

        return state

    # @observe(name="llm-grafana-uid-extraction")
    async def _extract_grafana_uid_with_llm(self, alert_payload: dict) -> str:
        """Use LLM to extract Grafana alert UID from complex alert payload with tracing."""
        with langfuse.start_as_current_span(name="llm-uid-extraction") as span:
            try:
                logger.info("Using LLM to extract Grafana alert UID")

                span.update(
                    input={"payload_size": len(str(alert_payload)) if alert_payload else 0},
                    metadata={"component": "llm_uid_extraction"}
                )

                # Create a prompt for LLM to find the UID
                extraction_prompt = f"""
You are a Grafana alert analyzer. Extract the alert UID from this alert payload.

Look for fields like:
- generatorURL containing /alertRules/[UID]
- alertUID, alert_uid, uid
- Any URL or reference containing an alphanumeric UID

Alert Payload:
{json.dumps(alert_payload, indent=2)}

If you find a Grafana alert UID, return ONLY the UID (alphanumeric string).
If no UID is found, return "NONE".
"""

                messages = [
                    {"role": "system", "content": "You are an expert at extracting Grafana alert UIDs from alert payloads."},
                    {"role": "user", "content": extraction_prompt}
                ]

                response = await self.llm.ainvoke(
                    messages, 
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                )

                extracted_uid = response.content if hasattr(response, 'content') else str(response)
                extracted_uid = extracted_uid.strip()

                if extracted_uid and extracted_uid != "NONE" and len(extracted_uid) > 5:
                    span.update(
                        output={"extracted_uid": extracted_uid, "extraction_successful": True},
                        metadata={"status": "success"}
                    )
                    logger.info(f"[{get_timestamp()}] LLM extracted Grafana UID: {extracted_uid}")
                    return extracted_uid
                else:
                    span.update(
                        output={"extraction_successful": False, "result": extracted_uid},
                        metadata={"status": "no_uid_found"}
                    )
                    logger.info("No Grafana UID found by LLM")
                    return ""

            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                logger.error(f"Failed to extract Grafana UID with LLM: {e}")
                return ""

    # @observe(name="grafana-alert-info-fetch")
    async def _fetch_grafana_alert_info(self, alert_uid: str) -> dict:
        """Fetch relevant alert rule info using get_alert_rule_by_uid tool with tracing."""
        with langfuse.start_as_current_span(name="fetch-grafana-alert-rule") as span:
            try:
                if not alert_uid:
                    span.update(
                        output={"fetch_attempted": False, "reason": "no_uid"},
                        metadata={"status": "skipped"}
                    )
                    logger.info("No alert UID provided, skipping Grafana alert rule fetch")
                    return {}

                if not self.mcp_client:
                    span.update(
                        output={"fetch_attempted": False, "reason": "no_mcp_client"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("No MCP client available for get_alert_rule_by_uid")
                    return {}

                span.update(
                    input={"alert_uid": alert_uid, "mcp_client_available": True},
                    metadata={"component": "grafana_fetch"}
                )

                # Check if get_alert_rule_by_uid tool is available
                available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
                tool_names = [tool.name for tool in available_tools] if available_tools else []

                has_alert_rule_tool = any('get_alert_rule_by_uid' in tool_name for tool_name in tool_names)
                if not has_alert_rule_tool:
                    span.update(
                        output={"fetch_attempted": False, "reason": "tool_not_available"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("get_alert_rule_by_uid tool not available")
                    return {}

                logger.info(f"[{get_timestamp()}] Fetching Grafana alert rule for UID: {alert_uid}")

                # Call the get_alert_rule_by_uid tool with tracing
                with langfuse.start_as_current_span(name="mcp-tool-call") as tool_span:
                    raw_result = await self.mcp_client.call_tool_direct("get_alert_rule_by_uid", {
                        "uid": alert_uid
                    })

                    tool_span.update(
                        input={"tool": "get_alert_rule_by_uid", "uid": alert_uid},
                        output={"result_received": bool(raw_result)},
                        metadata={"tool_execution": "grafana_alert_rule"}
                    )

                if not raw_result:
                    span.update(
                        output={"fetch_attempted": True, "result_received": False},
                        metadata={"status": "no_result"}
                    )
                    logger.warning("No alert rule info returned from get_alert_rule_by_uid")
                    return {}

                # Handle both string and dict responses from MCP tool
                if isinstance(raw_result, str):
                    try:
                        full_result = json.loads(raw_result)
                        logger.info("Parsed string response from MCP tool to dict")
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Could not parse MCP tool response as JSON: {json_err}")
                        logger.warning(f"Raw response: {raw_result[:200]}...")
                        return {}
                elif isinstance(raw_result, dict):
                    full_result = raw_result
                    logger.info("Received dict response from MCP tool")
                else:
                    logger.warning(f"Unexpected response type from MCP tool: {type(raw_result)}")
                    return {}

                # Extract only relevant data to reduce payload size
                relevant_data = {
                    "title": full_result.get("title", "Unknown Alert"),
                    "condition": full_result.get("condition", "C"),
                    "annotations": full_result.get("annotations", {}),
                    "labels": full_result.get("labels", {}),
                    "alert_expressions": []
                }

                # Extract LogQL/PromQL expressions from alert rule data
                if 'data' in full_result and isinstance(full_result['data'], list):
                    for data_item in full_result['data']:
                        if 'model' in data_item and 'expr' in data_item['model']:
                            expr_info = {
                                "refId": data_item.get('refId', 'Unknown'),
                                "expr": data_item['model']['expr'],
                                "queryType": data_item.get('queryType', 'range'),
                                "datasourceUid": data_item.get('datasourceUid', '')
                            }
                            relevant_data["alert_expressions"].append(expr_info)

                span.update(
                    output={
                        "fetch_successful": True,
                        "alert_title": relevant_data.get('title'),
                        "expressions_count": len(relevant_data["alert_expressions"])
                    },
                    metadata={"status": "success"}
                )

                logger.info(f"[{get_timestamp()}] Successfully fetched alert rule: {relevant_data.get('title')}")
                logger.info(f"[{get_timestamp()}] Found {len(relevant_data['alert_expressions'])} expressions")

                return relevant_data

            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                logger.error(f"Failed to fetch Grafana alert rule info: {e}")
                return {}

    # @observe(name="logql-generation")
    async def _generate_logql_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 4: Generate LogQL queries using LLM with comprehensive tracing."""
        with langfuse.start_as_current_span(name="generate-logql") as span:
            logger.info("STEP 4: Generating LogQL queries")

            try:
                logger.info(f"[{get_timestamp()}] LogQL node - state keys: {list(state.keys())}")
                logger.info(f"[{get_timestamp()}] LogQL node - service: {state.get('service')}")
                logger.info(f"LogQL node - timestamp: {state.get('timestamp')}")
                logger.info(f"[{get_timestamp()}] LogQL node - dependencies: {state.get('service_dependencies')}")
                span.update(
                    input={
                        "service": state.get('service'),
                        "dependencies": state.get('service_dependencies'),
                        "grafana_info_available": bool(state.get('grafana_alert_info'))
                    },
                    metadata={"step": "logql_generation", "workflow_position": 4}
                )

                # Initialize queries list if not exists
                if "generated_logql_queries" not in state or not state["generated_logql_queries"]:
                    state["generated_logql_queries"] = []
                    logger.info("Initialized empty logql_queries list")

                # Create LLM prompt for LogQL generation
                logql_generation_prompt = f"""
You are an expert SRE engineer specializing in LogQL query generation for Grafana Loki.

**CURRENT CONTEXT:**
- Alert: {state['alertname']}
- Service: {state['service']}
- Dependencies: {state.get('service_dependencies', [])}
- Alert Timestamp: {state['timestamp']}
- Grafana Alert Rule: {state.get('grafana_alert_info', {}).get('title', 'No alert rule data')}
- Alert Expressions: {[expr.get('expr', '') for expr in state.get('grafana_alert_info', {}).get('alert_expressions', [])]}
- Alert Annotations: {state.get('grafana_alert_info', {}).get('annotations', {})}

**TASK:**
Generate comprehensive LogQL queries for both primary service and its dependencies:

**QUERY TYPES TO GENERATE:**
1. **Primary Service Error Logs**: ERROR/WARN/FATAL logs for the main service
2. **Primary Service Info Logs**: INFO logs from primary service for context
3. **Dependency Service Logs**: Important logs from service dependencies

**FOR EACH SERVICE (Primary + Dependencies):**
- Use container_name field (NOT 'container')
- Use namespace "paylater" for all services
- Correct LogQL syntax: {{container_name="servicename", namespace="paylater"}} |~ "(?i)(error|failed|exception|info)"
- Time range: 5 minutes around alert timestamp
- Limit: 15 logs per query

**CONSTRAINTS:**
- Use datasourceUid: "f699f82a-3e72-4bfd-b993-56db8dd58997"
- Use |~ for regex patterns, |= for exact matches
- Include both container_name and namespace labels
- NO system/kubernetes logs (kubelet, kube-proxy, etc.)
- Limit each query to 15 results
- MAXIMUM 6 queries total - prioritize most relevant services
- Generate queries for primary service + top 2 dependencies

**OUTPUT FORMAT:**
Return a JSON list of queries for multiple services:
[
  {{
    "phase": "1",
    "service": "{state['service']}",
    "query": "{{container_name=\\"{state['service']}\\", namespace=\\"paylater\\"}} |~ \\"(?i)(error|failed|exception|fatal)\\"",
    "start_time": "2025-09-16T04:28:13Z",
    "end_time": "2025-09-16T04:33:13Z",
    "limit": 15,
    "datasourceUid": "f699f82a-3e72-4bfd-b993-56db8dd58997"
  }},
  {{
    "phase": "2",
    "service": "{state['service']}",
    "query": "{{container_name=\\"{state['service']}\\", namespace=\\"paylater\\"}} |~ \\"(?i)(info)\\"",
    "start_time": "2025-09-16T04:28:13Z",
    "end_time": "2025-09-16T04:33:13Z",
    "limit": 15,
    "datasourceUid": "f699f82a-3e72-4bfd-b993-56db8dd58997"
  }}
]

Generate queries ONLY for the primary service and its dependencies from this validated list: {state.get('service_dependencies', [])}.
Do NOT generate queries for any services not in this list. Only use these exact service names. Do not include any Python-style dicts, only JSON.
"""

                # Use LLM to generate LogQL queries with tracing
                messages = [
                    SystemMessage(content="You are a log analysis specialist helping to investigate application issues. Generate accurate queries to find relevant application logs that explain problems."),
                    HumanMessage(content=logql_generation_prompt)
                ]

                with langfuse.start_as_current_span(name="llm-logql-generation") as llm_span:
                    response = await self.llm.ainvoke(
                        messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                    )

                    llm_span.update(
                        input={"prompt_length": len(logql_generation_prompt)},
                        output={"response_length": len(response.content)},
                        metadata={"llm_task": "logql_generation"}
                    )

                try:
                    # Parse the JSON response
                    import re
                    json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                    if json_match:
                        queries_json = json_match.group(0)
                        new_queries = json.loads(queries_json)

                        # Add to existing queries
                        state["generated_logql_queries"].extend(new_queries)
                        logger.info(f"[{get_timestamp()}] Generated {len(new_queries)} LogQL queries")

                    else:
                        raise ValueError("No valid JSON found in LLM response")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse LogQL queries: {e}")
                    # Create fallback query with safe timestamp handling
                    with langfuse.start_as_current_span(name="fallback-query-creation") as fallback_span:
                        try:
                            timestamp_str = str(state["timestamp"])
                            service_name = str(state["service"])
                            logger.info(f"Creating fallback query for service: {service_name}, timestamp: {timestamp_str}")

                            # Safe timestamp parsing
                            if timestamp_str and timestamp_str != "":
                                try:
                                    timestamp_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    logger.info(f"Parsed timestamp successfully: {timestamp_obj}")
                                except ValueError as ts_error:
                                    logger.error(f"Timestamp parsing failed: {ts_error}")
                                    timestamp_obj = datetime.now()
                            else:
                                logger.warning("Empty timestamp, using current time")
                                timestamp_obj = datetime.now()

                            start_time = (timestamp_obj - timedelta(minutes=2)).isoformat()
                            end_time = (timestamp_obj + timedelta(minutes=3)).isoformat()

                            fallback_query = {
                                "phase": "1",
                                "service": service_name,
                                "query": f'{{container_name="{service_name}", namespace="paylater"}} |~ "(?i)(error|failed|exception|fatal)"',
                                "start_time": start_time,
                                "end_time": end_time,
                                "limit": 15,
                                "datasourceUid": "f699f82a-3e72-4bfd-b993-56db8dd58997"
                            }

                            # Safely append to the list
                            if isinstance(state["generated_logql_queries"], list):
                                state["generated_logql_queries"].append(fallback_query)
                                logger.info("Created and added fallback LogQL query")
                            else:
                                state["generated_logql_queries"] = [fallback_query]
                                logger.info("Initialized queries list with fallback query")

                            fallback_span.update(
                                output={"fallback_query_created": True},
                                metadata={"fallback_reason": "llm_parsing_failed"}
                            )

                        except Exception as fallback_error:
                            logger.error(f"Failed to create fallback query: {fallback_error}")
                            import traceback
                            logger.error(f"Fallback error stack trace: {traceback.format_exc()}")
                            state["error"] = f"Failed to generate LogQL queries: {str(e)}"

                            fallback_span.update(
                                output={"error": str(fallback_error)},
                                metadata={"status": "error"}
                            )
                            return state

                state["current_step"] = "logql_generated"

                span.update(
                    output={
                        "total_queries_generated": len(state['generated_logql_queries']),
                        "queries_created_successfully": True
                    },
                    metadata={"status": "success", "workflow_position": 4}
                )

                logger.info(f"[{get_timestamp()}] Total LogQL queries generated: {len(state['generated_logql_queries'])}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating LogQL queries: {error_msg}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"LogQL generation stack trace: {traceback.format_exc()}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 4}
                )

        return state

    # @observe(name="logs-fetching")
    async def _fetch_logs_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 5: Execute LogQL queries to fetch logs with comprehensive tracing."""
        with langfuse.start_as_current_span(name="fetch-logs") as span:
            logger.info("STEP 5: Fetching logs using generated LogQL queries")

            try:
                logger.info(f"[{get_timestamp()}] Fetch logs - queries to execute: {len(state.get('generated_logql_queries', []))}")
                queries_count = len(state.get('generated_logql_queries', []))
                span.update(
                    input={"queries_to_execute": queries_count},
                    metadata={"step": "logs_fetching", "workflow_position": 5}
                )

                # Initialize fetched_logs if not exists
                if "fetched_logs" not in state:
                    state["fetched_logs"] = {}
                    logger.info("Initialized empty fetched_logs dict")

                # Get MCP tools for log querying
                if not self.mcp_client:
                    span.update(
                        output={"logs_fetched": False, "reason": "no_mcp_client"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("No MCP client available, skipping log fetching")
                    state["fetched_logs"] = {}
                    state["current_step"] = "logs_fetched"
                    state["logs_need_more_fetching"] = False
                    return state

                # Check if query_loki_logs tool is available
                available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
                tool_names = [tool.name for tool in available_tools] if available_tools else []
                logger.info(f"[{get_timestamp()}] Available MCP tools: {tool_names}")

                has_loki_tool = any('query_loki_logs' in tool_name.lower() for tool_name in tool_names)

                if not has_loki_tool:
                    span.update(
                        output={"logs_fetched": False, "reason": "no_loki_tool"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("No query_loki_logs tool available, continuing without logs")
                    state["fetched_logs"] = {}
                    state["current_step"] = "logs_fetched"
                    state["logs_need_more_fetching"] = False
                    return state

                logger.info("Found query_loki_logs tool, proceeding with log queries")

                queries_to_execute = state.get("generated_logql_queries", [])
                # Limit to maximum 6 queries to avoid overwhelming the system
                if len(queries_to_execute) > 6:
                    queries_to_execute = queries_to_execute[:6]
                    logger.info(f"[{get_timestamp()}] Limited queries from {len(state.get('generated_logql_queries', []))} to 6 for performance")

                logger.info(f"[{get_timestamp()}] Executing {len(queries_to_execute)} LogQL queries")

                # Initialize tracking variables for tool call tracing
                mcp_calls = 0
                mcp_errors = 0
                tool_calls_made = []
                successful_queries = 0

                # Execute each query with comprehensive tool call tracing
                for i, query_config in enumerate(queries_to_execute):
                    try:
                        if not isinstance(query_config, dict):
                            logger.error(f"Query config {i+1} is not a dict: {type(query_config)}")
                            continue

                        logger.info(f"[{get_timestamp()}] Executing LogQL query {i+1}: {query_config.get('query', 'No query')}")

                        # **ENHANCED: Track individual tool call with Langfuse**
                        with langfuse.start_as_current_span(
                            name=f"[tool-called]-query_loki_logs-{i+1}"
                        ) as tool_call_span:
                            # Safely build query parameters using the correct MCP tool format
                            query_params = {}
                            try:
                                # Use the correct parameter names as shown in your example
                                query_params = {
                                    "datasourceUid": str(query_config.get("datasourceUid", "f699f82a-3e72-4bfd-b993-56db8dd58997")),
                                    "logql": str(query_config.get("query", "")),  # Use 'logql' instead of 'query'
                                    "startRfc3339": str(query_config.get("start_time", "")),  # Use 'startRfc3339' instead of 'start'
                                    "endRfc3339": str(query_config.get("end_time", "")),      # Use 'endRfc3339' instead of 'end'
                                    "limit": int(query_config.get("limit", 15))
                                }
                                logger.info(f"[{get_timestamp()}] Query params (corrected format): {query_params}")

                                tool_info = {
                                    "tool_name": "query_loki_logs",
                                    "tool_id": f"loki_query_{i+1}",
                                    "tool_args": query_params,
                                    "execution_order": len(tool_calls_made) + 1,
                                    "query_index": i + 1
                                }
                                tool_calls_made.append(tool_info)
                                mcp_calls += 1

                                tool_call_span.update(
                                    input={
                                        "tool_name": "query_loki_logs",
                                        "tool_args": query_params,
                                        "query_index": i + 1
                                    },
                                    metadata={
                                        "tool_type": "mcp_loki_query",
                                        "execution_order": tool_info['execution_order'],
                                        "agent_type": "correlation_agent"
                                    }
                                )

                            except Exception as param_error:
                                logger.error(f"Failed to build query params: {param_error}")
                                tool_call_span.update(
                                    output={"error": str(param_error)},
                                    metadata={"status": "parameter_error"}
                                )
                                continue

                            # Execute the query using MCP tool with direct call method
                            try:
                                result = await self.mcp_client.call_tool_direct("query_loki_logs", query_params)
                                successful_queries += 1
                                logger.info(f"[{get_timestamp()}] Query {i+1} executed successfully")

                                tool_call_span.update(
                                    output={
                                        "execution_successful": True,
                                        "result_received": bool(result)
                                    },
                                    metadata={"status": "success"}
                                )

                            except Exception as query_error:
                                mcp_errors += 1
                                logger.error(f"Query execution failed: {query_error}")
                                tool_call_span.update(
                                    output={"error": str(query_error)},
                                    metadata={"status": "execution_error"}
                                )
                                continue

                        # **ENHANCED: Track tool response with Langfuse**
                        with langfuse.start_as_current_span(
                            name=f"[tool-result]-query_loki_logs-{i+1}"
                        ) as result_span:
                            # Store results safely
                            try:
                                phase = str(query_config.get("phase", "unknown"))
                                service = str(query_config.get("service", "unknown"))
                                query_key = f"phase_{phase}_{service}"

                                # Ensure fetched_logs is a dict
                                if not isinstance(state["fetched_logs"], dict):
                                    state["fetched_logs"] = {}

                                state["fetched_logs"][query_key] = {
                                    "query": str(query_config.get("query", "")),
                                    "results": result,
                                    "phase": phase,
                                    "service": service
                                }

                                # Better result counting for Loki data
                                result_count = "N/A"
                                if isinstance(result, list):
                                    result_count = len(result)
                                elif isinstance(result, dict):
                                    if "data" in result and isinstance(result["data"], dict):
                                        if "result" in result["data"] and isinstance(result["data"]["result"], list):
                                            result_count = get_result_count(result)
                                        elif "resultType" in result["data"]:
                                            result_count = f"1 {result['data']['resultType']}"
                                    else:
                                        result_count = f"dict with {len(result)} keys"
                                elif result is not None:
                                    result_count = f"1 {type(result).__name__}"

                                result_span.update(
                                    input={"query_key": query_key},
                                    output={
                                        "result_count": str(result_count),
                                        "result_type": type(result).__name__,
                                        "stored_successfully": True
                                    },
                                    metadata={
                                        "message_type": "tool_response",
                                        "agent_type": "correlation_agent",
                                        "query_phase": phase,
                                        "service": service
                                    }
                                )

                                logger.info(f"[{get_timestamp()}] Stored logs for {query_key}: {result_count} entries")
                                logger.info(f"[{get_timestamp()}] Result type: {type(result).__name__}")
                                if isinstance(result, dict) and "data" in result:
                                    logger.info(f"[{get_timestamp()}] Data keys: {list(result['data'].keys()) if isinstance(result['data'], dict) else 'Not a dict'}")

                            except Exception as store_error:
                                result_span.update(
                                    output={"error": str(store_error)},
                                    metadata={"status": "storage_error"}
                                )
                                logger.error(f"Failed to store query results: {store_error}")

                    except Exception as query_error:
                        logger.error(f"Failed to execute query {i+1}: {query_error}")
                        logger.error(f"Query error type: {type(query_error).__name__}")
                        import traceback
                        logger.error(f"Query error stack trace: {traceback.format_exc()}")

                # **ENHANCED: Tool execution summary**
                with langfuse.start_as_current_span(name="logs-fetching-summary") as summary_span:
                    summary_span.update(
                        output={
                            "total_queries_attempted": len(queries_to_execute),
                            "successful_queries": successful_queries,
                            "mcp_calls_made": mcp_calls,
                            "mcp_errors": mcp_errors,
                            "error_rate": mcp_errors / mcp_calls if mcp_calls > 0 else 0,
                            "logs_stored": len(state["fetched_logs"])
                        },
                        metadata={
                            "agent_type": "correlation_agent",
                            "step": "logs_fetching_complete"
                        }
                    )

                # Analyze if more logs need to be fetched
                state["logs_need_more_fetching"] = await self._should_fetch_more_logs_logic(state)
                state["current_step"] = "logs_fetched"

                span.update(
                    output={
                        "logs_fetched_successfully": successful_queries > 0,
                        "total_queries_executed": successful_queries,
                        "total_errors": mcp_errors,
                        "logs_keys_stored": list(state["fetched_logs"].keys())
                    },
                    metadata={"status": "success", "workflow_position": 5}
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error fetching logs: {error_msg}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Log fetch stack trace: {traceback.format_exc()}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 5}
                )

        return state

    # @observe(name="logs-fetching-analysis")
    async def _should_fetch_more_logs_logic(self, state: CorrelationAgentState) -> bool:
        """Determine if more logs need to be fetched based on current results with tracing."""
        with langfuse.start_as_current_span(name="logs-fetching-decision-logic") as span:
            try:
                fetched_logs = state.get("fetched_logs", {})

                span.update(
                    input={"fetched_logs_count": len(fetched_logs)},
                    metadata={"decision_logic": "logs_fetching"}
                )

                # Check Phase 1 results
                phase1_results = [logs for logs in fetched_logs.values() if logs.get("phase") == "1"]

                if not phase1_results or all(not logs.get("results") for logs in phase1_results):
                    # No Phase 1 results, check if we need Phase 2 (INFO logs)
                    phase2_exists = any(logs.get("phase") == "2" for logs in fetched_logs.values())
                    if not phase2_exists:
                        span.update(
                            output={"decision": True, "reason": "need_phase2_queries"},
                            metadata={"status": "fetch_more"}
                        )
                        logger.info("No Phase 1 results found, need to generate Phase 2 queries")
                        return True

                    # Check if we need Phase 3 (dependency logs)
                    phase3_exists = any(logs.get("phase") == "3" for logs in fetched_logs.values())
                    if not phase3_exists:
                        span.update(
                            output={"decision": True, "reason": "need_phase3_queries"},
                            metadata={"status": "fetch_more"}
                        )
                        logger.info("No errors found, need to generate Phase 3 dependency queries")
                        return True

                # If we have relevant results, no need to fetch more
                span.update(
                    output={"decision": False, "reason": "sufficient_results"},
                    metadata={"status": "analyze"}
                )
                return False

            except Exception as e:
                span.update(
                    output={"error": str(e), "decision": False},
                    metadata={"status": "error"}
                )
                logger.error(f"Error determining if more logs needed: {e}")
                return False

    # @observe(name="log-correlation-analysis")
    async def _analyze_log_correlation_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 6: Analyze fetched logs for correlation with comprehensive tracing."""
        with langfuse.start_as_current_span(name="analyze-log-correlation") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 6: Analyzing log correlation")

            try:
                fetched_logs = state.get("fetched_logs", {})

                span.update(
                    input={
                        "fetched_logs_count": len(fetched_logs),
                        "logs_available": bool(fetched_logs),
                        "alert_name": state.get('alertname'),
                        "service": state.get('service')
                    },
                    metadata={"step": "log_correlation_analysis", "workflow_position": 6}
                )

                if not fetched_logs:
                    state["log_correlation_result"] = "NO LOGS FOUND - No logs retrieved for the alert timeframe"
                    # Convert CorrelationArray to dict to avoid unhashable type error
                    correlation_obj = CorrelationArray(correlated_logs=[
                        CorrelatedLog(
                            timestamp=datetime.now().isoformat(),
                            message="No logs found in the specified time window",
                            level="INFO",
                            reasoning="No logs were retrieved from any of the services during the alert timeframe"
                        )
                    ])
                    state["structured_correlation"] = correlation_obj.model_dump()
                    state["current_step"] = "log_correlation_complete"

                    span.update(
                        output={"correlation_result": "no_logs_found"},
                        metadata={"status": "no_data", "workflow_position": 6}
                    )
                    return state

                # Use LLM to analyze the logs with tracing
                correlation_analysis_prompt = f"""
You are investigating an application issue to help the engineering team understand what went wrong.

**The Problem:**
- Issue: {state['alertname']}
- Service Affected: {state['service']}
- Severity: {state['severity']}
- When It Happened: {state['timestamp']}

**Application Logs to Review:**
{json.dumps(fetched_logs, indent=2)}

**Your Investigation:**
Please analyze the application logs and provide a clear report:

1. **Initial Assessment:** Start with [RELEVANT/NOT RELEVANT/NO LOGS FOUND]

2. **What the Logs Show:**
   - Extract important log entries that explain the problem
   - Include when and where the errors occurred (timestamps, service locations)
   - Identify error messages and system failures
   - Explain how these issues relate to the alert

3. **Investigation Summary:**
   - Explain in simple terms why these log entries matter
   - Identify what likely caused the problem based on the evidence
   - Show how this problem might have affected other parts of the system

**Report Format:**
Write your findings as a clear investigation report that includes:
- Whether the logs help explain the issue
- Detailed analysis of important log entries
- Impact on the system and users
- Evidence-based conclusions about what happened
"""

                messages = [
                    SystemMessage(content="You are a system investigator helping engineering teams understand application problems. Write clear, business-friendly reports that explain technical issues in plain language."),
                    HumanMessage(content=correlation_analysis_prompt)
                ]

                with langfuse.start_as_current_span(name="llm-log-correlation-analysis") as llm_span:
                    response = await self.llm.ainvoke(
                        messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                    )

                    correlation_result = response.content

                    llm_span.update(
                        input={"logs_analyzed": len(fetched_logs), "prompt_length": len(correlation_analysis_prompt)},
                        output={"analysis_length": len(correlation_result)},
                        metadata={"llm_task": "log_correlation"}
                    )

                # Create structured correlation with tracing
                try:
                    with langfuse.start_as_current_span(name="structured-correlation-creation") as struct_span:
                        correlation_llm = self.llm.with_structured_output(CorrelationArray)

                        structured_prompt = f"""
                        Based on the log correlation analysis below, create structured output in JSON.

                        CORRELATION ANALYSIS:
                        {correlation_result}

                        Extract correlated log entries and organize them with:
                        - timestamp: Log timestamp
                        - pod: Kubernetes pod name (if available)
                        - instance: Instance identifier (if available)
                        - level: Log level (ERROR, INFO, WARN, etc.)
                        - stream: Log stream (stdout/stderr, if available)
                        - job: Job name (if available)
                        - node: Node name (if available)
                        - namespace: Namespace (if available)
                        - message: Complete log message with all details (transaction_id, error_code, thresholds, etc., separated by new lines)
                        - reasoning: Why this log is relevant to the alert

                        Set "is_metric_based" to true for downstream metrics analysis.
                        make the output in structured format

                        """

                        structured_correlation = await correlation_llm.ainvoke([
                            {"role": "system", "content": "Create structured correlation output from analysis."},
                            {"role": "user", "content": structured_prompt}
                        ], config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    })

                        # Convert to dict to avoid unhashable type error
                        state["structured_correlation"] = structured_correlation.model_dump() if hasattr(structured_correlation, 'model_dump') else structured_correlation

                        struct_span.update(
                            output={"structured_correlation_created": True},
                            metadata={"status": "success"}
                        )

                except Exception as e:
                    logger.error(f"Failed to create structured correlation: {e}")
                    fallback_correlation = self._create_fallback_correlation_structure(correlation_result)
                    state["structured_correlation"] = fallback_correlation.model_dump() if hasattr(fallback_correlation, 'model_dump') else fallback_correlation

                state["log_correlation_result"] = correlation_result
                state["is_metric_based"] = True
                state["current_step"] = "log_correlation_complete"

                # Save correlation results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize incident key for filename (replace invalid characters)
                safe_incident_key = state['incident_key'].replace(':', '_').replace('/', '_').replace('\\', '_')
                correlation_file = self.logs_dir / f"log_correlation_{safe_incident_key}_{timestamp}.txt"

                with open(correlation_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("LOG CORRELATION ANALYSIS REPORT\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Alert: {state['alertname']}\n")
                    f.write(f"Service: {state['service']}\n")
                    f.write(f"Timestamp: {state['timestamp']}\n")
                    f.write("=" * 80 + "\n")
                    f.write("FETCHED LOGS SUMMARY:\n")
                    for key, logs in fetched_logs.items():
                        f.write(f"- {key}: {len(logs.get('results', [])) if isinstance(logs.get('results'), list) else 'N/A'} entries\n")
                    f.write("=" * 80 + "\n")
                    f.write(correlation_result + "\n")
                    f.write("=" * 80 + "\n")

                # Store correlation analysis in database immediately
                try:
                    results = {"structured_correlation": state.get("structured_correlation")}
                    await self._store_analysis_in_database_new(state, results)
                    logger.info("Stored correlation analysis in database immediately")
                except Exception as db_error:
                    logger.error(f"Failed to store correlation analysis in database: {db_error}")

                # Add JIRA comment immediately after correlation analysis completion
                await self._add_jira_comment_for_analysis(
                    state,
                    "correlation",
                    correlation_result
                )

                span.update(
                    output={
                        "correlation_analysis_completed": True,
                        "analysis_length": len(correlation_result),
                        "structured_correlation_created": bool(state.get("structured_correlation")),
                        "file_saved": str(correlation_file)
                    },
                    metadata={"status": "success", "workflow_position": 6}
                )

                logger.info(f"[{get_timestamp()}] Log correlation analysis complete, saved to {correlation_file}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in log correlation analysis: {error_msg}")
                state["log_correlation_result"] = f"Log correlation analysis failed: {error_msg}"
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 6}
                )

        return state

    # @observe(name="service-dependencies-tool-creation")
    def _create_service_dependencies_tool(self):
        """Create service dependencies tool that returns JSON format expected by the node with tracing."""
        with langfuse.start_as_current_span(name="create-service-dependencies-tool") as span:
            from langchain_core.tools import tool

            @tool
            def get_service_dependencies(service_name: str) -> str:
                """
                Get service dependencies recursively for a given service name to check all related services that might be impacted.

                Args:
                    service_name (str): Name of the service to get dependencies for

                Returns:
                    str: JSON string containing service dependencies information with all recursive dependencies
                """
                def get_all_dependencies(service, deps_data, visited=None):
                    """Recursively get all dependencies for a service"""
                    if visited is None:
                        visited = set()

                    if service in visited:
                        return []  # Prevent infinite loops in circular dependencies

                    visited.add(service)
                    all_deps = []

                    if service in deps_data:
                        direct_deps = deps_data[service].get('dependencies', [])
                        all_deps.extend(direct_deps)

                        # Recursively get dependencies of dependencies
                        for dep in direct_deps:
                            recursive_deps = get_all_dependencies(dep, deps_data, visited.copy())
                            all_deps.extend(recursive_deps)

                    return list(set(all_deps))  # Remove duplicates

                try:
                    # Import and use the service_dependency tool
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    from service_dependency import execute_memgraph_query

                    # Create Cypher query to get dependencies for the service
                    query = f"""
                    MATCH (s:Service {{name: '{service_name}'}})-[r:DEPENDS_ON]->(t:Service)
                    RETURN s.name AS source, t.name AS dependency, t.namespace AS namespace
                    """

                    logger.info(f"[{get_timestamp()}] Executing Memgraph query for service: {service_name}")
                    logger.info(f"[{get_timestamp()}] Query: {query}")

                    # Execute the query
                    result = execute_memgraph_query(query)
                    logger.info(f"[{get_timestamp()}] Memgraph query result: {result}")

                    # Parse the response format: [{'source': 'oemconnector', 'dependency': 'catalogue', 'namespace': 'paylater'}]
                    dependencies = []
                    namespace = "unknown"

                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict) and item.get('source') == service_name:
                                dep = item.get('dependency')
                                if dep and dep not in dependencies:
                                    dependencies.append(dep)
                                namespace = item.get('namespace', namespace)

                        logger.info(f"[{get_timestamp()}] Parsed dependencies for {service_name}: {dependencies}")

                        # Create the expected JSON response format
                        response_data = {
                            "service": service_name,
                            "direct_dependencies": dependencies,
                            "all_dependencies": dependencies,  # For now, using direct dependencies
                            "namespace": namespace,
                            "all_services_to_check": [service_name] + dependencies,
                            "dependency_count": len(dependencies),
                            "services_to_monitor": len([service_name] + dependencies)
                        }

                        return json.dumps(response_data, indent=2)

                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Memgraph query error: {result['error']}")
                        return json.dumps({"error": f"Memgraph query failed: {result['error']}"})

                    else:
                        logger.warning(f"No dependencies found for service: {service_name}")
                        return json.dumps({
                            "service": service_name,
                            "direct_dependencies": [],
                            "all_dependencies": [],
                            "namespace": "unknown",
                            "all_services_to_check": [service_name],
                            "message": f"No dependencies found for {service_name} in Memgraph"
                        })

                except Exception as e:
                    logger.error(f"Error querying Memgraph for service dependencies: {e}")
                    return json.dumps({"error": f"Failed to query Memgraph: {str(e)}"})

            span.update(
                output={"tool_created": True},
                metadata={"status": "success"}
            )

            return get_service_dependencies

    # @observe(name="promql-generation")
    async def _generate_promql_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 7: Generate PromQL queries for metrics correlation with comprehensive tracing."""
        with langfuse.start_as_current_span(name="generate-promql") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 7: Generating PromQL queries")

            try:
                span.update(
                    input={
                        "service": state.get('service'),
                        "dependencies": state.get('service_dependencies'),
                        "correlation_available": bool(state.get('log_correlation_result'))
                    },
                    metadata={"step": "promql_generation", "workflow_position": 7}
                )

                # Initialize queries list if not exists
                if "generated_promql_queries" not in state or not state["generated_promql_queries"]:
                    state["generated_promql_queries"] = []

                # Get correlation report for context
                correlation_report = state.get("log_correlation_result", "")

                # Create LLM prompt for PromQL generation
                promql_generation_prompt = f"""
You are a metrics correlation agent that analyzes correlation reports and fetches relevant Prometheus metrics.

**ALERT INFORMATION:**
- Alert: {state['alertname']}
- Service: {state['service']}
- Severity: {state['severity']}
- Timestamp: {state['timestamp']}
- Dependencies: {state.get('service_dependencies', [])}
- Grafana Alert Rule: {state.get('grafana_alert_info', {}).get('title', 'No alert rule data')}
- Original Alert Expressions: {[expr.get('expr', '') for expr in state.get('grafana_alert_info', {}).get('alert_expressions', [])]}
- Alert Condition: {state.get('grafana_alert_info', {}).get('condition', 'C')}

**CORRELATION REPORT:**
{correlation_report}

**WORKFLOW:**
Read the alert info, there may be 'expr' field in it which would have custom promql used for calculating. Use that to query the prometheus and keep results consistent.

1. **READ CORRELATION REPORT**: Review the provided correlation report to identify which metric is responsible for the alert. Else use the uid from generator url to get alert info.
Always check for cpu/memory request and limit changes over the timeline as they could be changed and are often the root cause of issues.

2. **GET SERVICE DEPENDENCIES**: Already available: {state.get('service_dependencies', [])}

3. **QUERY METRICS**: For each service (primary + dependencies):
    - Include both container and namespace labels 
    -USE 'container' label not 'container_name'

   - Query the alert-causing metric identified from correlation report
   - Query 2 most relevant related metrics (e.g., CPU, memory, error rate, request rate)
   - Use datasource UID "NxyYHrE4k" for all Prometheus queries
   - Time range: Around alert timestamp

**CONSTRAINTS:**
- Use datasource UID "NxyYHrE4k" for ALL Prometheus queries
- MAXIMUM 9 queries total - prioritize most critical metrics only
- Limit to 2-3 metrics per service (1 alert metric + 1-2 related)
- Query primary service + max 2-3 key dependencies only
- **TIME RANGE LIMIT**: Maximum 2 minutes time difference between startTime and endTime
- Include actual metric values and trends
- Always query for cpu/memory  request and limits if they changed over time
- Grafana Alert Rule
- Keep analysis concise and factual
- Choose appropriate queryType:
  * "range" for time-series data (rates, trends, usage over time)
  * "instant" for current/latest values (current status, instant metrics)

**OUTPUT FORMAT:**
Return a JSON list of PromQL queries:
[
  {{
    "service": "primary_service",
    "metric_type": "alert_causing",
    "expr": "sum(rate(container_cpu_usage_seconds_total{{container=\"servicename\",namespace=\"namespace\"}}[5m]))",
    "queryType": "range",
    "startTime": "2024-01-01T10:03:00Z",
    "endTime": "2024-01-01T10:05:00Z",
    "stepSeconds": 15,
    "datasourceUid": "NxyYHrE4k"
  }},
  {{
    "service": "primary_service",
    "metric_type": "current_memory",
    "expr": "container_memory_usage_bytes{{container=\"servicename\",namespace=\"namespace\"}}",
    "queryType": "instant",
    "startTime": "2024-01-01T10:05:00Z",
    "endTime": "2024-01-01T10:05:00Z",
    "stepSeconds": 15,
    "datasourceUid": "NxyYHrE4k"
  }}
]

Generate PromQL queries ONLY for services from this validated list: {state.get('service_dependencies', [])}.
Do NOT generate queries for any services not in this list. Only use these exact service names for metrics correlation analysis.
"""

                # Use LLM to generate PromQL queries with tracing
                messages = [
                    SystemMessage(content="You are a performance monitoring specialist helping to investigate application performance issues. Generate accurate queries to find relevant system metrics that explain problems."),
                    HumanMessage(content=promql_generation_prompt)
                ]

                with langfuse.start_as_current_span(name="llm-promql-generation") as llm_span:
                    response = await self.llm.ainvoke(
                        messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                    )

                    llm_span.update(
                        input={"prompt_length": len(promql_generation_prompt)},
                        output={"response_length": len(response.content)},
                        metadata={"llm_task": "promql_generation"}
                    )

                try:
                    # Parse the JSON response
                    import re
                    json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                    if json_match:
                        queries_json = json_match.group(0)
                        new_queries = json.loads(queries_json)

                        # Add to existing queries
                        state["generated_promql_queries"].extend(new_queries)
                        logger.info(f"[{get_timestamp()}] Generated {len(new_queries)} PromQL queries")

                    else:
                        raise ValueError("No valid JSON found in LLM response")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse PromQL queries: {e}")
                    # Create fallback queries with tracing
                    with langfuse.start_as_current_span(name="fallback-promql-creation") as fallback_span:
                        alert_time = datetime.fromisoformat(state["timestamp"].replace('Z', '+00:00'))
                        start_time = (alert_time - timedelta(minutes=2)).isoformat()
                        end_time = (alert_time + timedelta(minutes=3)).isoformat()

                        fallback_queries = [
                            {
                                "service": state["service"],
                                "metric_type": "cpu_usage",
                                "expr": f'sum(rate(container_cpu_usage_seconds_total{{container="{state["service"]}",namespace="paylater"}}[5m]))',
                                "queryType": "range",
                                "startTime": start_time,
                                "endTime": end_time,
                                "stepSeconds": 15,
                                "datasourceUid": "NxyYHrE4k"
                            },
                            {
                                "service": state["service"],
                                "metric_type": "memory_usage",
                                "expr": f'sum(container_memory_usage_bytes{{container="{state["service"]}",namespace="paylater"}})',
                                "queryType": "instant",
                                "startTime": start_time,
                                "endTime": end_time,
                                "stepSeconds": 15,
                                "datasourceUid": "NxyYHrE4k"
                            }
                        ]
                        state["generated_promql_queries"].extend(fallback_queries)

                        fallback_span.update(
                            output={"fallback_queries_created": len(fallback_queries)},
                            metadata={"fallback_reason": "llm_parsing_failed"}
                        )

                state["current_step"] = "promql_generated"

                span.update(
                    output={
                        "total_queries_generated": len(state['generated_promql_queries']),
                        "queries_created_successfully": True
                    },
                    metadata={"status": "success", "workflow_position": 7}
                )

                logger.info(f"[{get_timestamp()}] Total PromQL queries generated: {len(state['generated_promql_queries'])}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating PromQL queries: {error_msg}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 7}
                )

        return state

    # @observe(name="metrics-fetching")
    async def _fetch_metrics_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 8: Execute PromQL queries to fetch metrics with comprehensive tracing."""
        with langfuse.start_as_current_span(name="fetch-metrics") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 8: Fetching metrics using generated PromQL queries")

            try:
                queries_count = len(state.get('generated_promql_queries', []))
                span.update(
                    input={"queries_to_execute": queries_count},
                    metadata={"step": "metrics_fetching", "workflow_position": 8}
                )

                # Initialize fetched_metrics if not exists
                if "fetched_metrics" not in state:
                    state["fetched_metrics"] = {}

                # Check if MCP client is available for metrics querying
                if not self.mcp_client:
                    span.update(
                        output={"metrics_fetched": False, "reason": "no_mcp_client"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("No MCP client available, skipping metrics fetching")
                    state["fetched_metrics"] = {}
                    state["current_step"] = "metrics_fetched"
                    state["metrics_need_more_fetching"] = False
                    return state

                # Check if Prometheus tools are available
                available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
                tool_names = [tool.name for tool in available_tools] if available_tools else []

                has_prometheus_tool = any('query_prometheus' in tool_name.lower() for tool_name in tool_names)

                if not has_prometheus_tool:
                    span.update(
                        output={"metrics_fetched": False, "reason": "no_prometheus_tool"},
                        metadata={"status": "skipped"}
                    )
                    logger.warning("No query_prometheus tool available, continuing without metrics")
                    state["fetched_metrics"] = {}
                    state["current_step"] = "metrics_fetched"
                    state["metrics_need_more_fetching"] = False
                    return state

                logger.info("Found query_prometheus tool, proceeding with metrics queries")
                queries_to_execute = state.get("generated_promql_queries", [])

                # Limit to maximum 6 queries to avoid overwhelming the system
                if len(queries_to_execute) > 9:
                    queries_to_execute = queries_to_execute[:9]
                    logger.info(f"[{get_timestamp()}] Limited PromQL queries from {len(state.get('generated_promql_queries', []))} to 9 for performance")

                logger.info(f"[{get_timestamp()}] Executing {len(queries_to_execute)} PromQL queries")

                # Initialize tracking variables for tool call tracing
                mcp_calls = 0
                mcp_errors = 0
                tool_calls_made = []
                successful_queries = 0

                # Execute each query with comprehensive tool call tracing
                for i, query_config in enumerate(queries_to_execute):
                    try:
                        if not isinstance(query_config, dict):
                            logger.error(f"PromQL query config {i+1} is not a dict: {type(query_config)}")
                            continue

                        logger.info(f"[{get_timestamp()}] Executing PromQL query {i+1}: {query_config.get('expr', query_config.get('query', 'No query'))}")

                        # **ENHANCED: Track individual tool call with Langfuse**
                        with langfuse.start_as_current_span(
                            name=f"[tool-called]-query_prometheus-{i+1}"
                        ) as tool_call_span:
                            # Safely build query parameters
                            prometheus_params = {}
                            try:
                                # Safe step conversion
                                step_value = query_config.get("stepSeconds", query_config.get("step", "15"))
                                if isinstance(step_value, str):
                                    step_value = step_value.replace("s", "")
                                step_seconds = int(step_value)

                                prometheus_params = {
                                    "expr": str(query_config.get("expr", query_config.get("query", ""))),
                                    "queryType": str(query_config.get("queryType", "range")),
                                    "startTime": str(query_config.get("startTime", query_config.get("start_time", ""))),
                                    "endTime": str(query_config.get("endTime", query_config.get("end_time", ""))),
                                    "stepSeconds": step_seconds,
                                    "datasourceUid": str(query_config.get("datasourceUid", "NxyYHrE4k"))
                                }
                                logger.info(f"[{get_timestamp()}] PromQL params: {prometheus_params}")

                                tool_info = {
                                    "tool_name": "query_prometheus",
                                    "tool_id": f"prometheus_query_{i+1}",
                                    "tool_args": prometheus_params,
                                    "execution_order": len(tool_calls_made) + 1,
                                    "query_index": i + 1
                                }
                                tool_calls_made.append(tool_info)
                                mcp_calls += 1

                                tool_call_span.update(
                                    input={
                                        "tool_name": "query_prometheus",
                                        "tool_args": prometheus_params,
                                        "query_index": i + 1
                                    },
                                    metadata={
                                        "tool_type": "mcp_prometheus_query",
                                        "execution_order": tool_info['execution_order'],
                                        "agent_type": "correlation_agent"
                                    }
                                )

                            except Exception as param_error:
                                logger.error(f"Failed to build PromQL params: {param_error}")
                                tool_call_span.update(
                                    output={"error": str(param_error)},
                                    metadata={"status": "parameter_error"}
                                )
                                continue

                            # Execute the query using MCP tool with direct call method
                            try:
                                result = await self.mcp_client.call_tool_direct("query_prometheus", prometheus_params)
                                successful_queries += 1
                                logger.info(f"[{get_timestamp()}]  PromQL query {i+1} executed successfully")
                                logger.info(f"[{get_timestamp()}]  Result is  {result}")
                                logger.info(f"[{get_timestamp()}]  Result is  {type(result)}")


                                tool_call_span.update(
                                    output={
                                        "execution_successful": True,
                                        "result_received": bool(result)
                                    },
                                    metadata={"status": "success"}
                                )

                            except Exception as query_error:
                                mcp_errors += 1
                                logger.error(f"PromQL query execution failed: {query_error}")
                                tool_call_span.update(
                                    output={"error": str(query_error)},
                                    metadata={"status": "execution_error"}
                                )
                                continue

                        # **ENHANCED: Track tool response with Langfuse**
                        with langfuse.start_as_current_span(
                            name=f"[tool-result]-query_prometheus-{i+1}"
                        ) as result_span:
                            # Parse JSON string result if needed
                            if isinstance(result, str):
                                try:
                                    import json
                                    result = json.loads(result)
                                    logger.info(f"[{get_timestamp()}] Parsed JSON string result for query {i+1}")
                                except (json.JSONDecodeError, ValueError) as parse_error:
                                    logger.warning(f"Could not parse result as JSON: {parse_error}")

                            # Store results with consistent key format matching filter function
                            try:
                                # Use query_{i+1} format to match the filter function
                                query_key = f"query_{i+1}"

                                # Ensure fetched_metrics is a dict
                                if not isinstance(state["fetched_metrics"], dict):
                                    state["fetched_metrics"] = {}

                                # Determine if query was successful
                                has_results = False
                                if isinstance(result, dict):
                                    if "data" in result and isinstance(result["data"], dict):
                                        if "result" in result["data"] and isinstance(result["data"]["result"], list):
                                            has_results = is_successful_promql_result(result)
                                elif isinstance(result, list):
                                    # Handle direct list results from JSON parsing
                                    has_results = len(result) > 0

                                    # Store with status
                                    state["fetched_metrics"][query_key] = {
                                        "query": str(query_config.get("expr", query_config.get("query", ""))),
                                        "results": result,
                                        "service": str(query_config.get("service", "unknown")),
                                        "metric_type": str(query_config.get("metric_type", "unknown")),
                                        "status": "success" if has_results else "empty",
                                        "query_config": query_config  # Store full config for filtering
                                    }

                                    # Better result counting for Prometheus data
                                    result_count = "N/A"
                                    if isinstance(result, list):
                                        result_count = len(result)
                                    elif isinstance(result, dict):
                                        if "data" in result and isinstance(result["data"], dict):
                                            if "result" in result["data"] and isinstance(result["data"]["result"], list):
                                                result_count = get_result_count(result)
                                            elif "resultType" in result["data"]:
                                                result_count = f"1 {result['data']['resultType']}"
                                        else:
                                            result_count = f"dict with {len(result)} keys"
                                    elif result is not None:
                                        result_count = f"1 {type(result).__name__}"

                                    result_span.update(
                                        input={"query_key": query_key},
                                        output={
                                            "result_count": str(result_count),
                                            "result_type": type(result).__name__,
                                            "stored_successfully": True,
                                            "has_results": has_results
                                        },
                                        metadata={
                                            "message_type": "tool_response",
                                            "agent_type": "correlation_agent",
                                            "service": query_config.get("service", "unknown"),
                                            "metric_type": query_config.get("metric_type", "unknown")
                                        }
                                    )

                                    logger.info(f"[{get_timestamp()}] Stored metrics for {query_key}: {result_count} entries (status: {'success' if has_results else 'empty'})")
                                    logger.info(f"[{get_timestamp()}] Result type: {type(result).__name__}")
                                    if isinstance(result, dict) and "data" in result:
                                        logger.info(f"[{get_timestamp()}] Data keys: {list(result['data'].keys()) if isinstance(result['data'], dict) else 'Not a dict'}")

                            except Exception as store_error:
                                result_span.update(
                                    output={"error": str(store_error)},
                                    metadata={"status": "storage_error"}
                                )
                                logger.error(f"Failed to store PromQL results: {store_error}")
                                import traceback
                                logger.error(f"PromQL store error stack trace: {traceback.format_exc()}")

                                # Store failed query
                                query_key = f"query_{i+1}"
                                state["fetched_metrics"][query_key] = {
                                    "query": str(query_config.get("expr", query_config.get("query", ""))),
                                    "status": "failed",
                                    "error": str(store_error),
                                    "query_config": query_config
                                }

                    except Exception as query_error:
                        logger.error(f"Failed to execute PromQL query {i+1}: {query_error}")
                        logger.error(f"PromQL query error type: {type(query_error).__name__}")

                # **ENHANCED: Tool execution summary**
                with langfuse.start_as_current_span(name="metrics-fetching-summary") as summary_span:
                    summary_span.update(
                        output={
                            "total_queries_attempted": len(queries_to_execute),
                            "successful_queries": successful_queries,
                            "mcp_calls_made": mcp_calls,
                            "mcp_errors": mcp_errors,
                            "error_rate": mcp_errors / mcp_calls if mcp_calls > 0 else 0,
                            "metrics_stored": len(state["fetched_metrics"])
                        },
                        metadata={
                            "agent_type": "correlation_agent",
                            "step": "metrics_fetching_complete"
                        }
                    )

                # Analyze if more metrics need to be fetched
                state["metrics_need_more_fetching"] = await self._should_fetch_more_metrics_logic(state)
                state["current_step"] = "metrics_fetched"

                span.update(
                    output={
                        "metrics_fetched_successfully": successful_queries > 0,
                        "total_queries_executed": successful_queries,
                        "total_errors": mcp_errors,
                        "metrics_keys_stored": list(state["fetched_metrics"].keys())
                    },
                    metadata={"status": "success", "workflow_position": 8}
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error fetching metrics: {error_msg}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 8}
                )

        return state

    # @observe(name="metrics-fetching-analysis")
    async def _should_fetch_more_metrics_logic(self, state: CorrelationAgentState) -> bool:
        """Determine if more metrics need to be fetched based on current results with tracing."""
        with langfuse.start_as_current_span(name="metrics-fetching-decision-logic") as span:
            try:
                fetched_metrics = state.get("fetched_metrics", {})
                current_queries = len(state.get("generated_promql_queries", []))

                span.update(
                    input={
                        "fetched_metrics_count": len(fetched_metrics),
                        "current_queries": current_queries
                    },
                    metadata={"decision_logic": "metrics_fetching"}
                )

                # Hard limit: Never fetch more if we've hit the 6 query limit
                if current_queries >= 9:
                    logger.info("✋ Already executed maximum 9 PromQL queries, stopping")
                    span.update(
                        output={"decision": False, "reason": "max_queries_reached"},
                        metadata={"status": "limit_reached"}
                    )
                    return False

                # Count total metric entries across all queries to assess data sufficiency
                total_metric_entries = 0
                for key, metrics in fetched_metrics.items():
                    if metrics.get("results"):
                        if isinstance(metrics["results"], list):
                            total_metric_entries += len(metrics["results"])
                        elif isinstance(metrics["results"], dict) and "data" in metrics["results"]:
                            data = metrics["results"]["data"]
                            if isinstance(data, dict) and "result" in data and isinstance(data["result"], list):
                                total_metric_entries += len(data["result"])

                # If we have any meaningful metrics, we have enough data
                if total_metric_entries > 0:
                    logger.info(f"[{get_timestamp()}] Found {total_metric_entries} metric entries across {len(fetched_metrics)} queries - sufficient data")
                    span.update(
                        output={"decision": False, "reason": "sufficient_data", "total_entries": total_metric_entries},
                        metadata={"status": "sufficient"}
                    )
                    return False

                # Conservative approach: Don't fetch more to avoid query loops
                logger.info("No meaningful metrics found, but stopping to avoid query loops")
                span.update(
                    output={"decision": False, "reason": "avoid_loops"},
                    metadata={"status": "conservative"}
                )
                return False

            except Exception as e:
                span.update(
                    output={"error": str(e), "decision": False},
                    metadata={"status": "error"}
                )
                logger.error(f"Error determining if more metrics needed: {e}")
                return False
    
    # @observe(name="metrics-correlation-analysis")
    async def _analyze_metrics_correlation_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 9: Analyze fetched metrics for correlation with comprehensive tracing."""
        with langfuse.start_as_current_span(name="analyze-metrics-correlation") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 9: Analyzing metrics correlation")

            try:
                fetched_metrics = state.get("fetched_metrics", {})
                correlation_report = state.get("log_correlation_result", "")

                span.update(
                    input={
                        "fetched_metrics_count": len(fetched_metrics),
                        "correlation_report_available": bool(correlation_report),
                        "alert_name": state.get('alertname'),
                        "service": state.get('service')
                    },
                    metadata={"step": "metrics_correlation_analysis", "workflow_position": 9}
                )

                if not fetched_metrics:
                    state["metrics_correlation_result"] = "No metrics data available for analysis"
                    state["current_step"] = "metrics_correlation_complete"

                    span.update(
                        output={"metrics_result": "no_metrics_data"},
                        metadata={"status": "no_data", "workflow_position": 9}
                    )
                    return state

                # Use LLM to analyze the metrics with comprehensive tracing
                metrics_analysis_prompt = f"""
You are analyzing system performance data to understand what happened during an application issue.

**The Problem:**
- Issue: {state['alertname']}
- Service Affected: {state['service']}
- Severity: {state['severity']}
- When It Happened: {state['timestamp']}

**Previous Investigation Findings:**
{correlation_report}

**System Performance Data:**
{json.dumps(fetched_metrics, indent=2)}

**Your Analysis:**
Please review the performance data and explain what happened:

**Report Structure:**
- **Primary Issue**: The main performance problem that triggered the alert (with actual numbers)
- **Related Performance Issues**: Other performance problems that happened at the same time (2 key metrics per service)
- **System Impact**: How this affected other parts of the application or users
- **Performance Timeline**: Key changes and patterns over time

**Guidelines:**
- Use actual performance numbers and trends from the data
- Explain how the performance issues relate to the alert
- Show how this affected other systems or services
- Draw clear conclusions based on the evidence

Write a clear performance analysis report that helps the team understand what happened to system performance.
"""

                messages = [
                    SystemMessage(content="You are a performance analyst helping engineering teams understand system performance issues. Write clear reports that explain technical performance data in business-friendly language."),
                    HumanMessage(content=metrics_analysis_prompt)
                ]

                with langfuse.start_as_current_span(name="llm-metrics-correlation-analysis") as llm_span:
                    response = await self.llm.ainvoke(
                        messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                    )

                    metrics_result = response.content

                    llm_span.update(
                        input={
                            "metrics_analyzed": len(fetched_metrics),
                            "prompt_length": len(metrics_analysis_prompt),
                            "correlation_report_length": len(correlation_report)
                        },
                        output={"analysis_length": len(metrics_result)},
                        metadata={"llm_task": "metrics_correlation"}
                    )

                # Create structured metrics analysis with tracing
                try:
                    with langfuse.start_as_current_span(name="structured-metrics-creation") as struct_span:
                        structured_metrics = {
                            "alert_metric": "Primary metric identified from analysis",
                            "supporting_metrics": [],
                            "service_impact": "Impact assessment",
                            "summary": metrics_result,
                            "analysis_timestamp": datetime.now().isoformat()
                        }

                        state["structured_metrics"] = structured_metrics

                        struct_span.update(
                            output={"structured_metrics_created": True},
                            metadata={"status": "success"}
                        )

                except Exception as e:
                    logger.error(f"Failed to create structured metrics: {e}")
                    state["structured_metrics"] = {"summary": metrics_result}

                state["metrics_correlation_result"] = metrics_result
                state["current_step"] = "metrics_correlation_complete"

                # Save metrics analysis results with tracing
                with langfuse.start_as_current_span(name="save-metrics-analysis-file") as file_span:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Sanitize incident key for filename (replace invalid characters)
                    safe_incident_key = state['incident_key'].replace(':', '_').replace('/', '_').replace('\\', '_')
                    metrics_file = self.logs_dir / f"metrics_correlation_{safe_incident_key}_{timestamp}.txt"

                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write("METRICS CORRELATION ANALYSIS REPORT\n")
                        f.write("=" * 80 + "\n")
                        f.write(f"Alert: {state['alertname']}\n")
                        f.write(f"Service: {state['service']}\n")
                        f.write(f"Timestamp: {state['timestamp']}\n")
                        f.write("=" * 80 + "\n")
                        f.write("FETCHED METRICS SUMMARY:\n")
                        for key, metrics in fetched_metrics.items():
                            f.write(f"- {key}: {metrics.get('metric_type', 'N/A')}\n")
                        f.write("=" * 80 + "\n")
                        f.write(metrics_result + "\n")
                        f.write("=" * 80 + "\n")

                    file_span.update(
                        output={"file_saved": str(metrics_file)},
                        metadata={"status": "success"}
                    )

                # Store metrics analysis in database immediately with tracing
                try:
                    with langfuse.start_as_current_span(name="store-metrics-database") as db_span:
                        results = {
                            "metrics_analysis": metrics_result,
                            "structured_metrics": state.get("structured_metrics", {})
                        }
                        await self._store_analysis_in_database_new(state, results)
                        logger.info("Stored metrics analysis in database immediately")

                        db_span.update(
                            output={"database_storage_successful": True},
                            metadata={"status": "success"}
                        )

                except Exception as db_error:
                    logger.error(f"Failed to store metrics analysis in database: {db_error}")

                # Add JIRA comment immediately after metrics analysis completion with tracing
                with langfuse.start_as_current_span(name="add-jira-comment-metrics") as jira_span:
                    await self._add_jira_comment_for_analysis(
                        state,
                        "metrics",
                        metrics_result
                    )

                    jira_span.update(
                        output={"jira_comment_added": True},
                        metadata={"status": "success"}
                    )

                span.update(
                    output={
                        "metrics_analysis_completed": True,
                        "analysis_length": len(metrics_result),
                        "structured_metrics_created": bool(state.get("structured_metrics")),
                        "file_saved": str(metrics_file),
                        "database_stored": True,
                        "jira_updated": True
                    },
                    metadata={"status": "success", "workflow_position": 9}
                )

                logger.info(f"[{get_timestamp()}] Metrics correlation analysis complete, saved to {metrics_file}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in metrics correlation analysis: {error_msg}")
                state["metrics_correlation_result"] = f"Metrics correlation analysis failed: {error_msg}"
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 9}
                )

        return state

    # @observe(name="correlation-summary-generation")
    async def _generate_correlation_summary_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 9.5: Generate comprehensive correlation summary and filter PromQL queries with comprehensive tracing."""
        with langfuse.start_as_current_span(name="generate-correlation-summary") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 9.5: Generating correlation summary and filtering PromQL queries")

            try:
                # Get correlation and metrics analysis from previous steps
                log_correlation = state.get("log_correlation_result", "")
                metrics_correlation = state.get("metrics_correlation_result", "")

                span.update(
                    input={
                        "log_correlation_available": bool(log_correlation),
                        "metrics_correlation_available": bool(metrics_correlation),
                        "log_correlation_length": len(log_correlation) if log_correlation else 0,
                        "metrics_correlation_length": len(metrics_correlation) if metrics_correlation else 0
                    },
                    metadata={"step": "correlation_summary_generation", "workflow_position": 9.5}
                )

                if not log_correlation and not metrics_correlation:
                    logger.warning("No correlation or metrics analysis available for summary")
                    state["correlation_summary"] = "No correlation data available for summary generation"
                    state["filtered_promql_queries"] = []

                    span.update(
                        output={"summary_result": "no_data_available"},
                        metadata={"status": "no_data", "workflow_position": 9.5}
                    )
                    return state

                # Generate focused correlation summary with comprehensive tracing
                summary_prompt = f"""
                You are an expert SRE correlation analyst. Create a concise, insightful summary that explains exactly how the log correlation findings and the metrics analysis relate to the specific alert below. Do not include recommendations or generic guidance. Focus only on evidence and how it maps to the alert condition.

                **ALERT CONTEXT:**
                - Alert Name: {state['alertname']}
                - Service: {state['service']}
                - Severity: {state['severity']}
                - Description: {state['description']}
                - Timestamp: {state.get('timestamp', 'Unknown')}

                **LOG CORRELATION REPORT:**
                {log_correlation if log_correlation else "No log correlation analysis was performed"}

                **METRICS REPORT:**
                {metrics_correlation if metrics_correlation else "No metrics analysis was performed"}

                **INSTRUCTIONS:**
                - Explain how the log patterns relate to the alert: key messages, error rates, components, time windows.
                - Explain how the metrics behavior relates to the alert: which metrics breached, thresholds, magnitude, timing vs alert.
                - If logs or metrics alone explain the alert, state that; if both contribute, describe how they reinforce each other.
                - Be specific about timing relationships between evidence and the alert trigger.
                - Do NOT include remediation steps, recommendations, or impact statements.

                **REQUIRED MARKDOWN OUTPUT:**

                # Correlation Summary for Alert: {state['alertname']}

                ## How Logs Relate to the Alert
                - [Key log findings that directly map to the alert condition, with timing]

                ## How Metrics Relate to the Alert
                - [Key metric changes/breaches that directly map to the alert condition, with thresholds and timing]

                ## Additional Alert Details
                - Grafana Alert Rule: {state.get('grafana_alert_info', {}).get('title', 'N/A')}
                - Alert Expressions: {[expr.get('expr', '') for expr in state.get('grafana_alert_info', {}).get('alert_expressions', [])] if state.get('grafana_alert_info', {}).get('alert_expressions') else 'N/A'}

                ## Timeline of Evidence vs Alert
                - [Short, ordered bullets showing when evidence appeared vs when the alert fired]

                ## Synthesis: Why This Alert Fired
                - [1-3 bullets connecting the evidence to the alert condition; no recommendations]

                Keep it precise and strictly evidence-to-alert mapping.
                """

                messages = [
                    SystemMessage(content="You are an expert SRE correlation analyst. Create focused markdown analysis that directly explains why a specific alert fired, using only relevant log and metrics evidence."),
                    HumanMessage(content=summary_prompt)
                ]

                with langfuse.start_as_current_span(name="llm-correlation-summary-generation") as llm_span:
                    response = await self.llm.ainvoke(
                        messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    }
                    )

                    correlation_summary = response.content

                    llm_span.update(
                        input={"summary_prompt_length": len(summary_prompt)},
                        output={"summary_length": len(correlation_summary)},
                        metadata={"llm_task": "correlation_summary"}
                    )

                logger.info(f"[{get_timestamp()}] Generated correlation summary, length: {len(correlation_summary)} characters")
                state["correlation_summary"] = correlation_summary

                # Filter PromQL queries - only keep successful ones correlated to alert with tracing
                with langfuse.start_as_current_span(name="filter-promql-queries") as filter_span:
                    await self._filter_promql_queries_for_storage(state)

                    filter_span.update(
                        output={"filtered_queries_count": len(state.get("filtered_promql_queries", []))},
                        metadata={"status": "success"}
                    )

                # Store results in database and call feedback endpoint with tracing
                with langfuse.start_as_current_span(name="store-correlation-summary") as store_span:
                    await self._store_correlation_summary_and_feedback(state)

                    store_span.update(
                        output={"storage_attempted": True},
                        metadata={"status": "success"}
                    )

                state["current_step"] = "correlation_summary_generated"

                span.update(
                    output={
                        "correlation_summary_generated": True,
                        "summary_length": len(correlation_summary),
                        "filtered_queries_count": len(state.get("filtered_promql_queries", [])),
                        "storage_completed": True
                    },
                    metadata={"status": "success", "workflow_position": 9.5}
                )

                logger.info("Correlation summary generation completed successfully")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating correlation summary: {error_msg}")
                state["correlation_summary"] = f"Correlation summary generation failed: {error_msg}"
                state["filtered_promql_queries"] = []
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 9.5}
                )

        return state

    # @observe(name="promql-query-filtering")
    async def _filter_promql_queries_for_storage(self, state: CorrelationAgentState):
        """Filter PromQL queries to store only successful ones that correlate to the alert with comprehensive tracing."""
        with langfuse.start_as_current_span(name="filter-promql-queries") as span:
            try:
                generated_queries = state.get("generated_promql_queries", [])
                fetched_metrics = state.get("fetched_metrics", {})

                span.update(
                    input={
                        "generated_queries_count": len(generated_queries),
                        "fetched_metrics_count": len(fetched_metrics)
                    },
                    metadata={"component": "promql_query_filtering"}
                )

                if not generated_queries:
                    logger.warning("No PromQL queries to filter")
                    state["filtered_promql_queries"] = []

                    span.update(
                        output={"filtered_queries": 0, "reason": "no_queries"},
                        metadata={"status": "no_data"}
                    )
                    return

                # Get successful queries with results
                successful_queries = []
                for i, query_config in enumerate(generated_queries):
                    query_key = f"query_{i+1}"
                    if query_key in fetched_metrics and fetched_metrics[query_key].get("status") == "success":
                        results = fetched_metrics[query_key].get("results", {})
                        # Check if has actual data - handle both dict and list formats
                        has_data = False
                        if isinstance(results, dict):
                            has_data = bool(results.get("data", {}).get("result"))
                        elif isinstance(results, list):
                            has_data = len(results) > 0

                        if has_data:
                            successful_queries.append({
                                "query_config": query_config,
                                "results": results,
                                "query_index": i + 1
                            })

                        success_span.update(
                            output={"successful_queries_count": len(successful_queries)},
                            metadata={"status": "success"}
                        )

                    if not successful_queries:
                        logger.warning("No successful PromQL queries found")
                        state["filtered_promql_queries"] = []

                        span.update(
                            output={"filtered_queries": 0, "reason": "no_successful_queries"},
                            metadata={"status": "no_data"}
                        )
                        return

                    # Use LLM to filter queries that actually correlate to the alert with tracing
                    filter_prompt = f"""
                    You are analyzing PromQL queries to determine which ones are relevant for explaining this specific alert.

                    **ALERT CONTEXT:**
                    - Alert: {state['alertname']}
                    - Service: {state['service']}
                    - Severity: {state['severity']}
                    - Description: {state['description']}

                **SUCCESSFUL PROMQL QUERIES:**
                {json.dumps([{"index": i+1, "query": q["query_config"].get("expr", ""), "service": q["query_config"].get("service", ""), "metric_type": q["query_config"].get("metric_type", "")} for i, q in enumerate(successful_queries)], indent=2)}

                **TASK:**
                Based on the PromQL expressions, services, and metric types, determine which queries are directly relevant to explaining this alert.
                Consider:
                - Does the metric_type relate to the alert condition?
                - Does the PromQL query (expr) measure something that could cause or correlate with this alert?
                - Is this query from the same service or related infrastructure?

                Return a JSON array of query indices (1-based) that are relevant to explaining this alert.
                If unsure, include the query - it's better to have more context than less.

                Example response: [1, 2, 3]

                    Return only the JSON array, no other text.
                    """

                    messages = [
                        SystemMessage(content="You are a metrics correlation expert. Filter queries to only include those that help explain a specific alert condition."),
                        HumanMessage(content=filter_prompt)
                    ]

                logger.info(f"[{get_timestamp()}] PromQL filtering prompt: {filter_prompt}")
                response = await self.llm.ainvoke(messages, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    })
                logger.info(f"[{get_timestamp()}] LLM response for query filtering: {response.content}")

                try:
                    relevant_indices = json.loads(response.content.strip())
                    if not isinstance(relevant_indices, list):
                        raise ValueError("Response is not a list")
                    logger.info(f"[{get_timestamp()}] Parsed relevant indices: {relevant_indices}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Could not parse LLM response for query filtering: {e}. Using all successful queries.")
                    relevant_indices = [q["query_index"] for q in successful_queries]
                    logger.info(f"[{get_timestamp()}] Fallback relevant indices: {relevant_indices}")

                # Build filtered queries in the required format
                filtered_queries = []
                for query_data in successful_queries:
                    if query_data["query_index"] in relevant_indices:
                        query_config = query_data["query_config"]

                        # Extract step_seconds with safe conversion
                        step_value = query_config.get("stepSeconds", query_config.get("step", "15"))
                        if isinstance(step_value, str):
                            step_value = step_value.replace("s", "")
                        try:
                            step_seconds = int(step_value)
                        except (ValueError, TypeError):
                            step_seconds = 15  # Default fallback

                        filtered_queries.append({
                            "query": query_config.get("expr", ""),
                            "start_time": query_config.get("startTime", ""),
                            "end_time": query_config.get("endTime", ""),
                            "metric_name": query_config.get("metric_type", query_config.get("service", "Unknown metric")),
                            "step_seconds": step_seconds,
                            "query_type": query_config.get("queryType", "range")
                        })

                        state["filtered_promql_queries"] = filtered_queries

                        span.update(
                            output={
                                "filtered_queries_count": len(filtered_queries),
                                "successful_queries_count": len(successful_queries),
                                "relevant_indices": relevant_indices
                            },
                            metadata={"status": "success"}
                        )

                        logger.info(f"[{get_timestamp()}] Filtered {len(filtered_queries)} relevant PromQL queries from {len(successful_queries)} successful queries")

            except Exception as e:
                logger.error(f"Error filtering PromQL queries: {e}")
                state["filtered_promql_queries"] = []

                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )

    # @observe(name="correlation-summary-storage")
    async def _store_correlation_summary_and_feedback(self, state: CorrelationAgentState):
        """Store correlation summary and filtered PromQL queries in database, then call feedback endpoint with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-correlation-summary-feedback") as span:
            if not self.SessionLocal:
                logger.warning("Database not configured, skipping correlation summary storage")

                span.update(
                    output={"storage_attempted": False, "reason": "no_database"},
                    metadata={"status": "skipped"}
                )
                return

            try:
                # Get incident ID from state with tracing
                with langfuse.start_as_current_span(name="extract-incident-id") as extract_span:
                    incident_key = state["incident_key"]
                    incident_id = None
                    if incident_key.startswith("incidents:"):
                        parts = incident_key.split(":")
                        if len(parts) >= 2:
                            try:
                                incident_id = int(parts[1])
                                extract_span.update(
                                    output={"incident_id": incident_id, "extraction_successful": True},
                                    metadata={"status": "success"}
                                )
                            except ValueError:
                                logger.error(f"Could not extract incident ID from key: {incident_key}")
                                extract_span.update(
                                    output={"error": f"Invalid incident ID in key: {incident_key}"},
                                    metadata={"status": "error"}
                                )
                                return
                    else:
                        logger.error(f"Invalid incident key format: {incident_key}")
                        extract_span.update(
                            output={"error": f"Invalid incident key format: {incident_key}"},
                            metadata={"status": "error"}
                        )
                        return

                span.update(
                    input={"incident_id": incident_id, "incident_key": incident_key},
                    metadata={"component": "database_storage"}
                )

                # Store correlation summary and PromQL queries in single transaction with tracing and retry logic
                max_retries = 3
                retry_delay = 1
                storage_success = False

                for attempt in range(1, max_retries + 1):
                    session = self.SessionLocal()
                    try:
                        with langfuse.start_as_current_span(name="database-transaction") as db_span:
                            # Store correlation summary
                            correlation_summary = state.get("correlation_summary", "")
                            summary_stored = False
                            if correlation_summary:
                                await self._store_individual_analysis(session, incident_id, "correlation_summary", correlation_summary)
                                summary_stored = True

                            # Store filtered PromQL queries
                            filtered_queries = state.get("filtered_promql_queries", [])
                            queries_stored = 0
                            if filtered_queries:
                                await self._store_individual_analysis(session, incident_id, "promql_queries", filtered_queries)
                                queries_stored = len(filtered_queries)

                            from datetime import datetime
                            from sqlalchemy import text

                            session.execute(
                                text("UPDATE incidents SET mtta = :nowutc WHERE id = :id"),
                                {"nowutc": datetime.utcnow(), "id": incident_id}
                            )

                            session.commit()

                            storage_success = True
                            logger.info(f"[{get_timestamp()}] PostgreSQL storage succeeded on attempt {attempt}")
                            db_span.update(
                                output={
                                    "summary_stored": summary_stored,
                                    "queries_stored": queries_stored,
                                    "mtta_updated": True,
                                    "attempts": attempt
                                },
                                metadata={"status": "success"}
                            )

                        span.update(
                            output={
                                "storage_successful": True,
                                "incident_id": incident_id,
                                "summary_stored": bool(correlation_summary),
                                "queries_stored": len(filtered_queries),
                                "attempts": attempt
                            },
                            metadata={"status": "success"}
                        )

                        logger.info(f"[{get_timestamp()}] Successfully stored correlation summary and feedback for incident {incident_id}")
                        break

                    except Exception as e:
                        logger.warning(f"PostgreSQL storage attempt {attempt} failed: {e}")
                        session.rollback()
                        if attempt < max_retries:
                            logger.info(f"[{get_timestamp()}] Retrying PostgreSQL storage in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"All {max_retries} PostgreSQL storage attempts failed")
                            span.update(
                                output={"error": str(e), "attempts": attempt},
                                metadata={"status": "error"}
                            )
                            raise e
                    finally:
                        session.close()

            except Exception as e:
                logger.error(f"Error in correlation summary storage: {e}")

                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )

    # @observe(name="results-storage")
    async def _store_results_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 10: Store results in Redis and PostgreSQL with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-results") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 10: Storing correlation and metrics results")

            try:
                # Prepare results for storage with tracing
                with langfuse.start_as_current_span(name="prepare-results-data") as prep_span:
                    results = {
                        "correlation_analysis": state.get("log_correlation_result", ""),
                        "structured_correlation": state.get("structured_correlation"),  # Already a dict
                        "metrics_analysis": state.get("metrics_correlation_result", ""),
                        "structured_metrics": state.get("structured_metrics", {}),
                        "correlation_summary": state.get("correlation_summary", ""),
                        "filtered_promql_queries": state.get("filtered_promql_queries", []),
                        "is_metric_based": state.get("is_metric_based", True),
                        "current_step": state.get("current_step", "completed"),
                        "error": state.get("error"),
                        "analysis_timestamp": datetime.now().isoformat(),
                        "fetched_logs_summary": {key: len(logs.get("results", [])) if isinstance(logs.get("results"), list) else 0
                                               for key, logs in state.get("fetched_logs", {}).items()},
                        "fetched_metrics_summary": list(state.get("fetched_metrics", {}).keys())
                    }

                    prep_span.update(
                        output={
                            "results_keys": list(results.keys()),
                            "correlation_size": len(results["correlation_analysis"]),
                            "metrics_size": len(results["metrics_analysis"]),
                            "summary_size": len(results["correlation_summary"])
                        },
                        metadata={"status": "success"}
                    )

                span.update(
                    input={
                        "results_keys": list(results.keys()),
                        "correlation_available": bool(results["correlation_analysis"]),
                        "metrics_available": bool(results["metrics_analysis"]),
                        "summary_available": bool(results["correlation_summary"])
                    },
                    metadata={"step": "results_storage", "workflow_position": 10}
                )

                # Store in Redis with comprehensive tracing
                try:
                    with langfuse.start_as_current_span(name="redis-storage") as redis_span:
                        await self.store_results_in_redis(state["incident_key"], results)
                        state["redis_stored"] = True

                        redis_span.update(
                            input={"incident_key": state["incident_key"], "results_size": len(str(results))},
                            output={"redis_stored": True},
                            metadata={"status": "success"}
                        )

                        logger.info("Results stored in Redis successfully")

                except Exception as redis_error:
                    logger.error(f"Failed to store in Redis: {redis_error}")
                    state["redis_stored"] = False

                # Store in PostgreSQL database using storage_manager
                try:
                    with langfuse.start_as_current_span(name="postgresql-storage") as db_span:
                        success = await self.storage_manager.store_all_correlation_results(
                            state["incident_key"],
                            state,
                            results
                        )
                        state["postgres_stored"] = success

                        db_span.update(
                            input={"incident_key": state["incident_key"], "results_keys": list(results.keys())},
                            output={"postgres_stored": success},
                            metadata={"status": "success" if success else "failed"}
                        )

                        if success:
                            logger.info("Results stored in PostgreSQL successfully")
                        else:
                            logger.error("Failed to store in PostgreSQL")

                except Exception as db_error:
                    logger.error(f"Failed to store in PostgreSQL: {db_error}")
                    state["postgres_stored"] = False

                state["current_step"] = "results_stored"

                span.update(
                    output={
                        "redis_stored": state.get("redis_stored", False),
                        "postgres_stored": state.get("postgres_stored", False),
                        "storage_completed": True
                    },
                    metadata={"status": "success", "workflow_position": 10}
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error storing results: {error_msg}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 10}
                )

        return state

    # @observe(name="jira-update")
    async def _update_jira_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 11: Jira update step - now handled individually after each analysis with comprehensive tracing."""
        with langfuse.start_as_current_span(name="update-jira") as span:
            span.update(session_id=str(uuid.uuid4().hex))
            logger.info("STEP 11: Jira comments already added individually after each analysis")

            try:
                span.update(
                    input={
                        "jira_comments_added_individually": True,
                        "correlation_analysis_available": bool(state.get("log_correlation_result")),
                        "metrics_analysis_available": bool(state.get("metrics_correlation_result"))
                    },
                    metadata={"step": "jira_update", "workflow_position": 11}
                )

                # Individual Jira comments are now added immediately after correlation and metrics analysis
                # This follows the same pattern as main agents folder
                state["jira_updated"] = True  # Set to true since individual comments were added
                state["current_step"] = "jira_updated"

                span.update(
                    output={
                        "jira_updated": True, 
                        "method": "individual_comments",
                        "comments_added_after_correlation": bool(state.get("log_correlation_result")),
                        "comments_added_after_metrics": bool(state.get("metrics_correlation_result"))
                    },
                    metadata={"status": "success", "workflow_position": 11}
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in Jira update step: {error_msg}")
                state["error"] = error_msg

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 11}
                )

        return state
    
    # @observe(name="jira-comment-analysis")
    async def _add_jira_comment_for_analysis(self, state: CorrelationAgentState, analysis_type: str, analysis_content: str):
        """Add a Jira comment for specific analysis step using jira_manager."""
        await self.jira_manager.add_analysis_comment(state, analysis_type, analysis_content)

    # @observe(name="fallback-jira-comment")
    def _fallback_jira_comment(self, state: CorrelationAgentState) -> str:
        """Fallback Jira comment if LLM generation fails with tracing."""
        with langfuse.start_as_current_span(name="generate-fallback-jira-comment") as span:
            try:
                incident_key = state.get("incident_key", "Unknown")
                alert_name = state.get("alertname", "Unknown Alert")
                service = state.get("service", "Unknown Service")
                severity = state.get("severity", "Unknown")

                log_analysis = state.get("log_correlation_result", "No log analysis available")
                metrics_analysis = state.get("metrics_correlation_result", "No metrics analysis available")

                span.update(
                    input={
                        "incident_key": incident_key,
                        "alert_name": alert_name,
                        "service": service,
                        "severity": severity,
                        "log_analysis_length": len(log_analysis),
                        "metrics_analysis_length": len(metrics_analysis)
                    },
                    metadata={"component": "fallback_comment_generation"}
                )

                fallback_comment = f"""*Automated Correlation Analysis Results*

*Incident Details:*
• Alert: {alert_name}
• Service: {service}
• Severity: {severity}
• Incident Key: {incident_key}

*Log Correlation Analysis:*
{log_analysis[:500]}{"..." if len(log_analysis) > 500 else ""}

*Metrics Correlation Analysis:*
{metrics_analysis[:500]}{"..." if len(metrics_analysis) > 500 else ""}

_Generated by Correlation Agent at {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}_"""

                span.update(
                    output={
                        "fallback_comment_generated": True,
                        "comment_length": len(fallback_comment),
                        "log_truncated": len(log_analysis) > 500,
                        "metrics_truncated": len(metrics_analysis) > 500
                    },
                    metadata={"status": "success"}
                )

                return fallback_comment

            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                logger.error(f"Error generating fallback Jira comment: {e}")
                return "*Automated Correlation Analysis Results* - Error generating comment"

    # @observe(name="unicode-sanitization")
    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize Unicode characters for Jira compatibility."""
        return utils.sanitize_unicode(text)

    # @observe(name="workflow-completion")
    async def _complete_workflow_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 11: Complete the workflow."""
        logger.info("STEP 11: Completing correlation workflow")
        return self.workflow_completion.complete_workflow(state)

    # @observe(name="database-analysis-storage")
    async def _store_analysis_in_database_new(self, state: CorrelationAgentState, results: Dict[str, Any]):
        """Store analysis results in PostgreSQL database using the same pattern as main agents with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-analysis-database") as span:
            span.update(session_id=str(uuid.uuid4().hex))

            if not self.SessionLocal:
                logger.warning("Database not configured, skipping storage")
                span.update(
                    output={"storage_attempted": False, "reason": "no_database_config"},
                    metadata={"status": "skipped"}
                )
                return

            try:
                # Extract incident ID from state with comprehensive tracing
                with langfuse.start_as_current_span(name="extract-incident-id-from-key") as extract_span:
                    incident_key = state["incident_key"]
                    incident_id = None
                    extraction_method = "unknown"

                    # Extract incident ID from incident_key if it contains one
                    if incident_key.startswith("incidents:"):
                        # New folder format: incidents:123:main
                        parts = incident_key.split(":")
                        if len(parts) >= 2 and parts[1].isdigit():
                            incident_id = int(parts[1])
                            extraction_method = "incidents_format"
                    elif incident_key.startswith("incident:"):
                        # Old format: incident:123
                        parts = incident_key.split(":")
                        if len(parts) == 2 and parts[1].isdigit():
                            incident_id = int(parts[1])
                            extraction_method = "incident_format"
                    elif incident_key.isdigit():
                        # Direct ID
                        incident_id = int(incident_key)
                        extraction_method = "direct_id"

                    extract_span.update(
                        input={"incident_key": incident_key, "key_format": "unknown"},
                        output={
                            "incident_id": incident_id,
                            "extraction_successful": bool(incident_id),
                            "extraction_method": extraction_method
                        },
                        metadata={"status": "success" if incident_id else "failed"}
                    )

                if not incident_id:
                    logger.warning(f"Could not extract incident ID from key: {incident_key}, skipping database storage")
                    span.update(
                        output={"storage_completed": False, "reason": "invalid_incident_key"},
                        metadata={"status": "skipped"}
                    )
                    return

                logger.info(f"[{get_timestamp()}] Storing structured correlation results for incident ID {incident_id}")

                span.update(
                    input={
                        "incident_id": incident_id,
                        "results_keys": list(results.keys()),
                        "correlation_available": bool(results.get("structured_correlation")),
                        "metrics_available": bool(results.get("metrics_analysis"))
                    },
                    metadata={"component": "database_storage"}
                )

                # Database transaction with comprehensive tracing and retry logic
                max_retries = 3
                retry_delay = 1
                transaction_success = False

                for attempt in range(1, max_retries + 1):
                    session = self.SessionLocal()
                    try:
                        with langfuse.start_as_current_span(name="database-transaction-correlation") as db_span:
                            operations_completed = []

                            # Store correlation analysis using individual field updates
                            await self._store_individual_analysis(session, incident_id, "correlation", results.get("structured_correlation"))
                            operations_completed.append("correlation")

                            # Store metrics analysis
                            if results.get("metrics_analysis"):
                                await self._store_individual_analysis(session, incident_id, "metrics", results.get("metrics_analysis"))
                                operations_completed.append("metrics")

                            # Update additional fields specific to correlation agent
                            service_deps = state.get("service_dependencies", [])
                            if service_deps:
                                deps_json = json.dumps(service_deps)
                                session.execute(
                                    text("UPDATE incidents SET dependencies = :data WHERE id = :id"),
                                    {"data": deps_json, "id": incident_id}
                                )
                                logger.info(f"[{get_timestamp()}] Updated dependencies with {len(service_deps)} services")
                                operations_completed.append("dependencies")

                            # Update alert_metadata field with tracing
                            with langfuse.start_as_current_span(name="update-alert-metadata") as metadata_span:
                                grafana_alert_info = state.get("grafana_alert_info")
                                if grafana_alert_info:
                                    # Get existing metadata first
                                    existing_result = session.execute(
                                        text("SELECT alert_metadata FROM incidents WHERE id = :id"),
                                        {"id": incident_id}
                                    ).fetchone()

                                    existing_metadata = {}
                                    if existing_result and existing_result[0]:
                                        try:
                                            existing_metadata = json.loads(existing_result[0])
                                        except:
                                            existing_metadata = {}

                                    existing_metadata.update({
                                        "grafana_alert_info": grafana_alert_info,
                                        "processed_by_correlation_agent": True,
                                        "processing_timestamp": datetime.now().isoformat(),
                                        "alertname": state.get("alertname"),
                                        "severity": state.get("severity"),
                                        "timestamp": state.get("timestamp")
                                    })

                                    session.execute(
                                        text("UPDATE incidents SET alert_metadata = :data WHERE id = :id"),
                                        {"data": json.dumps(existing_metadata), "id": incident_id}
                                    )
                                    logger.info("Updated alert_metadata with Grafana alert info")
                                    operations_completed.append("alert_metadata")

                                    metadata_span.update(
                                        output={"metadata_updated": True, "grafana_info_included": True},
                                        metadata={"status": "success"}
                                    )

                            # Update execution metrics and last updated by
               
                            operations_completed.append("execution_metrics")

                            session.commit()

                            transaction_success = True
                            logger.info(f"[{get_timestamp()}] PostgreSQL transaction succeeded on attempt {attempt}")
                            db_span.update(
                                output={
                                    "operations_completed": operations_completed,
                                    "total_operations": len(operations_completed),
                                    "commit_successful": True,
                                    "attempts": attempt
                                },
                                metadata={"status": "success"}
                            )

                            logger.info(f"[{get_timestamp()}] Successfully updated incident {incident_id} in database with structured data")

                        span.update(
                            output={
                                "storage_successful": True,
                                "incident_id": incident_id,
                                "operations_completed": operations_completed,
                                "attempts": attempt
                            },
                            metadata={"status": "success"}
                        )
                        break

                    except Exception as db_error:
                        logger.warning(f"PostgreSQL transaction attempt {attempt} failed: {db_error}")
                        session.rollback()
                        if attempt < max_retries:
                            logger.info(f"[{get_timestamp()}] Retrying PostgreSQL transaction in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"All {max_retries} PostgreSQL transaction attempts failed")
                            span.update(
                                output={"error": str(db_error), "transaction_rolled_back": True, "attempts": attempt},
                                metadata={"status": "error"}
                            )
                            raise db_error

                    finally:
                        session.close()

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error storing analysis in database: {error_msg}")

                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error"}
                )
                if 'session' in locals():
                    try:
                        session.rollback()
                        session.close()
                    except:
                        pass
                raise

    # @observe(name="individual-analysis-storage")
    async def _store_individual_analysis(self, session, incident_id: int, analysis_type: str, structured_data: Any):
        """Store individual analysis results in database - same pattern as main agents with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-individual-analysis") as span:
            try:
                span.update(
                    input={
                        "incident_id": incident_id,
                        "analysis_type": analysis_type,
                        "data_type": type(structured_data).__name__,
                        "data_size": len(str(structured_data)) if structured_data else 0
                    },
                    metadata={"component": "individual_analysis_storage"}
                )

                if analysis_type == "correlation":
                    # Update correlation_result field
                    if structured_data:
                        with langfuse.start_as_current_span(name="store-correlation-data") as corr_span:
                            data_json = json.dumps(structured_data, indent=2) if isinstance(structured_data, dict) else json.dumps(structured_data.model_dump()) if hasattr(structured_data, 'model_dump') else json.dumps(str(structured_data))
                            session.execute(
                                text("UPDATE incidents SET correlation_result = :data WHERE id = :id"),
                                {"data": data_json, "id": incident_id}
                            )
                            log_count = len(structured_data.get('correlated_logs', [])) if isinstance(structured_data, dict) else 0
                            logger.info(f"[{get_timestamp()}] Successfully updated correlation field in database for incident {incident_id} with {log_count} structured logs")

                            corr_span.update(
                                output={"correlation_stored": True, "log_count": log_count},
                                metadata={"status": "success"}
                            )

                elif analysis_type == "metrics":
                    # Update metric_insights field
                    with langfuse.start_as_current_span(name="store-metrics-data") as metrics_span:
                        metrics_text = structured_data if isinstance(structured_data, str) else str(structured_data)
                        session.execute(
                            text("UPDATE incidents SET metric_insights = :data WHERE id = :id"),
                            {"data": metrics_text, "id": incident_id}
                        )
                        logger.info(f"[{get_timestamp()}] Successfully updated metrics field in database for incident {incident_id}")

                        metrics_span.update(
                            output={"metrics_stored": True, "text_length": len(metrics_text)},
                            metadata={"status": "success"}
                        )

                elif analysis_type == "correlation_summary":
                    # Update correlation_summary field
                    if structured_data:
                        with langfuse.start_as_current_span(name="store-summary-data") as summary_span:
                            summary_text = structured_data if isinstance(structured_data, str) else str(structured_data)
                            session.execute(
                                text("UPDATE incidents SET correlation_summary = :data WHERE id = :id"),
                                {"data": summary_text, "id": incident_id}
                            )
                            logger.info(f"[{get_timestamp()}] Successfully updated correlation_summary field in database for incident {incident_id}")

                            summary_span.update(
                                output={"summary_stored": True, "summary_length": len(summary_text)},
                                metadata={"status": "success"}
                            )

                elif analysis_type == "promql_queries":
                    # Update correlation_metrics_promql field
                    if structured_data:
                        with langfuse.start_as_current_span(name="store-promql-data") as promql_span:
                            queries_json = json.dumps(structured_data, indent=2) if isinstance(structured_data, (list, dict)) else json.dumps(str(structured_data))
                            session.execute(
                                text("UPDATE incidents SET correlation_metrics_promql = :data WHERE id = :id"),
                                {"data": queries_json, "id": incident_id}
                            )
                            logger.info(f"[{get_timestamp()}] Successfully updated correlation_metrics_promql field in database for incident {incident_id}")

                            promql_span.update(
                                output={
                                    "promql_stored": True,
                                    "queries_count": len(structured_data) if isinstance(structured_data, list) else 1
                                },
                                metadata={"status": "success"}
                            )
                else:
                    logger.warning(f"Unknown analysis type: {analysis_type}")
                    span.update(
                        output={"storage_attempted": False, "reason": "unknown_analysis_type"},
                        metadata={"status": "skipped"}
                    )
                    return

                span.update(
                    output={
                        "storage_successful": True,
                        "analysis_type": analysis_type,
                        "incident_id": incident_id
                    },
                    metadata={"status": "success"}
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error storing {analysis_type} analysis in database: {e}")

                span.update(
                    output={"error": error_msg, "analysis_type": analysis_type},
                    metadata={"status": "error"}
                )
                raise
    
    # @observe(name="fallback-correlation-structure")
    def _create_fallback_correlation_structure(self, correlation_result: str) -> CorrelationArray:
        """Create fallback structured correlation from text result."""
        return utils.create_fallback_correlation_structure(correlation_result)

    # @observe(name="metric-decision-extraction")
    def _extract_metric_decision_from_correlation(self, structured_correlation) -> bool:
        """Extract metric-based decision from correlation analysis."""
        return utils.extract_metric_decision_from_correlation(structured_correlation)

    # @observe(name="redis-incident-retrieval")
    async def get_incident_from_redis(self, incident_key: str) -> Optional[Dict[str, Any]]:
        """Get incident information from Redis folder structure with comprehensive tracing."""
        with langfuse.start_as_current_span(name="get-incident-from-redis") as span:
            span.update(session_id=str(uuid.uuid4().hex))

            if not self.redis_client:
                logger.warning("Redis not configured")
                span.update(
                    output={"incident_retrieved": False, "reason": "redis_not_configured"},
                    metadata={"status": "skipped"}
                )
                return None

            try:
                span.update(
                    input={"incident_key": incident_key, "redis_available": True},
                    metadata={"component": "redis_incident_retrieval"}
                )

                incident_data = None
                retrieval_method = "unknown"
                attempts_made = []

                # Handle both old format (incident:123) and new folder format (incidents:123:main)
                with langfuse.start_as_current_span(name="redis-key-resolution") as resolve_span:
                    if incident_key.startswith("incidents:") and ":main" in incident_key:
                        # New folder format
                        incident_data = self.redis_client.get(incident_key)
                        retrieval_method = "new_folder_format"
                        attempts_made.append("new_folder_format")

                    elif incident_key.startswith("incident:"):
                        # Old format, convert to new format
                        incident_id = incident_key.split(":")[-1]
                        new_key = f"incidents:{incident_id}:main"
                        incident_data = self.redis_client.get(new_key)
                        attempts_made.append("converted_to_new_format")

                        if incident_data:
                            retrieval_method = "converted_to_new_format"
                        else:
                            # Fallback to old format
                            incident_data = self.redis_client.get(incident_key)
                            retrieval_method = "old_format_fallback"
                            attempts_made.append("old_format_fallback")
                    else:
                        # Direct key lookup
                        incident_data = self.redis_client.get(incident_key)
                        retrieval_method = "direct_key_lookup"
                        attempts_made.append("direct_key_lookup")

                    resolve_span.update(
                        output={
                            "retrieval_method": retrieval_method,
                            "attempts_made": attempts_made,
                            "data_found": bool(incident_data)
                        },
                        metadata={"status": "success"}
                    )

                # Parse JSON data if found
                parsed_data = None
                if incident_data:
                    with langfuse.start_as_current_span(name="parse-incident-json") as parse_span:
                        try:
                            parsed_data = json.loads(incident_data)
                            data_keys = list(parsed_data.keys()) if isinstance(parsed_data, dict) else []

                            parse_span.update(
                                output={
                                    "parsing_successful": True,
                                    "data_type": type(parsed_data).__name__,
                                    "data_keys_count": len(data_keys),
                                    "data_keys": data_keys[:10] if len(data_keys) <= 10 else data_keys[:10] + ["..."]
                                },
                                metadata={"status": "success"}
                            )

                        except json.JSONDecodeError as json_err:
                            parse_span.update(
                                output={"error": str(json_err), "raw_data_length": len(incident_data)},
                                metadata={"status": "error"}
                            )
                            logger.error(f"Failed to parse incident JSON: {json_err}")
                            return None

                span.update(
                    output={
                        "incident_retrieved": bool(parsed_data),
                        "retrieval_method": retrieval_method,
                        "attempts_made": attempts_made,
                        "data_size": len(str(parsed_data)) if parsed_data else 0
                    },
                    metadata={"status": "success" if parsed_data else "not_found"}
                )

                if parsed_data:
                    logger.info(f"[{get_timestamp()}] Successfully retrieved incident {incident_key} using method: {retrieval_method}")
                else:
                    logger.warning(f"Incident {incident_key} not found in Redis after {len(attempts_made)} attempts")

                return parsed_data

            except Exception as e:
                span.update(
                    output={"error": str(e), "incident_retrieved": False},
                    metadata={"status": "error"}
                )
                logger.error(f"Error getting incident from Redis: {e}")
                return None

    # @observe(name="redis-results-storage")
    async def store_results_in_redis(self, incident_key: str, results: Dict[str, Any]):
        """Store correlation and metrics results in Redis folder structure with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-results-in-redis") as span:
            span.update(session_id=str(uuid.uuid4().hex))

            if not self.redis_client:
                logger.warning("Redis not configured")
                span.update(
                    output={"storage_attempted": False, "reason": "redis_not_configured"},
                    metadata={"status": "skipped"}
                )
                return

            try:
                span.update(
                    input={
                        "incident_key": incident_key,
                        "results_keys": list(results.keys()),
                        "results_size": len(str(results)),
                        "redis_available": True
                    },
                    metadata={"component": "redis_results_storage"}
                )

                # Get existing incident data with tracing
                with langfuse.start_as_current_span(name="get-existing-incident-data") as get_span:
                    incident_data = await self.get_incident_from_redis(incident_key)

                    get_span.update(
                        output={
                            "existing_data_found": bool(incident_data),
                            "existing_data_size": len(str(incident_data)) if incident_data else 0
                        },
                        metadata={"status": "success"}
                    )

                if incident_data:
                    # Update with results with tracing
                    with langfuse.start_as_current_span(name="merge-results-with-existing") as merge_span:
                        original_size = len(str(incident_data))
                        incident_data.update(results)
                        updated_size = len(str(incident_data))

                        merge_span.update(
                            output={
                                "merge_successful": True,
                                "original_size": original_size,
                                "updated_size": updated_size,
                                "size_increase": updated_size - original_size
                            },
                            metadata={"status": "success"}
                        )

                    # Store updated main data with tracing
                    with langfuse.start_as_current_span(name="store-main-incident-data") as main_span:
                        self.redis_client.set(incident_key, json.dumps(incident_data), ex=604800)  # 7 days expiry

                        main_span.update(
                            output={
                                "main_data_stored": True,
                                "expiry_seconds": 604800,
                                "data_size": len(json.dumps(incident_data))
                            },
                            metadata={"status": "success"}
                        )

                    # Store individual components in folder structure with tracing
                    if incident_key.startswith("incidents:") and ":main" in incident_key:
                        with langfuse.start_as_current_span(name="store-folder-structure-components") as folder_span:
                            folder_prefix = incident_key.replace(":main", "")
                            components_stored = []

                            # Store correlation results separately
                            if "correlation_analysis" in results:
                                correlation_key = f"{folder_prefix}:correlation"
                                correlation_data = {
                                    "analysis": results.get("correlation_analysis", ""),
                                    "structured_correlation": results.get("structured_correlation"),
                                    "timestamp": results.get("analysis_timestamp")
                                }
                                self.redis_client.set(correlation_key, json.dumps(correlation_data), ex=604800)
                                components_stored.append("correlation")

                            # Store metrics results separately  
                            if "metrics_analysis" in results:
                                metrics_key = f"{folder_prefix}:metrics"
                                metrics_data = {
                                    "analysis": results.get("metrics_analysis", ""),
                                    "prometheus_metrics": results.get("prometheus_metrics", {}),
                                    "timestamp": results.get("analysis_timestamp")
                                }
                                self.redis_client.set(metrics_key, json.dumps(metrics_data), ex=604800)
                                components_stored.append("metrics")

                            folder_span.update(
                                output={
                                    "folder_prefix": folder_prefix,
                                    "components_stored": components_stored,
                                    "components_count": len(components_stored)
                                },
                                metadata={"status": "success"}
                            )

                    span.update(
                        output={
                            "storage_successful": True,
                            "incident_key": incident_key,
                            "main_data_updated": True,
                            "folder_structure_used": incident_key.startswith("incidents:") and ":main" in incident_key
                        },
                        metadata={"status": "success"}
                    )

                    logger.info(f"[{get_timestamp()}] Updated incident {incident_key} in Redis with results in folder structure")
                else:
                    logger.warning(f"Incident {incident_key} not found in Redis")
                    span.update(
                        output={"storage_successful": False, "reason": "incident_not_found"},
                        metadata={"status": "not_found"}
                    )

            except Exception as e:
                span.update(
                    output={"error": str(e), "storage_successful": False},
                    metadata={"status": "error"}
                )
                logger.error(f"Error storing results in Redis: {e}")

    # @observe(name="incident-analysis-orchestration")
    async def analyze_incident(self, incident_key: str, langfuse_trace_context: None) -> Dict[str, Any]:
        """Main method to analyze an incident using the new LangGraph workflow with comprehensive tracing."""
        with langfuse.start_as_current_span(
            name="analyze-incident",
            trace_context=langfuse_trace_context
        ) as span:
            span.update_trace(session_id=langfuse_trace_context.get("session_id"))
            logger.info(f"[{get_timestamp()}] Starting incident analysis with LangGraph workflow for key: {incident_key}")

            global current_trace_id
            current_trace_id=langfuse_trace_context.get("trace_id")
            
            global current_observation_id
            current_observation_id = langfuse_trace_context.get("parent_span_id")
            
            global _global_session_id
            _global_session_id =  langfuse_trace_context.get("session_id")
             
            try:
                span.update(
                    input={"incident_key": incident_key},
                    metadata={"component": "incident_analysis_orchestration"}
                )

                # Initialize MCP client with tracing
                with langfuse.start_as_current_span(name="initialize-mcp-client") as mcp_span:
                    await self.initialize_mcp_client()

                    mcp_span.update(
                        output={"mcp_client_initialized": bool(self.mcp_client)},
                        metadata={"status": "success"}
                    )

                # Get incident data from Redis with tracing
                with langfuse.start_as_current_span(name="retrieve-incident-data") as retrieve_span:
                    incident_data = await self.get_incident_from_redis(incident_key)
                    if not incident_data:
                        raise ValueError(f"Incident {incident_key} not found in Redis")

                    retrieve_span.update(
                        output={
                            "incident_data_retrieved": True,
                            "data_size": len(str(incident_data)),
                            "data_keys": list(incident_data.keys()) if isinstance(incident_data, dict) else []
                        },
                        metadata={"status": "success"}
                    )

                # Initialize new workflow state with comprehensive tracing
                with langfuse.start_as_current_span(name="create-workflow-state") as state_span:
                    logger.info("🏗️ Creating initial state for LangGraph workflow")

                    # Validate incident_data to avoid unhashable type errors
                    logger.info(f"[{get_timestamp()}] Incident data type: {type(incident_data)}")
                    if not isinstance(incident_data, dict):
                        logger.error(f"Incident data is not a dict: {type(incident_data)}")
                        raise ValueError(f"Expected incident_data to be dict, got {type(incident_data)}")

                    # Create safe copies of any dict values to prevent unhashable type errors
                    safe_incident_data = dict(incident_data) if incident_data else {}

                    logger.info("Building initial state dictionary...")
                    try:
                        initial_state: CorrelationAgentState = {
                            "alert_payload": safe_incident_data,
                            "alert_status": safe_incident_data.get("status", "firing"),
                            # Use standardized fields with fallback support for legacy payloads
                            "alertname": (
                                safe_incident_data.get("alertname") or
                                safe_incident_data.get("alert_name") or
                                "Unknown Alert"
                            ),
                            "severity": safe_incident_data.get("severity", "medium"),
                            "service": (
                                safe_incident_data.get("service") or
                                safe_incident_data.get("service_name") or
                                safe_incident_data.get("container_name") or
                                "unknown"
                            ),
                            "description": (
                                safe_incident_data.get("description") or
                                safe_incident_data.get("alert_context") or
                                safe_incident_data.get("message") or
                                ""
                            ),
                            "timestamp": safe_incident_data.get("timestamp", datetime.now().isoformat()),
                            "incident_key": incident_key,

                            # Processing data
                            "service_dependencies": [],
                            "grafana_alert_info": {},

                            # Log processing workflow
                            "generated_logql_queries": [],
                            "fetched_logs": {},
                            "log_correlation_result": "",
                            "structured_correlation": None,

                            # Metrics processing workflow
                            "generated_promql_queries": [],
                            "fetched_metrics": {},
                            "metrics_correlation_result": "",
                            "structured_metrics": None,

                            # RCA processing workflow
                            "rca_analysis": "",
                            "structured_rca": None,

                            # Correlation summary workflow
                            "correlation_summary": None,
                            "filtered_promql_queries": None,

                            # Control flow
                            "current_step": "starting",
                            "error": None,
                            "completed": False,
                            "is_metric_based": True,
                            "logs_need_more_fetching": False,
                            "metrics_need_more_fetching": False,

                            # Redis and Database integration
                            "redis_stored": False,
                            "postgres_stored": False,
                            "jira_updated": False
                        }

                        state_span.update(
                            output={
                                "initial_state_created": True,
                                "state_keys_count": len(initial_state),
                                "alertname": initial_state["alertname"],
                                "service": initial_state["service"],
                                "severity": initial_state["severity"]
                            },
                            metadata={"status": "success"}
                        )

                        logger.info("Initial state created successfully")

                    except Exception as state_error:
                        logger.error(f"Error creating initial state: {state_error}")
                        logger.error(f"State error type: {type(state_error).__name__}")
                        import traceback
                        logger.error(f"State creation stack trace: {traceback.format_exc()}")

                        state_span.update(
                            output={"error": str(state_error)},
                            metadata={"status": "error"}
                        )
                        raise

                # Execute LangGraph workflow with tracing
                with langfuse.start_as_current_span(name="execute-langgraph-workflow") as workflow_span:
                    logger.info("Executing LangGraph correlation workflow")
                    logger.info(f"[{get_timestamp()}] Initial state keys: {list(initial_state.keys())}")
                    logger.info(f"[{get_timestamp()}] Initial state types: {[(k, type(v).__name__) for k, v in initial_state.items()]}")

                    result = await self.graph.ainvoke(initial_state, config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": langfuse.get_current_trace_id(),
                            "langfuse_tags": ["correlation_agent"]
                        }
                    })

                    workflow_span.update(
                        output={
                            "workflow_executed": True,
                            "result_type": type(result).__name__,
                            "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                            "workflow_completed": result.get("completed", False) if isinstance(result, dict) else False
                        },
                        metadata={"status": "success"}
                    )

                    logger.info("LangGraph workflow execution completed")
                    logger.info(f"[{get_timestamp()}] Result keys: {list(result.keys()) if isinstance(result, dict) else f'Result type: {type(result).__name__}'}")

                # Prepare final results with tracing
                with langfuse.start_as_current_span(name="prepare-final-results") as final_span:
                    final_results = {
                        "correlation_analysis": result.get("log_correlation_result", ""),
                        "structured_correlation": result.get("structured_correlation"),  # Already a dict
                        "metrics_analysis": result.get("metrics_correlation_result", ""),
                        "structured_metrics": result.get("structured_metrics", {}),
                        "is_metric_based": result.get("is_metric_based", True),
                        "current_step": result.get("current_step", "completed"),
                        "error": result.get("error"),
                        "completed": result.get("completed", False),
                        "analysis_timestamp": datetime.now().isoformat(),
                        "workflow_summary": {
                            "total_logql_queries": len(result.get("generated_logql_queries", [])),
                            "total_promql_queries": len(result.get("generated_promql_queries", [])),
                            "logs_fetched": len(result.get("fetched_logs", {})),
                            "metrics_fetched": len(result.get("fetched_metrics", {})),
                            "redis_stored": result.get("redis_stored", False),
                            "postgres_stored": result.get("postgres_stored", False)
                        }
                    }

                    final_span.update(
                        output={
                            "final_results_prepared": True,
                            "correlation_available": bool(final_results["correlation_analysis"]),
                            "metrics_available": bool(final_results["metrics_analysis"]),
                            "workflow_summary": final_results["workflow_summary"]
                        },
                        metadata={"status": "success"}
                    )

                span.update(
                    output={
                        "analysis_successful": True,
                        "incident_key": incident_key,
                        "final_step": final_results.get("current_step"),
                        "workflow_completed": final_results.get("completed", False),
                        "error_occurred": bool(final_results.get("error"))
                    },
                    metadata={"status": "success"}
                )

                logger.info(f"[{get_timestamp()}] LangGraph workflow completed for incident: {incident_key}")
                logger.info(f"[{get_timestamp()}] Final status: {result.get('current_step', 'unknown')}")
                return final_results

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in LangGraph workflow for incident {incident_key}: {error_msg}")

                error_results = {
                    "error": error_msg,
                    "current_step": "workflow_failed",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "completed": False
                }

                # Store error results with tracing
                with langfuse.start_as_current_span(name="store-error-results") as error_span:
                    await self.store_results_in_redis(incident_key, error_results)

                    error_span.update(
                        output={"error_results_stored": True},
                        metadata={"status": "success"}
                    )

                span.update(
                    output={
                        "analysis_successful": False,
                        "error": error_msg,
                        "error_results_stored": True
                    },
                    metadata={"status": "error"}
                )

                raise

    # @observe(name="rca-agent-delegation")
    async def _rca_analysis_node(self, state: CorrelationAgentState) -> CorrelationAgentState:
        """STEP 12: Call RCA Agent for Root Cause Analysis."""
        logger.info("STEP 12: Calling RCA agent for analysis")
        return await self.workflow_completion.delegate_to_rca_agent(
            state,
            current_trace_id,
            current_observation_id,
            _global_session_id
        )

    # @observe(name="client-cleanup")
    async def close(self):
        """Close connections and cleanup resources with comprehensive tracing."""
        with langfuse.start_as_current_span(name="correlation-agent-cleanup") as span:
            span.update(session_id=str(uuid.uuid4().hex))

            try:
                cleanup_operations = []

                # Close MCP client connection
                if hasattr(self, 'mcp_client') and self.mcp_client:
                    with langfuse.start_as_current_span(name="close-mcp-client") as mcp_span:
                        try:
                            await self.mcp_client.close()
                            cleanup_operations.append("mcp_client")

                            mcp_span.update(
                                output={"mcp_client_closed": True},
                                metadata={"status": "success"}
                            )
                        except Exception as mcp_error:
                            mcp_span.update(
                                output={"error": str(mcp_error)},
                                metadata={"status": "error"}
                            )
                            logger.error(f"Error closing MCP client: {mcp_error}")

                # Close Redis connection
                if hasattr(self, 'redis_client') and self.redis_client:
                    with langfuse.start_as_current_span(name="close-redis-client") as redis_span:
                        try:
                            self.redis_client.close()
                            cleanup_operations.append("redis_client")

                            redis_span.update(
                                output={"redis_client_closed": True},
                                metadata={"status": "success"}
                            )
                        except Exception as redis_error:
                            redis_span.update(
                                output={"error": str(redis_error)},
                                metadata={"status": "error"}
                            )
                            logger.error(f"Error closing Redis client: {redis_error}")

                # Close database connections
                if hasattr(self, 'engine') and self.engine:
                    with langfuse.start_as_current_span(name="close-database-engine") as db_span:
                        try:
                            self.engine.dispose()
                            cleanup_operations.append("database_engine")

                            db_span.update(
                                output={"database_engine_closed": True},
                                metadata={"status": "success"}
                            )
                        except Exception as db_error:
                            db_span.update(
                                output={"error": str(db_error)},
                                metadata={"status": "error"}
                            )
                            logger.error(f"Error closing database engine: {db_error}")

                span.update(
                    output={
                        "cleanup_successful": True,
                        "operations_completed": cleanup_operations,
                        "total_operations": len(cleanup_operations)
                    },
                    metadata={"status": "success"}
                )

                logger.info(f"[{get_timestamp()}] CorrelationAgent cleanup completed: {cleanup_operations}")

            except Exception as e:
                span.update(
                    output={"error": str(e), "cleanup_successful": False},
                    metadata={"status": "error"}
                )
                logger.error(f"Error during CorrelationAgent cleanup: {e}")
            # """Close connections"""
            # if self.mcp_client:
            #     await self.mcp_client.close()
            # if self.redis_client:
            #     self.redis_client.close()