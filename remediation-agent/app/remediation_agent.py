"""
Simple Remediation Agent using LangGraph
Receives RCA analysis, generates remediation recommendations, stores in Redis and PostgreSQL, creates Jira tickets
"""

import logging
import json
import os
import redis
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm_config import LLMConfig
from app.models import RemediationStructured
from app.mcp_config import get_mcp_config
from app.mcp_client import LangChainMCPClient
from app.prompts import (
    get_remediation_analysis_prompt,
    get_remediation_system_message,
    get_structured_remediation_prompt,
    get_structured_remediation_system_message,
    get_fallback_remediation_text
)
from app.utils import (
    create_fallback_structured_remediation,
    fetch_rca_correlation_from_redis,
    fetch_complete_incident_from_redis,
    store_remediation_in_redis,
    store_remediation_in_database,
    add_jira_comment,
    calculate_performance_metrics
)

# Langfuse imports for comprehensive observability
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

# Initialize Langfuse client
load_dotenv()
langfuse = get_client()
logger = logging.getLogger(__name__)


@dataclass
class RemediationState:
    """Simple state for remediation processing"""
    incident: Dict[str, Any]
    rca_analysis: str
    correlation_data: str
    remediation_analysis: str
    incident_id: str
    structured_remediation_data: Optional[RemediationStructured] = None
    error: str = None


class SimpleRemediationAgent:
    """Simple Remediation Agent with LangGraph"""

    def __init__(self, redis_url: str = None, database_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.database_url = database_url or os.getenv("DATABASE_URL")

        # Initialize predefined trace ID for consistent tracing
        self.predefined_trace_id = langfuse.create_trace_id()

        # Initialize Langfuse callback handler for LLM tracing
        self.langfuse_handler = CallbackHandler()

        # Initialize Redis
        self.redis_client = None
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")

        # Initialize MCP client with custom wrapper
        self.mcp_client = None
        self.mcp_tools = []
        try:
            mcp_config = get_mcp_config()
            if mcp_config:
                self.mcp_client = LangChainMCPClient("dummy_url")
                logger.info("MCP client initialized for Remediation Agent")
            else:
                logger.warning("No MCP configuration available")
        except Exception as e:
            logger.warning(f"MCP client initialization failed: {e}")

        # Initialize LLM
        self.llm = LLMConfig.get_llm()

        # Build graph
        self.graph = self._build_workflow_graph()
        logger.info("Simple Remediation Agent initialized")

    @observe(name="build-remediation-workflow-graph")
    def _build_workflow_graph(self):
        """Build simple remediation workflow graph"""
        with langfuse.start_as_current_span(name="remediation-workflow-graph-build") as span:
            workflow = StateGraph(RemediationState)

            workflow.add_node("analyze_remediation", self._analyze_remediation_node)
            workflow.add_node("store_results", self._store_results_node)

            workflow.add_edge("analyze_remediation", "store_results")
            workflow.add_edge("store_results", END)
            workflow.set_entry_point("analyze_remediation")

            compiled_graph = workflow.compile()

            span.update(
                output={"workflow_nodes": ["analyze_remediation", "store_results"]},
                metadata={"component": "remediation_workflow", "status": "success"}
            )

            return compiled_graph

    async def _analyze_remediation_node(self, state: RemediationState) -> RemediationState:
        """
        STEP 1: Analyze remediation based on RCA and correlation data.

        Args:
            state: Current remediation state

        Returns:
            Updated state with remediation analysis
        """
        with langfuse.start_as_current_span(name="remediation-analysis") as span:
            logger.info("STEP 1: Starting remediation analysis")

            span.update(
                input={
                    "incident_id": state.incident_id,
                    "has_rca_analysis": bool(state.rca_analysis),
                    "has_correlation_data": bool(state.correlation_data)
                },
                metadata={"workflow_step": "remediation_analysis", "step_order": 1}
            )

            try:
                incident = state.incident
                rca_analysis = state.rca_analysis
                correlation_data = state.correlation_data

                if not rca_analysis:
                    logger.warning("No RCA analysis available for remediation")
                    state.remediation_analysis = "Remediation analysis could not be completed: No RCA data available"
                    state.error = "insufficient_data"
                    span.update(
                        output={"error": "insufficient_data"},
                        metadata={"status": "insufficient_data"}
                    )
                    return state

                logger.info("Starting remediation analysis with RCA data")

                # Generate remediation analysis using prompts
                remediation_prompt = get_remediation_analysis_prompt(incident, rca_analysis, correlation_data)
                system_message = get_remediation_system_message()

                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=remediation_prompt)
                ]

                with langfuse.start_as_current_span(name="[llm-call]-remediation-generation") as llm_span:
                    logger.info("Sending remediation prompt to LLM...")

                    llm_span.update(
                        input={
                            "prompt_length": len(remediation_prompt),
                            "rca_analysis_length": len(rca_analysis)
                        },
                        metadata={"component": "remediation_generation", "model": "llm"}
                    )

                    response = await self.llm.ainvoke(
                        messages,
                        config={
                            "callbacks": [self.langfuse_handler],
                            "metadata": {
                                "langfuse_trace_id": self.predefined_trace_id,
                                "langfuse_tags": ["remediation-agent"],
                                "component": "remediation_analysis_generation"
                            }
                        }
                    )

                    remediation_steps = response.content

                    llm_span.update(
                        output={
                            "remediation_length": len(remediation_steps),
                            "has_content": len(remediation_steps.strip()) >= 50
                        }
                    )

                logger.info(f"Remediation analysis completed, length: {len(remediation_steps)} characters")

                if not remediation_steps or len(remediation_steps.strip()) < 50:
                    logger.error("Remediation analysis response is empty or too short")
                    remediation_steps = get_fallback_remediation_text(incident)

                # Create structured remediation output
                structured_remediation = await self._create_structured_remediation(
                    incident, rca_analysis, remediation_steps
                )

                state.remediation_analysis = remediation_steps
                state.structured_remediation_data = structured_remediation
                logger.info("Remediation analysis node completed successfully")

                span.update(
                    output={
                        "remediation_analysis_length": len(remediation_steps),
                        "has_structured_data": bool(state.structured_remediation_data),
                        "status": "success"
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error in remediation analysis: {e}")
                state.remediation_analysis = f"Remediation analysis failed: {str(e)}"
                state.error = str(e)
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def _create_structured_remediation(
        self,
        incident: Dict[str, Any],
        rca_analysis: str,
        remediation_steps: str
    ) -> RemediationStructured:
        """
        Create structured remediation output from analysis text.

        Args:
            incident: Incident details dictionary
            rca_analysis: Root cause analysis text
            remediation_steps: Generated remediation steps

        Returns:
            Structured remediation data
        """
        with langfuse.start_as_current_span(name="[llm-call]-structure-remediation") as struct_span:
            try:
                remediation_llm = self.llm.with_structured_output(RemediationStructured)

                structured_prompt = get_structured_remediation_prompt(incident, rca_analysis, remediation_steps)
                system_message = get_structured_remediation_system_message()

                struct_span.update(
                    input={"remediation_analysis_length": len(remediation_steps)},
                    metadata={"operation": "structure_conversion"}
                )

                structured_remediation = await remediation_llm.ainvoke(
                    [
                        SystemMessage(content=system_message),
                        HumanMessage(content=structured_prompt)
                    ],
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": self.predefined_trace_id,
                            "langfuse_tags": ["remediation-agent"],
                            "component": "remediation_structuring"
                        }
                    }
                )

                # Validate structured remediation
                if hasattr(structured_remediation, 'model_dump'):
                    data = structured_remediation.model_dump()
                    logger.info(f"Structured remediation data: {data}")
                else:
                    raise ValueError("Structured remediation is not a Pydantic model")

                struct_span.update(
                    output={"structured_successfully": True, "validation_passed": True}
                )

                return structured_remediation

            except Exception as structured_error:
                logger.error(f"Structured remediation generation failed: {structured_error}")
                logger.warning("Creating fallback structured remediation from text output")

                # Create fallback structured remediation
                structured_remediation = create_fallback_structured_remediation(incident)

                struct_span.update(
                    output={"structured_successfully": False, "fallback_created": True}
                )
                logger.info(f"Created fallback structured remediation: {structured_remediation.model_dump()}")

                return structured_remediation

    async def _store_results_node(self, state: RemediationState) -> RemediationState:
        """
        STEP 2: Store remediation results in Redis, PostgreSQL, and Jira using MCP.

        Args:
            state: Current remediation state

        Returns:
            Updated state after storage operations
        """
        with langfuse.start_as_current_span(name="store-remediation-results") as span:
            logger.info("STEP 2: Storing remediation results")

            span.update(
                input={
                    "incident_id": state.incident_id,
                    "has_remediation_analysis": bool(state.remediation_analysis),
                    "has_structured_data": bool(state.structured_remediation_data)
                },
                metadata={"workflow_step": "store_results", "step_order": 2}
            )

            try:
                incident_id = state.incident_id
                postgresql_stored = False
                redis_stored = False

                # Store in PostgreSQL using MCP execute_query
                if self.mcp_client and self.mcp_tools:
                    postgresql_stored = await store_remediation_in_database(
                        self.mcp_client,
                        self.mcp_tools,
                        incident_id,
                        state.structured_remediation_data if state.structured_remediation_data else state.remediation_analysis
                    )

                # Store in Redis (fallback/cache)
                redis_stored = await store_remediation_in_redis(
                    self.redis_client,
                    incident_id,
                    state.remediation_analysis,
                    bool(state.rca_analysis),
                    bool(state.correlation_data),
                    bool(state.error)
                )

                # Add Jira comment for remediation analysis
                if state.remediation_analysis and not state.error:
                    with langfuse.start_as_current_span(name="[tool-called]-jira-remediation-comment") as jira_span:
                        try:
                            jira_comment_added = await add_jira_comment(
                                self.mcp_client,
                                self.mcp_tools,
                                self.llm,
                                self.langfuse_handler,
                                self.predefined_trace_id,
                                state.incident,
                                state.remediation_analysis
                            )
                            jira_span.update(
                                output={"comment_added": jira_comment_added},
                                metadata={"operation": "jira_comment"}
                            )
                        except Exception as e:
                            logger.warning(f"JIRA comment failed: {e}")
                            jira_span.update(
                                output={"comment_added": False, "error": str(e)}
                            )

                span.update(
                    output={
                        "postgresql_stored": postgresql_stored,
                        "redis_stored": redis_stored,
                        "jira_comment_attempted": bool(state.remediation_analysis and not state.error),
                        "storage_complete": True
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error storing remediation results: {e}")
                state.error = f"Storage failed: {str(e)}"
                span.update(
                    output={"error": str(e), "storage_complete": False},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def analyze_remediation(
        self,
        incident: Dict[str, Any],
        rca_analysis: str = "",
        correlation_data: str = "",
        langfuse_trace_context=None
    ) -> str:
        """
        Main entry point for remediation analysis.

        Args:
            incident: Incident details dictionary
            rca_analysis: Root cause analysis text
            correlation_data: Correlation analysis text
            langfuse_trace_context: Optional trace context for Langfuse

        Returns:
            Remediation analysis text
        """
        with langfuse.start_as_current_span(name="Remediation-Agent-Main-Execution", trace_context=langfuse_trace_context) as main_span:
            start_time = datetime.now()
            incident_id = incident.get('incident_id', f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            logger.info(f"Starting remediation analysis for incident: {incident_id}")
            logger.info(f"Has RCA analysis: {bool(rca_analysis)}")
            logger.info(f"Has correlation data: {bool(correlation_data)}")

            main_span.update(
                input={
                    "incident_id": incident_id,
                    "incident_data": incident,
                    "has_rca_analysis": bool(rca_analysis),
                    "has_correlation_data": bool(correlation_data),
                    "rca_analysis_length": len(rca_analysis),
                    "correlation_data_length": len(correlation_data)
                },
                metadata={
                    "component": "remediation_agent_main",
                    "workflow": "full_remediation_analysis",
                    "trace_id": self.predefined_trace_id
                }
            )

            try:
                # Setup MCP tools if not already done
                if self.mcp_client and not self.mcp_tools:
                    with langfuse.start_as_current_span(name="setup-mcp-tools") as mcp_span:
                        await self._setup_mcp_tools()
                        mcp_span.update(
                            output={"tools_setup": len(self.mcp_tools) > 0, "tools_count": len(self.mcp_tools)},
                            metadata={"component": "mcp_setup"}
                        )

                # Create initial state
                initial_state = RemediationState(
                    incident=incident,
                    rca_analysis=rca_analysis,
                    correlation_data=correlation_data,
                    remediation_analysis="",
                    incident_id=incident_id
                )

                # Run the graph
                with langfuse.start_as_current_span(name="execute-remediation-workflow") as workflow_span:
                    workflow_span.update(
                        input={"initial_state": {"incident_id": incident_id}},
                        metadata={"workflow_step": "graph_execution", "component": "langgraph"}
                    )

                    result = await self.graph.ainvoke(
                        initial_state,
                        config={
                            "callbacks": [self.langfuse_handler],
                            "metadata": {
                                "langfuse_trace_id": self.predefined_trace_id,
                                "langfuse_tags": ["remediation-agent"],
                                "component": "jira_comment_generation"
                            }
                        }
                    )

                    workflow_span.update(
                        output={
                            "workflow_completed": True,
                            "result_type": type(result).__name__,
                            "has_remediation_analysis": bool(result.remediation_analysis if hasattr(result, 'remediation_analysis') else result.get("remediation_analysis"))
                        }
                    )

                # Handle both dict and RemediationState object returns
                remediation_analysis = ""
                if isinstance(result, dict):
                    remediation_analysis = result.get("remediation_analysis", "No remediation analysis generated")
                else:
                    remediation_analysis = result.remediation_analysis

                # Calculate performance metrics
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                performance_factors, efficiency_score = calculate_performance_metrics(
                    execution_time,
                    remediation_analysis,
                    result.error if hasattr(result, 'error') else result.get("error"),
                    bool(rca_analysis)
                )

                # Score the execution
                main_span.score(
                    name="remediation-main-workflow-efficiency",
                    value=efficiency_score,
                    comment=f"Performance factors: {', '.join(performance_factors)}"
                )

                main_span.update(
                    output={
                        "remediation_analysis": remediation_analysis,
                        "execution_time_seconds": execution_time,
                        "performance_score": efficiency_score,
                        "status": "completed"
                    },
                    metadata={
                        "execution_time": execution_time,
                        "analysis_length": len(remediation_analysis),
                        "status": "success"
                    }
                )

                return remediation_analysis

            except Exception as e:
                logger.error(f"Error in remediation main workflow: {e}")
                main_span.update(
                    output={"error": str(e), "status": "failed"},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )
                return f"Remediation analysis failed: {str(e)}"

    async def fetch_complete_incident_data_from_redis(
        self,
        incident_key: str,
        langfuse_trace_context=None
    ) -> Dict[str, Any]:
        """
        Fetch complete incident data from Redis using incident_key.

        Args:
            incident_key: Redis key for incident (e.g., incidents:197:main)
            langfuse_trace_context: Optional trace context for Langfuse

        Returns:
            Dictionary containing complete incident data
        """
        return await fetch_complete_incident_from_redis(
            self.redis_client,
            incident_key,
            langfuse_trace_context
        )

    async def fetch_data_from_redis(self, incident_id: str) -> tuple[str, str]:
        """
        Fetch RCA analysis and correlation data from Redis.

        Args:
            incident_id: Incident identifier

        Returns:
            Tuple of (rca_analysis, correlation_data)
        """
        return await fetch_rca_correlation_from_redis(self.redis_client, incident_id)

    async def _setup_mcp_tools(self):
        """Setup MCP tools for Jira and Postgres integration."""
        with langfuse.start_as_current_span(name="setup-mcp-tools") as span:
            span.update(
                input={"has_mcp_client": bool(self.mcp_client)},
                metadata={"component": "mcp_tool_setup"}
            )

            if not self.mcp_client:
                logger.warning("MCP client not initialized, skipping tool setup")
                span.update(
                    output={"tools_setup": False, "reason": "no_mcp_client"},
                    metadata={"status": "skipped"}
                )
                return

            try:
                with langfuse.start_as_current_span(name="[mcp-operation]-connect-and-get-tools") as mcp_span:
                    # Connect to MCP servers and get tools using custom client
                    await self.mcp_client.connect()
                    self.mcp_tools = self.mcp_client.get_tools()
                    logger.info(f"Retrieved {len(self.mcp_tools)} MCP tools")

                    # Log available tools
                    if self.mcp_tools:
                        jira_tools = [tool for tool in self.mcp_tools if 'jira' in tool.name.lower()]
                        postgres_tools = [tool for tool in self.mcp_tools if 'postgres' in tool.name.lower() or 'sql' in tool.name.lower()]

                        logger.info(f"Available Jira tools: {[tool.name for tool in jira_tools]}")
                        logger.info(f"Available Postgres tools: {[tool.name for tool in postgres_tools]}")

                        mcp_span.update(
                            output={
                                "total_tools": len(self.mcp_tools),
                                "jira_tools": len(jira_tools),
                                "postgres_tools": len(postgres_tools)
                            },
                            metadata={"operation": "mcp_tool_retrieval"}
                        )

                span.update(
                    output={
                        "tools_setup": True,
                        "total_tools": len(self.mcp_tools),
                        "connection_successful": True
                    },
                    metadata={"status": "success"}
                )

            except Exception as e:
                logger.error(f"Failed to setup MCP tools: {e}")
                span.update(
                    output={"tools_setup": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

    async def setup_mcp_tools(self):
        """Public method to setup MCP tools."""
        await self._setup_mcp_tools()
