"""
Simple RCA Agent using LangGraph
Receives request, generates RCA analysis, stores in Redis and PostgreSQL
"""

import logging
import json
import os
import redis
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Import MCP client and LLM config
from .mcp_client import LangChainMCPClient
from app.llm_config import LLMConfig

# Import prompts and utility functions
from app.prompts import (
    get_prompt,
    get_system_prompt,
    get_attribution_text,
    get_reference_source_note
)
from app.utils import (
    sanitize_unicode_for_jira,
    sort_memgraph_results_by_similarity,
    extract_node_id_from_response,
    fetch_incident_data_from_redis,
    store_rca_in_redis,
    store_rca_in_database,
    update_memgraph_node_id_in_database,
    create_fallback_structured_rca,
    format_jira_comment_footer,
    calculate_rca_performance_score
)

# Langfuse imports for comprehensive observability
from langfuse import get_client
from langfuse.langchain import CallbackHandler


os.environ["LANGFUSE_AUTO_TRACE"] = "false"
os.environ["LANGFUSE_TRACE_EVERYTHING"] = "false"

# Initialize Langfuse client
load_dotenv()
langfuse = get_client()
logger = logging.getLogger(__name__)

current_trace_id=""
current_observation_id=""
_global_session_id=""

class RCAStructured(BaseModel):
    """Structured RCA analysis format"""
    incident_summary: List[str] = Field(description="Array of incident summary points")
    root_cause_analysis: List[str] = Field(description="Array of root cause analysis points")
    log_evidence: List[str] = Field(description="Array of log evidence points")

@dataclass
class RCAState:
    """Enhanced state for RCA processing with memgraph integration"""
    incident: Dict[str, Any]
    correlation_data: str
    metrics_analysis: str
    correlation_summary: str
    rca_analysis: str
    incident_id: str
    memgraph_results: List[Dict[str, Any]] = None
    memgraph_node_id: Optional[int] = None
    similarity_threshold: float = 0.9
    has_high_similarity_rca: bool = False
    existing_rca_content: str = ""
    is_human_assisted_rca: bool = False
    reference_rca_generated_by: str = ""  # "human", "ai", or "SRE Agent"
    structured_rca_data: Optional[RCAStructured] = None  # Store structured RCA data
    reference_rcas: Optional[List[Dict[str, Any]]] = None  # Up to 2 reference RCAs to guide new generation
    error: str = None

class SimpleRCAAgent:
    """Simple RCA Agent with LangGraph"""

    def __init__(self, redis_url: str = None, database_url: str = None, mcp_sse_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.database_url = database_url or os.getenv("DATABASE_URL")

        # Initialize langfuse handler for tracing
        self.langfuse_handler = CallbackHandler()

        # Initialize Redis
        self.redis_client = None
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")

        # Initialize MCP client for database and tool operations
        self.mcp_client = None
        if mcp_sse_url:
            self.mcp_client = LangChainMCPClient(mcp_sse_url)
            logger.info("MCP client initialized")

        # Initialize LLM
        self.llm = LLMConfig.get_llm()

        # Build graph
        self.graph = self._build_graph()
        logger.info("Simple RCA Agent initialized")

    def _build_graph(self):
        """Build memgraph-enhanced RCA workflow"""
        with langfuse.start_as_current_span(name="rca-workflow-graph-build") as span:
            workflow = StateGraph(RCAState)

            # Step 1: Fetch correlation summary from Redis
            workflow.add_node("fetch_correlation_summary", self._fetch_correlation_summary_node)

            # Step 2: Query memgraph for similar incidents
            workflow.add_node("query_memgraph", self._query_memgraph_node)

            # Step 3a: Use existing RCA if high similarity found
            workflow.add_node("existing_rca", self._existing_rca_node)

            # Step 3b: Generate new RCA and insert into memgraph
            workflow.add_node("generate_new_rca", self._generate_new_rca_node)

            # Step 4: Call remediation agent
            workflow.add_node("call_remediation", self._call_remediation_node)

            # Step 5: Store results in database
            workflow.add_node("store_results", self._store_results_node)

            # Set up edges
            workflow.add_edge("fetch_correlation_summary", "query_memgraph")
            workflow.add_conditional_edges(
                "query_memgraph",
                self._should_use_existing_rca,
                {
                    "existing_rca": "existing_rca",
                    "generate_new_rca": "generate_new_rca"
                }
            )
            workflow.add_edge("existing_rca", "store_results")
            workflow.add_edge("generate_new_rca", "store_results")
            workflow.add_edge("store_results", "call_remediation")
            workflow.add_edge("call_remediation", END)
            workflow.set_entry_point("fetch_correlation_summary")

            compiled_graph = workflow.compile()
            
            span.update(
                output={"workflow_nodes": ["fetch_correlation_summary", "query_memgraph", "existing_rca", "generate_new_rca", "call_remediation", "store_results"]},
                metadata={"component": "rca_workflow", "status": "success"}
            )
            
            return compiled_graph

    async def _generate_new_rca_node(self, state: RCAState) -> RCAState:
        """STEP 2b: Generate new RCA and insert into memgraph"""
        with langfuse.start_as_current_span(name="generate-new-rca") as span:
            logger.info("STEP 2b: Generating new RCA and storing in memgraph")
            
            span.update(
                input={
                    "has_correlation_data": bool(state.correlation_data),
                    "has_metrics_analysis": bool(state.metrics_analysis),
                    "has_reference_rcas": bool(state.reference_rcas),
                    "incident_id": state.incident_id
                },
                metadata={"workflow_step": "generate_rca", "step_order": 3}
            )

            try:
                incident = state.incident
                correlation_data = state.correlation_data
                metrics_analysis = state.metrics_analysis

                if not correlation_data and not metrics_analysis:
                    logger.warning("No correlation data or metrics analysis available for RCA")
                    state.rca_analysis = "RCA Analysis could not be completed: No correlation data or metrics analysis available"
                    state.error = "insufficient_data"
                    span.update(
                        output={"error": "insufficient_data"},
                        metadata={"status": "insufficient_data"}
                    )
                    return state

                logger.info("Starting RCA analysis with correlation data and metrics analysis")

                # Build reference RCAs section
                reference_rcas_section = ""
                if state.reference_rcas:
                    refs_text = "\n".join([
                        f"- Generated By: {(r.get('generated_by') or 'unknown').title()} | "
                        f"Similarity: {(r.get('similarity') or 0)*100:.1f}% | "
                        f"Timestamp: {r.get('timestamp', 'N/A')}\n\n{r.get('rca', '')}"
                        for r in state.reference_rcas
                    ])
                    reference_rcas_section = f"\n**REFERENCE RCAs (for guidance):**\n{refs_text}"

                # Get prompt from prompts.py
                rca_prompt = get_prompt("rca-generation", {
                    "incident_id": incident.get('incident_id', 'Unknown'),
                    "alert_name": incident.get('alert_name', 'Unknown'),
                    "description": incident.get('description', 'No description'),
                    "priority": incident.get('priority', 'medium'),
                    "correlation_data": correlation_data if correlation_data else "No correlation data available",
                    "metrics_analysis": metrics_analysis if metrics_analysis else "No metrics analysis available",
                    "reference_rcas_section": reference_rcas_section
                })

                messages = [
                    SystemMessage(content=get_system_prompt("rca-generation")),
                    HumanMessage(content=rca_prompt)
                ]

                with langfuse.start_as_current_span(name="[llm-call]-rca-generation") as llm_span:
                    logger.info("Sending RCA prompt to LLM...")
                    
                    llm_span.update(
                        input={
                            "prompt_length": len(rca_prompt),
                            "has_reference_guidance": bool(state.reference_rcas)
                        },
                        metadata={"component": "rca_generation", "model": "llm"}
                    )
                    
                    response = await self.llm.ainvoke(
                        messages,
                        config={
                            "callbacks": [self.langfuse_handler],
                            "metadata": {
                                "langfuse_trace_id": current_trace_id,
                                "langfuse_tags": ["rca_agent"],
                                "component": "rca_analysis_generation"
                            }
                        }
                    )
                    
                    rca_analysis = response.content
                    
                    llm_span.update(
                        output={
                            "rca_length": len(rca_analysis),
                            "has_content": len(rca_analysis.strip()) >= 50
                        }
                    )

                logger.info(f"RCA analysis completed, length: {len(rca_analysis)} characters")

                if not rca_analysis or len(rca_analysis.strip()) < 50:
                    logger.error("RCA analysis response is empty or too short")
                    rca_analysis = f"""
                    INCIDENT SUMMARY:
                    • Incident ID: {incident.get('incident_id', 'Unknown')} requires investigation
                    • RCA analysis completed but response was minimal
                    • Manual investigation required for complete analysis

                    ROOT CAUSE ANALYSIS:
                    • Unable to complete automated analysis due to insufficient data
                    • Recommend manual log review and metrics investigation
                    • Check system components mentioned in incident for failures

                    LOG EVIDENCE:
                    • Manual review of correlation data required
                    • Check for patterns in error messages and timestamps
                    • Investigate system dependencies and failure points

                    METRICS EVIDENCE:
                    • Manual review of metrics analysis required
                    • Check for anomalies in system performance indicators
                    • Investigate resource utilization patterns
                    """

                # Create structured output using the already generated RCA analysis
                with langfuse.start_as_current_span(name="[llm-call]-structure-rca") as struct_span:
                    try:
                        rca_llm = self.llm.with_structured_output(RCAStructured)

                        # Get structuring prompt from prompts.py
                        structured_rca_prompt = get_prompt("rca-structuring", {
                            "rca_analysis": rca_analysis
                        })

                        struct_span.update(
                            input={"rca_analysis_length": len(rca_analysis)},
                            metadata={"operation": "structure_conversion"}
                        )

                        structured_rca = await rca_llm.ainvoke(
                            [
                                SystemMessage(content=get_system_prompt("rca-structuring")),
                                HumanMessage(content=structured_rca_prompt)
                            ],
                            config={
                                "callbacks": [self.langfuse_handler],
                                "metadata": {
                                    "langfuse_trace_id": current_trace_id,
                                    "langfuse_tags": ["rca_agent"],
                                    "component": "rca_structuring"
                                }
                            }
                        )

                        logger.info(f"Structured RCA analysis completed: {type(structured_rca)}")

                        # Validate structured output
                        if hasattr(structured_rca, 'model_dump'):
                            data = structured_rca.model_dump()
                            logger.info(f"Structured RCA data: {data}")
                            # Validate the fields exist and are lists
                            if not isinstance(data.get('incident_summary'), list):
                                raise ValueError("incident_summary is not a list")
                            if not isinstance(data.get('root_cause_analysis'), list):
                                raise ValueError("root_cause_analysis is not a list")
                            if not isinstance(data.get('log_evidence'), list):
                                raise ValueError("log_evidence is not a list")
                        else:
                            raise ValueError("Structured RCA is not a Pydantic model")

                        struct_span.update(
                            output={"structured_successfully": True, "validation_passed": True}
                        )

                    except Exception as structured_error:
                        logger.error(f"Structured RCA generation failed: {structured_error}")
                        logger.warning("Creating fallback structured RCA from text output")
                        # Create a fallback structured RCA using utility function
                        structured_rca = create_fallback_structured_rca(
                            incident.get('incident_id', 'Unknown'),
                            "normal"
                        )
                        struct_span.update(
                            output={"structured_successfully": False, "fallback_created": True}
                        )
                        logger.info(f"Created fallback structured RCA: {structured_rca.model_dump()}")

                state.rca_analysis = rca_analysis
                state.structured_rca_data = structured_rca  # Store structured data in state

                # Insert new incident into memgraph after RCA generation
                with langfuse.start_as_current_span(name="[tool-called]-insert-memgraph") as memgraph_span:
                    try:
                        await self._insert_into_memgraph(state, rca_analysis)
                        memgraph_span.update(
                            output={"inserted": True},
                            metadata={"operation": "memgraph_insert"}
                        )
                    except Exception as e:
                        logger.warning(f"Memgraph insertion failed: {e}")
                        memgraph_span.update(
                            output={"inserted": False, "error": str(e)}
                        )

                # Add JIRA comment immediately after RCA analysis completion
                with langfuse.start_as_current_span(name="[tool-called]-jira-comment") as jira_span:
                    try:
                        await self.add_jira_comment_for_rca(incident, rca_analysis)
                        jira_span.update(
                            output={"comment_added": True},
                            metadata={"operation": "jira_comment"}
                        )
                    except Exception as e:
                        logger.warning(f"JIRA comment failed: {e}")
                        jira_span.update(
                            output={"comment_added": False, "error": str(e)}
                        )

                logger.info("RCA analysis node completed successfully")
                
                span.update(
                    output={
                        "rca_analysis_length": len(rca_analysis),
                        "has_structured_data": bool(state.structured_rca_data),
                        "memgraph_inserted": True,
                        "jira_comment_added": True
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error in RCA analysis: {e}")
                state.rca_analysis = f"RCA analysis failed: {str(e)}"
                state.error = str(e)
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def _call_remediation_node(self, state: RCAState) -> RCAState:
        """STEP 4: Call remediation agent with RCA analysis results"""
        with langfuse.start_as_current_span(name="call-remediation-agent") as span:
            logger.info("STEP 4: Calling remediation agent with RCA results")
            
            span.update(
                input={
                    "has_rca_analysis": bool(state.rca_analysis),
                    "incident_id": state.incident_id
                },
                metadata={"workflow_step": "remediation_call", "step_order": 5}
            )

            try:
                from .tools.call_remediation_agent import send_to_remediation_agent

                if not state.rca_analysis:
                    logger.warning("No RCA analysis available for remediation agent")
                    state.error = "No RCA analysis for remediation"
                    span.update(
                        output={"called": False, "reason": "no_rca_analysis"},
                        metadata={"status": "skipped"}
                    )
                    return state

                with langfuse.start_as_current_span(name="[tool-called]-send-to-remediation") as remediation_span:
                    # Send request to remediation agent (fire and forget)
                    remediation_status = await send_to_remediation_agent(
                        incident_id=state.incident_id,
                        incident_data=state.incident,
                        rca_analysis=state.rca_analysis,
                        correlation_data=state.correlation_data,
                        current_trace_id=current_trace_id,
                        current_observation_id=current_observation_id,
                        session_id=_global_session_id
                    )
                    
                    remediation_span.update(
                        input={
                            "incident_id": state.incident_id,
                            "rca_length": len(state.rca_analysis)
                        },
                        output={"status": remediation_status},
                        metadata={"operation": "remediation_agent_call"}
                    )

                logger.info(f"Remediation agent request sent: {remediation_status}")
                
                span.update(
                    output={
                        "called": True,
                        "status": remediation_status
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error sending request to remediation agent: {e}")
                # Don't set error - this is not critical for RCA completion
                span.update(
                    output={"called": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def _store_results_node(self, state: RCAState) -> RCAState:
        """STEP 5: Store RCA results in Redis and PostgreSQL using MCP"""
        with langfuse.start_as_current_span(name="store-rca-results") as span:
            logger.info("STEP 5: Storing RCA results")
            
            span.update(
                input={
                    "incident_id": state.incident_id,
                    "has_rca_analysis": bool(state.rca_analysis),
                    "has_structured_data": bool(state.structured_rca_data)
                },
                metadata={"workflow_step": "store_results", "step_order": 6}
            )

            try:
                incident_id = state.incident_id
                rca_analysis = state.rca_analysis

                # Store in PostgreSQL using MCP execute_query tool
                storage_success = False
                if self.mcp_client:
                    with langfuse.start_as_current_span(name="[database-store]-rca-analysis") as db_span:
                        structured_data = state.structured_rca_data if state.structured_rca_data else rca_analysis
                        storage_success = await store_rca_in_database(
                            self.mcp_client,
                            incident_id,
                            structured_data,
                            max_retries=3
                        )

                        db_span.update(
                            output={"stored": storage_success, "method": "mcp_execute_query"},
                            metadata={"operation": "postgresql_store"}
                        )

                        if storage_success:
                            logger.info("Structured RCA analysis stored in PostgreSQL using MCP")
                        else:
                            logger.error("Failed to store RCA analysis in PostgreSQL")

                # Store in Redis if available (fallback)
                redis_success = False
                if self.redis_client:
                    with langfuse.start_as_current_span(name="[redis-store]-rca-results") as redis_span:
                        redis_success = await store_rca_in_redis(
                            self.redis_client,
                            incident_id,
                            rca_analysis,
                            state.correlation_data,
                            state.metrics_analysis,
                            state.error,
                            ttl=3600
                        )

                        redis_span.update(
                            output={"stored": redis_success, "ttl_hours": 1},
                            metadata={"operation": "redis_store"}
                        )

                        if redis_success:
                            logger.info(f"Stored RCA analysis in Redis for incident {incident_id}")
                        else:
                            logger.warning("Redis storage failed")

                span.update(
                    output={
                        "postgresql_stored": storage_success,
                        "redis_stored": redis_success,
                        "storage_complete": True
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error storing RCA results: {e}")
                state.error = f"Storage failed: {str(e)}"
                span.update(
                    output={"error": str(e), "storage_complete": False},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def _fetch_correlation_summary_node(self, state: RCAState) -> RCAState:
        """STEP 0.5: Fetch correlation summary from Redis (from correlation agent)"""
        with langfuse.start_as_current_span(name="fetch-correlation-summary") as span:
            logger.info("STEP 0.5: Fetching correlation summary from Redis")
            
            span.update(
                input={"incident_id": state.incident_id},
                metadata={"workflow_step": "fetch_correlation", "step_order": 1}
            )

            try:
                incident_id = state.incident_id
                correlation_data = ""
                metrics_analysis = ""
                correlation_summary = ""

                if self.redis_client:
                    with langfuse.start_as_current_span(name="[redis-fetch]-correlation-data") as redis_span:
                        try:
                            incident_key = f"incidents:{incident_id}:main"
                            incident_raw = self.redis_client.get(incident_key)
                            
                            redis_span.update(
                                input={"redis_key": incident_key},
                                metadata={"operation": "redis_get"}
                            )
                            
                            if incident_raw:
                                incident_data = json.loads(incident_raw.decode('utf-8'))
                                correlation_summary = incident_data.get("correlation_summary", "")
                                correlation_data = incident_data.get("correlation_analysis", "")
                                metrics_analysis = incident_data.get("metrics_analysis", "")
                                
                                redis_span.update(
                                    output={
                                        "has_correlation_summary": bool(correlation_summary),
                                        "has_correlation_data": bool(correlation_data),
                                        "has_metrics_analysis": bool(metrics_analysis)
                                    }
                                )
                                
                                if correlation_summary:
                                    logger.info(f"Retrieved correlation summary from Redis for incident {incident_id}")
                                else:
                                    logger.warning(f"No correlation summary found in incident data for {incident_id}")
                            else:
                                logger.warning(f"No incident data found in Redis for key: {incident_key}")
                                redis_span.update(output={"found": False})

                        except Exception as e:
                            logger.warning(f"Failed to fetch data from Redis: {e}")
                            redis_span.update(metadata={"error": str(e)})
                else:
                    logger.warning("No Redis client available")

                # Update state with all fetched data
                state.correlation_data = correlation_data
                state.metrics_analysis = metrics_analysis
                state.correlation_summary = correlation_summary

                # Get additional incident details for memgraph queries
                incident = state.incident
                if not state.correlation_summary and not correlation_data and not metrics_analysis:
                    logger.warning("No correlation summary, correlation data, or metrics analysis available")
                    state.error = "No correlation data available for RCA"
                    span.update(
                        output={"error": "No correlation data available"},
                        metadata={"status": "insufficient_data"}
                    )
                    return state

                logger.info("Data fetch summary:")
                logger.info(f"  - Correlation data: {'Available' if correlation_data else 'Not available'}")
                logger.info(f"  - Metrics analysis: {'Available' if metrics_analysis else 'Not available'}")
                logger.info(f"  - Correlation summary: {'Available' if correlation_summary else 'Not available'}")

                if correlation_summary:
                    logger.info("Will use correlation summary for memgraph similarity search")
                else:
                    logger.warning("No correlation summary available - will use correlation data for RCA generation")

                span.update(
                    output={
                        "correlation_data_available": bool(correlation_data),
                        "metrics_analysis_available": bool(metrics_analysis),
                        "correlation_summary_available": bool(correlation_summary),
                        "status": "success"
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error fetching correlation summary: {e}")
                state.error = f"Failed to fetch correlation data: {str(e)}"
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def _query_memgraph_node(self, state: RCAState) -> RCAState:
        """STEP 1: Query memgraph for similar incidents using React Agent"""
        with langfuse.start_as_current_span(name="query-memgraph-similarities") as span:
            logger.info("STEP 1: Querying memgraph for similar incidents")
            
            span.update(
                input={
                    "has_correlation_summary": bool(state.correlation_summary),
                    "similarity_threshold": state.similarity_threshold
                },
                metadata={"workflow_step": "memgraph_query", "step_order": 2}
            )

            try:
                from .tools.memgraph_tools import query_log_tool, filter_high_similarity_results

                correlation_summary = state.correlation_summary
                if not correlation_summary:
                    logger.warning("No correlation summary available for memgraph query")
                    state.memgraph_results = []
                    state.has_high_similarity_rca = False
                    span.update(
                        output={"results_count": 0, "has_high_similarity": False},
                        metadata={"status": "no_correlation_summary"}
                    )
                    return state

                # Prepare context for the tool call
                incident = state.incident
                source = incident.get('service', incident.get('source', ''))
                alert_type = 'error'  # Default, could be made dynamic

                with langfuse.start_as_current_span(name="[tool-called]-query_log_tool") as tool_span:
                    logger.info("Calling query_log_tool directly for similarity query")
                    
                    tool_input = {
                        "query_log": correlation_summary,
                        "top_k": 6,
                        "source": source,
                        "alert_type": alert_type
                    }
                    
                    tool_span.update(
                        input=tool_input,
                        metadata={"tool": "query_log_tool", "operation": "similarity_search"}
                    )
                    
                    tool_response = await query_log_tool.ainvoke(tool_input, config={
                                "callbacks": [self.langfuse_handler],
                                "metadata": {
                                    "langfuse_trace_id": current_trace_id,
                                    "langfuse_tags": ["rca_agent"]
                                }
                            })
                    
                    tool_span.update(
                        output={
                            "success": tool_response.get("success", False) if isinstance(tool_response, dict) else True,
                            "results_count": len(tool_response.get("results", [])) if isinstance(tool_response, dict) else 0
                        }
                    )

                memgraph_results = tool_response.get("results", []) if isinstance(tool_response, dict) else []
                if tool_response.get("success") is False:
                    logger.warning(f"Memgraph query reported error: {tool_response.get('error', 'Unknown error')}")

                # Process the extracted memgraph results
                if memgraph_results:
                    with langfuse.start_as_current_span(name="[tool-result]-similarity-filtering") as filter_span:
                        # Filter for high similarity results (>= threshold) and exclude 100%
                        high_similarity_results = filter_high_similarity_results(memgraph_results, state.similarity_threshold)
                        high_similarity_results = [r for r in high_similarity_results if (r.get("similarity") or 0) < 1.1]
                        
                        filter_span.update(
                            input={
                                "raw_results_count": len(memgraph_results),
                                "similarity_threshold": state.similarity_threshold
                            },
                            output={
                                "filtered_results_count": len(high_similarity_results),
                                "has_high_similarity": len(high_similarity_results) > 0
                            }
                        )

                        logger.info(f"Post-filter results (>= {state.similarity_threshold} and < 1.1): {len(high_similarity_results)}")

                    state.memgraph_results = memgraph_results
                    state.has_high_similarity_rca = len(high_similarity_results) > 0

                    if state.has_high_similarity_rca:
                        with langfuse.start_as_current_span(name="[processing]-similarity-sorting") as sort_span:
                            # Use utility function for sorting and reference selection
                            sorted_results, refs, attribution_source = sort_memgraph_results_by_similarity(
                                high_similarity_results,
                                state.similarity_threshold
                            )

                            logger.info(f"Candidates after sorting (top 6 preview): {[{'gen': (r.get('generated_by') or '').lower(), 'sim': round((r.get('similarity') or 0)*100,1), 'ts': r.get('timestamp','N/A')} for r in sorted_results[:6]]}")

                            state.reference_rcas = refs

                            # Set attribution based on utility function result
                            if attribution_source == "human":
                                state.reference_rca_generated_by = "human"
                                state.is_human_assisted_rca = True
                            else:
                                state.reference_rca_generated_by = "SRE Agent"
                                state.is_human_assisted_rca = False

                            # Logging: best human score, best agent score, and which references are passed
                            human_refs = [r for r in refs if (r.get("generated_by") or "").lower() == "human"]
                            non_human_refs = [r for r in refs if (r.get("generated_by") or "").lower() != "human"]

                            best_human = human_refs[0] if human_refs else None
                            best_agent = non_human_refs[0] if non_human_refs else None

                            if best_human:
                                logger.info(f"Best human RCA similarity: {round((best_human.get('similarity') or 0)*100,1)}% @ {best_human.get('timestamp','N/A')}")
                            else:
                                logger.info("No human RCA present in top candidates")

                            if best_agent:
                                logger.info(f"Best agent RCA similarity: {round((best_agent.get('similarity') or 0)*100,1)}% @ {best_agent.get('timestamp','N/A')}")
                            else:
                                logger.info("No agent RCA present in top candidates")

                            logger.info(f"Passing up to 2 reference RCAs: {[{'gen': (r.get('generated_by') or '').lower(), 'sim': round((r.get('similarity') or 0)*100,1), 'ts': r.get('timestamp','N/A')} for r in state.reference_rcas]}")
                            logger.info(f"Attribution set to: {state.reference_rca_generated_by} (based on {len(human_refs)} human refs out of {len(refs)} total)")

                            # Best result for existing RCA content (for legacy compatibility)
                            best_result = sorted_results[0]
                            state.existing_rca_content = best_result.get("rca", "")

                            sort_span.update(
                                output={
                                    "sorted_count": len(sorted_results),
                                    "reference_count": len(refs),
                                    "attribution": state.reference_rca_generated_by
                                }
                            )
                else:
                    logger.info(f"Found {len(memgraph_results)} results but none above similarity threshold {state.similarity_threshold}")
                    logger.info("Will proceed to normal RCA generation")

            except Exception as e:
                logger.error(f"Error in memgraph React agent: {e}")
                state.memgraph_results = []
                state.has_high_similarity_rca = False
                state.is_human_assisted_rca = False
                state.reference_rca_generated_by = ""
                
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    def _should_use_existing_rca(self, state: RCAState) -> str:
        """Conditional branch: Determine whether to use existing RCA or generate new one"""
        with langfuse.start_as_current_span(name="routing-decision") as span:
            decision = "existing_rca" if (state.has_high_similarity_rca and state.existing_rca_content) else "generate_new_rca"
            
            span.update(
                input={
                    "has_high_similarity_rca": state.has_high_similarity_rca,
                    "has_existing_content": bool(state.existing_rca_content),
                    "similarity_threshold": state.similarity_threshold
                },
                output={
                    "routing_decision": decision,
                    "will_use_existing": decision == "existing_rca"
                },
                metadata={"workflow_step": "routing", "step_completed": True}
            )
            
            if decision == "existing_rca":
                logger.info("High similarity RCA found - using guided RCA generation")
            else:
                logger.info("No high similarity RCA found - proceeding to normal RCA generation")
            
            return decision

    async def _existing_rca_node(self, state: RCAState) -> RCAState:
        """STEP 2a: Generate RCA guided by existing high-similarity RCA content"""
        with langfuse.start_as_current_span(name="use-existing-rca-guided") as span:
            logger.info("STEP 2a: Generating RCA guided by existing similar incident")
            
            span.update(
                input={
                    "existing_rca_length": len(state.existing_rca_content),
                    "is_human_assisted": state.is_human_assisted_rca,
                    "reference_generated_by": state.reference_rca_generated_by,
                    "has_reference_rcas": bool(state.reference_rcas)
                },
                metadata={"workflow_step": "guided_rca", "step_order": 3}
            )

            try:
                incident = state.incident
                correlation_summary = state.correlation_summary
                existing_rca = state.existing_rca_content

                # Get reference source note and attribution text from prompts.py
                reference_source_note = get_reference_source_note(state.reference_rca_generated_by)
                attribution_text = get_attribution_text(state.reference_rca_generated_by)

                # Build additional references section
                additional_references_section = ""
                if state.reference_rcas:
                    refs_text = "\n".join([
                        f"- Generated By: {(r.get('generated_by') or 'unknown').title()} | "
                        f"Similarity: {(r.get('similarity') or 0)*100:.1f}% | "
                        f"Timestamp: {r.get('timestamp', 'N/A')}\n\n{r.get('rca', '')}"
                        for r in state.reference_rcas
                    ])
                    additional_references_section = f"\n**ADDITIONAL REFERENCE RCAs (top matches):**\n{refs_text}"

                # Build human weightage note
                human_weightage_note = ""
                if state.is_human_assisted_rca:
                    human_weightage_note = "6. **GIVE HIGHER WEIGHTAGE** to the reference RCA as it contains human expert insights"

                # Get guided RCA prompt from prompts.py
                guided_rca_prompt = get_prompt("rca-guided-generation", {
                    "reference_source_note": reference_source_note,
                    "incident_id": incident.get('incident_id', 'Unknown'),
                    "alert_name": incident.get('alert_name', 'Unknown'),
                    "description": incident.get('description', 'No description'),
                    "priority": incident.get('priority', 'medium'),
                    "correlation_summary": correlation_summary,
                    "existing_rca": existing_rca,
                    "additional_references_section": additional_references_section,
                    "human_weightage_note": human_weightage_note,
                    "attribution_text": attribution_text
                })

                messages = [
                    SystemMessage(content=get_system_prompt("rca-guided-generation")),
                    HumanMessage(content=guided_rca_prompt)
                ]

                with langfuse.start_as_current_span(name="[llm-call]-guided-rca-generation") as llm_span:
                    logger.info("Generating guided RCA using existing similar incident...")
                    
                    llm_span.update(
                        input={
                            "prompt_length": len(guided_rca_prompt),
                            "has_human_guidance": state.is_human_assisted_rca,
                            "reference_count": len(state.reference_rcas) if state.reference_rcas else 0
                        },
                        metadata={"component": "guided_rca_generation", "model": "llm"}
                    )
                    
                    response = await self.llm.ainvoke(
                        messages,
                        config={
                            "callbacks": [self.langfuse_handler],
                            "metadata": {
                                "langfuse_trace_id": current_trace_id,
                                "langfuse_tags": ["rca_agent"]
                            }
                        }
                    )
                    
                    rca_analysis = response.content
                    
                    llm_span.update(
                        output={
                            "rca_length": len(rca_analysis),
                            "has_content": len(rca_analysis.strip()) >= 50
                        }
                    )

                logger.info(f"Guided RCA analysis completed, length: {len(rca_analysis)}")

                if not rca_analysis or len(rca_analysis.strip()) < 50:
                    logger.error("Guided RCA analysis response is empty or too short")
                    rca_analysis = f"""
                    # RCA for Incident {incident.get('incident_id', 'Unknown')} (Guided by Similar Incident)

                    ## Incident Summary
                    • Incident {incident.get('incident_id', 'Unknown')} is similar to a previously analyzed incident
                    • Correlation summary indicates system issues that match known patterns
                    • Analysis guided by high-confidence similar incident

                    ## Root Cause Analysis
                    • Based on similar incident patterns, likely root cause involves system components
                    • Current incident shows similar symptoms to past incident
                    • Recommend following similar resolution approach as reference incident

                    ## Evidence from Current Incident
                    • Correlation summary shows patterns consistent with reference incident
                    • System behavior aligns with previously observed failure modes
                    • Timeline and symptoms match known incident patterns

                    ## Similarity Assessment
                    • High similarity (>90%) with past incident suggests same root cause
                    • Current incident context matches reference patterns
                    • Resolution approach should follow proven successful methods

                    ## Recommended Actions
                    • Follow resolution steps that worked for similar past incident
                    • Monitor same metrics that were crucial in past incident
                    • Apply same fixes that successfully resolved reference incident
                    """

                state.rca_analysis = rca_analysis

                # Create structured output using the already generated guided RCA analysis
                with langfuse.start_as_current_span(name="[llm-call]-structure-guided-rca") as struct_span:
                    try:
                        rca_llm = self.llm.with_structured_output(RCAStructured)

                        # Get structuring prompt from prompts.py
                        structured_rca_prompt = get_prompt("rca-structuring", {
                            "rca_analysis": rca_analysis
                        })

                        struct_span.update(
                            input={"guided_rca_length": len(rca_analysis)},
                            metadata={"operation": "structure_guided_rca"}
                        )

                        structured_rca = await rca_llm.ainvoke(
                            [
                                SystemMessage(content=get_system_prompt("rca-structuring")),
                                HumanMessage(content=structured_rca_prompt)
                            ],
                            config={
                                "callbacks": [self.langfuse_handler],
                                "metadata": {
                                    "langfuse_trace_id": current_trace_id,
                                    "langfuse_tags": ["rca_agent"]
                                }
                            }
                        )

                        logger.info(f"Structured guided RCA analysis completed: {type(structured_rca)}")

                        # Validate structured output
                        if hasattr(structured_rca, 'model_dump'):
                            data = structured_rca.model_dump()
                            logger.info(f"Structured guided RCA data: {data}")
                            # Validate the fields exist and are lists
                            if not isinstance(data.get('incident_summary'), list):
                                raise ValueError("incident_summary is not a list")
                            if not isinstance(data.get('root_cause_analysis'), list):
                                raise ValueError("root_cause_analysis is not a list")
                            if not isinstance(data.get('log_evidence'), list):
                                raise ValueError("log_evidence is not a list")
                        else:
                            raise ValueError("Structured guided RCA is not a Pydantic model")

                        struct_span.update(
                            output={"structured_successfully": True, "validation_passed": True}
                        )

                    except Exception as structured_error:
                        logger.error(f"Structured guided RCA generation failed: {structured_error}")
                        logger.warning("Creating fallback structured RCA from guided text output")
                        # Create a fallback structured RCA using utility function
                        structured_rca = create_fallback_structured_rca(
                            incident.get('incident_id', 'Unknown'),
                            "guided"
                        )
                        struct_span.update(
                            output={"structured_successfully": False, "fallback_created": True}
                        )
                        logger.info(f"Created fallback structured guided RCA: {structured_rca.model_dump()}")

                # Store structured data in state
                state.structured_rca_data = structured_rca

                # Insert current incident and its RCA into memgraph and persist node_id
                with langfuse.start_as_current_span(name="[tool-called]-insert-guided-memgraph") as memgraph_span:
                    try:
                        await self._insert_into_memgraph(state, rca_analysis)
                        memgraph_span.update(
                            output={"inserted": True},
                            metadata={"operation": "memgraph_insert_guided"}
                        )
                    except Exception as e:
                        logger.warning(f"Memgraph insertion failed: {e}")
                        memgraph_span.update(
                            output={"inserted": False, "error": str(e)}
                        )

                # Add JIRA comment immediately after RCA analysis completion
                with langfuse.start_as_current_span(name="[tool-called]-jira-guided-comment") as jira_span:
                    try:
                        await self.add_jira_comment_for_rca(incident, rca_analysis)
                        jira_span.update(
                            output={"comment_added": True},
                            metadata={"operation": "jira_comment_guided"}
                        )
                    except Exception as e:
                        logger.warning(f"JIRA comment failed: {e}")
                        jira_span.update(
                            output={"comment_added": False, "error": str(e)}
                        )

                logger.info("Guided RCA analysis node completed successfully")
                
                span.update(
                    output={
                        "guided_rca_length": len(rca_analysis),
                        "has_structured_data": bool(state.structured_rca_data),
                        "memgraph_inserted": True,
                        "jira_comment_added": True,
                        "attribution_included": True
                    },
                    metadata={"status": "success", "step_completed": True}
                )

            except Exception as e:
                logger.error(f"Error in guided RCA analysis: {e}")
                state.rca_analysis = f"Guided RCA analysis failed: {str(e)}"
                state.error = str(e)
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return state

    async def initialize_mcp_client(self):
        """Initialize MCP client connection"""
        with langfuse.start_as_current_span(name="initialize-mcp-client") as span:
            logger.info("Starting MCP client initialization...")
            
            span.update(
                input={"has_mcp_client": bool(self.mcp_client)},
                metadata={"component": "mcp_initialization"}
            )
            
            if self.mcp_client:
                try:
                    logger.info("Attempting to connect MCP client...")
                    await self.mcp_client.connect()
                    logger.info("MCP client connected successfully")

                    span.update(
                        output={"connected": True, "status": "success"},
                        metadata={"status": "connected"}
                    )

                except Exception as e:
                    logger.error(f"Failed to connect MCP client: {e}")
                    import traceback
                    logger.error(f"MCP connection stack trace: {traceback.format_exc()}")
                    self.mcp_client = None

                    span.update(
                        output={"connected": False, "error": str(e)},
                        metadata={"status": "failed", "error_type": type(e).__name__}
                    )
            else:
                logger.warning("No MCP client configured - continuing without MCP tools")
                span.update(
                    output={"connected": False, "reason": "no_client_configured"},
                    metadata={"status": "skipped"}
                )

    async def analyze_root_cause(self, incident: Dict[str, Any],langfuse_trace_context=None) -> str:
        """Main entry point for memgraph-enhanced RCA analysis workflow"""
        with langfuse.start_as_current_span(name="rca_agent-main-execution",trace_context=langfuse_trace_context) as main_span:
            start_time = datetime.now()
            incident_id = incident.get('incident_id', f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            logger.info(f"Starting memgraph-enhanced RCA analysis for incident: {incident_id}")
            
            main_span.update(
                input={
                    "incident_id": incident_id,
                    "incident_data": incident,
                    "alert_name": incident.get('alert_name', 'Unknown'),
                    "priority": incident.get('priority', 'medium')
                },
                metadata={
                    "component": "rca_agent_main",
                    "workflow": "full_rca_analysis",
                    "trace_id": current_trace_id
                }
            )

            try:
                # Initialize MCP client connection like correlation agent
                with langfuse.start_as_current_span(name="initialize-mcp-connection") as init_span:
                    await self.initialize_mcp_client()
                    init_span.update(
                        output={"mcp_client_available": bool(self.mcp_client)},
                        metadata={"component": "mcp_initialization"}
                    )

                # Create initial state - the new workflow fetches correlation data from Redis
                initial_state = RCAState(
                    incident=incident,
                    correlation_data="",  # Will be fetched by fetch_correlation_summary_node
                    metrics_analysis="",  # Will be fetched by fetch_correlation_summary_node
                    rca_analysis="",
                    incident_id=incident_id,
                    correlation_summary=""
                )

                # Run the new memgraph-enhanced graph
                with langfuse.start_as_current_span(name="execute-rca-workflow") as workflow_span:
                    workflow_span.update(
                        input={"initial_state": {"incident_id": incident_id}},
                        metadata={"workflow_step": "graph_execution", "component": "langgraph"}
                    )
                    
                    result = await self.graph.ainvoke(initial_state, config={
                                "callbacks": [self.langfuse_handler],
                                "metadata": {
                                    "langfuse_trace_id": current_trace_id,
                                    "langfuse_tags": ["rca_agent"]
                                }
                            }
                        )
                    
                    workflow_span.update(
                        output={
                            "workflow_completed": True,
                            "result_type": type(result).__name__,
                            "has_rca_analysis": bool(result.rca_analysis if hasattr(result, 'rca_analysis') else result.get("rca_analysis"))
                        }
                    )

                # Handle both dict and RCAState object returns
                rca_analysis = ""
                if isinstance(result, dict):
                    rca_analysis = result.get("rca_analysis", "No RCA analysis generated")
                else:
                    rca_analysis = result.rca_analysis

                # Calculate performance metrics using utility function
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                has_error = bool(result.error if hasattr(result, 'error') else result.get("error"))

                efficiency_score, performance_factors = calculate_rca_performance_score(
                    execution_time,
                    len(rca_analysis),
                    has_error
                )
                
                # Score the execution
                main_span.score(
                    name="rca-main-workflow-efficiency",
                    value=efficiency_score,
                    comment=f"Performance factors: {', '.join(performance_factors)}"
                )

                main_span.update(
                    output={
                        "rca_analysis": rca_analysis,
                        "execution_time_seconds": execution_time,
                        "performance_score": efficiency_score,
                        "status": "completed"
                    },
                    metadata={
                        "execution_time": execution_time,
                        "analysis_length": len(rca_analysis),
                        "status": "success"
                    }
                )

                return rca_analysis

            except Exception as e:
                logger.error(f"Error in RCA main workflow: {e}")
                main_span.update(
                    output={"error": str(e), "status": "failed"},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )
                return f"RCA analysis failed: {str(e)}"

    async def fetch_complete_incident_data_from_redis(self, incident_key: str, langfuse_trace_context=None) -> Dict[str, Any]:
        """Fetch complete incident data from Redis using incident_key (e.g., incidents:197:main)"""
        with langfuse.start_as_current_span(
            name="fetch-complete-incident-data",
            trace_context=langfuse_trace_context
        ) as span:
            span.update(
                input={"incident_key": incident_key},
                metadata={"operation": "redis_fetch_complete", "component": "data_retrieval"}
            )

            span.update_trace(session_id=langfuse_trace_context.get("session_id"))

            global current_trace_id
            current_trace_id = langfuse_trace_context.get("trace_id")

            global current_observation_id
            current_observation_id = langfuse_trace_context.get("parent_span_id")

            global _global_session_id
            _global_session_id = langfuse_trace_context.get("session_id")

            if not self.redis_client:
                logger.warning("Redis client not available")
                span.update(
                    output={"success": False, "reason": "no_redis_client"},
                    metadata={"status": "unavailable"}
                )
                return {}

            try:
                # Use utility function for Redis fetch
                parsed_data = await fetch_incident_data_from_redis(self.redis_client, incident_key)

                if parsed_data:
                    logger.info(f"Retrieved complete incident data from Redis for key: {incident_key}")
                    logger.info(f"Available fields: {list(parsed_data.keys())}")

                    span.update(
                        output={
                            "success": True,
                            "data_retrieved": True,
                            "fields_available": list(parsed_data.keys()),
                            "field_count": len(parsed_data.keys())
                        },
                        metadata={"status": "success"}
                    )

                    return parsed_data
                else:
                    logger.warning(f"No incident data found in Redis for key: {incident_key}")
                    span.update(
                        output={"success": False, "data_retrieved": False, "reason": "not_found"},
                        metadata={"status": "not_found"}
                    )
                    return {}

            except Exception as e:
                logger.error(f"Error fetching incident data from Redis: {e}")
                span.update(
                    output={"success": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )
                return {}

    async def _insert_into_memgraph(self, state: RCAState, rca_analysis: str):
        """Insert new incident and RCA into memgraph using React agent with HTTP tools"""
        with langfuse.start_as_current_span(name="insert-into-memgraph") as span:
            logger.info("Inserting new incident into memgraph database")
            
            span.update(
                input={
                    "incident_id": state.incident_id,
                    "rca_analysis_length": len(rca_analysis),
                    "has_correlation_summary": bool(state.correlation_summary)
                },
                metadata={"operation": "memgraph_insert", "component": "memgraph_integration"}
            )

            try:
                # Import memgraph tools
                from .tools.memgraph_tools import get_memgraph_tools

                with langfuse.start_as_current_span(name="[tool-initialization]-memgraph-tools") as tool_init_span:
                    # Get memgraph tools for React agent
                    memgraph_tools = get_memgraph_tools()
                    logger.info(f"Using memgraph tools: {[tool.name for tool in memgraph_tools]}")
                    
                    tool_init_span.update(
                        output={"available_tools": [tool.name for tool in memgraph_tools]},
                        metadata={"operation": "tool_loading"}
                    )

                # Create React agent with memgraph tools
                from langgraph.prebuilt import create_react_agent
                
                with langfuse.start_as_current_span(name="[agent-creation]-react-agent") as agent_span:
                    react_agent = create_react_agent(self.llm, memgraph_tools)
                    agent_span.update(
                        output={"agent_created": True, "tools_count": len(memgraph_tools)},
                        metadata={"operation": "agent_initialization"}
                    )

                # Prepare the prompt for inserting into memgraph
                incident = state.incident
                correlation_summary = state.correlation_summary

                insert_prompt = f"""
                You need to insert a new incident log and its RCA analysis into the memgraph database.

                **TASK:** Call the insert_log_tool with the following information:

                **Parameters for insert_log_tool:**
                - log: "{correlation_summary}"
                - rca: "{rca_analysis}"
                - source: "{incident.get('service', incident.get('source', 'unknown'))}"
                - alert_type: "error"
                - namespace: "{incident.get('namespace', '')}"

                **IMPORTANT:** You must call the insert_log_tool exactly once with these parameters.
                The tool will return a node_id which represents the unique identifier in the memgraph database.
                """

                # Run React agent to insert into memgraph with streaming
                logger.info("Running React agent to insert into memgraph")

                # Prepare inputs and config like correlation agent
                inputs = {
                    "messages": [("human", insert_prompt)]
                }

                config = {
                    'configurable': {
                        'thread_id': f"rca_memgraph_insert_{state.incident_id}",
                        'recursion_limit': 20
                    },
                    "callbacks": [self.langfuse_handler],
                    "metadata": {
                        "langfuse_trace_id": current_trace_id,
                        "langfuse_tags": ["rca_agent"]
                    }
                }

                # Process with streaming to capture intermediate results and extract node_id
                node_id = None
                final_result = None
                tool_calls_made = 0

                with langfuse.start_as_current_span(name="[agent-execution]-memgraph-insert") as exec_span:
                    async for chunk in react_agent.astream(inputs, config):
                        # Process each chunk
                        for node_name, node_data in chunk.items():
                            logger.info(f"Agent Node: {node_name}")

                            if 'messages' in node_data:
                                for msg in node_data['messages']:
                                    # Track tool calls
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_calls_made += 1
                                            tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                                            tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})

                                            with langfuse.start_as_current_span(name=f"[tool-execution]-{tool_name}") as tool_exec_span:
                                                logger.info(f"Agent calling tool: {tool_name}")
                                                logger.info(f"   Tool args: {tool_args}")

                                                tool_exec_span.update(
                                                    input={"tool_name": tool_name, "tool_args": tool_args},
                                                    metadata={"operation": "tool_call"}
                                                )

                                    # Track tool responses and extract node_id
                                    if hasattr(msg, 'name') and msg.name:
                                        try:
                                            content = getattr(msg, 'content', 'No content')
                                            logger.info(f"Tool response from {msg.name}: {content}")

                                            with langfuse.start_as_current_span(name=f"[tool-result]-{msg.name}") as tool_result_span:
                                                # Extract node_id from tool response using utility function
                                                if msg.name == 'insert_log_tool' and content:
                                                    try:
                                                        extracted_id = extract_node_id_from_response(content)
                                                        if extracted_id:
                                                            node_id = extracted_id

                                                        tool_result_span.update(
                                                            input={"response_content": content},
                                                            output={"node_id_extracted": node_id, "extraction_successful": bool(node_id)},
                                                            metadata={"operation": "node_id_extraction"}
                                                        )

                                                    except Exception as extract_error:
                                                        logger.warning(f"Error extracting node_id from response: {extract_error}")
                                                        tool_result_span.update(
                                                            output={"extraction_error": str(extract_error)},
                                                            metadata={"status": "extraction_failed"}
                                                        )

                                        except Exception as e:
                                            logger.error(f"Error handling tool response from {msg.name}: {e}")

                        # Keep track of the final result
                        final_result = chunk

                    exec_span.update(
                        output={
                            "tool_calls_made": tool_calls_made,
                            "node_id_extracted": node_id,
                            "execution_completed": True
                        }
                    )

                if node_id:
                    logger.info(f"Successfully inserted into memgraph, node_id: {node_id}")
                    state.memgraph_node_id = node_id

                    # Update the database with memgraph_node_id using utility function
                    with langfuse.start_as_current_span(name="[database-update]-memgraph-node-id") as db_update_span:
                        update_success = await update_memgraph_node_id_in_database(
                            self.mcp_client,
                            state.incident_id,
                            node_id,
                            max_retries=3
                        )
                        db_update_span.update(
                            input={"incident_id": state.incident_id, "node_id": node_id},
                            output={"database_updated": update_success},
                            metadata={"operation": "database_update"}
                        )
                else:
                    logger.warning("Could not extract node_id from memgraph insert response")

                span.update(
                    output={
                        "memgraph_inserted": bool(node_id),
                        "node_id": node_id,
                        "tool_calls_made": tool_calls_made,
                        "status": "success" if node_id else "partial_success"
                    },
                    metadata={"status": "success" if node_id else "warning"}
                )

            except Exception as e:
                logger.error(f"Error inserting into memgraph: {e}")
                # Don't fail the RCA process if memgraph insert fails
                span.update(
                    output={"memgraph_inserted": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

    async def add_jira_comment_for_rca(self, incident_data: Dict[str, Any], rca_analysis: str):
        """Add a Jira comment for RCA analysis using the same pattern as correlation agent"""
        with langfuse.start_as_current_span(name="add-jira-rca-comment") as span:
            span.update(
                input={
                    "has_jira_ticket_id": bool(incident_data.get("jira_ticket_id")),
                    "rca_analysis_length": len(rca_analysis)
                },
                metadata={"operation": "jira_comment", "component": "jira_integration"}
            )
            
            try:
                # Get the Jira ticket ID from incident data
                jira_ticket_id = incident_data.get("jira_ticket_id")

                if not jira_ticket_id:
                    logger.warning("No Jira ticket ID found for RCA analysis comment")
                    span.update(
                        output={"comment_added": False, "reason": "no_ticket_id"},
                        metadata={"status": "skipped"}
                    )
                    return

                # Check if MCP client and Jira tools are available
                if not self.mcp_client:
                    logger.warning("No MCP client available, skipping Jira comment")
                    span.update(
                        output={"comment_added": False, "reason": "no_mcp_client"},
                        metadata={"status": "skipped"}
                    )
                    return

                with langfuse.start_as_current_span(name="[validation]-jira-tools-availability") as validation_span:
                    available_tools = self.mcp_client.tools if hasattr(self.mcp_client, 'tools') else []
                    tool_names = [tool.name for tool in available_tools] if available_tools else []

                    has_jira_tool = any('jira_add_comment' in tool_name for tool_name in tool_names)

                    validation_span.update(
                        input={"available_tools": tool_names},
                        output={"jira_tool_available": has_jira_tool},
                        metadata={"operation": "tool_validation"}
                    )

                    if not has_jira_tool:
                        logger.warning("jira_add_comment tool not available, skipping Jira comment")
                        span.update(
                            output={"comment_added": False, "reason": "jira_tool_unavailable"},
                            metadata={"status": "skipped"}
                        )
                        return

                # Use the same variable pattern as correlation agent
                jira_variables = {
                    "analysis_type": "rca",
                    "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown Alert")),
                    "severity": incident_data.get("severity", "Unknown"),
                    "analysis_content": rca_analysis,
                    "title": "RCA"
                }

                with langfuse.start_as_current_span(name="[llm-call]-jira-comment-generation") as llm_span:
                    # Get JIRA formatter prompt from Langfuse (same as correlation agent)
                    jira_formatter_prompt = get_correlation_prompt("jira-formatter", jira_variables)
                    logger.info(f"Retrieved JIRA formatter prompt from Langfuse for RCA analysis")

                    # Use the same user prompt pattern as correlation agent
                    user_prompt = f"**Task:** Create focused markdown comment showing ONLY the rca analysis results based on the provided content and context."

                    llm_span.update(
                        input={
                            "prompt_variables": jira_variables,
                            "user_prompt_length": len(user_prompt)
                        },
                        metadata={"component": "jira_comment_generation"}
                    )

                    # Generate comment using LLM (same pattern as correlation agent)
                    response = await self.llm.ainvoke([
                        {"role": "system", "content": jira_formatter_prompt},
                        {"role": "user", "content": user_prompt}
                    ], config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": current_trace_id,
                            "langfuse_tags": ["rca_agent"]
                        }
                    })
                    
                    llm_span.update(
                        output={"comment_generated": True, "comment_length": len(response.content)},
                        metadata={"status": "success"}
                    )

                with langfuse.start_as_current_span(name="[processing]-comment-formatting") as format_span:
                    # Add footer using utility function
                    markdown_comment = f"{response.content}{format_jira_comment_footer()}"

                    # Sanitize Unicode characters using utility function
                    sanitized_comment = sanitize_unicode_for_jira(markdown_comment)

                    format_span.update(
                        input={"raw_comment_length": len(markdown_comment)},
                        output={"sanitized_comment_length": len(sanitized_comment)},
                        metadata={"operation": "comment_sanitization"}
                    )

                jira_params = {
                    "issue_key": jira_ticket_id,
                    "comment": sanitized_comment
                }

                logger.info(f"Adding RCA comment to Jira ticket: {jira_ticket_id}")

                max_retries = 3
                retry_count = 0
                success = False

                with langfuse.start_as_current_span(name="[tool-called]-jira-add-comment-with-retries") as retry_span:
                    while retry_count < max_retries and not success:
                        try:
                            with langfuse.start_as_current_span(name=f"[retry-attempt]-{retry_count + 1}") as attempt_span:
                                result = await self.mcp_client.call_tool_direct("jira_add_comment", jira_params)
                                success = True
                                
                                attempt_span.update(
                                    input={"attempt_number": retry_count + 1, "jira_params": jira_params},
                                    output={"success": True, "result": result},
                                    metadata={"operation": "jira_tool_call"}
                                )
                                
                                logger.info(f"Successfully added RCA analysis comment to Jira ticket (attempt {retry_count + 1})")

                        except Exception as retry_error:
                            retry_count += 1
                            logger.warning(f"Attempt {retry_count} failed for RCA Jira comment: {retry_error}")

                            attempt_span.update(
                                input={"attempt_number": retry_count},
                                output={"success": False, "error": str(retry_error)},
                                metadata={"status": "failed"}
                            )

                            if retry_count < max_retries:
                                import asyncio
                                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                            else:
                                logger.error(f"All {max_retries} attempts failed for RCA Jira comment")
                                raise retry_error

                    retry_span.update(
                        output={
                            "final_success": success,
                            "attempts_made": retry_count + (1 if success else 0),
                            "max_retries": max_retries
                        }
                    )

                span.update(
                    output={
                        "comment_added": success,
                        "jira_ticket_id": jira_ticket_id,
                        "attempts_made": retry_count + (1 if success else 0),
                        "status": "success" if success else "failed"
                    },
                    metadata={"status": "success" if success else "error"}
                )

            except Exception as e:
                logger.error(f"Failed to add RCA Jira comment: {e}")
                # Don't set error state as this shouldn't stop the workflow
                span.update(
                    output={"comment_added": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )
