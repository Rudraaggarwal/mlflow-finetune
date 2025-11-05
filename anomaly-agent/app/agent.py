import os
import logging
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.llm import LLMConfig
from config.mcp_servers import get_mcp_config

# Langfuse imports for comprehensive observability
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import uuid

# Initialize Langfuse client
load_dotenv()
langfuse = get_client()
logger = logging.getLogger(__name__)

class AnomalyAgentState(TypedDict):
    """State for the anomaly detection workflow"""
    # Input data
    alert_payload: Dict[str, Any]
    alert_status: str
    alertname: str
    severity: str
    service: str
    description: str
    timestamp: str
    # Processing data
    service_dependencies: List[str]
    jira_ticket_id: Optional[str]
    incident_id: Optional[str]
    database_result: Optional[str]
    # Control flow
    current_step: str
    error: Optional[str]
    completed: bool
    
_global_session_id=""

class AnomalyAgent:
    """Anomaly Detection Agent with conditional workflow graph and comprehensive observability"""

    def __init__(self):
        """Initialize the AnomalyAgent with comprehensive tracing setup."""
        # Initialize MCP client
        self.mcp_client = None
        self.mcp_tools = []
        
        # Initialize Langfuse handler and trace ID
        self.langfuse_handler = CallbackHandler()
        self.predefined_trace_id = langfuse.create_trace_id()
        
        try:
            mcp_config = get_mcp_config()
            if mcp_config:
                self.mcp_client = MultiServerMCPClient(mcp_config)
                logger.info("✅ MCP client initialized for anomaly workflow")
            else:
                logger.warning("⚠️ No MCP configuration available")
        except Exception as e:
            logger.warning(f"⚠️ MCP client initialization failed: {e}")

        # Initialize LLM
        self.llm = LLMConfig.get_llm()
        
        # Build workflow graph
        self.graph = self._build_workflow_graph()
        logger.info("✅ Anomaly Workflow Agent initialized")

    # @observe(name="workflow-graph-construction")
    def _build_workflow_graph(self):
        """Build the conditional workflow graph with observability."""
        with langfuse.start_as_current_span(name="anomaly-workflow-graph-build") as span:
            span.update(
                input={"agent_type": "anomaly_detection"},
                metadata={"component": "workflow-builder", "graph_type": "conditional"}
            )
            
            try:
                workflow = StateGraph(AnomalyAgentState)
                
                # Add nodes
                workflow.add_node("check_alert_status", self._check_alert_status_node)
                workflow.add_node("handle_resolved_alert", self._handle_resolved_alert_node)
                workflow.add_node("check_recent_incidents", self._check_recent_incidents_node)
                workflow.add_node("parse_alert", self._parse_alert_node)
                workflow.add_node("get_service_dependencies", self._get_service_dependencies_node)
                workflow.add_node("create_jira_ticket", self._create_jira_ticket_node)
                workflow.add_node("store_incident", self._store_incident_node)
                workflow.add_node("notify_correlation_agent", self._notify_correlation_agent_node)
                workflow.add_node("complete_workflow", self._complete_workflow_node)

                # Add conditional edges
                workflow.add_conditional_edges(
                    "check_alert_status",
                    self._should_handle_resolved,
                    {
                        "resolved": "handle_resolved_alert",
                        "active": "parse_alert"
                    }
                )

                # Check recent incidents conditional flow
                workflow.add_conditional_edges(
                    "check_recent_incidents",
                    self._check_incident_decision,
                    {
                        "continue": "get_service_dependencies",
                        "duplicate": END,
                        "error": END
                    }
                )

                # Sequential flow for active alerts with error handling
                workflow.add_conditional_edges(
                    "parse_alert",
                    self._check_for_errors,
                    {
                        "continue": "check_recent_incidents",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "get_service_dependencies",
                    self._check_for_errors,
                    {
                        "continue": "create_jira_ticket",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "create_jira_ticket",
                    self._check_for_errors,
                    {
                        "continue": "store_incident",
                        "error": END
                    }
                )
                workflow.add_conditional_edges(
                    "store_incident",
                    self._check_for_errors,
                    {
                        "continue": "notify_correlation_agent",
                        "error": END
                    }
                )
                workflow.add_edge("notify_correlation_agent", "complete_workflow")

                # End nodes
                workflow.add_edge("handle_resolved_alert", END)
                workflow.add_edge("complete_workflow", END)

                # Set entry point
                workflow.set_entry_point("check_alert_status")
                
                app = workflow.compile()
                print(app.get_graph().draw_ascii())
                
                span.update(
                    output={
                        "nodes_count": 8,
                        "conditional_edges": 4,
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

    # @observe(name="mcp-tools-setup")
    async def setup_mcp_tools(self):
        """Setup MCP tools for the workflow with observability."""
        with langfuse.start_as_current_span(name="mcp-tools-initialization") as span:
            if not self.mcp_client:
                span.update(
                    output={"tools_setup": False, "reason": "no_mcp_client"},
                    metadata={"status": "skipped"}
                )
                logger.warning("MCP client not initialized, skipping tool setup")
                return

            try:
                span.update(
                    input={"mcp_client_available": True},
                    metadata={"component": "mcp-setup"}
                )
                
                self.mcp_tools = await self.mcp_client.get_tools()
                
                jira_tools = [tool for tool in self.mcp_tools if 'jira' in tool.name.lower()]
                postgres_tools = [tool for tool in self.mcp_tools if 'describe_table' in tool.name.lower() or 'execute_query' in tool.name.lower()]
                
                span.update(
                    output={
                        "total_tools": len(self.mcp_tools),
                        "jira_tools": [tool.name for tool in jira_tools],
                        "postgres_tools": [tool.name for tool in postgres_tools]
                    },
                    metadata={"status": "success"}
                )
                
                logger.info(f"Retrieved {len(self.mcp_tools)} MCP tools for anomaly workflow")
                logger.info(f"Available Jira tools: {[tool.name for tool in jira_tools]}")
                logger.info(f"Available Postgres tools: {[tool.name for tool in postgres_tools]}")
                
            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                logger.error(f"Failed to setup MCP tools: {e}")
                raise

    # @observe(name="alert-status-check")
    async def _check_alert_status_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """STEP 0: Check if alert is resolved or active with comprehensive tracing."""
        with langfuse.start_as_current_span(name="check-alert-status") as span:
            logger.info("STEP 0: Checking alert status")
            
            try:
                alert_payload = state["alert_payload"]
                
                span.update(
                    input={
                        "payload_type": type(alert_payload).__name__,
                        "payload_preview": str(alert_payload)[:200]
                    },
                    metadata={"step": "alert_status_check", "workflow_position": 0}
                )
                
                # Handle case where alert_payload might be a string (JSON or LLM-wrapped)
                if isinstance(alert_payload, str):
                    with langfuse.start_as_current_span(name="payload-parsing") as parse_span:
                        try:
                            # First try direct JSON parsing
                            alert_payload = json.loads(alert_payload)
                            state["alert_payload"] = alert_payload
                            parse_span.update(
                                output={"parsing_method": "direct_json"},
                                metadata={"status": "success"}
                            )
                        except json.JSONDecodeError:
                            # Check if this is an LLM-wrapped payload
                            if "Alert Data:" in alert_payload and "{" in alert_payload:
                                try:
                                    # Extract JSON from LLM-wrapped text
                                    start_idx = alert_payload.find("{")
                                    end_idx = alert_payload.rfind("}") + 1
                                    json_part = alert_payload[start_idx:end_idx]
                                    alert_payload = json.loads(json_part)
                                    state["alert_payload"] = alert_payload
                                    parse_span.update(
                                        output={"parsing_method": "llm_wrapped_extraction"},
                                        metadata={"status": "success"}
                                    )
                                    logger.info("Successfully extracted JSON from LLM-wrapped payload")
                                except (json.JSONDecodeError, ValueError) as e:
                                    logger.error(f"Failed to extract JSON from LLM-wrapped payload: {e}")
                                    alert_payload = {"alertname": "Unknown Alert", "status": "firing", "severity": "warning"}
                                    state["alert_payload"] = alert_payload
                                    parse_span.update(
                                        output={"parsing_method": "fallback", "error": str(e)},
                                        metadata={"status": "fallback"}
                                    )
                            else:
                                logger.error(f"Failed to parse alert payload as JSON: {alert_payload}")
                                alert_payload = {"alertname": "Unknown Alert", "status": "firing", "severity": "warning"}
                                state["alert_payload"] = alert_payload
                                parse_span.update(
                                    output={"parsing_method": "fallback"},
                                    metadata={"status": "fallback"}
                                )

                # Extract alert status with type checking
                if not isinstance(alert_payload, dict):
                    error_msg = f"Invalid alert payload type: {type(alert_payload)}"
                    logger.error(error_msg)
                    state["error"] = error_msg
                    span.update(
                        output={"error": error_msg},
                        metadata={"status": "error"}
                    )
                    return state

                alert_status = alert_payload.get("status", "").lower()
                logger.info(f"Extracted alert status: {alert_status}")

                # Extract basic alert info with format detection
                state["alert_status"] = alert_status
                
                # Check if this is the new metric-based payload format
                if "metric" in alert_payload and "current_value" in alert_payload:
                    with langfuse.start_as_current_span(name="metric-format-processing") as metric_span:
                        # New payload format with metric data
                        state["alertname"] = f"Anomaly Alert: {alert_payload.get('metric', 'Unknown Metric')}"
                        state["severity"] = "critical" if alert_payload.get("status") == "firing" else "warning"
                        # Infer service from metric payload (container or metric suffix)
                        inferred_service = self._infer_service_from_metric(alert_payload, fallback=state.get("service", "unknown"))
                        state["service"] = inferred_service
                        
                        # Create description from metric data
                        current_val = alert_payload.get("current_value", 0)
                        normal_val = alert_payload.get("normal value (mean)", 0)
                        upper_band = alert_payload.get("upper_band", 0)
                        lower_band = alert_payload.get("lower_band", 0)
                        state["description"] = f"Metric '{alert_payload.get('metric', 'unknown')}' anomaly detected. Current: {current_val}, Normal: {normal_val}, Bands: [{lower_band}, {upper_band}]"
                        state["timestamp"] = alert_payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                        
                        metric_span.update(
                            input={"format_type": "metric_based"},
                            output={
                                "alertname": state["alertname"],
                                "severity": state["severity"],
                                "service": state["service"]
                            },
                            metadata={"format": "new_metric_format"}
                        )
                else:
                    with langfuse.start_as_current_span(name="legacy-format-processing") as legacy_span:
                        # Legacy payload format
                        state["alertname"] = alert_payload.get("alertname", "")
                        state["severity"] = alert_payload.get("severity", "")
                        state["service"] = alert_payload.get("service", alert_payload.get("namespace", ""))
                        state["description"] = alert_payload.get("description", alert_payload.get("summary", ""))
                        state["timestamp"] = alert_payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                        
                        legacy_span.update(
                            input={"format_type": "legacy"},
                            output={
                                "alertname": state["alertname"],
                                "severity": state["severity"],
                                "service": state["service"]
                            },
                            metadata={"format": "legacy_format"}
                        )

                state["current_step"] = "check_alert_status_complete"
                
                span.update(
                    output={
                        "alert_status": alert_status,
                        "alertname": state["alertname"],
                        "severity": state["severity"],
                        "service": state["service"],
                        "step_completed": True
                    },
                    metadata={"status": "success", "workflow_position": 0}
                )
                
                logger.info(f"Alert status: {alert_status}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error checking alert status: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 0}
                )
                
        return state

    # @observe(name="workflow-routing-decision")
    def _should_handle_resolved(self, state: AnomalyAgentState) -> str:
        """Decide whether to handle resolved alert or continue with active alert processing."""
        with langfuse.start_as_current_span(name="resolved-alert-routing") as span:
            alert_status = state.get("alert_status")
            decision = "resolved" if alert_status == "resolved" else "active"
            
            span.update(
                input={"alert_status": alert_status},
                output={"routing_decision": decision},
                metadata={"workflow_step": "routing"}
            )
            
            return decision

    # @observe(name="error-check-routing")
    def _check_for_errors(self, state: AnomalyAgentState) -> str:
        """Check if there are errors that should stop the workflow."""
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

    # @observe(name="resolved-alert-handling")
    async def _handle_resolved_alert_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """Handle resolved alert by updating database"""
        logger.info("STEP 0.1: Handling resolved alert")

        try:
            # Search for similar alerts in database
            alertname = state["alertname"]

            # Get Postgres tools
            postgres_tools = [tool for tool in self.mcp_tools if 'describe_table' in tool.name.lower() or 'execute_query' in tool.name.lower()]
            if postgres_tools:
                query_tool = next((tool for tool in postgres_tools if 'query' in tool.name.lower()), postgres_tools[0])

                # Search for similar active incidents
                search_query = f"""
                SELECT id FROM incidents
                WHERE alertname = '{alertname}'
                AND status IN ('Active', 'Open', 'Analyzed')
                ORDER BY created_date DESC
                LIMIT 3
                """

                search_result = await query_tool.ainvoke({"query": search_query})

                if search_result:
                    # Update incidents to resolved
                    current_time = datetime.now(timezone.utc).isoformat()

                    for incident in search_result:
                        incident_id = incident.get('id')
                        update_query = f"""
                        UPDATE incidents
                        SET status = 'Resolved', resolution_time = '{current_time}', mttr = '{current_time}'
                        WHERE id = {incident_id}
                        """

                        await query_tool.ainvoke({"query": update_query})

                    logger.info(f"Updated {len(search_result)} incidents to resolved status")

            state["current_step"] = "resolved_alert_handled"
            state["completed"] = True

        except Exception as e:
            logger.error(f"Error handling resolved alert: {e}")
            state["error"] = str(e)

        return state

    async def _check_recent_incidents_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """Check for recent incidents using React agent with MCP tools"""
        logger.info("🔍 Checking for recent incidents and handling deduplication")

        try:
            # Initialize deduplication fields
            state["existing_incident_id"] = None
            state["is_duplicate_alert"] = False
            state["incident_counter"] = 1
            state["incident_analysis"] = None

            if not self.mcp_client or not self.mcp_tools:
                logger.warning("MCP tools not available, skipping incident check")
                state["current_step"] = "incident_check_skipped"
                return state

            # Create React agent with MCP tools
            react_agent = create_react_agent(self.llm, self.mcp_tools)

            alertname = state.get("alertname", "unknown")
            service = state.get("service", "unknown")

            # Prepare the query for the React agent with detailed instructions
            query = f"""
                I need to check for recent incidents and handle alert deduplication for this Grafana alert.

                Alert details:
                - Alert Name: {alertname}
                - Service: {service}
                - Current Time: {datetime.now().isoformat()}
                - Alert Status: {state.get("alert_status", "firing")}

                Business Rules:
                1. If duplicate alert found with status "Active" within 1 hour → DUPLICATE (ignore, still processing)
                2. If duplicate alert found with status "Resolved" within 1 hour → NEW (follow normal flow)
                3. If no duplicate found → NEW (normal alert processing)

                Technical Steps:

                1) **Check for recent incidents (last 1 hour):**
                SELECT
                    id,
                    name,
                    service,
                    status,
                    created_date,
                    incident_counter,
                    jira_url,
                    name
                FROM incidents
                WHERE name = '{alertname}'
                AND service = '{service}'
                AND created_date >= (NOW() AT TIME ZONE 'utc') - INTERVAL '1 hour'
                ORDER BY created_date DESC LIMIT 5

                2) **Handle based on business rules:**

                - If found and status = "Active":
                → Do NOT update anything
                → This alert is still being processed, ignore duplicate

                - If found and status = "Resolved":
                → Follow normal alert processing flow (create new JIRA, new DB entry, etc.)
                → The resolved incident remains as-is

                - If no matching incidents found:
                → Normal alert processing continues

                3) **IMPORTANT - End your response with exactly one of these:**
                - "STATUS: DUPLICATE" - if found Active incident (ignore alert)
                - "STATUS: NEW" - for resolved incidents or no matching incidents (normal flow)
            """

            # Execute React agent with streaming for intermediate steps
            logger.info(f" Executing React agent for incident analysis with streaming with query{query}")

            inputs = {"messages": [HumanMessage(content=query)]}
            config = {
                "configurable": {
                    "thread_id": f"anomaly-incident-check-{alertname}-{datetime.now().timestamp()}"
                }
            }

            # Process with streaming to capture intermediate results with detailed logging
            final_result = None
            intermediate_messages = []
            mcp_calls = 0
            mcp_errors = 0
            tool_calls_made = []

            try:
                # Stream React agent execution to see intermediate steps
                async for chunk in react_agent.astream(inputs, config):
                    # Process each chunk
                    for node_name, node_data in chunk.items():
                        logger.info(f"🔄 React Agent Node: {node_name}")

                    if 'messages' in node_data:
                        for msg in node_data['messages']:
                            if hasattr(msg, 'content'):
                                logger.debug(f"Triage Agent message: {msg.content}")

                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                with langfuse.start_as_current_span(name="tool-calls-batch") as batch_span:
                                    batch_tools = []
                                    
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                                        tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                                        tool_id = tool_call.get('id', '') if isinstance(tool_call, dict) else getattr(tool_call, 'id', '')

                                        logger.info(f"🔧 Triage Agent calling tool: {tool_name}")
                                        logger.info(f"   Tool args: {tool_args}")

                                        # Track tool info
                                        tool_info = {
                                            "tool_name": tool_name,
                                            "tool_id": tool_id,
                                            "tool_args": tool_args,
                                            "execution_order": len(tool_calls_made) + 1,
                                            "node_name": node_name
                                        }
                                        tool_calls_made.append(tool_info)
                                        batch_tools.append(tool_info)

                                        # **CREATE INDIVIDUAL TOOL CALL SPANS**
                                        with langfuse.start_as_current_span(
                                            name=f"[tool-called]-{tool_name}"
                                        ) as tool_span:
                                            tool_span.update(
                                                input={
                                                    "tool_name": tool_name,
                                                    "tool_args": tool_args,
                                                    "tool_id": tool_id,
                                                    "node": node_name
                                                },
                                                metadata={
                                                    "tool_type": "triage_agent_action",
                                                    "execution_order": tool_info['execution_order'],
                                                    "agent_type": "triage_agent"
                                                }
                                            )

                                        # Count database and JIRA MCP tool calls
                                        if any(db_type in tool_name.lower() for db_type in ['postgres', 'database', 'execute_query', 'describe_table']):
                                            mcp_calls += 1
                                            with langfuse.start_as_current_span(name="mcp-database-tool-identified") as db_span:
                                                db_span.update(
                                                    input={"tool_name": tool_name, "tool_type": "database"},
                                                    metadata={"mcp_tool_category": "database"}
                                                )
                                        elif any(jira_type in tool_name.lower() for jira_type in ['jira', 'ticket', 'comment']):
                                            mcp_calls += 1
                                            with langfuse.start_as_current_span(name="mcp-jira-tool-identified") as jira_span:
                                                jira_span.update(
                                                    input={"tool_name": tool_name, "tool_type": "jira"},
                                                    metadata={"mcp_tool_category": "jira"}
                                                )

                                    # Update batch span
                                    batch_span.update(
                                        output={
                                            "batch_size": len(batch_tools),
                                            "tools_in_batch": [t['tool_name'] for t in batch_tools],
                                            "total_mcp_calls": mcp_calls
                                        },
                                        metadata={
                                            "node_name": node_name,
                                            "agent_type": "triage_agent"
                                        }
                                    )

                            if hasattr(msg, 'name') and msg.name:
                                try:
                                    content = getattr(msg, 'content', 'No content')
                                    logger.info(f"🔍 Tool response from {msg.name}: {str(content)[:200]}..." if len(str(content)) > 200 else f"🔍 Tool response from {msg.name}: {content}")

                                    # **CREATE TOOL RESPONSE SPANS**
                                    with langfuse.start_as_current_span(
                                        name=f"[tool-result]-{msg.name}"
                                    ) as result_span:
                                        result_span.update(
                                            input={"tool_name": msg.name},
                                            output={
                                                "content_preview": str(content)[:500],
                                                "full_length": len(str(content)),
                                                "has_content": bool(content)
                                            },
                                            metadata={
                                                "message_type": "tool_response",
                                                "content_length": len(str(content)),
                                                "agent_type": "triage_agent",
                                                "node_name": node_name
                                            }
                                        )

                                        # **ERROR CHECK WITH TRACING**
                                        is_error = isinstance(content, str) and ('error' in content.lower() or 'failed' in content.lower())
                                        if is_error:
                                            if any(tool_type in msg.name.lower() for tool_type in ['postgres', 'database', 'jira', 'execute_query']):
                                                mcp_errors += 1
                                                logger.warning(f"❌ MCP tool error in {msg.name}: {content}")
                                                
                                                # **CREATE ERROR SPAN**
                                                with langfuse.start_as_current_span(name="mcp-tool-error") as error_span:
                                                    error_span.update(
                                                        input={"tool_name": msg.name, "error_content": str(content)[:200]},
                                                        output={"mcp_errors_total": mcp_errors},
                                                        metadata={
                                                            "error_type": "mcp_tool_error",
                                                            "tool_category": "database" if any(db in msg.name.lower() for db in ['postgres', 'database']) else "jira"
                                                        }
                                                    )
                                        
                                        result_span.update(
                                            metadata={
                                                "has_error": is_error,
                                                "total_mcp_errors": mcp_errors
                                            }
                                        )

                                except Exception as e:
                                    logger.error(f"Error handling tool response from {msg.name}: {e}")
                                    
                                    # **CREATE ERROR HANDLING SPAN**
                                    with langfuse.start_as_current_span(name="tool-response-error") as error_span:
                                        error_span.update(
                                            input={"tool_name": msg.name, "error": str(e)},
                                            metadata={"error_type": "tool_response_processing_error"}
                                        )

                            intermediate_messages.append(msg)

                    # Keep track of the final result
                    final_result = chunk

                logger.info(f"📊 Incident Check Summary: {mcp_calls} MCP calls made, {mcp_errors} errors encountered")

            except Exception as streaming_error:
                logger.error(f"Error during streaming execution: {streaming_error}")
                final_result = None

            # Extract final content from the streaming results
            final_message = "Processing completed"
            if final_result:
                for node_name, node_data in final_result.items():
                    if 'messages' in node_data and node_data['messages']:
                        final_msg = node_data['messages'][-1]
                        if hasattr(final_msg, 'content') and final_msg.content:
                            final_message = final_msg.content
                            break

            logger.info(f"✅ Final React agent response: {final_message[:200]}..." if len(final_message) > 200 else f"✅ Final React agent response: {final_message}")

            # Parse the STATUS line from final message to determine flow
            if "STATUS: DUPLICATE" in final_message:
                state["is_duplicate_alert"] = True
                state["current_step"] = "duplicate_alert_handled"
                logger.info("✅ STATUS: DUPLICATE - Active incident found, ignoring duplicate (still processing)")

            elif "STATUS: NEW" in final_message:
                state["current_step"] = "no_recent_incidents"
                logger.info("✅ STATUS: NEW - Proceeding with normal alert processing (resolved incident or no match)")

            else:
                # Fallback if STATUS line is not found
                state["current_step"] = "no_recent_incidents"
                logger.warning("⚠️ No clear STATUS found in response, proceeding as new incident")

            # Store the agent's analysis for reference
            state["incident_analysis"] = final_message

        except Exception as e:
            logger.error(f"❌ Error in React agent incident check: {e}")
            state["error"] = str(e)
            state["current_step"] = "incident_check_error"

        return state

    def _check_incident_decision(self, state: AnomalyAgentState) -> str:
        """Decision function for incident checking flow"""
        if state.get("error"):
            logger.error("Error detected, stopping workflow")
            return "error"

        if state.get("is_duplicate_alert"):
            logger.info("Duplicate alert detected (Active status), ending workflow")
            return "duplicate"

        logger.info("Continuing with normal flow (NEW or resolved incident)")
        return "continue"

    async def _parse_alert_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """STEP 1: Parse alert and extract information with comprehensive tracing."""
        with langfuse.start_as_current_span(name="parse-alert") as span:
            logger.info("STEP 1: Parsing alert")
            
            try:
                alert_payload = state["alert_payload"]
                
                span.update(
                    input={
                        "payload_type": type(alert_payload).__name__,
                        "payload_keys": list(alert_payload.keys()) if isinstance(alert_payload, dict) else []
                    },
                    metadata={"step": "alert_parsing", "workflow_position": 1}
                )
                
                if not isinstance(alert_payload, dict):
                    error_msg = f"Parse alert failed - invalid payload type: {type(alert_payload)}"
                    logger.error(error_msg)
                    state["error"] = error_msg
                    span.update(
                        output={"error": error_msg},
                        metadata={"status": "error"}
                    )
                    return state

                # Extract information from different payload formats
                if "alerts" in alert_payload and alert_payload["alerts"]:
                    with langfuse.start_as_current_span(name="grafana-webhook-parsing") as grafana_span:
                        # Grafana webhook format
                        first_alert = alert_payload["alerts"][0]
                        labels = first_alert.get("labels", {})
                        state["alertname"] = labels.get("alertname", "Unknown Alert")
                        state["severity"] = labels.get("severity", "medium")
                        state["service"] = labels.get("namespace", labels.get("application", "unknown"))
                        state["timestamp"] = first_alert.get("startsAt", datetime.now(timezone.utc).isoformat())
                        
                        grafana_span.update(
                            input={"format": "grafana_webhook", "alerts_count": len(alert_payload["alerts"])},
                            output={
                                "alertname": state["alertname"],
                                "severity": state["severity"],
                                "service": state["service"]
                            },
                            metadata={"parsing_method": "grafana_webhook"}
                        )
                        
                        # Generate description using LLM
                        await self._generate_alert_description(state, alert_payload)
                        
                elif "metric" in alert_payload and "current_value" in alert_payload:
                    with langfuse.start_as_current_span(name="metric-based-parsing") as metric_span:
                        # New metric-based payload format
                        logger.info("Detected new metric-based payload format")
                        
                        # Infer alert name and service deterministically (no LLM)
                        inferred_alert_name, inferred_service = await self._infer_alert_name_from_metric(alert_payload)
                        state["alertname"] = inferred_alert_name
                        state["severity"] = "critical" if alert_payload.get("status") == "firing" else "warning"
                        state["service"] = inferred_service
                        
                        # Create description from metric data
                        current_val = alert_payload.get("current_value", 0)
                        normal_val = alert_payload.get("normal value (mean)", 0)
                        upper_band = alert_payload.get("upper_band", 0)
                        lower_band = alert_payload.get("lower_band", 0)
                        metric_name = alert_payload.get("metric", "unknown")
                        usage_data = alert_payload.get("usage data", "")
                        
                        description = f"Metric '{metric_name}' anomaly detected."
                        description += f" Current: {current_val}, Normal: {normal_val}"
                        description += f", Bands: [{lower_band}, {upper_band}]"
                        if usage_data:
                            description += f". Usage data: {usage_data}"
                        state["description"] = description
                        state["timestamp"] = alert_payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                        
                        metric_span.update(
                            input={"format": "metric_based", "metric": metric_name},
                            output={
                                "alertname": state["alertname"],
                                "severity": state["severity"],
                                "description_length": len(description)
                            },
                            metadata={"parsing_method": "metric_based"}
                        )
                        
                else:
                    with langfuse.start_as_current_span(name="direct-format-parsing") as direct_span:
                        # Direct alert format fallback
                        state["alertname"] = alert_payload.get("alertname", "Unknown Alert")
                        state["severity"] = alert_payload.get("severity", "medium")
                        state["service"] = alert_payload.get("service", alert_payload.get("namespace", "unknown"))
                        state["description"] = alert_payload.get("description", "No description available")
                        state["timestamp"] = alert_payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                        
                        direct_span.update(
                            input={"format": "direct"},
                            output={
                                "alertname": state["alertname"],
                                "severity": state["severity"],
                                "service": state["service"]
                            },
                            metadata={"parsing_method": "direct_format"}
                        )

                state["current_step"] = "parse_alert_complete"
                
                span.update(
                    output={
                        "alertname": state["alertname"],
                        "severity": state["severity"],
                        "service": state["service"],
                        "description_length": len(state["description"]),
                        "step_completed": True
                    },
                    metadata={"status": "success", "workflow_position": 1}
                )
                
                logger.info(f"Parsed alert: {state['alertname']} - {state['severity']}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error parsing alert: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 1}
                )
                
        return state

    # @observe(name="llm-alert-name-inference")
    async def _infer_alert_name_from_metric(self, alert_payload: Dict[str, Any]) -> tuple[str, str]:
        """Infer alert name and service from metric payload using a single LLM call returning JSON.

        Returns: (alert_name, service_name)
        """
        with langfuse.start_as_current_span(name="alert-name-generation") as span:
            try:
                metric_name_raw = alert_payload.get("metric", "unknown") or "unknown"
                current_value = alert_payload.get("current_value", 0)
                mean_value = alert_payload.get("normal value (mean)", 0)
                upper_band = alert_payload.get("upper_band", 0)
                lower_band = alert_payload.get("lower_band", 0)
                status = str(alert_payload.get("status", "")).lower()
                expr = alert_payload.get("expr", "")
                usage = alert_payload.get("usage data", "")

                # Best-effort local inference for service to help LLM
                seed_service = self._infer_service_from_metric(alert_payload, fallback="unknown")

                llm_prompt = f"""
You are an SRE assistant. Infer a concise alert name and the service involved from a metric-based anomaly payload. Return ONLY valid JSON.

Metric Payload (normalized):
- metric: {metric_name_raw}
- current_value: {current_value}
- mean_value: {mean_value}
- upper_band: {upper_band}
- lower_band: {lower_band}
- status: {status}
- expr: {expr}
- seed_service: {seed_service}
- usage_data_preview: {usage}

Rules:
- alert_name: 2-5 words, clear (e.g., "High CPU Usage", "Memory Usage Anomaly").
- service: prefer container from usage data; otherwise infer from metric suffix (e.g., memory_usage_oemconnector -> oemconnector). Lowercase.
- If unsure, use seed_service when reasonable.
- Output strictly as JSON with keys: alert_name, service
"""

                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content="You produce only strict JSON responses for downstream parsing."),
                    HumanMessage(content=llm_prompt)
                ]

                response = await self.llm.ainvoke(
                    messages,
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": self.predefined_trace_id,
                            "langfuse_tags": ["anomaly-agent", "alert-naming"],
                            "component": "alert_name_service_generation"
                        }
                    }
                )

                raw = (response.content or "").strip()
                import re, json
                match = re.search(r"\{[\s\S]*\}", raw)
                json_text = match.group(0) if match else raw
                data = json.loads(json_text)
                alert_name = str(data.get("alert_name") or "").strip()
                service_name = str(data.get("service") or "").strip().lower()

                if not alert_name:
                    alert_name = f"Anomaly: {metric_name_raw}"
                if not service_name or service_name == "unknown":
                    service_name = seed_service

                span.update(
                    input={
                        "metric": metric_name_raw,
                        "seed_service": seed_service,
                    },
                    output={"alert_name": alert_name, "service": service_name},
                    metadata={"status": "success"}
                )

                return alert_name, service_name
            except Exception as e:
                span.update(
                    output={"error": str(e), "fallback_used": True},
                    metadata={"status": "fallback"}
                )
                # Fallback: deterministic
                try:
                    service_name = self._infer_service_from_metric(alert_payload, fallback="unknown")
                    metric_name_raw = alert_payload.get('metric', 'Unknown Metric')
                    return f"Anomaly: {metric_name_raw}", service_name
                except Exception:
                    return "Anomaly Alert", "unknown"

    # @observe(name="llm-description-generation")
    async def _generate_alert_description(self, state: AnomalyAgentState, alert_payload: Dict[str, Any]):
        """Generate alert description using LLM with comprehensive tracing."""
        with langfuse.start_as_current_span(name="alert-description-generation") as span:
            try:
                if "alerts" in alert_payload and alert_payload["alerts"]:
                    first_alert = alert_payload["alerts"][0]
                    labels = first_alert.get("labels", {})
                    annotations = first_alert.get("annotations", {})
                    
                    span.update(
                        input={
                            "alertname": labels.get("alertname", "Unknown"),
                            "severity": labels.get("severity", "unknown"),
                            "application": labels.get("application", "unknown"),
                            "namespace": labels.get("namespace", "unknown")
                        },
                        metadata={"component": "llm_description_generation"}
                    )
                    
                    # Create prompt for LLM
                    llm_prompt = f"""
                    Analyze this Grafana alert and create a concise technical description:
                    Alert: {labels.get("alertname", "Unknown")}
                    Severity: {labels.get("severity", "unknown")}
                    Application: {labels.get("application", "unknown")}
                    Namespace: {labels.get("namespace", "unknown")}
                    Container: {labels.get("container_name", "unknown")}
                    Pod: {labels.get("pod_name", "unknown")}
                    Node: {labels.get("node", "unknown")}
                    Annotations:
                    Description: {annotations.get("description", "No description")}
                    Summary: {annotations.get("summary", "No summary")}
                    Values: {first_alert.get("values", {})}
                    Create a clear, technical description of what's happening in 1-2 sentences.
                    """
                    
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [
                        SystemMessage(content="You are an SRE expert. Create concise, technical alert descriptions in markdown format"),
                        HumanMessage(content=llm_prompt)
                    ]
                    
                    response = await self.llm.ainvoke(
                        messages,
                        config={
                            "callbacks": [self.langfuse_handler],
                            "metadata": {
                                "langfuse_trace_id": self.predefined_trace_id,
                                "langfuse_tags": ["anomaly-agent", "description-generation"],
                                "component": "alert_description_generation"
                            }
                        }
                    )
                    
                    state["description"] = response.content.strip()
                    
                    span.update(
                        output={
                            "description_generated": True,
                            "description_length": len(state["description"])
                        },
                        metadata={"status": "success"}
                    )
                    
                    logger.info(f"Generated LLM description: {state['description']}")
                    
            except Exception as e:
                span.update(
                    output={"error": str(e), "fallback_used": True},
                    metadata={"status": "fallback"}
                )
                logger.warning(f"Failed to generate LLM description: {e}")
                # Fallback to annotation description
                if "alerts" in alert_payload and alert_payload["alerts"]:
                    annotations = alert_payload["alerts"][0].get("annotations", {})
                    state["description"] = annotations.get("description", annotations.get("summary", "No description available"))

    # @observe(name="service-dependencies-retrieval")
    async def _get_service_dependencies_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """Get service dependencies with tracing."""
        with langfuse.start_as_current_span(name="get-service-dependencies") as span:
            logger.info("STEP 1.1: Getting service dependencies")
            
            try:
                span.update(
                    input={"service": state.get("service")},
                    metadata={"step": "service_dependencies", "workflow_position": "1.1"}
                )
                
                # For Paylater application, always search for oemconnector and catalogueserv only
                # The prompt specifies to always search for oemconnector and catalogueserv only
                # Paylater is application name inside which these are services
                state["service_dependencies"] = ["oemconnector", "catalogueserv"]
                state["current_step"] = "service_dependencies_complete"
                
                span.update(
                    output={
                        "dependencies": state["service_dependencies"],
                        "dependencies_count": len(state["service_dependencies"])
                    },
                    metadata={"status": "success", "workflow_position": "1.1"}
                )
                
                logger.info(f"Service dependencies retrieved: {state['service_dependencies']}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error getting service dependencies: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": "1.1"}
                )
                
        return state

    # @observe(name="jira-ticket-creation")
    async def _create_jira_ticket_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """STEP 2: Create Jira ticket with comprehensive tracing."""
        with langfuse.start_as_current_span(name="create-jira-ticket") as span:
            logger.info("STEP 2: Creating Jira ticket")
            
            try:
                span.update(
                    input={
                        "alertname": state.get('alertname'),
                        "service": state.get('service'),
                        "severity": state.get('severity'),
                        "description_length": len(state.get('description', ''))
                    },
                    metadata={"step": "jira_creation", "workflow_position": 2}
                )
                
                # Get Jira tools
                jira_tools = [tool for tool in self.mcp_tools if 'jira' in tool.name.lower()]
                if not jira_tools:
                    raise Exception("No Jira tools available")

                create_tool = next((tool for tool in jira_tools if 'jira_create_issue' in tool.name.lower()), jira_tools[0])
                logger.info(f"Using Jira tool: {create_tool.name}")

                # Prepare ticket data with safe access
                current_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000+0000")
                alert_timestamp = state.get("timestamp", datetime.now(timezone.utc).isoformat())

                # Format description with metric data if available
                with langfuse.start_as_current_span(name="jira-description-formatting") as desc_span:
                    description = await self._format_jira_description(state)
                    # Store the human-readable description for later use in database
                    state["jira_description"] = description
                    logger.info(f"Jira description generated and stored: {description[:100]}..." if len(description) > 100 else f"Jira description generated and stored: {description}")
                    desc_span.update(
                        output={"description_length": len(description)},
                        metadata={"formatting_step": "jira_description"}
                    )

                ticket_data = {
                    "project_key": "PRCINC",
                    "summary": f"Alert: {state['alertname']} - {state['service']} service issue",
                    "issue_type": "Incident",
                    "description": description,
                    "assignee": "712020:0018661f-7bc3-43f2-98a8-130040b4b71c",
                    "additional_fields": {
                        "reporter": {
                            "accountId": "712020:0018661f-7bc3-43f2-98a8-130040b4b71c"
                        },
                        "customfield_10044": [{"value": "EMI"}],
                        "customfield_10344": {"value": "Alert"},
                        "customfield_10335": {"value": "Single Component Failure"},
                        "customfield_10065": {"value": "Sev 1"},
                        "customfield_10537": [{"accountId": "712020:e3cefd80-5e61-4a6a-a72c-2e504b08e11c"}],
                        "customfield_10085": {"value": "Moderate / Limited"},
                        "customfield_10538": f"{state['service']} service experiencing {state['alertname']}",
                        "customfield_10237": current_timestamp,
                        "customfield_10539": alert_timestamp,
                        "customfield_10069": {"accountId": "557058:ca5a0eee-2f7d-478f-9358-42de1e3c64fb"},
                        "customfield_10540": {"value": "Not Applicable"},
                        "customfield_10337": {"value": "Change Failure"},
                        "customfield_10338": {"value": "Resolved (Permanently)"}
                    }
                }

                # Create Jira ticket with retries and tool execution tracing
                with langfuse.start_as_current_span(name="jira-tool-execution") as tool_span:
                    tool_span.update(
                        input={
                            "tool_name": create_tool.name,
                            "project_key": ticket_data["project_key"],
                            "summary": ticket_data["summary"]
                        },
                        metadata={"tool_execution": "jira_create"}
                    )

                    max_retries = 3
                    retry_count = 0
                    last_error = None
                    jira_result = None

                    while retry_count < max_retries and jira_result is None:
                        try:
                            jira_result = await create_tool.ainvoke(ticket_data)
                        except Exception as retry_error:
                            last_error = retry_error
                            retry_count += 1
                            logger.warning(f"Attempt {retry_count} failed for Jira create: {retry_error}")
                            if retry_count < max_retries:
                                import asyncio
                                await asyncio.sleep(2 ** retry_count)
                            else:
                                break

                    tool_span.update(
                        output={
                            "result_type": type(jira_result).__name__ if jira_result is not None else "None",
                            "result_received": bool(jira_result),
                            "retries": retry_count,
                            "last_error": str(last_error) if last_error else None
                        },
                        metadata={"status": "success" if jira_result else "error"}
                    )

                logger.info(f"Jira result: {jira_result}")

                if isinstance(jira_result, str):
                    try:
                        jira_result = json.loads(jira_result)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode Jira result: {jira_result}")
                        raise

                if jira_result and "issue" in jira_result:
                    state["jira_ticket_id"] = jira_result["issue"].get("key", "")
                    logger.info(f"Created Jira ticket: {state['jira_ticket_id']}")
                else:
                    raise Exception("Failed to create Jira ticket")

                state["current_step"] = "jira_ticket_created"
                
                span.update(
                    output={
                        "jira_ticket_id": state["jira_ticket_id"],
                        "ticket_created": True
                    },
                    metadata={"status": "success", "workflow_position": 2}
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error creating Jira ticket: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 2}
                )
                
        return state

    # @observe(name="jira-description-formatting")
    async def _format_jira_description(self, state: AnomalyAgentState) -> str:
        """Generate Jira description using LLM based on alert payload with tracing."""
        with langfuse.start_as_current_span(name="format-jira-description") as span:
            alert_payload = state.get("alert_payload", {})
            
            # Type checking for alert_payload
            if not isinstance(alert_payload, dict):
                logger.warning(f"Alert payload is not dict in format_jira_description: {type(alert_payload)}")
                alert_payload = {}

            span.update(
                input={
                    "alertname": state.get('alertname', 'Unknown'),
                    "service": state.get('service', 'Unknown'),
                    "severity": state.get('severity', 'Unknown'),
                    "payload_keys": list(alert_payload.keys()) if alert_payload else []
                },
                metadata={"component": "jira_description_formatter"}
            )

            try:
                # Create LLM prompt for generating Jira description (summarize metric payloads clearly)
                llm_prompt = f"""Create a concise, factual Jira incident description based solely on the provided alert data. Keep it brief and professional.
                    Alert Data:
                    - Alert: {state.get('alertname', 'Unknown')}
                    - Service: {state.get('service', 'Unknown')}
                    - Severity: {state.get('severity', 'Unknown')}
                    - Status: {state.get('description', 'No description')}
                    - Time: {state.get('timestamp', 'Unknown')}

                    Raw Alert Payload:
                    {json.dumps(alert_payload, indent=2)}

                    Dependencies: {', '.join(state.get('service_dependencies', []))}

                    Requirements:
                    - Write 3-4 sentences maximum
                    - State facts only from the alert data
                    - If metric payload is present (e.g., fields like "usage data", "current_value", "normal value (mean)", "upper_band", "lower_band", "expr"), summarize it clearly:
                      • Name the metric and container (if present)
                      • Compare current vs mean with % deviation
                      • Mention bands and whether current breached the upper band
                      • Briefly describe the time window covered by the series (start→end UTC)
                    - No emojis or excessive formatting
                    - Focus on what happened and when, not recommendations
                    - Mention affected service and severity
                    Generate a brief, factual incident description."""

                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content="You are an expert SRE creating professional Jira incident descriptions. Be concise but comprehensive, focusing on actionable information."),
                    HumanMessage(content=llm_prompt)
                ]

                response = await self.llm.ainvoke(
                    messages,
                    config={
                        "callbacks": [self.langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": self.predefined_trace_id,
                            "langfuse_tags": ["anomaly-agent", "jira-description"],
                            "component": "jira_description_generation"
                        }
                    }
                )
                
                description = response.content.strip()
                
                span.update(
                    output={
                        "description_generated": True,
                        "description_length": len(description),
                        "method": "llm_generated"
                    },
                    metadata={"status": "success"}
                )
                
                logger.info(f"Generated LLM-based Jira description: {description[:200]}...")
                return description
                
            except Exception as e:
                span.update(
                    output={"error": str(e), "fallback_used": True, "method": "template_fallback"},
                    metadata={"status": "fallback"}
                )
                logger.warning(f"Failed to generate LLM description: {e}, falling back to template")
                # Fallback to template-based description
                return self._fallback_jira_description(state, alert_payload)

    # @observe(name="jira-description-fallback")
    def _fallback_jira_description(self, state: AnomalyAgentState, alert_payload: dict) -> str:
        """Fallback template-based Jira description if LLM fails with tracing."""
        with langfuse.start_as_current_span(name="fallback-jira-description") as span:
            span.update(
                input={"fallback_reason": "llm_generation_failed"},
                metadata={"component": "template_description"}
            )
            
            description = f"Hi Team,\n\nWe got an alert for {state.get('alertname', 'Unknown')}.\n\n"
            description += f"Service: {state.get('service', 'Unknown')}\n"
            description += f"Severity: {state.get('severity', 'Unknown')}\n"
            description += f"Description: {state.get('description', 'No description')}\n\n"

            # Check for metric data with safe access and summarize clearly
            metric_info_added = False
            if isinstance(alert_payload, dict):
                # Check for new payload format with direct metric fields
                if "metric" in alert_payload and "current_value" in alert_payload:
                    current_value = alert_payload.get("current_value", 0)
                    normal_value = alert_payload.get("normal value (mean)", 0)
                    upper_band = alert_payload.get("upper_band", 0)
                    lower_band = alert_payload.get("lower_band", 0)
                    metric_name = alert_payload.get("metric", "Unknown metric")
                    usage_data = alert_payload.get("usage data", "")
                    expr = alert_payload.get("expr", "")

                    # Calculate percentage deviation
                    if normal_value > 0:
                        difference = current_value - normal_value
                        percentage_deviation = (difference / normal_value) * 100
                    else:
                        percentage_deviation = 0

                    # Derive series window if usage_data is present
                    window_text = ""
                    try:
                        series = usage_data[0].get("values", []) if isinstance(usage_data, list) and usage_data else []
                        if series:
                            start_ts = series[0][0]
                            end_ts = series[-1][0]
                            from datetime import datetime, timezone
                            start_iso = datetime.fromtimestamp(int(start_ts), tz=timezone.utc).isoformat()
                            end_iso = datetime.fromtimestamp(int(end_ts), tz=timezone.utc).isoformat()
                            window_text = f" (window: {start_iso} → {end_iso})"
                    except Exception:
                        window_text = ""

                    breached_text = "breached upper band" if upper_band and current_value >= upper_band else "within bands"

                    description += f"Metric Analysis ({metric_name}){window_text}:\n"
                    description += f"- Current vs mean: {current_value:.4f} vs {normal_value:.4f} ({percentage_deviation:.2f}% deviation, {breached_text})\n"
                    description += f"- Bands: lower={lower_band:.4f}, upper={upper_band:.4f}\n"
                    if expr:
                        description += f"- Expr: {expr}\n"
                    description += "\n"
                    metric_info_added = True

            description += f"Dependencies: {', '.join(state.get('service_dependencies', []))}\n"
            description += f"Timestamp: {state['timestamp']}\n\n"
            description += "Please investigate and take necessary action."
            
            span.update(
                output={
                    "description_length": len(description),
                    "metric_info_included": metric_info_added
                },
                metadata={"status": "success"}
            )

            return description

    # @observe(name="incident-storage")
    async def _store_incident_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """STEP 3: Store incident in database with comprehensive tracing."""
        with langfuse.start_as_current_span(name="store-incident") as span:
            logger.info("STEP 3: Storing incident in database")
            
            try:
                span.update(
                    input={
                        "alertname": state.get('alertname'),
                        "service": state.get('service'),
                        "jira_ticket_id": state.get('jira_ticket_id')
                    },
                    metadata={"step": "incident_storage", "workflow_position": 3}
                )
                
                # Get Postgres tools
                postgres_tools = [tool for tool in self.mcp_tools if 'describe_table' in tool.name.lower() or 'execute_query' in tool.name.lower()]
                if not postgres_tools:
                    raise Exception("No Postgres tools available")

                query_tool = next((tool for tool in postgres_tools if 'query' in tool.name.lower()), postgres_tools[0])
                logger.info(f"Using Postgres tool: {query_tool.name}")

                # First describe table structure
                with langfuse.start_as_current_span(name="table-structure-validation") as table_span:
                    describe_query = "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'incidents'"
                    table_result = await query_tool.ainvoke({"query": describe_query})
                    
                    table_span.update(
                        output={"table_structure_validated": bool(table_result)},
                        metadata={"validation_step": "table_structure"}
                    )

                # Prepare incident data
                current_time = datetime.now(timezone.utc).isoformat()
                jira_base_url = os.getenv("JIRA_URL", "https://your-jira-domain.atlassian.net")
                jira_url = f"{jira_base_url}/browse/{state['jira_ticket_id']}"

                # Format dependencies array for PostgreSQL
                dependencies_str = "'{" + ",".join(state["service_dependencies"]) + "}'"

                # Use the human-readable Jira description for database storage
                # Fallback to basic description if Jira description is not available
                db_description = state.get("jira_description", state.get("description", "No description available"))

                # Debug logging to verify which description is being used
                logger.info(f"Database storage - jira_description available: {bool(state.get('jira_description'))}")
                logger.info(f"Database storage - using description: {db_description[:100]}..." if len(db_description) > 100 else f"Database storage - using description: {db_description}")

                # Insert incident with database execution tracing
                with langfuse.start_as_current_span(name="database-insert-execution") as db_span:
                    insert_query = f"""
                    INSERT INTO incidents (
                        name, risk_level, service, description, status,
                        mttd, created_date, dependencies, jira_url
                    ) VALUES (
                        '{state["alertname"].replace("'", "''")}',
                        '{state["severity"]}',
                        '{state["service"].replace("'", "''")}',
                        '{db_description.replace("'", "''")}',
                        'Active',
                        '{current_time}',
                        '{current_time}',
                        {dependencies_str},
                        '{jira_url}'
                    ) RETURNING id;"""
                    
                    db_span.update(
                        input={
                            "query_type": "INSERT",
                            "table": "incidents",
                            "tool_name": query_tool.name
                        },
                        metadata={"database_operation": "insert_incident"}
                    )

                    db_result = await query_tool.ainvoke({"query": insert_query})
                    
                    db_span.update(
                        output={
                            "result_type": type(db_result).__name__,
                            "result_received": bool(db_result)
                        },
                        metadata={"status": "success"}
                    )

                logger.info(f"Raw database results: {db_result}")

                # Ensure it's a list of dicts
                if isinstance(db_result, str):
                    try:
                        db_result = json.loads(db_result)
                    except Exception as e:
                        logger.error(f"Failed to parse db_result: {e}")
                        db_result = []

                if db_result and isinstance(db_result, list) and len(db_result) > 0:
                    state["incident_id"] = str(db_result[0].get("id", ""))
                    logger.info(f"Stored incident with ID: {state['incident_id']}")
                else:
                    raise Exception("Failed to get incident ID from database")

                state["current_step"] = "incident_stored"
                
                span.update(
                    output={
                        "incident_id": state["incident_id"],
                        "incident_stored": True
                    },
                    metadata={"status": "success", "workflow_position": 3}
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error storing incident: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 3}
                )
                
        return state

    # @observe(name="correlation-agent-notification")
    async def _notify_correlation_agent_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """STEP 4: Notify correlation agent with comprehensive tracing."""
        with langfuse.start_as_current_span(name="notify-correlation-agent") as span:
            logger.info("STEP 4: Notifying correlation agent")
            
            try:
                span.update(
                    input={
                        "incident_id": state.get('incident_id'),
                        "jira_ticket_id": state.get('jira_ticket_id'),
                        "alert_payload_type": type(state.get('alert_payload'))
                    },
                    metadata={"step": "correlation_notification", "workflow_position": 4}
                )
                
                if not state.get("incident_id") or not state.get("jira_ticket_id"):
                    raise Exception("Missing incident_id or jira_ticket_id")

                # Import the correlation agent tool
                from tools.call_correlation_agent import send_alert_to_correlation_agent

                # Prepare alert context without metric data with safe access
                alert_payload = state.get("alert_payload", {})
                if isinstance(alert_payload, dict):
                    alert_payload_clean = alert_payload.copy()
                    if "metric_data" in alert_payload_clean:
                        del alert_payload_clean["metric_data"]
                else:
                    logger.warning(f"Alert payload is not dict in notify step: {type(alert_payload)}")
                    alert_payload_clean = {}

                # Call correlation agent with retry logic and tracing
                max_retries = 3
                for attempt in range(max_retries):
                    with langfuse.start_as_current_span(name=f"correlation-agent-call-attempt-{attempt + 1}") as attempt_span:
                        try:
                            attempt_span.update(
                                input={
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "incident_id": state["incident_id"],
                                    "service_name": state["service"]
                                },
                                metadata={"retry_attempt": attempt + 1}
                            )
                            
                            result = await send_alert_to_correlation_agent(
                                incident_id=state["incident_id"],
                                alert_context=f"Alert: {state['alertname']} for service {state['service']}",
                                jira_ticket_id=state["jira_ticket_id"],
                                alert_payload=json.dumps(alert_payload_clean),
                                service_name=state["service"],
                                severity=state["severity"],
                                alert_name=state["alertname"],
                                timestamp=state["timestamp"],
                                global_session_id=_global_session_id
                            )
                            
                            attempt_span.update(
                                output={"notification_result": str(result)[:200], "success": True},
                                metadata={"status": "success"}
                            )
                            
                            logger.info(f"Successfully notified correlation agent: {result}")
                            break
                            
                        except Exception as e:
                            attempt_span.update(
                                output={"error": str(e), "will_retry": attempt < max_retries - 1},
                                metadata={"status": "error"}
                            )
                            
                            if attempt < max_retries - 1:
                                logger.warning(f"Correlation agent notification failed (attempt {attempt + 1}): {e}")
                                await asyncio.sleep(5)
                            else:
                                raise e

                state["current_step"] = "correlation_agent_notified"
                
                span.update(
                    output={
                        "correlation_agent_notified": True,
                        "notification_successful": True
                    },
                    metadata={"status": "success", "workflow_position": 4}
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error notifying correlation agent: {error_msg}")
                state["error"] = error_msg
                
                span.update(
                    output={"error": error_msg},
                    metadata={"status": "error", "workflow_position": 4}
                )
                
        return state

    # @observe(name="workflow-completion")
    async def _complete_workflow_node(self, state: AnomalyAgentState) -> AnomalyAgentState:
        """Complete the workflow with tracing."""
        with langfuse.start_as_current_span(name="complete-workflow") as span:
            logger.info("Workflow completed successfully")
            
            state["completed"] = True
            state["current_step"] = "workflow_complete"
            
            span.update(
                output={
                    "workflow_completed": True,
                    "final_step": "workflow_complete"
                },
                metadata={"status": "success", "workflow_position": "final"}
            )
            
        return state

    def _infer_service_from_metric(self, payload: Dict[str, Any], fallback: str = "unknown") -> str:
        """Infer service/container name from metric-based payload.

        Priority:
        1) usage data[0].metric.container
        2) parse from metric name (suffix after last underscore)
        3) fallback provided
        """
        try:
            usage = payload.get("usage data")
            if isinstance(usage, list) and usage:
                first = usage[0]
                metric_obj = first.get("metric", {}) if isinstance(first, dict) else {}
                container = metric_obj.get("container")
                if isinstance(container, str) and container.strip():
                    return container.strip()

            metric_name = payload.get("metric")
            if isinstance(metric_name, str) and metric_name:
                parts = metric_name.split("_")
                if len(parts) >= 2 and parts[-1]:
                    return parts[-1]
        except Exception:
            pass
        return fallback

    # @observe(name="anomaly-agent-execution")
    async def process_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point to process an alert with comprehensive observability."""
        with langfuse.start_as_current_span(
            name="Anomaly-agent-execution",
            trace_context={"trace_id": langfuse.create_trace_id()},
            metadata={
                "agent_type": "anomaly_detection",
                "execution_start": True
            }
        ) as main_span:
            
            logger.info("Starting anomaly detection workflow")
            
            global _global_session_id
            _global_session_id = str(uuid.uuid4().hex)
            
            main_span.update_trace(session_id = _global_session_id)
            
            main_span.update(
                input={
                    "alert_payload_type": type(alert_payload).__name__,
                    "alert_payload_keys": list(alert_payload.keys()) if isinstance(alert_payload, dict) else [],
                    "agent_initialized": bool(self.graph)
                },
                metadata={
                    "agent_type": "anomaly_detection",
                    "workflow_type": "conditional_graph",
                    "execution_start": True
                }
            )
            
            try:
                # Setup MCP tools if not already done
                if self.mcp_client and not self.mcp_tools:
                    await self.setup_mcp_tools()

                # Create initial state
                initial_state = AnomalyAgentState(
                    alert_payload=alert_payload,
                    alert_status="",
                    alertname="",
                    severity="",
                    service="",
                    description="",
                    timestamp="",
                    service_dependencies=[],
                    jira_ticket_id=None,
                    incident_id=None,
                    database_result=None,
                    current_step="starting",
                    error=None,
                    completed=False
                )

                # Run the workflow with enhanced tracing
                with langfuse.start_as_current_span(name="workflow-graph-execution") as workflow_span:
                    workflow_span.update(
                        input={"initial_state_prepared": True},
                        metadata={"component": "langgraph_execution"}
                    )
                    
                    result = await self.graph.ainvoke(initial_state)
                    
                    workflow_span.update(
                        output={
                            "workflow_completed": result.get("completed", False),
                            "final_step": result.get("current_step"),
                            "has_error": bool(result.get("error"))
                        },
                        metadata={"status": "success" if not result.get("error") else "error"}
                    )

                # Prepare final result
                final_result = {
                    "success": result.get("completed", False),
                    "incident_id": result.get("incident_id"),
                    "jira_ticket_id": result.get("jira_ticket_id"),
                    "current_step": result.get("current_step"),
                    "error": result.get("error")
                }
                
                # Calculate execution metrics
                execution_success = result.get("completed", False) and not result.get("error")
                has_incident_id = bool(result.get("incident_id"))
                has_jira_ticket = bool(result.get("jira_ticket_id"))
                
                main_span.update(
                    output={
                        "execution_summary": final_result,
                        "metrics": {
                            "workflow_completed": execution_success,
                            "incident_created": has_incident_id,
                            "jira_ticket_created": has_jira_ticket,
                            "correlation_agent_notified": result.get("current_step") == "workflow_complete"
                        }
                    },
                    metadata={
                        "execution_completed": True,
                        "success": execution_success,
                        "agent_type": "anomaly_detection"
                    }
                )
                
                # Performance scoring
                efficiency_score = 1.0
                performance_factors = []
                
                if not execution_success:
                    efficiency_score -= 0.5
                    performance_factors.append("workflow_failed")
                if result.get("error"):
                    efficiency_score -= 0.3
                    performance_factors.append("has_errors")
                if not has_incident_id:
                    efficiency_score -= 0.2
                    performance_factors.append("no_incident_created")
                if not has_jira_ticket:
                    efficiency_score -= 0.2
                    performance_factors.append("no_jira_ticket")
                
                efficiency_score = max(0.0, efficiency_score)
                
                main_span.score(
                    name="anomaly-workflow-efficiency",
                    value=efficiency_score,
                    comment=f"Anomaly detection workflow completed. " +
                        f"Performance factors: {', '.join(performance_factors) if performance_factors else 'optimal'}"
                )
                
                return final_result
                
            except Exception as e:
                # Comprehensive error tracking
                error_context = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "workflow_context": {
                        "mcp_tools_setup": bool(self.mcp_tools),
                        "graph_initialized": bool(self.graph)
                    }
                }
                
                main_span.update(
                    output=error_context,
                    metadata={
                        "execution_failed": True,
                        "error_type": type(e).__name__,
                        "agent_type": "anomaly_detection"
                    }
                )
                
                main_span.score(
                    name="anomaly-workflow-error",
                    value=0.0,
                    comment=f"Anomaly workflow failed: {str(e)}"
                )
                
                logger.error(f"Anomaly detection workflow failed: {e}")
                raise
                
            finally:
                # Ensure all traces are flushed
                try:
                    langfuse.flush()
                    logger.info("Langfuse traces flushed successfully")
                except Exception as flush_error:
                    logger.error(f"Failed to flush Langfuse traces: {flush_error}")
