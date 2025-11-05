"""
Simple Remediation Agent using LangGraph
Receives RCA analysis, generates remediation recommendations, stores in Redis and PostgreSQL, creates Jira tickets
"""

import logging
import json
import os
import redis
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.llm_config import LLMConfig
from app.models import Alert, RemediationStructured, RemediationRecommendation
from app.mcp_config import get_mcp_config
from app.mcp_client import LangChainMCPClient
from app.langfuse_prompts import get_correlation_prompt

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
    structured_remediation_data: Optional[RemediationStructured] = None  # Store structured remediation data
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
                logger.info("✅ Redis client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Redis initialization failed: {e}")

        # Initialize Database
        self.db_session = None
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                self.db_session = SessionLocal
                logger.info("✅ Database initialized")
            except Exception as e:
                logger.warning(f"⚠️ Database initialization failed: {e}")

        # Initialize MCP client with custom wrapper
        self.mcp_client = None
        self.mcp_tools = []
        try:
            mcp_config = get_mcp_config()
            if mcp_config:
                # Use the custom LangChainMCPClient wrapper like RCA agent
                self.mcp_client = LangChainMCPClient("dummy_url")  # URL not used with config
                logger.info("✅ MCP client initialized for Remediation Agent")
            else:
                logger.warning("⚠️ No MCP configuration available")
        except Exception as e:
            logger.warning(f"⚠️ MCP client initialization failed: {e}")

        # Initialize LLM
        self.llm = LLMConfig.get_llm()

        # Build graph
        self.graph = self._build_graph()
        logger.info("✅ Simple Remediation Agent initialized")

    @observe(name="build-remediation-workflow-graph")
    def _build_graph(self):
        """Build simple remediation workflow"""
        with langfuse.start_as_current_span(name="remediation-workflow-graph-build") as span:
            workflow = StateGraph(RemediationState)

            workflow.add_node("analyze_remediation", self._remediation_analysis_node)
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

    async def _remediation_analysis_node(self, state: RemediationState) -> RemediationState:
        """STEP 1: Enhanced Remediation Agent Node - adapted from correlation agent"""
        with langfuse.start_as_current_span(name="remediation-analysis") as span:
            logger.info("🔧 STEP 1: Starting remediation analysis")
            
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

                # Enhanced prompt for remediation analysis
                remediation_prompt = f"""
                You are an expert SRE remediation specialist. Generate SHORT, focused remediation steps based on RCA findings.

                **INCIDENT DETAILS:**
                - Incident ID: {incident.get('incident_id', 'Unknown')}
                - Alert Name: {incident.get('alert_name', 'Unknown')}
                - Description: {incident.get('description', 'No description')}
                - Priority: {incident.get('priority', 'medium')}

                **RCA ANALYSIS:**
                {rca_analysis}

                **CORRELATION DATA:**
                {correlation_data if correlation_data else "No correlation data available"}

                **TASK:** Generate comprehensive remediation recommendations with:
                1. Stop Service Degradation - immediate actions to prevent further degradation
                2. Emergency Containment - containment measures to isolate the issue
                3. User Communication - what to communicate to users/stakeholders
                4. System Stabilization - steps to stabilize the affected systems
                5. Service Restart - procedures to restart affected services
                6. Monitoring - monitoring adjustments needed
                7. Code Snippet - technical fixes or configuration changes
                8. Success Criteria - how to verify the fix worked
                9. Rollback Plan - fallback procedures if remediation fails

                Be specific with commands, configuration changes, and technical details.
                Focus on actionable steps that can be implemented immediately.
                """

                messages = [
                    SystemMessage(content="You are an expert SRE remediation specialist. Generate SHORT, focused remediation steps based on RCA findings. Be specific with commands and technical details."),
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
                    remediation_steps = f"""
                    **STOP SERVICE DEGRADATION:**
                    • Incident {incident.get('incident_id', 'Unknown')} requires immediate attention
                    • Monitor system metrics for further degradation
                    • Implement circuit breaker if applicable

                    **EMERGENCY CONTAINMENT:**
                    • Isolate affected services/pods if possible
                    • Scale down problematic components temporarily
                    • Implement traffic throttling if needed

                    **USER COMMUNICATION:**
                    • Notify stakeholders of ongoing incident
                    • Provide ETA for resolution if available
                    • Update status page with current information

                    **SYSTEM STABILIZATION:**
                    • Check system resources (CPU, memory, disk)
                    • Verify database connections and health
                    • Review recent deployments for potential rollback

                    **SERVICE RESTART:**
                    • Restart affected services using standard procedures
                    • Verify service health after restart
                    • Check dependency services for issues

                    **MONITORING:**
                    • Increase monitoring frequency for affected systems
                    • Set up additional alerts for related metrics
                    • Monitor for cascade failures

                    **CODE SNIPPET:**
                    # Check service status
                    kubectl get pods -n production
                    # Restart service if needed
                    kubectl rollout restart deployment/service-name

                    **SUCCESS CRITERIA:**
                    • System metrics return to normal ranges
                    • Error rates drop below threshold
                    • User-facing functionality restored

                    **ROLLBACK PLAN:**
                    • Rollback to previous stable version if fixes fail
                    • Restore from backup if data corruption occurred
                    • Escalate to senior engineering team if needed
                    """

                # Create structured remediation output
                with langfuse.start_as_current_span(name="[llm-call]-structure-remediation") as struct_span:
                    try:
                        remediation_llm = self.llm.with_structured_output(RemediationStructured)

                        structured_remediation_prompt = f"""
                        You are a specialized Remediation Agent for SRE operations.

                        **INCIDENT DETAILS:**
                        - Incident ID: {incident.get('incident_id', 'Unknown')}
                        - Alert Name: {incident.get('alert_name', 'Unknown')}
                        - Description: {incident.get('description', 'No description')}
                        - Priority: {incident.get('priority', 'medium')}

                        **RCA ANALYSIS:**
                        {rca_analysis}

                        **REMEDIATION ANALYSIS:**
                        {remediation_steps}

                        **TASK:** Create structured RemediationStructured response based on the remediation analysis.

                        Extract and organize remediation steps into these categories:
                        - stop_service_degradation: Actions to prevent further service degradation
                        - emergency_containment: Emergency containment measures
                        - user_communication: User communication requirements
                        - system_stabilization: System stabilization steps
                        - service_restart: Service restart procedures
                        - monitoring: Monitoring adjustments needed
                        - code_snippet: Code examples for technical fixes
                        - success_criteria: Criteria to verify successful resolution
                        - rollback_plan: Rollback procedures if actions fail

                        **IMPORTANT:** Only use information from the remediation analysis provided above.
                        """

                        struct_span.update(
                            input={"remediation_analysis_length": len(remediation_steps)},
                            metadata={"operation": "structure_conversion"}
                        )

                        structured_remediation = await remediation_llm.ainvoke(
                            [
                                SystemMessage(content="You are an expert SRE remediation specialist. Create structured output from remediation text."),
                                HumanMessage(content=structured_remediation_prompt)
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

                    except Exception as structured_error:
                        logger.error(f"Structured remediation generation failed: {structured_error}")
                        logger.warning("Creating fallback structured remediation from text output")
                        # Create a fallback structured remediation
                        structured_remediation = RemediationStructured(
                            recommendation=RemediationRecommendation(
                                stop_service_degradation=[
                                    f"Investigate incident {incident.get('incident_id', 'Unknown')} immediately",
                                    "Monitor system metrics for degradation patterns"
                                ],
                                emergency_containment=[
                                    "Isolate affected systems from traffic if necessary",
                                    "Implement emergency circuit breakers"
                                ],
                                user_communication=[
                                    "Notify stakeholders of ongoing incident investigation",
                                    "Provide regular updates on remediation progress"
                                ],
                                system_stabilization=[
                                    "Check system health and resource utilization",
                                    "Verify all critical system dependencies"
                                ],
                                service_restart=[
                                    "Restart affected services using standard procedures",
                                    "Verify service health after restart"
                                ],
                                monitoring=[
                                    "Increase monitoring frequency for affected systems",
                                    "Set up additional alerts for key metrics"
                                ],
                                code_snippet="# Manual investigation required\nkubectl get pods -n production\nkubectl logs -f deployment/affected-service",
                                success_criteria=[
                                    "System metrics return to baseline",
                                    "Error rates drop below acceptable thresholds"
                                ],
                                rollback_plan=[
                                    "Rollback to previous stable deployment if needed",
                                    "Escalate to senior engineering team for complex issues"
                                ]
                            )
                        )
                        struct_span.update(
                            output={"structured_successfully": False, "fallback_created": True}
                        )
                        logger.info(f"Created fallback structured remediation: {structured_remediation.model_dump()}")

                state.remediation_analysis = remediation_steps
                state.structured_remediation_data = structured_remediation  # Store structured data in state
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

    async def _store_results_node(self, state: RemediationState) -> RemediationState:
        """STEP 2: Store remediation results in Redis, PostgreSQL, and Jira using MCP"""
        with langfuse.start_as_current_span(name="store-remediation-results") as span:
            logger.info("💾 STEP 2: Storing remediation results")
            
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
                remediation_analysis = state.remediation_analysis
                timestamp = datetime.now().isoformat()

                # Store using individual analysis pattern like correlation agent
                from sqlalchemy import text
                if self.db_session:
                    with langfuse.start_as_current_span(name="[database-store]-individual-analysis") as db_span:
                        max_retries = 3
                        retry_delay = 1
                        storage_success = False

                        for attempt in range(1, max_retries + 1):
                            session = self.db_session()
                            try:
                                # Store structured remediation result using individual field updates (same pattern as correlation agent)
                                structured_data = state.structured_remediation_data if state.structured_remediation_data else remediation_analysis
                                await self._store_individual_analysis(session, incident_id, "remediation", structured_data)
                                session.commit()

                                storage_success = True
                                logger.info(f"✅ PostgreSQL storage succeeded on attempt {attempt}")
                                db_span.update(
                                    output={"stored": True, "table": "individual_analysis", "attempts": attempt},
                                    metadata={"operation": "postgresql_store"}
                                )
                                logger.info("✅ Structured remediation analysis stored in PostgreSQL using individual analysis pattern")
                                break
                            except Exception as e:
                                logger.warning(f"PostgreSQL storage attempt {attempt} failed: {e}")
                                session.rollback()
                                if attempt < max_retries:
                                    logger.info(f"Retrying PostgreSQL storage in {retry_delay}s...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                else:
                                    logger.error(f"❌ All {max_retries} PostgreSQL storage attempts failed")
                                    db_span.update(
                                        output={"stored": False, "error": str(e), "attempts": attempt}
                                    )
                            finally:
                                session.close()

                # Store in Redis if available (fallback)
                if self.redis_client:
                    with langfuse.start_as_current_span(name="[redis-store]-remediation-results") as redis_span:
                        try:
                            redis_key = f"incidents:{incident_id}:remediation"
                            redis_data = {
                                "incident_id": incident_id,
                                "remediation_analysis": remediation_analysis,
                                "timestamp": timestamp,
                                "status": "completed" if not state.error else "failed",
                                "has_rca_data": bool(state.rca_analysis),
                                "has_correlation_data": bool(state.correlation_data)
                            }
                            self.redis_client.setex(redis_key, 3600, json.dumps(redis_data))  # 1 hour TTL
                            
                            redis_span.update(
                                input={"redis_key": redis_key},
                                output={"stored": True, "ttl_hours": 1},
                                metadata={"operation": "redis_store"}
                            )
                            
                            logger.info(f"✅ Stored remediation analysis in Redis: {redis_key}")
                        except Exception as e:
                            logger.warning(f"⚠️ Redis storage failed: {e}")
                            redis_span.update(
                                output={"stored": False, "error": str(e)}
                            )

                # Add Jira comment for remediation analysis (same pattern as correlation agent)
                if state.remediation_analysis and not state.error:
                    with langfuse.start_as_current_span(name="[tool-called]-jira-remediation-comment") as jira_span:
                        try:
                            await self.add_jira_comment_for_remediation(state.incident, state.remediation_analysis)
                            jira_span.update(
                                output={"comment_added": True},
                                metadata={"operation": "jira_comment"}
                            )
                        except Exception as e:
                            logger.warning(f"JIRA comment failed: {e}")
                            jira_span.update(
                                output={"comment_added": False, "error": str(e)}
                            )

                span.update(
                    output={
                        "postgresql_stored": self.db_session is not None,
                        "redis_stored": self.redis_client is not None,
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

    async def analyze_remediation(self, incident: Dict[str, Any], rca_analysis: str = "", correlation_data: str = "",langfuse_trace_context=None) -> str:
        """Main entry point for remediation analysis"""
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
                        await self.setup_mcp_tools()
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
                
                performance_factors = []
                if execution_time < 45:
                    performance_factors.append("fast_execution")
                if len(remediation_analysis) > 100:
                    performance_factors.append("comprehensive_remediation")
                if not (result.error if hasattr(result, 'error') else result.get("error")):
                    performance_factors.append("error_free")
                if bool(rca_analysis):
                    performance_factors.append("rca_guided")

                # Calculate efficiency score
                efficiency_score = min(1.0, len(performance_factors) * 0.25)
                
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

    async def fetch_complete_incident_data_from_redis(self, incident_key: str, langfuse_trace_context=None) -> Dict[str, Any]:
        """Fetch complete incident data from Redis using incident_key (e.g., incidents:197:main)"""
        with langfuse.start_as_current_span(
            name="fetch-complete-incident-data-remediation",
            trace_context=langfuse_trace_context            
        ) as span:
            span.update(
                input={"incident_key": incident_key},
                metadata={"operation": "redis_fetch_complete", "component": "data_retrieval"}
            )
            span.update_trace(session_id=langfuse_trace_context.get("session_id"))
            
            if not self.redis_client:
                logger.warning("Redis client not available")
                span.update(
                    output={"success": False, "reason": "no_redis_client"},
                    metadata={"status": "unavailable"}
                )
                return {}

            try:
                with langfuse.start_as_current_span(name="[redis-operation]-fetch-complete-data") as redis_span:
                    # Handle both old format (incident:123) and new folder format (incidents:123:main)
                    incident_data = None
                    key_attempts = []
                    
                    if incident_key.startswith("incidents:") and ":main" in incident_key:
                        # New folder format - direct lookup
                        key_attempts.append(incident_key)
                        incident_data = self.redis_client.get(incident_key)
                    elif incident_key.startswith("incident:"):
                        # Old format, convert to new format
                        incident_id = incident_key.split(":")[-1]
                        new_key = f"incidents:{incident_id}:main"
                        key_attempts.extend([new_key, incident_key])
                        incident_data = self.redis_client.get(new_key)
                        if not incident_data:
                            # Fallback to old format
                            incident_data = self.redis_client.get(incident_key)
                    else:
                        # Direct key lookup
                        key_attempts.append(incident_key)
                        incident_data = self.redis_client.get(incident_key)

                    redis_span.update(
                        input={"original_key": incident_key, "key_attempts": key_attempts},
                        output={"data_found": bool(incident_data)},
                        metadata={"operation": "redis_get_multiple_attempts"}
                    )

                if incident_data:
                    with langfuse.start_as_current_span(name="[data-processing]-parse-incident-json") as parse_span:
                        parsed_data = json.loads(incident_data)
                        
                        parse_span.update(
                            input={"data_size_bytes": len(incident_data)},
                            output={"parsed_fields": list(parsed_data.keys()), "field_count": len(parsed_data.keys())},
                            metadata={"operation": "json_parse"}
                        )
                        
                        logger.info(f"✅ Retrieved complete incident data from Redis for key: {incident_key}")
                        logger.info(f"📋 Available fields: {list(parsed_data.keys())}")

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
                    logger.warning(f"⚠️ No incident data found in Redis for key: {incident_key}")
                    span.update(
                        output={"success": False, "data_retrieved": False, "reason": "not_found"},
                        metadata={"status": "not_found"}
                    )
                    return {}

            except Exception as e:
                logger.error(f"❌ Error fetching incident data from Redis: {e}")
                span.update(
                    output={"success": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )
                return {}

    async def fetch_data_from_redis(self, incident_id: str) -> tuple[str, str]:
        """Fetch RCA analysis and correlation data from Redis"""
        with langfuse.start_as_current_span(name="fetch-rca-correlation-data") as span:
            span.update(
                input={"incident_id": incident_id},
                metadata={"operation": "redis_fetch_rca_correlation", "component": "data_retrieval"}
            )
            
            rca_analysis = ""
            correlation_data = ""

            if self.redis_client:
                try:
                    with langfuse.start_as_current_span(name="[redis-fetch]-rca-analysis") as rca_span:
                        # Fetch RCA analysis
                        rca_key = f"rca_analysis:{incident_id}"
                        rca_raw = self.redis_client.get(rca_key)
                        
                        rca_span.update(
                            input={"redis_key": rca_key},
                            metadata={"operation": "redis_get_rca"}
                        )
                        
                        if rca_raw:
                            rca_data = json.loads(rca_raw.decode('utf-8'))
                            rca_analysis = rca_data.get('rca_analysis', '')
                            logger.info(f"✅ Retrieved RCA analysis from Redis for incident {incident_id}")
                            rca_span.update(output={"rca_found": True, "rca_length": len(rca_analysis)})
                        else:
                            rca_span.update(output={"rca_found": False})

                    with langfuse.start_as_current_span(name="[redis-fetch]-correlation-data") as corr_span:
                        # Fetch correlation data
                        correlation_key = f"correlation_data:{incident_id}"
                        correlation_raw = self.redis_client.get(correlation_key)
                        
                        corr_span.update(
                            input={"redis_key": correlation_key},
                            metadata={"operation": "redis_get_correlation"}
                        )
                        
                        if correlation_raw:
                            correlation_data = correlation_raw.decode('utf-8')
                            logger.info(f"✅ Retrieved correlation data from Redis for incident {incident_id}")
                            corr_span.update(output={"correlation_found": True, "correlation_length": len(correlation_data)})
                        else:
                            corr_span.update(output={"correlation_found": False})

                    span.update(
                        output={
                            "rca_retrieved": bool(rca_analysis),
                            "correlation_retrieved": bool(correlation_data),
                            "rca_length": len(rca_analysis),
                            "correlation_length": len(correlation_data)
                        },
                        metadata={"status": "success"}
                    )

                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch data from Redis: {e}")
                    span.update(
                        output={"success": False, "error": str(e)},
                        metadata={"status": "error", "error_type": type(e).__name__}
                    )

            return rca_analysis, correlation_data

    async def setup_mcp_tools(self):
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

    async def add_jira_comment_for_remediation(self, incident_data: Dict[str, Any], remediation_analysis: str):
        """Add a Jira comment for Remediation analysis using the same pattern as correlation agent"""
        with langfuse.start_as_current_span(name="add-jira-remediation-comment") as span:
            span.update(
                input={
                    "has_jira_ticket_id": bool(incident_data.get("jira_ticket_id")),
                    "remediation_analysis_length": len(remediation_analysis)
                },
                metadata={"operation": "jira_comment", "component": "jira_integration"}
            )
            
            try:
                # Get the Jira ticket ID from incident data
                jira_ticket_id = incident_data.get("jira_ticket_id")

                if not jira_ticket_id:
                    logger.warning(f"⚠️ No Jira ticket ID found for Remediation analysis comment")
                    span.update(
                        output={"comment_added": False, "reason": "no_ticket_id"},
                        metadata={"status": "skipped"}
                    )
                    return

                # Check if MCP client and Jira tools are available
                if not self.mcp_client:
                    logger.warning("⚠️ No MCP client available, skipping Jira comment")
                    span.update(
                        output={"comment_added": False, "reason": "no_mcp_client"},
                        metadata={"status": "skipped"}
                    )
                    return

                # Setup MCP tools if not already done
                if not self.mcp_tools:
                    with langfuse.start_as_current_span(name="[setup]-mcp-tools-for-jira") as setup_span:
                        await self.setup_mcp_tools()
                        setup_span.update(
                            output={"tools_setup_completed": True, "tools_count": len(self.mcp_tools)}
                        )

                with langfuse.start_as_current_span(name="[validation]-jira-tools-availability") as validation_span:
                    available_tools = self.mcp_tools if self.mcp_tools else []
                    tool_names = [tool.name for tool in available_tools] if available_tools else []

                    has_jira_tool = any('jira_add_comment' in tool_name for tool_name in tool_names)
                    
                    validation_span.update(
                        input={"available_tools": tool_names},
                        output={"jira_tool_available": has_jira_tool},
                        metadata={"operation": "tool_validation"}
                    )
                    
                    if not has_jira_tool:
                        logger.warning("⚠️ jira_add_comment tool not available, skipping Jira comment")
                        logger.info(f"Available tools: {tool_names}")
                        span.update(
                            output={"comment_added": False, "reason": "jira_tool_unavailable"},
                            metadata={"status": "skipped"}
                        )
                        return

                # Use the same variable pattern as correlation agent
                jira_variables = {
                    "analysis_type": "remediation",
                    "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown Alert")),
                    "severity": incident_data.get("severity", "Unknown"),
                    "analysis_content": remediation_analysis,
                    "title": "Remediation"
                }

                with langfuse.start_as_current_span(name="[llm-call]-jira-comment-generation") as llm_span:
                    # Get JIRA formatter prompt from Langfuse (same as correlation agent)
                    jira_formatter_prompt = get_correlation_prompt("jira-formatter", jira_variables)
                    logger.info(f"Retrieved JIRA formatter prompt from Langfuse for Remediation analysis")

                    # Use the same user prompt pattern as correlation agent
                    user_prompt = f"**Task:** Create focused markdown comment showing ONLY the remediation analysis results based on the provided content and context."

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
                            "langfuse_trace_id": self.predefined_trace_id,
                            "langfuse_tags": ["remediation-agent"],
                            "component": "jira_comment_generation"
                        }
                    })
                    
                    llm_span.update(
                        output={"comment_generated": True, "comment_length": len(response.content)},
                        metadata={"status": "success"}
                    )

                with langfuse.start_as_current_span(name="[processing]-comment-formatting") as format_span:
                    # Add header and footer (same pattern as correlation agent)
                    markdown_comment = f"""{response.content}

---
*Remediation analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"""

                    # Sanitize Unicode characters
                    sanitized_comment = self._sanitize_unicode(markdown_comment)
                    
                    format_span.update(
                        input={"raw_comment_length": len(markdown_comment)},
                        output={"sanitized_comment_length": len(sanitized_comment)},
                        metadata={"operation": "comment_sanitization"}
                    )

                # Add comment to Jira ticket using correct tool name
                jira_params = {
                    "issue_key": jira_ticket_id,
                    "comment": sanitized_comment
                }

                logger.info(f"🎫 Adding Remediation comment to Jira ticket: {jira_ticket_id}")

                # Add retry logic for Jira comment failures (like correlation agent)
                max_retries = 3
                retry_count = 0
                success = False

                with langfuse.start_as_current_span(name="[tool-called]-jira-add-comment-with-retries") as retry_span:
                    while retry_count < max_retries and not success:
                        try:
                            with langfuse.start_as_current_span(name=f"[retry-attempt]-{retry_count + 1}") as attempt_span:
                                await self.mcp_client.call_tool_direct("jira_add_comment", jira_params)
                                success = True
                                
                                attempt_span.update(
                                    input={"attempt_number": retry_count + 1, "jira_params": jira_params},
                                    output={"success": True},
                                    metadata={"operation": "jira_tool_call"}
                                )
                                
                                logger.info(f"✅ Successfully added Remediation analysis comment to Jira ticket (attempt {retry_count + 1})")
                                
                        except Exception as retry_error:
                            retry_count += 1
                            logger.warning(f"⚠️ Attempt {retry_count} failed for Remediation Jira comment: {retry_error}")
                            
                            attempt_span.update(
                                input={"attempt_number": retry_count},
                                output={"success": False, "error": str(retry_error)},
                                metadata={"status": "failed"}
                            )
                            
                            if retry_count < max_retries:
                                import asyncio
                                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                            else:
                                logger.error(f"❌ All {max_retries} attempts failed for Remediation Jira comment")
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
                logger.error(f"❌ Failed to add Remediation Jira comment: {e}")
                # Don't set error state as this shouldn't stop the workflow
                span.update(
                    output={"comment_added": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize Unicode characters for Jira compatibility"""
        try:
            # Replace problematic Unicode characters
            sanitized = text.encode('ascii', 'ignore').decode('ascii')
            # Replace common Unicode punctuation
            replacements = {
                '"': '"', '"': '"', ''': "'", ''': "'",
                '–': '-', '—': '-', '…': '...'
            }
            for old, new in replacements.items():
                sanitized = sanitized.replace(old, new)
            return sanitized
        except Exception:
            return text

    async def _store_individual_analysis(self, session, incident_id: int, analysis_type: str, structured_data: Any):
        """Store individual analysis results in database - same pattern as correlation agent"""
        with langfuse.start_as_current_span(name="store-individual-analysis-remediation") as span:
            span.update(
                input={
                    "incident_id": incident_id,
                    "analysis_type": analysis_type,
                    "has_structured_data": bool(structured_data)
                },
                metadata={"operation": "database_store", "component": "database"}
            )
            
            try:
                from sqlalchemy import text

                if analysis_type == "remediation":
                    with langfuse.start_as_current_span(name="[database-operation]-update-remediation-result") as db_span:
                        # Update remediation_result field
                        if structured_data:
                            data_json = json.dumps(structured_data, indent=2) if isinstance(structured_data, dict) else json.dumps(structured_data.model_dump()) if hasattr(structured_data, 'model_dump') else json.dumps(str(structured_data))
                            
                            db_span.update(
                                input={"data_json_length": len(data_json), "data_type": type(structured_data).__name__},
                                metadata={"operation": "remediation_result_update"}
                            )
                            
                            session.execute(
                                text("UPDATE incidents SET remediation_result = :data WHERE id = :id"),
                                {"data": data_json, "id": incident_id}
                            )
                            
                            db_span.update(
                                output={"update_executed": True, "field_updated": "remediation_result"},
                                metadata={"status": "success"}
                            )
                            
                            logger.info(f"Successfully updated remediation_result field in database for incident {incident_id}")
                else:
                    logger.warning(f"Unknown analysis type: {analysis_type}")
                    span.update(
                        output={"stored": False, "reason": "unknown_analysis_type"},
                        metadata={"status": "error"}
                    )
                    return

                logger.info(f"Successfully stored {analysis_type} analysis in database for incident {incident_id}")

            except Exception as e:
                logger.error(f"Failed to store {analysis_type} analysis in database: {e}")
                raise e