"""
Utility functions for Remediation Agent
Contains helper functions for data processing, storage, and external integrations
"""

import logging
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from langfuse import get_client
from app.models import RemediationStructured, RemediationRecommendation
from app.prompts import format_jira_comment_footer

logger = logging.getLogger(__name__)
langfuse = get_client()


def sanitize_unicode(text: str) -> str:
    """
    Sanitize Unicode characters for Jira compatibility.

    Args:
        text: Input text with potential unicode characters

    Returns:
        Sanitized text safe for Jira
    """
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


def create_fallback_structured_remediation(incident: Dict[str, Any]) -> RemediationStructured:
    """
    Create fallback structured remediation when LLM structuring fails.

    Args:
        incident: Incident details dictionary

    Returns:
        RemediationStructured object with fallback data
    """
    return RemediationStructured(
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


async def fetch_rca_correlation_from_redis(
    redis_client,
    incident_id: str
) -> Tuple[str, str]:
    """
    Fetch RCA analysis and correlation data from Redis.

    Args:
        redis_client: Redis client instance
        incident_id: Incident identifier

    Returns:
        Tuple of (rca_analysis, correlation_data)
    """
    with langfuse.start_as_current_span(name="fetch-rca-correlation-data") as span:
        span.update(
            input={"incident_id": incident_id},
            metadata={"operation": "redis_fetch_rca_correlation", "component": "data_retrieval"}
        )

        rca_analysis = ""
        correlation_data = ""

        if redis_client:
            try:
                with langfuse.start_as_current_span(name="[redis-fetch]-rca-analysis") as rca_span:
                    # Fetch RCA analysis
                    rca_key = f"rca_analysis:{incident_id}"
                    rca_raw = redis_client.get(rca_key)

                    rca_span.update(
                        input={"redis_key": rca_key},
                        metadata={"operation": "redis_get_rca"}
                    )

                    if rca_raw:
                        rca_data = json.loads(rca_raw.decode('utf-8'))
                        rca_analysis = rca_data.get('rca_analysis', '')
                        logger.info(f"Retrieved RCA analysis from Redis for incident {incident_id}")
                        rca_span.update(output={"rca_found": True, "rca_length": len(rca_analysis)})
                    else:
                        rca_span.update(output={"rca_found": False})

                with langfuse.start_as_current_span(name="[redis-fetch]-correlation-data") as corr_span:
                    # Fetch correlation data
                    correlation_key = f"correlation_data:{incident_id}"
                    correlation_raw = redis_client.get(correlation_key)

                    corr_span.update(
                        input={"redis_key": correlation_key},
                        metadata={"operation": "redis_get_correlation"}
                    )

                    if correlation_raw:
                        correlation_data = correlation_raw.decode('utf-8')
                        logger.info(f"Retrieved correlation data from Redis for incident {incident_id}")
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
                logger.warning(f"Failed to fetch data from Redis: {e}")
                span.update(
                    output={"success": False, "error": str(e)},
                    metadata={"status": "error", "error_type": type(e).__name__}
                )

        return rca_analysis, correlation_data


async def fetch_complete_incident_from_redis(
    redis_client,
    incident_key: str,
    langfuse_trace_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Fetch complete incident data from Redis using incident_key.

    Args:
        redis_client: Redis client instance
        incident_key: Redis key for incident (e.g., incidents:197:main)
        langfuse_trace_context: Optional trace context for Langfuse

    Returns:
        Dictionary containing complete incident data
    """
    with langfuse.start_as_current_span(
        name="fetch-complete-incident-data-remediation",
        trace_context=langfuse_trace_context
    ) as span:
        span.update(
            input={"incident_key": incident_key},
            metadata={"operation": "redis_fetch_complete", "component": "data_retrieval"}
        )
        if langfuse_trace_context:
            span.update_trace(session_id=langfuse_trace_context.get("session_id"))

        if not redis_client:
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
                    incident_data = redis_client.get(incident_key)
                elif incident_key.startswith("incident:"):
                    # Old format, convert to new format
                    incident_id = incident_key.split(":")[-1]
                    new_key = f"incidents:{incident_id}:main"
                    key_attempts.extend([new_key, incident_key])
                    incident_data = redis_client.get(new_key)
                    if not incident_data:
                        # Fallback to old format
                        incident_data = redis_client.get(incident_key)
                else:
                    # Direct key lookup
                    key_attempts.append(incident_key)
                    incident_data = redis_client.get(incident_key)

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


async def store_remediation_in_redis(
    redis_client,
    incident_id: str,
    remediation_analysis: str,
    has_rca_data: bool,
    has_correlation_data: bool,
    has_error: bool
) -> bool:
    """
    Store remediation results in Redis with TTL.

    Args:
        redis_client: Redis client instance
        incident_id: Incident identifier
        remediation_analysis: Remediation analysis text
        has_rca_data: Whether RCA data was available
        has_correlation_data: Whether correlation data was available
        has_error: Whether an error occurred

    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False

    with langfuse.start_as_current_span(name="[redis-store]-remediation-results") as redis_span:
        try:
            redis_key = f"incidents:{incident_id}:remediation"
            redis_data = {
                "incident_id": incident_id,
                "remediation_analysis": remediation_analysis,
                "timestamp": datetime.now().isoformat(),
                "status": "completed" if not has_error else "failed",
                "has_rca_data": has_rca_data,
                "has_correlation_data": has_correlation_data
            }
            redis_client.setex(redis_key, 3600, json.dumps(redis_data))  # 1 hour TTL

            redis_span.update(
                input={"redis_key": redis_key},
                output={"stored": True, "ttl_hours": 1},
                metadata={"operation": "redis_store"}
            )

            logger.info(f"Stored remediation analysis in Redis: {redis_key}")
            return True
        except Exception as e:
            logger.warning(f"Redis storage failed: {e}")
            redis_span.update(
                output={"stored": False, "error": str(e)}
            )
            return False


async def store_remediation_in_database(
    mcp_client,
    mcp_tools: List,
    incident_id: str,
    structured_data: Any,
    max_retries: int = 3
) -> bool:
    """
    Store remediation analysis in database using MCP execute_query tool.

    Args:
        mcp_client: MCP client instance
        mcp_tools: List of available MCP tools
        incident_id: Incident identifier
        structured_data: Structured remediation data
        max_retries: Maximum retry attempts

    Returns:
        True if successful, False otherwise
    """
    with langfuse.start_as_current_span(name="[database-store]-remediation-analysis") as db_span:
        # Find execute_query tool
        query_tool = None
        for tool in mcp_tools:
            if 'execute_query' in tool.name.lower() or 'postgres' in tool.name.lower():
                query_tool = tool
                break

        if not query_tool:
            logger.warning("No database query tool available")
            db_span.update(
                output={"stored": False, "reason": "no_query_tool"},
                metadata={"status": "unavailable"}
            )
            return False

        try:
            # Prepare data for storage
            data_json = json.dumps(structured_data, indent=2) if isinstance(structured_data, dict) \
                else json.dumps(structured_data.model_dump()) if hasattr(structured_data, 'model_dump') \
                else json.dumps(str(structured_data))

            # Escape single quotes for SQL
            escaped_data = data_json.replace("'", "''")

            # Build UPDATE query
            update_query = f"""
            UPDATE incidents
            SET remediation_result = '{escaped_data}'
            WHERE id = {incident_id}
            RETURNING id;
            """

            db_span.update(
                input={
                    "query_type": "UPDATE",
                    "table": "incidents",
                    "incident_id": incident_id,
                    "tool_name": query_tool.name
                },
                metadata={"database_operation": "update_remediation"}
            )

            # Execute with retries
            retry_delay = 1
            for attempt in range(1, max_retries + 1):
                try:
                    db_result = await query_tool.ainvoke({"query": update_query})

                    db_span.update(
                        output={
                            "result_type": type(db_result).__name__,
                            "result_received": bool(db_result),
                            "attempts": attempt
                        },
                        metadata={"status": "success"}
                    )

                    logger.info(f"Successfully stored remediation in database on attempt {attempt}")
                    logger.info(f"Raw database results: {db_result}")
                    return True

                except Exception as e:
                    logger.warning(f"Database storage attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        logger.info(f"Retrying database storage in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} database storage attempts failed")
                        db_span.update(
                            output={"stored": False, "error": str(e), "attempts": attempt},
                            metadata={"status": "error"}
                        )
                        return False

        except Exception as e:
            logger.error(f"Failed to prepare database query: {e}")
            db_span.update(
                output={"stored": False, "error": str(e)},
                metadata={"status": "error", "error_type": type(e).__name__}
            )
            return False


async def add_jira_comment(
    mcp_client,
    mcp_tools: List,
    llm,
    langfuse_handler,
    predefined_trace_id: str,
    incident_data: Dict[str, Any],
    remediation_analysis: str
) -> bool:
    """
    Add Jira comment for remediation analysis.

    Args:
        mcp_client: MCP client instance
        mcp_tools: List of available MCP tools
        llm: Language model instance
        langfuse_handler: Langfuse callback handler
        predefined_trace_id: Trace ID for Langfuse
        incident_data: Incident details dictionary
        remediation_analysis: Remediation analysis text

    Returns:
        True if successful, False otherwise
    """
    from app.langfuse_prompts import get_correlation_prompt

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
                logger.warning("No Jira ticket ID found for Remediation analysis comment")
                span.update(
                    output={"comment_added": False, "reason": "no_ticket_id"},
                    metadata={"status": "skipped"}
                )
                return False

            # Check if MCP client is available
            if not mcp_client:
                logger.warning("No MCP client available, skipping Jira comment")
                span.update(
                    output={"comment_added": False, "reason": "no_mcp_client"},
                    metadata={"status": "skipped"}
                )
                return False

            with langfuse.start_as_current_span(name="[validation]-jira-tools-availability") as validation_span:
                available_tools = mcp_tools if mcp_tools else []
                tool_names = [tool.name for tool in available_tools] if available_tools else []

                has_jira_tool = any('jira_add_comment' in tool_name for tool_name in tool_names)

                validation_span.update(
                    input={"available_tools": tool_names},
                    output={"jira_tool_available": has_jira_tool},
                    metadata={"operation": "tool_validation"}
                )

                if not has_jira_tool:
                    logger.warning("jira_add_comment tool not available, skipping Jira comment")
                    logger.info(f"Available tools: {tool_names}")
                    span.update(
                        output={"comment_added": False, "reason": "jira_tool_unavailable"},
                        metadata={"status": "skipped"}
                    )
                    return False

            # Prepare Jira comment variables
            jira_variables = {
                "analysis_type": "remediation",
                "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown Alert")),
                "severity": incident_data.get("severity", "Unknown"),
                "analysis_content": remediation_analysis,
                "title": "Remediation"
            }

            with langfuse.start_as_current_span(name="[llm-call]-jira-comment-generation") as llm_span:
                # Get JIRA formatter prompt from Langfuse
                jira_formatter_prompt = get_correlation_prompt("jira-formatter", jira_variables)
                logger.info("Retrieved JIRA formatter prompt from Langfuse for Remediation analysis")

                user_prompt = "**Task:** Create focused markdown comment showing ONLY the remediation analysis results based on the provided content and context."

                llm_span.update(
                    input={
                        "prompt_variables": jira_variables,
                        "user_prompt_length": len(user_prompt)
                    },
                    metadata={"component": "jira_comment_generation"}
                )

                # Generate comment using LLM
                response = await llm.ainvoke([
                    {"role": "system", "content": jira_formatter_prompt},
                    {"role": "user", "content": user_prompt}
                ], config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_trace_id": predefined_trace_id,
                        "langfuse_tags": ["remediation-agent"],
                        "component": "jira_comment_generation"
                    }
                })

                llm_span.update(
                    output={"comment_generated": True, "comment_length": len(response.content)},
                    metadata={"status": "success"}
                )

            with langfuse.start_as_current_span(name="[processing]-comment-formatting") as format_span:
                # Add header and footer
                markdown_comment = f"{response.content}{format_jira_comment_footer()}"

                # Sanitize Unicode characters
                sanitized_comment = sanitize_unicode(markdown_comment)

                format_span.update(
                    input={"raw_comment_length": len(markdown_comment)},
                    output={"sanitized_comment_length": len(sanitized_comment)},
                    metadata={"operation": "comment_sanitization"}
                )

            # Add comment to Jira ticket
            jira_params = {
                "issue_key": jira_ticket_id,
                "comment": sanitized_comment
            }

            logger.info(f"Adding Remediation comment to Jira ticket: {jira_ticket_id}")

            # Add retry logic for Jira comment failures
            max_retries = 3
            retry_count = 0
            success = False

            with langfuse.start_as_current_span(name="[tool-called]-jira-add-comment-with-retries") as retry_span:
                while retry_count < max_retries and not success:
                    try:
                        with langfuse.start_as_current_span(name=f"[retry-attempt]-{retry_count + 1}") as attempt_span:
                            await mcp_client.call_tool_direct("jira_add_comment", jira_params)
                            success = True

                            attempt_span.update(
                                input={"attempt_number": retry_count + 1, "jira_params": jira_params},
                                output={"success": True},
                                metadata={"operation": "jira_tool_call"}
                            )

                            logger.info(f"Successfully added Remediation analysis comment to Jira ticket (attempt {retry_count + 1})")

                    except Exception as retry_error:
                        retry_count += 1
                        logger.warning(f"Attempt {retry_count} failed for Remediation Jira comment: {retry_error}")

                        attempt_span.update(
                            input={"attempt_number": retry_count},
                            output={"success": False, "error": str(retry_error)},
                            metadata={"status": "failed"}
                        )

                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        else:
                            logger.error(f"All {max_retries} attempts failed for Remediation Jira comment")

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

            return success

        except Exception as e:
            logger.error(f"Failed to add Remediation Jira comment: {e}")
            span.update(
                output={"comment_added": False, "error": str(e)},
                metadata={"status": "error", "error_type": type(e).__name__}
            )
            return False


def calculate_performance_metrics(
    execution_time: float,
    remediation_analysis: str,
    has_error: bool,
    has_rca_analysis: bool
) -> Tuple[List[str], float]:
    """
    Calculate performance metrics for remediation analysis.

    Args:
        execution_time: Time taken for execution in seconds
        remediation_analysis: Generated remediation analysis text
        has_error: Whether an error occurred
        has_rca_analysis: Whether RCA analysis was available

    Returns:
        Tuple of (performance_factors list, efficiency_score)
    """
    performance_factors = []
    if execution_time < 45:
        performance_factors.append("fast_execution")
    if len(remediation_analysis) > 100:
        performance_factors.append("comprehensive_remediation")
    if not has_error:
        performance_factors.append("error_free")
    if has_rca_analysis:
        performance_factors.append("rca_guided")

    # Calculate efficiency score
    efficiency_score = min(1.0, len(performance_factors) * 0.25)

    return performance_factors, efficiency_score
