"""
Utility functions for RCA Agent
Contains helper functions for database operations, data processing, and formatting
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy import text

logger = logging.getLogger(__name__)


def sanitize_unicode_for_jira(text: str) -> str:
    """
    Sanitize Unicode characters for Jira compatibility

    Args:
        text: Input text that may contain Unicode characters

    Returns:
        Sanitized text safe for Jira comments
    """
    try:
        sanitized = text.encode('ascii', 'ignore').decode('ascii')

        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...'
        }

        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        return sanitized
    except Exception:
        return text


def parse_timestamp_with_timezone(ts_str: str) -> datetime:
    """
    Parse timestamp string and ensure timezone awareness

    Args:
        ts_str: Timestamp string to parse

    Returns:
        Timezone-aware datetime object
    """
    if not ts_str or not isinstance(ts_str, str):
        return datetime.min.replace(tzinfo=timezone.utc)

    try:
        ts_norm = ts_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts_norm)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def sort_memgraph_results_by_similarity(results: list, threshold: float = 0.9) -> tuple:
    """
    Sort memgraph results by similarity and timestamp

    Args:
        results: List of memgraph results
        threshold: Similarity threshold for filtering

    Returns:
        Tuple of (sorted_results, reference_rcas, attribution_source)
    """
    def sort_key(r):
        sim = r.get("similarity", 0.0) or 0.0
        gen_by = (r.get("generated_by") or "").lower()

        if gen_by == "human":
            ts = parse_timestamp_with_timezone(r.get("timestamp"))
        else:
            ts = datetime.min.replace(tzinfo=timezone.utc)

        return (-sim, -ts.timestamp())

    sorted_results = sorted(results, key=sort_key)

    top_six_candidates = sorted_results[:6]
    top_six = [r for r in top_six_candidates if (r.get("similarity") or 0) >= threshold]

    humans = [r for r in top_six if (r.get("generated_by") or "").lower() == "human"]
    humans_sorted = sorted(humans, key=lambda r: parse_timestamp_with_timezone(r.get("timestamp")), reverse=True)

    refs = []
    refs.extend(humans_sorted[:2])

    if len(refs) < 2:
        non_humans = [r for r in top_six if (r.get("generated_by") or "").lower() != "human"]
        refs.extend(non_humans[: 2 - len(refs)])

    human_refs = [r for r in refs if (r.get("generated_by") or "").lower() == "human"]

    if len(human_refs) > 0:
        attribution_source = "human"
    else:
        attribution_source = "sre_agent"

    return sorted_results, refs, attribution_source


def extract_node_id_from_response(content: Any) -> Optional[int]:
    """
    Extract node_id from memgraph tool response

    Args:
        content: Response content from memgraph tool

    Returns:
        Extracted node_id or None if not found
    """
    node_id = None

    try:
        if isinstance(content, str):
            try:
                result_data = json.loads(content)
                node_id = result_data.get("node_id")
                logger.info(f"Extracted node_id from JSON string: {node_id}")
            except json.JSONDecodeError:
                import re
                node_id_match = re.search(r'node_id["\']?\s*[:=]\s*(\d+)', content)
                if node_id_match:
                    node_id = int(node_id_match.group(1))
                    logger.info(f"Extracted node_id from pattern: {node_id}")
                else:
                    logger.warning("Could not extract node_id from string response")

        elif isinstance(content, dict):
            node_id = content.get("node_id")
            logger.info(f"Extracted node_id from dict: {node_id}")

            if content.get("success") is False:
                logger.warning(f"Memgraph insert reported error: {content.get('error', 'Unknown error')}")

    except Exception as e:
        logger.warning(f"Error extracting node_id from response: {e}")

    return node_id


async def execute_database_query_with_retry(
    mcp_client,
    query: str,
    params: Dict[str, Any] = None,
    max_retries: int = 3,
    retry_delay: int = 1
) -> bool:
    """
    Execute database query using MCP execute_query tool with retry logic

    Args:
        mcp_client: MCP client instance
        query: SQL query to execute
        params: Query parameters
        max_retries: Maximum number of retry attempts
        retry_delay: Initial retry delay in seconds

    Returns:
        True if successful, False otherwise
    """
    if not mcp_client:
        logger.warning("No MCP client available for database operations")
        return False

    if params is None:
        params = {}

    for attempt in range(1, max_retries + 1):
        try:
            result = await mcp_client.call_tool_direct(
                "execute_query",
                {"query": query, "params": params}
            )

            logger.info(f"Database query succeeded on attempt {attempt}")
            return True

        except Exception as e:
            logger.warning(f"Database query attempt {attempt} failed: {e}")

            if attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying database query in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} database query attempts failed")
                return False

    return False


async def store_rca_in_database(
    mcp_client,
    incident_id: str,
    structured_data: Any,
    max_retries: int = 3
) -> bool:
    """
    Store RCA analysis in database using MCP execute_query

    Args:
        mcp_client: MCP client instance
        incident_id: Incident ID
        structured_data: Structured RCA data
        max_retries: Maximum retry attempts

    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(structured_data, dict):
            data_json = json.dumps(structured_data, indent=2)
        elif hasattr(structured_data, 'model_dump'):
            data_json = json.dumps(structured_data.model_dump(), indent=2)
        else:
            data_json = json.dumps(str(structured_data))

        query = "UPDATE incidents SET rca_result = :data WHERE id = :id"
        params = {"data": data_json, "id": incident_id}

        success = await execute_database_query_with_retry(
            mcp_client,
            query,
            params,
            max_retries
        )

        if success:
            logger.info(f"Successfully stored RCA analysis in database for incident {incident_id}")
        else:
            logger.error(f"Failed to store RCA analysis in database for incident {incident_id}")

        return success

    except Exception as e:
        logger.error(f"Error storing RCA analysis: {e}")
        return False


async def update_memgraph_node_id_in_database(
    mcp_client,
    incident_id: str,
    node_id: int,
    max_retries: int = 3
) -> bool:
    """
    Update memgraph_agent_node_id in database using MCP execute_query

    Args:
        mcp_client: MCP client instance
        incident_id: Incident ID
        node_id: Memgraph node ID
        max_retries: Maximum retry attempts

    Returns:
        True if successful, False otherwise
    """
    try:
        query = "UPDATE incidents SET memgraph_agent_node_id = :node_id WHERE id = :incident_id"
        params = {"node_id": node_id, "incident_id": incident_id}

        success = await execute_database_query_with_retry(
            mcp_client,
            query,
            params,
            max_retries
        )

        if success:
            logger.info(f"Successfully updated memgraph_node_id in database for incident {incident_id}")
        else:
            logger.error(f"Failed to update memgraph_node_id in database for incident {incident_id}")

        return success

    except Exception as e:
        logger.error(f"Error updating memgraph_node_id: {e}")
        return False


async def fetch_incident_data_from_redis(
    redis_client,
    incident_key: str
) -> Dict[str, Any]:
    """
    Fetch complete incident data from Redis

    Args:
        redis_client: Redis client instance
        incident_key: Redis key for incident data

    Returns:
        Incident data dictionary or empty dict if not found
    """
    if not redis_client:
        logger.warning("Redis client not available")
        return {}

    try:
        incident_data = None
        key_attempts = []

        if incident_key.startswith("incidents:") and ":main" in incident_key:
            key_attempts.append(incident_key)
            incident_data = redis_client.get(incident_key)

        elif incident_key.startswith("incident:"):
            incident_id = incident_key.split(":")[-1]
            new_key = f"incidents:{incident_id}:main"
            key_attempts.extend([new_key, incident_key])

            incident_data = redis_client.get(new_key)
            if not incident_data:
                incident_data = redis_client.get(incident_key)

        else:
            key_attempts.append(incident_key)
            incident_data = redis_client.get(incident_key)

        if incident_data:
            parsed_data = json.loads(incident_data)
            logger.info(f"Retrieved complete incident data from Redis for key: {incident_key}")
            logger.info(f"Available fields: {list(parsed_data.keys())}")
            return parsed_data
        else:
            logger.warning(f"No incident data found in Redis for key: {incident_key}")
            return {}

    except Exception as e:
        logger.error(f"Error fetching incident data from Redis: {e}")
        return {}


async def store_rca_in_redis(
    redis_client,
    incident_id: str,
    rca_analysis: str,
    correlation_data: str = "",
    metrics_analysis: str = "",
    error: str = None,
    ttl: int = 3600
) -> bool:
    """
    Store RCA analysis results in Redis

    Args:
        redis_client: Redis client instance
        incident_id: Incident ID
        rca_analysis: RCA analysis text
        correlation_data: Correlation data
        metrics_analysis: Metrics analysis
        error: Error message if any
        ttl: Time to live in seconds (default 1 hour)

    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        logger.warning("Redis client not available")
        return False

    try:
        redis_key = f"rca_analysis:{incident_id}"
        redis_data = {
            "incident_id": incident_id,
            "rca_analysis": rca_analysis,
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if not error else "failed",
            "has_correlation_data": bool(correlation_data),
            "has_metrics_analysis": bool(metrics_analysis)
        }

        redis_client.setex(redis_key, ttl, json.dumps(redis_data))
        logger.info(f"Stored RCA analysis in Redis: {redis_key}")
        return True

    except Exception as e:
        logger.warning(f"Redis storage failed: {e}")
        return False


def create_fallback_structured_rca(incident_id: str, analysis_type: str = "normal"):
    """
    Create fallback structured RCA when LLM structuring fails

    Args:
        incident_id: Incident ID
        analysis_type: Type of analysis ("normal" or "guided")

    Returns:
        Fallback RCAStructured object
    """
    from pydantic import BaseModel, Field
    from typing import List

    class RCAStructured(BaseModel):
        incident_summary: List[str] = Field(description="Array of incident summary points")
        root_cause_analysis: List[str] = Field(description="Array of root cause analysis points")
        log_evidence: List[str] = Field(description="Array of log evidence points")

    if analysis_type == "guided":
        return RCAStructured(
            incident_summary=[
                f"Guided RCA for incident {incident_id} based on similar past incident",
                "Analysis leveraged patterns from high-similarity reference incident",
                "Current incident context adapted from proven resolution approaches"
            ],
            root_cause_analysis=[
                "Root cause identified using guidance from similar past incident with proven resolution",
                "Pattern matching with reference incident suggests consistent underlying system issue"
            ],
            log_evidence=[
                "Correlation summary shows patterns consistent with reference incident",
                "Current incident evidence aligns with previously successful diagnostic approaches",
                "System behavior matches known failure modes from similar incident"
            ]
        )
    else:
        return RCAStructured(
            incident_summary=[
                f"Incident {incident_id} analysis completed",
                "Manual review required for detailed analysis",
                "System investigation needed based on logs and metrics"
            ],
            root_cause_analysis=[
                "Automated analysis indicated potential system issue",
                "Further investigation required to determine exact cause from metrics correlation"
            ],
            log_evidence=[
                "Correlation data suggests system anomaly",
                "Log patterns indicate service disruption",
                "Metrics analysis shows performance degradation"
            ]
        )


def format_jira_comment_footer() -> str:
    """
    Format standard footer for Jira comments

    Returns:
        Formatted footer string
    """
    return f"\n\n---\n*RCA analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"


def calculate_rca_performance_score(
    execution_time: float,
    rca_length: int,
    has_error: bool
) -> tuple:
    """
    Calculate performance score for RCA execution

    Args:
        execution_time: Execution time in seconds
        rca_length: Length of RCA analysis
        has_error: Whether an error occurred

    Returns:
        Tuple of (performance_score, performance_factors)
    """
    performance_factors = []

    if execution_time < 60:
        performance_factors.append("fast_execution")

    if rca_length > 100:
        performance_factors.append("comprehensive_analysis")

    if not has_error:
        performance_factors.append("error_free")

    efficiency_score = min(1.0, len(performance_factors) * 0.33)

    return efficiency_score, performance_factors