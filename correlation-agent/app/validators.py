"""
Validation utilities for correlation engine.
Provides simplified validation for query results, especially PromQL and LogQL results.
"""

from typing import Any, Dict, List, Optional, Tuple


def is_successful_promql_result(result: Any) -> bool:
    """
    Check if a PromQL result is successful and has data.
    Simplified logic: Just check if result length > 0.

    Args:
        result: PromQL query result (can be dict, list, or other)

    Returns:
        True if result has data, False otherwise
    """
    if result is None:
        return False

    # Handle dictionary results (standard Prometheus response format)
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], dict):
            if "result" in result["data"] and isinstance(result["data"]["result"], list):
                return len(result["data"]["result"]) > 0
        return False

    # Handle list results
    if isinstance(result, list):
        return len(result) > 0

    # Handle string results
    if isinstance(result, str):
        return len(result) > 0

    # For any other type, check if it's truthy
    return bool(result)


def is_successful_logql_result(result: Any) -> bool:
    """
    Check if a LogQL result is successful and has data.
    Uses same simplified logic as PromQL.

    Args:
        result: LogQL query result

    Returns:
        True if result has data, False otherwise
    """
    return is_successful_promql_result(result)


def get_result_count(result: Any) -> int:
    """
    Get the count of items in a query result.

    Args:
        result: Query result (PromQL or LogQL)

    Returns:
        Count of result items, 0 if no results
    """
    if result is None:
        return 0

    # Handle dictionary results
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], dict):
            if "result" in result["data"] and isinstance(result["data"]["result"], list):
                return len(result["data"]["result"])
        return 0

    # Handle list results
    if isinstance(result, list):
        return len(result)

    # Handle string results
    if isinstance(result, str):
        return 1 if result else 0

    return 1 if result else 0


def validate_query_response(response: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate a query response and return status with optional error message.

    Args:
        response: Query response to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if response is None:
        return False, "Response is None"

    if isinstance(response, dict):
        # Check for error field
        if "error" in response:
            return False, response["error"]

        # Check for status field
        if "status" in response and response["status"] != "success":
            return False, f"Query status: {response['status']}"

        # Check if we have data
        if not is_successful_promql_result(response):
            return False, "No data in response"

    return True, None


def has_prometheus_data(metrics_data: Dict[str, Any]) -> bool:
    """
    Check if metrics data contains any Prometheus results.

    Args:
        metrics_data: Dictionary of fetched metrics

    Returns:
        True if any metrics have data
    """
    if not metrics_data:
        return False

    for query_name, result in metrics_data.items():
        if is_successful_promql_result(result):
            return True

    return False


def has_loki_data(logs_data: Dict[str, Any]) -> bool:
    """
    Check if logs data contains any Loki results.

    Args:
        logs_data: Dictionary of fetched logs

    Returns:
        True if any logs have data
    """
    if not logs_data:
        return False

    for query_name, result in logs_data.items():
        if is_successful_logql_result(result):
            return True

    return False


def count_total_metrics(metrics_data: Dict[str, Any]) -> int:
    """
    Count total metric entries across all queries.

    Args:
        metrics_data: Dictionary of fetched metrics

    Returns:
        Total count of metric entries
    """
    total = 0

    if not metrics_data:
        return 0

    for query_name, result in metrics_data.items():
        total += get_result_count(result)

    return total


def count_total_logs(logs_data: Dict[str, Any]) -> int:
    """
    Count total log entries across all queries.

    Args:
        logs_data: Dictionary of fetched logs

    Returns:
        Total count of log entries
    """
    total = 0

    if not logs_data:
        return 0

    for query_name, result in logs_data.items():
        total += get_result_count(result)

    return total


def is_valid_query_config(query_config: Any) -> bool:
    """
    Validate a query configuration object.

    Args:
        query_config: Query configuration to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(query_config, dict):
        return False

    # Check for required fields
    required_fields = ["service"]
    for field in required_fields:
        if field not in query_config:
            return False

    # Check for query expression (expr for PromQL, query for LogQL)
    if "expr" not in query_config and "query" not in query_config:
        return False

    return True


def filter_successful_queries(queries: List[Dict[str, Any]], results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter queries to only include those with successful results.

    Args:
        queries: List of query configurations
        results: Dictionary of query results

    Returns:
        List of queries that had successful results
    """
    successful = []

    for i, query in enumerate(queries):
        query_key = f"query_{i}"
        if query_key in results:
            result = results[query_key]
            if is_successful_promql_result(result):
                successful.append({
                    "query_config": query,
                    "result": result,
                    "index": i
                })

    return successful


def validate_incident_key(incident_key: Optional[str]) -> bool:
    """
    Validate an incident key format.

    Args:
        incident_key: Incident key to validate

    Returns:
        True if valid format
    """
    if not incident_key:
        return False

    if not isinstance(incident_key, str):
        return False

    # Should not be empty after stripping
    if not incident_key.strip():
        return False

    return True
