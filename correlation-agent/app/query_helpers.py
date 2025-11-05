"""
Query Helpers for Correlation Agent.
Handles LogQL and PromQL query processing logic.
"""

import logging
from typing import Dict, Any, List
from langfuse import get_client

from .validators import count_total_logs, count_total_metrics, has_loki_data, has_prometheus_data
from .logging_utils import get_timestamp

logger = logging.getLogger(__name__)
langfuse = get_client()


class QueryHelpers:
    """Helper for query processing operations."""

    @staticmethod
    async def should_fetch_more_logs_logic(state: Dict[str, Any]) -> bool:
        """Determine if more log fetching is needed."""
        try:
            fetched_logs = state.get("fetched_logs", {})
            generated_queries = state.get("generated_logql_queries", [])

            # Count total logs fetched
            total_logs = count_total_logs(fetched_logs)

            # Check if we have enough logs
            if total_logs >= 50:
                logger.info(f"Sufficient logs fetched ({total_logs}), proceeding to analysis")
                return False

            # Check if we've tried enough queries
            queries_executed = len(fetched_logs.keys())
            if queries_executed >= len(generated_queries):
                logger.info(f"All queries executed, proceeding to analysis")
                return False

            # Check if we have any Loki data
            if not has_loki_data(fetched_logs):
                logger.warning("No Loki data found, may need more fetching")
                return True

            logger.info(f"Have {total_logs} logs, continuing to fetch more")
            return total_logs < 50

        except Exception as e:
            logger.error(f"Error in should_fetch_more_logs_logic: {e}")
            return False

    @staticmethod
    async def should_fetch_more_metrics_logic(state: Dict[str, Any]) -> bool:
        """Determine if more metrics fetching is needed."""
        try:
            fetched_metrics = state.get("fetched_metrics", {})
            generated_queries = state.get("generated_promql_queries", [])

            # Count total metrics fetched
            total_metrics = count_total_metrics(fetched_metrics)

            # Check if we have enough metrics
            if total_metrics >= 10:
                logger.info(f"Sufficient metrics fetched ({total_metrics}), proceeding to analysis")
                return False

            # Check if we've tried enough queries
            queries_executed = len(fetched_metrics.keys())
            if queries_executed >= len(generated_queries):
                logger.info(f"All queries executed, proceeding to analysis")
                return False

            # Check if we have any Prometheus data
            if not has_prometheus_data(fetched_metrics):
                logger.warning("No Prometheus data found, may need more fetching")
                return True

            logger.info(f"Have {total_metrics} metrics, continuing to fetch more")
            return total_metrics < 10

        except Exception as e:
            logger.error(f"Error in should_fetch_more_metrics_logic: {e}")
            return False

    @staticmethod
    async def filter_promql_queries_for_storage(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter and prepare PromQL queries for storage."""
        with langfuse.start_as_current_span(name="filter-promql-queries") as span:
            try:
                generated_queries = state.get("generated_promql_queries", [])

                if not generated_queries:
                    logger.info("No PromQL queries to filter")
                    return []

                span.update(
                    input={"total_queries": len(generated_queries)},
                    metadata={"component": "promql_filtering"}
                )

                # Filter queries that returned data
                filtered_queries = []
                fetched_metrics = state.get("fetched_metrics", {})

                for query in generated_queries:
                    query_name = query.get("name", "")

                    # Check if this query returned data
                    if query_name in fetched_metrics:
                        metric_data = fetched_metrics[query_name]
                        if metric_data and has_prometheus_data({query_name: metric_data}):
                            filtered_queries.append(query)

                logger.info(f"Filtered {len(filtered_queries)} PromQL queries with data from {len(generated_queries)} total")

                span.update(
                    output={
                        "filtered_count": len(filtered_queries),
                        "total_count": len(generated_queries)
                    },
                    metadata={"status": "success"}
                )

                return filtered_queries

            except Exception as e:
                logger.error(f"Error filtering PromQL queries: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return []
