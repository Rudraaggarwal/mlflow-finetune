"""
Storage Manager for Correlation Agent.
Handles all database operations using MCP execute_query tool with retry logic.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from langfuse import get_client, observe

from .logging_utils import get_timestamp

logger = logging.getLogger(__name__)
langfuse = get_client()


class StorageManager:
    """Manages database storage operations using MCP execute_query tool."""

    def __init__(self, mcp_client=None):
        """Initialize storage manager with MCP client."""
        self.mcp_client = mcp_client

    async def _execute_query_with_retries(
        self,
        query: str,
        operation_name: str,
        max_retries: int = 3
    ) -> Optional[Any]:
        """Execute database query using MCP tool with retry logic."""
        if not self.mcp_client:
            logger.warning("No MCP client available for database operations")
            return None

        retry_count = 0
        success = False
        result = None

        while retry_count < max_retries and not success:
            try:
                with langfuse.start_as_current_span(name=f"mcp-execute-query-{operation_name}-attempt-{retry_count + 1}") as span:
                    result = await self.mcp_client.call_tool_direct(
                        "execute_query",
                        {"query": query}
                    )
                    success = True

                    span.update(
                        input={"query_preview": query[:200], "operation": operation_name, "attempt": retry_count + 1},
                        output={"execution_successful": True, "result_received": bool(result)},
                        metadata={"tool_type": "mcp_execute_query"}
                    )

                    logger.info(f"Database {operation_name} succeeded on attempt {retry_count + 1}")

            except Exception as retry_error:
                retry_count += 1
                logger.warning(f"Database {operation_name} attempt {retry_count} failed: {retry_error}")

                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed for database {operation_name}")
                    raise retry_error

        return result

    async def extract_incident_id(self, incident_key: str) -> Optional[int]:
        """Extract incident ID from incident key."""
        with langfuse.start_as_current_span(name="extract-incident-id") as span:
            incident_id = None
            extraction_method = "unknown"

            if incident_key.startswith("incidents:"):
                parts = incident_key.split(":")
                if len(parts) >= 2 and parts[1].isdigit():
                    incident_id = int(parts[1])
                    extraction_method = "incidents_format"
            elif incident_key.startswith("incident:"):
                parts = incident_key.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    incident_id = int(parts[1])
                    extraction_method = "incident_format"
            elif incident_key.isdigit():
                incident_id = int(incident_key)
                extraction_method = "direct_id"

            span.update(
                input={"incident_key": incident_key},
                output={
                    "incident_id": incident_id,
                    "extraction_successful": bool(incident_id),
                    "extraction_method": extraction_method
                },
                metadata={"status": "success" if incident_id else "failed"}
            )

            if not incident_id:
                logger.warning(f"Could not extract incident ID from key: {incident_key}")

            return incident_id

    async def store_correlation_data(
        self,
        incident_id: int,
        correlation_data: Any
    ) -> bool:
        """Store correlation analysis data."""
        try:
            data_json = json.dumps(
                correlation_data if isinstance(correlation_data, dict)
                else correlation_data.model_dump() if hasattr(correlation_data, 'model_dump')
                else str(correlation_data)
            )

            query = f"""
            UPDATE incidents
            SET correlation_result = '{data_json.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "store-correlation")

            log_count = len(correlation_data.get('correlated_logs', [])) if isinstance(correlation_data, dict) else 0
            logger.info(f"Stored correlation data for incident {incident_id} with {log_count} logs")
            return True

        except Exception as e:
            logger.error(f"Failed to store correlation data: {e}")
            return False

    async def store_metrics_data(
        self,
        incident_id: int,
        metrics_data: str
    ) -> bool:
        """Store metrics analysis data."""
        try:
            metrics_text = metrics_data if isinstance(metrics_data, str) else str(metrics_data)

            query = f"""
            UPDATE incidents
            SET metric_insights = '{metrics_text.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "store-metrics")
            logger.info(f"Stored metrics data for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store metrics data: {e}")
            return False

    async def store_correlation_summary(
        self,
        incident_id: int,
        summary: str
    ) -> bool:
        """Store correlation summary."""
        try:
            summary_text = summary if isinstance(summary, str) else str(summary)

            query = f"""
            UPDATE incidents
            SET correlation_summary = '{summary_text.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "store-summary")
            logger.info(f"Stored correlation summary for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store correlation summary: {e}")
            return False

    async def store_promql_queries(
        self,
        incident_id: int,
        queries: List[Dict[str, Any]]
    ) -> bool:
        """Store PromQL queries."""
        try:
            queries_json = json.dumps(queries)

            query = f"""
            UPDATE incidents
            SET correlation_metrics_promql = '{queries_json.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "store-promql-queries")
            logger.info(f"Stored {len(queries)} PromQL queries for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store PromQL queries: {e}")
            return False

    async def store_dependencies(
        self,
        incident_id: int,
        dependencies: List[str]
    ) -> bool:
        """Store service dependencies."""
        try:
            deps_json = json.dumps(dependencies)

            query = f"""
            UPDATE incidents
            SET dependencies = '{deps_json.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "store-dependencies")
            logger.info(f"Stored {len(dependencies)} dependencies for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store dependencies: {e}")
            return False

    async def update_alert_metadata(
        self,
        incident_id: int,
        grafana_info: Dict[str, Any],
        alert_name: str,
        severity: str,
        timestamp: str
    ) -> bool:
        """Update alert metadata with Grafana info."""
        try:
            # First get existing metadata
            fetch_query = f"SELECT alert_metadata FROM incidents WHERE id = {incident_id}"
            existing_result = await self._execute_query_with_retries(fetch_query, "fetch-metadata")

            existing_metadata = {}
            if existing_result and isinstance(existing_result, str):
                try:
                    existing_metadata = json.loads(existing_result)
                except:
                    existing_metadata = {}

            # Update metadata
            existing_metadata.update({
                "grafana_alert_info": grafana_info,
                "processed_by_correlation_agent": True,
                "processing_timestamp": datetime.now().isoformat(),
                "alertname": alert_name,
                "severity": severity,
                "timestamp": timestamp
            })

            metadata_json = json.dumps(existing_metadata)

            query = f"""
            UPDATE incidents
            SET alert_metadata = '{metadata_json.replace("'", "''")}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "update-metadata")
            logger.info(f"Updated alert metadata for incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update alert metadata: {e}")
            return False

    async def update_mtta(self, incident_id: int) -> bool:
        """Update MTTA (Mean Time To Acknowledge) timestamp."""
        try:
            now_utc = datetime.utcnow().isoformat()

            query = f"""
            UPDATE incidents
            SET mtta = '{now_utc}'
            WHERE id = {incident_id}
            """

            await self._execute_query_with_retries(query, "update-mtta")
            return True

        except Exception as e:
            logger.error(f"Failed to update MTTA: {e}")
            return False

    async def store_all_correlation_results(
        self,
        incident_key: str,
        state: Dict[str, Any],
        results: Dict[str, Any]
    ) -> bool:
        """Store all correlation results in database."""
        with langfuse.start_as_current_span(name="store-all-correlation-results") as span:
            try:
                # Extract incident ID
                incident_id = await self.extract_incident_id(incident_key)
                if not incident_id:
                    logger.warning(f"Skipping database storage - invalid incident key: {incident_key}")
                    return False

                logger.info(f"Storing correlation results for incident {incident_id}")

                operations_completed = []

                # Store correlation analysis
                if results.get("structured_correlation"):
                    if await self.store_correlation_data(incident_id, results["structured_correlation"]):
                        operations_completed.append("correlation")

                # Store metrics analysis
                if results.get("metrics_analysis"):
                    if await self.store_metrics_data(incident_id, results["metrics_analysis"]):
                        operations_completed.append("metrics")

                # Store dependencies
                service_deps = state.get("service_dependencies", [])
                if service_deps:
                    if await self.store_dependencies(incident_id, service_deps):
                        operations_completed.append("dependencies")

                # Update alert metadata
                grafana_info = state.get("grafana_alert_info")
                if grafana_info:
                    if await self.update_alert_metadata(
                        incident_id,
                        grafana_info,
                        state.get("alertname", ""),
                        state.get("severity", ""),
                        state.get("timestamp", "")
                    ):
                        operations_completed.append("alert_metadata")

                # Store correlation summary
                if results.get("correlation_summary"):
                    if await self.store_correlation_summary(incident_id, results["correlation_summary"]):
                        operations_completed.append("correlation_summary")

                # Store PromQL queries
                if results.get("filtered_promql_queries"):
                    if await self.store_promql_queries(incident_id, results["filtered_promql_queries"]):
                        operations_completed.append("promql_queries")

                # Update MTTA
                if await self.update_mtta(incident_id):
                    operations_completed.append("mtta_update")

                logger.info(f"Successfully stored {len(operations_completed)} operations for incident {incident_id}")

                span.update(
                    output={
                        "storage_successful": True,
                        "incident_id": incident_id,
                        "operations_completed": operations_completed,
                        "total_operations": len(operations_completed)
                    },
                    metadata={"status": "success"}
                )

                return True

            except Exception as e:
                logger.error(f"Failed to store correlation results: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return False
