"""
Redis Helper for Correlation Agent.
Handles Redis operations for incident data.
"""

import logging
import json
from typing import Dict, Any, Optional
from langfuse import get_client

from .logging_utils import get_timestamp

logger = logging.getLogger(__name__)
langfuse = get_client()


class RedisHelper:
    """Helper for Redis operations."""

    def __init__(self, redis_client=None):
        """Initialize Redis helper."""
        self.redis_client = redis_client

    async def get_incident_from_redis(self, incident_key: str) -> Optional[Dict[str, Any]]:
        """Get incident information from Redis folder structure."""
        with langfuse.start_as_current_span(name="redis-incident-retrieval") as span:
            try:
                if not self.redis_client:
                    logger.warning("Redis client not configured")
                    return None

                span.update(
                    input={"incident_key": incident_key},
                    metadata={"component": "redis_retrieval"}
                )

                logger.info(f"Retrieving incident from Redis: {incident_key}")

                # Try to get from Redis
                incident_data = self.redis_client.get(incident_key)

                if incident_data:
                    parsed_data = json.loads(incident_data)
                    logger.info(f"Retrieved incident from Redis with {len(parsed_data)} fields")

                    span.update(
                        output={
                            "incident_found": True,
                            "data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else [],
                            "data_size": len(str(parsed_data))
                        },
                        metadata={"status": "success"}
                    )

                    return parsed_data
                else:
                    logger.warning(f"Incident not found in Redis: {incident_key}")
                    span.update(
                        output={"incident_found": False},
                        metadata={"status": "not_found"}
                    )
                    return None

            except Exception as e:
                logger.error(f"Error retrieving incident from Redis: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return None

    async def store_results_in_redis(self, incident_key: str, results: Dict[str, Any]) -> bool:
        """Store correlation results in Redis."""
        with langfuse.start_as_current_span(name="redis-results-storage") as span:
            try:
                if not self.redis_client:
                    logger.warning("Redis client not configured")
                    return False

                span.update(
                    input={
                        "incident_key": incident_key,
                        "results_keys": list(results.keys()),
                        "results_size": len(str(results))
                    },
                    metadata={"component": "redis_storage"}
                )

                logger.info(f"Storing results in Redis for incident: {incident_key}")

                # Store results in Redis with expiration
                results_json = json.dumps(results)
                self.redis_client.setex(
                    f"{incident_key}:correlation_results",
                    86400,  # 24 hours expiration
                    results_json
                )

                logger.info(f"Successfully stored results in Redis")

                span.update(
                    output={
                        "storage_successful": True,
                        "expiration_seconds": 86400
                    },
                    metadata={"status": "success"}
                )

                return True

            except Exception as e:
                logger.error(f"Error storing results in Redis: {e}")
                span.update(
                    output={"error": str(e)},
                    metadata={"status": "error"}
                )
                return False
