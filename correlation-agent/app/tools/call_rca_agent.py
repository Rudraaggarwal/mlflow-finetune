import httpx
import os
import logging
import json
import redis
from typing import Any, Dict
from uuid import uuid4
from app.agent import langfuse

logger = logging.getLogger(__name__)

RCA_AGENT_URL = os.getenv("RCA_AGENT_URL", "http://localhost:8011")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def send_to_rca_agent(
    incident_id: str,
    incident_data: Dict[str, Any],
    correlation_data: str = "",
    metrics_analysis: str = "",
    current_trace_id = None,
    current_observation_id = None,
    global_session_id = None
) -> str:
    """
    Send request to RCA agent for root cause analysis (fire and forget).

    Args:
        incident_id (str): The incident ID
        incident_data (dict): Incident details
        correlation_data (str): Correlation analysis results
        metrics_analysis (str): Metrics analysis results

    Returns:
        str: Status message of the request
    """

    if not incident_id:
        error_msg = "Failed to call RCA agent: incident_id is required"
        logger.error(error_msg)
        return error_msg

    try:
        logger.info(f"Calling RCA agent for incident {incident_id}: {RCA_AGENT_URL}")
        logger.info(f"Incident data is {incident_data}")
        # Store correlation data and metrics in Redis for RCA agent to fetch
        try:
            redis_client = redis.from_url(REDIS_URL)

            # Store correlation data
            if correlation_data:
                correlation_key = f"correlation_data:{incident_id}"
                redis_client.setex(correlation_key, 3600, correlation_data)  # 1 hour TTL
                logger.info(f"Stored correlation data in Redis: {correlation_key}")

            # Store metrics analysis
            if metrics_analysis:
                metrics_key = f"metrics_analysis:{incident_id}"
                redis_client.setex(metrics_key, 3600, metrics_analysis)  # 1 hour TTL
                logger.info(f"Stored metrics analysis in Redis: {metrics_key}")

        except Exception as redis_error:
            logger.error(f"Failed to store data in Redis: {redis_error}")
            # Continue with RCA call even if Redis fails

        # Prepare the RCA payload
        rca_payload = {
            "incident_id": incident_id,
            "incident_key": incident_data.get("incident_key", incident_id),
            "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown")),
            "description": incident_data.get("description", "RCA analysis request"),
            "severity": incident_data.get("severity", "unknown"),
            "service": incident_data.get("service", "unknown"),
            "instance": incident_data.get("instance", "unknown"),
            "current_trace_id":current_trace_id,
            "current_observation_id":current_observation_id,
            "session_id": global_session_id
        }

        # Create A2A JSON-RPC protocol message format
        a2a_payload = {
            "jsonrpc": "2.0",
            "id": f"correlation-agent-{uuid4().hex}",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"correlation-rca-{uuid4().hex}",
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps(rca_payload)
                        }
                    ]
                }
            }
        }

        logger.info(f"RCA A2A payload: {json.dumps(a2a_payload, indent=2)}")

        # Make HTTP request to RCA agent using A2A protocol
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for RCA
            response = await client.post(
                RCA_AGENT_URL,
                json=a2a_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info(f"✅ RCA agent accepted incident {incident_id} successfully")
                return f"Incident {incident_id} sent to RCA agent successfully"

            elif response.status_code == 202:
                logger.info(f"✅ RCA agent accepted incident {incident_id} for background processing")
                return f"Incident {incident_id} queued for RCA analysis"

            else:
                error_msg = f"RCA agent returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Failed to send incident {incident_id} to RCA agent: {error_msg}"

    except httpx.TimeoutException:
        error_msg = f"Timeout calling RCA agent for incident {incident_id}"
        logger.error(error_msg)
        return error_msg

    except httpx.RequestError as e:
        error_msg = f"Network error calling RCA agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg

    except Exception as e:
        error_msg = f"Unexpected error calling RCA agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg