import httpx
import os
import logging
import json
import redis
from typing import Any, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)

REMEDIATION_AGENT_URL = os.getenv("REMEDIATION_AGENT_URL", "http://localhost:8012")
RCA_AGENT_URL = os.getenv("RCA_AGENT_URL", "http://localhost:8041")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def send_to_remediation_agent(
    incident_id: str,
    incident_data: Dict[str, Any],
    rca_analysis: str = "",
    correlation_data: str = "",
    current_trace_id = None,
    current_observation_id = None,
    session_id = None
) -> str:
    """
    Send request to remediation agent for remediation recommendations (fire and forget).

    Args:
        incident_id (str): The incident ID
        incident_data (dict): Incident details
        rca_analysis (str): RCA analysis results
        correlation_data (str): Correlation analysis results

    Returns:
        str: Status message of the request
    """

    if not incident_id:
        error_msg = "Failed to call remediation agent: incident_id is required"
        logger.error(error_msg)
        return error_msg

    try:
        logger.info(f"Calling remediation agent for incident {incident_id}: {REMEDIATION_AGENT_URL}")

        # If no RCA analysis provided, try to get it from RCA agent first
        if not rca_analysis:
            logger.info(f"No RCA analysis provided, calling RCA agent first for incident {incident_id}")
            # Import here to avoid circular imports
            from .call_rca_agent import call_rca_agent
            rca_result = await call_rca_agent(incident_id, incident_data, correlation_data, "")
            if rca_result.get("status") == "completed":
                rca_analysis = rca_result.get("rca_analysis", "")
                logger.info("Successfully obtained RCA analysis for remediation")
            else:
                logger.warning(f"Failed to get RCA analysis: {rca_result.get('error', 'Unknown error')}")

        # Store RCA analysis and correlation data in Redis for remediation agent to fetch
        try:
            redis_client = redis.from_url(REDIS_URL)

            # Store RCA analysis
            if rca_analysis:
                rca_key = f"rca_analysis:{incident_id}"
                rca_data = {
                    "incident_id": incident_id,
                    "rca_analysis": rca_analysis,
                    "timestamp": incident_data.get("timestamp", ""),
                    "status": "completed"
                }
                redis_client.setex(rca_key, 3600, json.dumps(rca_data))  # 1 hour TTL
                logger.info(f"Stored RCA analysis in Redis: {rca_key}")

            # Store correlation data
            if correlation_data:
                correlation_key = f"correlation_data:{incident_id}"
                redis_client.setex(correlation_key, 3600, correlation_data)  # 1 hour TTL
                logger.info(f"Stored correlation data in Redis: {correlation_key}")

        except Exception as redis_error:
            logger.error(f"Failed to store data in Redis: {redis_error}")
            # Continue with remediation call even if Redis fails

        # Prepare the remediation payload
        # Ensure incident_key is in the correct Redis format (incidents:ID:main)
        incident_key = incident_data.get("incident_key")
        if not incident_key or incident_key == incident_id:
            # Generate proper Redis key format if not provided
            incident_key = f"incidents:{incident_id}:main"

        remediation_payload = {
            "incident_id": incident_id,
            "incident_key": incident_key,
            "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown")),
            "description": incident_data.get("description", "Remediation analysis request"),
            "severity": incident_data.get("severity", "unknown"),
            "service": incident_data.get("service", "unknown"),
            "instance": incident_data.get("instance", "unknown"),
            "current_trace_id":current_trace_id,
            "current_observation_id":current_observation_id,
            "session_id":session_id
        }

        # Create A2A JSON-RPC protocol message format
        a2a_payload = {
            "jsonrpc": "2.0",
            "id": f"correlation-agent-{uuid4().hex}",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"correlation-remediation-{uuid4().hex}",
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps(remediation_payload)
                        }
                    ]
                }
            }
        }

        logger.info(f"Remediation A2A payload: {json.dumps(a2a_payload, indent=2)}")

        # Make HTTP request to remediation agent using A2A protocol
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for remediation
            response = await client.post(
                REMEDIATION_AGENT_URL,
                json=a2a_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info(f"✅ Remediation agent accepted incident {incident_id} successfully")
                return f"Incident {incident_id} sent to remediation agent successfully"

            elif response.status_code == 202:
                logger.info(f"✅ Remediation agent accepted incident {incident_id} for background processing")
                return f"Incident {incident_id} queued for remediation analysis"

            else:
                error_msg = f"Remediation agent returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Failed to send incident {incident_id} to remediation agent: {error_msg}"

    except httpx.TimeoutException:
        error_msg = f"Timeout calling remediation agent for incident {incident_id}"
        logger.error(error_msg)
        return error_msg

    except httpx.RequestError as e:
        error_msg = f"Network error calling remediation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg

    except Exception as e:
        error_msg = f"Unexpected error calling remediation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg