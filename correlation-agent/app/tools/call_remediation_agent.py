import httpx
import os
import logging
import json
import redis
from typing import Any, Dict
from uuid import uuid4
from app.agent import langfuse

logger = logging.getLogger(__name__)

REMEDIATION_AGENT_URL = os.getenv("REMEDIATION_AGENT_URL", "http://localhost:8042")
RCA_AGENT_URL = os.getenv("RCA_AGENT_URL", "http://localhost:8041")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def call_remediation_agent(
    incident_id: str,
    incident_data: Dict[str, Any],
    rca_analysis: str = "",
    correlation_data: str = ""
) -> Dict[str, Any]:
    """
    Call remediation agent for remediation recommendations.

    Args:
        incident_id (str): The incident ID
        incident_data (dict): Incident details
        rca_analysis (str): RCA analysis results
        correlation_data (str): Correlation analysis results

    Returns:
        dict: Remediation analysis results or error information
    """

    if not incident_id:
        error_msg = "Failed to call remediation agent: incident_id is required"
        logger.error(error_msg)
        return {"error": error_msg, "status": "failed"}

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

        # Check if this is a business-level alert to add special remediation context
        is_business_alert = incident_data.get("is_business_alert", False)
        has_internal_server_error = False

        # Check for Internal Server Error patterns in correlation data
        if correlation_data and isinstance(correlation_data, str):
            internal_server_patterns = [
                "Internal Server Error",
                "500 Internal Server Error",
                "Server Error",
                "CProductValidation::ProductValidate][INFORMATION] ResponseMsg: Internal Server Error occurred"
            ]
            has_internal_server_error = any(pattern in correlation_data for pattern in internal_server_patterns)

        # Prepare the remediation payload with business context
        remediation_payload = {
            "incident_id": incident_id,
            "incident_key": incident_data.get("incident_key", incident_id),
            "alert_name": incident_data.get("alert_name", incident_data.get("alertname", "Unknown")),
            "description": incident_data.get("description", "Remediation analysis request"),
            "severity": incident_data.get("severity", "unknown"),
            "service": incident_data.get("service", "unknown"),
            "instance": incident_data.get("instance", "unknown"),
            "current_trace_id": langfuse.get_current_trace_id(),
            "current_observation_id": langfuse.get_current_observation_id(),
            "is_business_alert": is_business_alert,
            "has_internal_server_error": has_internal_server_error
        }

        # Add business-specific remediation context
        if is_business_alert:
            remediation_payload["business_context"] = {
                "transaction_flow": "POS → PC (C++) → ValidationInterface (DCOM) → OEM Connector (Java) → OEM (Brand)",
                "critical_component": "OEM Connector",
                "payload_data": incident_data.get("payload", "Not available"),
                "vendor_notification_required": has_internal_server_error,
                "remediation_focus": "OEM Connector endpoint failures require vendor notification"
            }

        # Add special instructions for business alerts with Internal Server Errors
        if is_business_alert and has_internal_server_error:
            remediation_payload["special_instructions"] = (
                "BUSINESS CRITICAL: Internal Server Error detected in OEM Connector. "
                "This indicates OEM endpoint failure. Remediation MUST include: "
                "1. Immediate notification to OEM connector vendor about endpoint failures "
                "2. Business impact assessment for transaction processing "
                "3. Escalation to vendor support with specific error details "
                "4. Monitoring of vendor endpoint health status"
            )

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
                result = response.json()
                logger.info(f"Remediation agent completed analysis for incident {incident_id}")

                # Parse the result to extract remediation analysis
                if "result" in result and "artifacts" in result["result"]:
                    artifacts = result["result"]["artifacts"]
                    for artifact in artifacts:
                        if artifact.get("name") == "remediation_result":
                            remediation_content = artifact.get("parts", [{}])[0].get("text", "{}")
                            try:
                                remediation_data = json.loads(remediation_content)
                                return {
                                    "status": "completed",
                                    "remediation_analysis": remediation_data.get("remediation_analysis", ""),
                                    "incident_id": remediation_data.get("incident_id", incident_id),
                                    "timestamp": remediation_data.get("timestamp", "")
                                }
                            except json.JSONDecodeError:
                                return {
                                    "status": "completed",
                                    "remediation_analysis": remediation_content,
                                    "incident_id": incident_id
                                }

                return {
                    "status": "completed",
                    "remediation_analysis": "Remediation analysis completed but format was unexpected",
                    "incident_id": incident_id,
                    "raw_response": result
                }

            elif response.status_code == 202:
                logger.info(f"Remediation agent accepted incident {incident_id} for background processing")
                return {
                    "status": "processing",
                    "message": f"Remediation analysis queued for incident {incident_id}",
                    "incident_id": incident_id
                }

            else:
                error_msg = f"Remediation agent returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {
                    "status": "failed",
                    "error": error_msg,
                    "incident_id": incident_id
                }

    except httpx.TimeoutException:
        error_msg = f"Timeout calling remediation agent for incident {incident_id}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "incident_id": incident_id
        }

    except httpx.RequestError as e:
        error_msg = f"Network error calling remediation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "incident_id": incident_id
        }

    except Exception as e:
        error_msg = f"Unexpected error calling remediation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "incident_id": incident_id
        }