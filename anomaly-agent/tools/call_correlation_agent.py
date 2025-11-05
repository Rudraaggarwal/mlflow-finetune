import httpx
import os
import logging
import json
import redis
from typing import Any, Dict
from uuid import uuid4
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent import langfuse

logger = logging.getLogger(__name__)

CORRELATION_AGENT_URL = os.getenv("CORRELATION_AGENT_URL", "http://localhost:8040")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def create_standardized_incident_payload(
    incident_id_str: str,
    alert_name: str,
    alert_context: str,
    severity: str,
    service_name: str,
    jira_ticket_id: str,
    timestamp: str,
    raw_alert_payload: dict
) -> dict:
    """Use LLM to create a standardized incident payload structure for Redis storage"""
    try:
        # Import LLM config from main app
        from config.llm import LLMConfig

        # Initialize LLM using config
        llm = LLMConfig.get_llm()

        llm_prompt = f"""
Create a standardized incident data structure for correlation analysis. Extract and organize the key information into a consistent JSON format.

Input Data:
- Incident ID: {incident_id_str}
- Alert Name: {alert_name}
- Alert Context: {alert_context}
- Severity: {severity}
- Service: {service_name}
- Jira Ticket: {jira_ticket_id}
- Timestamp: {timestamp}

Raw Alert Payload:
{json.dumps(raw_alert_payload, indent=2)}

Create a JSON structure with these standardized fields:
{{
  "incident_id": "{incident_id_str}",
  "alert_name": "extracted or provided alert name",
  "alertname": "same as alert_name for compatibility",
  "service": "extracted service name",
  "service_name": "same as service for compatibility",
  "severity": "extracted severity level",
  "description": "clear description of what happened",
  "status": "active",
  "timestamp": "when the alert occurred",
  "jira_ticket_id": "{jira_ticket_id}",
  "namespace": "extracted namespace if available",
  "container_name": "extracted container name if available",
  "application": "extracted application name if available",
  "grafana_folder": "extracted grafana folder if available",
  "alert_payload": "copy the original raw payload",
  "alert_context": "{alert_context}",
  "source": "anomaly_agent",
  "created_at": "{timestamp}"
}}

Rules:
1. Extract service/container information from nested alert payload if available
2. Use provided values as fallback if extraction fails
3. Ensure alertname and alert_name have the same value
4. Ensure service and service_name have the same value
5. Keep the original alert_payload for reference
6. Return only valid JSON, no extra text

Generate the standardized JSON structure:
"""

        messages = [
            SystemMessage(content="You are a data standardization specialist. Create consistent, structured JSON from incident data. Return only valid JSON with no additional text."),
            HumanMessage(content=llm_prompt)
        ]

        response = await llm.ainvoke(messages)
        standardized_json = response.content.strip()

        # Parse and validate the JSON
        try:
            standardized_data = json.loads(standardized_json)
            logger.info("Successfully created standardized incident payload using LLM")
            return standardized_data
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            # Fallback to manual structure
            return create_fallback_incident_payload(
                incident_id_str, alert_name, alert_context, severity,
                service_name, jira_ticket_id, timestamp, raw_alert_payload
            )

    except Exception as e:
        logger.error(f"Error creating standardized payload with LLM: {e}")
        # Fallback to manual structure
        return create_fallback_incident_payload(
            incident_id_str, alert_name, alert_context, severity,
            service_name, jira_ticket_id, timestamp, raw_alert_payload
        )

def create_fallback_incident_payload(
    incident_id_str: str,
    alert_name: str,
    alert_context: str,
    severity: str,
    service_name: str,
    jira_ticket_id: str,
    timestamp: str,
    raw_alert_payload: dict
) -> dict:
    """Fallback function to create standardized payload if LLM fails"""
    return {
        "incident_id": incident_id_str,
        "alert_name": alert_name,
        "alertname": alert_name,  # For compatibility
        "service": service_name,
        "service_name": service_name,  # For compatibility
        "severity": severity,
        "description": alert_context,
        "status": "active",
        "timestamp": timestamp,
        "jira_ticket_id": jira_ticket_id,
        "namespace": "unknown",
        "container_name": service_name,  # Fallback
        "application": "unknown",
        "grafana_folder": "unknown",
        "alert_payload": raw_alert_payload,
        "alert_context": alert_context,
        "source": "anomaly_agent",
        "created_at": timestamp
    }


async def send_alert_to_correlation_agent(
    incident_id: str,
    alert_context: str,
    jira_ticket_id: str,
    alert_payload: str = "",
    service_name: str = "",
    severity: str = "unknown",
    alert_name: str = "",
    timestamp: str = "",
    global_session_id: str = ""
) -> str:
    """
    Send alert context and Jira ticket information to correlation agent for analysis.

    Args:
        incident_id (str): The database incident ID
        alert_context (str): Full alert context and details from Grafana
        jira_ticket_id (str): The Jira ticket ID that was created recently.
        alert_payload (str): Complete alert payload JSON with all metadata
        service_name (str): Service/namespace affected by the alert
        severity (str): Alert severity level
        alert_name (str): Name/type of the alert
        timestamp (str): Alert timestamp

    Returns:
        str: Status of the correlation agent request
    """
    
    # Validate required parameters
    if not incident_id or not str(incident_id).strip():
        error_msg = "Failed to send to correlation agent: incident_id is required and cannot be empty"
        logger.error(error_msg)
        return error_msg

    if not jira_ticket_id or not str(jira_ticket_id).strip():
        error_msg = "Failed to send to correlation agent: jira_ticket_id is required and cannot be empty"
        logger.error(error_msg)
        return error_msg

    if not alert_context or not str(alert_context).strip():
        error_msg = "Failed to send to correlation agent: alert_context is required and cannot be empty"
        logger.error(error_msg)
        return error_msg

    # Validate jira_ticket_id format
    jira_ticket_id = str(jira_ticket_id).strip()
    if not (jira_ticket_id.startswith('PRCINC-') or jira_ticket_id.startswith('AS-')):
        error_msg = f"Failed to send to correlation agent: jira_ticket_id format invalid. Expected PRCINC-XXX or AS-XXX, got: {jira_ticket_id}"
        logger.error(error_msg)
        return error_msg

    # Validate incident_id is numeric
    incident_id_str = str(incident_id).strip()
    if not incident_id_str.isdigit():
        error_msg = f"Failed to send to correlation agent: incident_id must be numeric, got: {incident_id}"
        logger.error(error_msg)
        return error_msg
    
    try:
        logger.info(f"Sending incident {incident_id_str} with Jira ticket {jira_ticket_id} to correlation agent: {CORRELATION_AGENT_URL}")

        # Parse alert_payload if provided
        parsed_alert_payload = None
        if alert_payload and alert_payload.strip():
            try:
                parsed_alert_payload = json.loads(alert_payload)
                logger.info("Successfully parsed alert payload JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse alert_payload as JSON: {e}")
                # Keep alert_payload as string if JSON parsing fails
                parsed_alert_payload = alert_payload

        # Create incident folder structure in Redis
        incident_folder = f"incidents:{incident_id_str}"
        incident_key = f"{incident_folder}:main"

        # Use LLM to create standardized incident payload for Redis storage
        standardized_incident_data = await create_standardized_incident_payload(
            incident_id_str=incident_id_str,
            alert_name=alert_name,
            alert_context=alert_context,
            severity=severity,
            service_name=service_name,
            jira_ticket_id=jira_ticket_id,
            timestamp=timestamp,
            raw_alert_payload=parsed_alert_payload
        )

        # Use the standardized data structure
        incident_data = standardized_incident_data

        # Store incident in Redis folder structure before calling correlation agent
        try:
            redis_client = redis.from_url(REDIS_URL)

            # Store main incident data
            redis_client.set(incident_key, json.dumps(incident_data), ex=604800)  # 7 days expiry

            # Store metadata for the incident folder
            metadata = {
                "incident_id": incident_id_str,
                "created_at": timestamp,
                "status": "processing",
                "components": ["main", "correlation", "metrics"]
            }
            redis_client.set(f"{incident_folder}:metadata", json.dumps(metadata), ex=604800)

            logger.info(f"Stored incident {incident_id_str} in Redis folder: {incident_folder}")
        except Exception as redis_error:
            logger.error(f"Failed to store incident in Redis: {redis_error}")
            # Continue with correlation agent call even if Redis fails

        # Prepare the correlation payload data
        correlation_payload = {
            "incident_key": incident_key,
            "incident_id": incident_id_str,
            "current_trace_id": langfuse.get_current_trace_id(),
            "current_observation_id": langfuse.get_current_observation_id(),
            "session_id": global_session_id
        }

        # Create A2A JSON-RPC protocol message format (as per test_a2a.py)
        a2a_payload = {
            "jsonrpc": "2.0",
            "id": f"anomaly-agent-{uuid4().hex}",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"anomaly-msg-{uuid4().hex}",
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps(correlation_payload)
                        }
                    ]
                }
            }
        }

        logger.info(f"A2A payload: {a2a_payload}")

        # Make HTTP request to correlation agent using A2A protocol
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                CORRELATION_AGENT_URL,  # Use base URL for A2A protocol
                json=a2a_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Correlation agent accepted incident {incident_id} successfully")
                return f"Incident {incident_id} with Jira ticket {jira_ticket_id} sent to correlation agent successfully. Status: {result.get('status', 'processing')}"
            
            elif response.status_code == 202:
                result = response.json()
                logger.info(f"Correlation agent accepted incident {incident_id} for background processing")
                return f"Incident {incident_id} queued for correlation analysis. Processing ID: {result.get('processing_id', 'unknown')}"
            
            else:
                error_msg = f"Correlation agent returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Failed to send incident {incident_id} to correlation agent. {error_msg}"
                
    except httpx.TimeoutException:
        error_msg = f"Timeout calling correlation agent for incident {incident_id}"
        logger.error(error_msg)
        return error_msg
        
    except httpx.RequestError as e:
        error_msg = f"Network error calling correlation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error calling correlation agent for incident {incident_id}: {str(e)}"
        logger.error(error_msg)
        return error_msg