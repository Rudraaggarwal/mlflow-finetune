"""
Grafana alert payload models for secure handling and validation.
"""
 
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, validator
import logging
 
logger = logging.getLogger(__name__)
 
 
class GrafanaAlertLabels(BaseModel):
    """Labels associated with a Grafana alert."""
    alertname: Optional[str] = Field(None)
    grafana_folder: Optional[str] = Field(None)
    instance: Optional[str] = Field(None)
    severity: Optional[str] = Field(None)
    model_config = ConfigDict(extra='allow')
 
    @validator('severity', pre=True, always=True)
    def normalize_severity(cls, v):
        """Normalize severity to lowercase."""
        if v:
            return str(v).lower()
        return v
 
 
class GrafanaAlertAnnotations(BaseModel):
    """Annotations associated with a Grafana alert."""
    summary: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    model_config = ConfigDict(extra='allow')
 
 
class GrafanaAlertValue(BaseModel):
    """Individual alert value in a Grafana alert payload."""
    status: Optional[str] = Field(None)
    labels: Optional[GrafanaAlertLabels] = None
    annotations: Optional[GrafanaAlertAnnotations] = None
    startsAt: Optional[str] = Field(None)
    endsAt: Optional[str] = Field(None)
    generatorURL: Optional[str] = Field(None)
    fingerprint: Optional[str] = Field(None)
    silenceURL: Optional[str] = Field(None)
    dashboardURL: Optional[str] = Field(None)
    panelURL: Optional[str] = Field(None)
    values: Optional[Dict[str, Any]] = None
    valueString: Optional[str] = Field(None)
    orgId: Optional[int] = Field(None, ge=0, le=999999)
    model_config = ConfigDict(extra='allow')
 
    @validator('status', pre=True, always=True)
    def normalize_status(cls, v):
        """Normalize status to lowercase."""
        if v:
            return str(v).lower()
        return v
 
    def get_timestamp(self) -> Optional[datetime]:
        """Parse and return the alert start timestamp."""
        if not self.startsAt:
            return None
        try:
            # Handle various timestamp formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"
            ]:
                try:
                    return datetime.strptime(self.startsAt, fmt)
                except ValueError:
                    continue
            logger.warning(f"Unable to parse timestamp: {self.startsAt}")
            return None
        except Exception as e:
            logger.error(f"Error parsing timestamp {self.startsAt}: {e}")
            return None
 
 
class GrafanaAlertPayload(BaseModel):
    """Complete Grafana alert webhook payload."""
    receiver: Optional[str] = Field(None)
    status: Optional[str] = Field(None)
    alerts: Optional[List[GrafanaAlertValue]] = Field(default_factory=list)
    groupLabels: Optional[Dict[str, Any]] = None
    commonLabels: Optional[Dict[str, Any]] = None
    commonAnnotations: Optional[Dict[str, Any]] = None
    externalURL: Optional[str] = Field(None)
    version: Optional[str] = Field(None)
    groupKey: Optional[str] = Field(None)
    truncatedAlerts: Optional[int] = Field(None, ge=0, le=10000)
    orgId: Optional[int] = Field(None, ge=0, le=999999)
    title: Optional[str] = Field(None)
    state: Optional[str] = Field(None)
    message: Optional[str] = Field(None)
    model_config = ConfigDict(extra='allow')
 
    def get_alerts_list(self) -> List[GrafanaAlertValue]:
        """Get alerts as a list, handling None and single alert cases."""
        if not self.alerts:
            return []
        if not isinstance(self.alerts, list):
            return [self.alerts]
        return self.alerts
 
    def get_primary_alert(self) -> Optional[GrafanaAlertValue]:
        """Get the primary (first) alert from the payload."""
        alerts = self.get_alerts_list()
        return alerts[0] if alerts else None
 
    def get_severity(self) -> str:
        """Extract severity from alerts, defaulting to 'info'."""
        primary_alert = self.get_primary_alert()
        if primary_alert and primary_alert.labels and primary_alert.labels.severity:
            return primary_alert.labels.severity
        return "info"
 
    def get_service_name(self) -> Optional[str]:
        """Extract service name from labels or annotations."""
        primary_alert = self.get_primary_alert()
        if not primary_alert:
            return None
        
        # Try various label fields for service identification
        if primary_alert.labels:
            for field in ['service', 'job', 'instance', 'namespace', 'pod']:
                value = getattr(primary_alert.labels, field, None)
                if value:
                    return str(value)
        
        # Try annotations
        if primary_alert.annotations and primary_alert.annotations.summary:
            # Extract service name from summary if it follows patterns
            summary = primary_alert.annotations.summary.lower()
            if 'service' in summary:
                # Simple extraction - can be enhanced based on naming conventions
                parts = summary.split()
                for i, part in enumerate(parts):
                    if part == 'service' and i + 1 < len(parts):
                        return parts[i + 1].strip('.,;:')
        
        return None
 
    def get_alert_summary(self) -> str:
        """Generate a concise summary of the alert for logging/display."""
        primary_alert = self.get_primary_alert()
        if not primary_alert:
            return "Unknown alert"
        
        parts = []
        
        # Add alertname if available
        if primary_alert.labels and primary_alert.labels.alertname:
            parts.append(f"Alert: {primary_alert.labels.alertname}")
        
        # Add status
        if primary_alert.status:
            parts.append(f"Status: {primary_alert.status}")
        
        # Add severity
        severity = self.get_severity()
        parts.append(f"Severity: {severity}")
        
        # Add service if identified
        service = self.get_service_name()
        if service:
            parts.append(f"Service: {service}")
        
        return " | ".join(parts)
 
    def to_agent_prompt(self) -> str:
        """Convert the alert payload to a natural language prompt for the agent."""
        summary = self.get_alert_summary()
        primary_alert = self.get_primary_alert()

        prompt_parts = [
            f"Process this Grafana alert: {summary}",
            f"Alert received at: {datetime.now().isoformat()}",
        ]

        if primary_alert:
            # Add timestamp information
            timestamp = primary_alert.get_timestamp()
            if timestamp:
                prompt_parts.append(f"Alert started at: {timestamp.isoformat()}")

            # Add description if available
            if primary_alert.annotations and primary_alert.annotations.description:
                prompt_parts.append(f"Description: {primary_alert.annotations.description}")

            # Add any dashboard/panel URLs for context
            if primary_alert.dashboardURL:
                prompt_parts.append(f"Dashboard: {primary_alert.dashboardURL}")

        # Add the full alert data as JSON for complete context
        import json
        try:
            alert_json = json.dumps(self.model_dump(), indent=2, default=str)
            prompt_parts.append(f"Full alert data:\n```json\n{alert_json}\n```")
        except Exception as e:
            logger.warning(f"Could not serialize alert to JSON: {e}")

        return "\n\n".join(prompt_parts)

    def get_complete_payload_json(self) -> str:
        """Get the complete alert payload as JSON string for passing to correlation agent."""
        import json
        try:
            return json.dumps(self.model_dump(), indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize complete payload to JSON: {e}")
            return "{}"
 
 
class GrafanaWebhookPayload(BaseModel):
    """Wrapper payload format that contains the actual alert payload."""
    event: Optional[str] = None
    service: Optional[str] = None  
    version: Optional[str] = None
    environment: Optional[str] = None
    hostname: Optional[str] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    level: Optional[str] = None
    payload: Optional[GrafanaAlertPayload] = None
    model_config = ConfigDict(extra='allow')
 
    def is_wrapper_format(self) -> bool:
        """Check if this is the new wrapper format."""
        return self.payload is not None
 
    def extract_alert_payload(self) -> GrafanaAlertPayload:
        """Extract the alert payload, handling both old and new formats."""
        if self.is_wrapper_format():
            return self.payload
        else:
            # This is direct alert payload format - convert to GrafanaAlertPayload
            payload_dict = self.model_dump()
            # Remove wrapper-specific fields that don't belong in GrafanaAlertPayload
            wrapper_fields = ['event', 'service', 'version', 'environment', 'hostname',
                            'correlation_id', 'request_id', 'timestamp', 'level']
            for field in wrapper_fields:
                payload_dict.pop(field, None)
            return GrafanaAlertPayload(**payload_dict)
 
    def get_individual_alerts(self) -> List[GrafanaAlertPayload]:
        """Extract individual alerts as separate GrafanaAlertPayload objects for processing."""
        alert_payload = self.extract_alert_payload()
        alerts = alert_payload.get_alerts_list()
        
        if not alerts:
            return []
            
        individual_payloads = []
        for alert in alerts:
            # Create a new payload for each individual alert
            individual_payload = GrafanaAlertPayload(
                receiver=alert_payload.receiver,
                status=alert_payload.status,
                alerts=[alert],  # Single alert
                groupLabels=alert_payload.groupLabels,
                commonLabels=alert_payload.commonLabels,
                commonAnnotations=alert_payload.commonAnnotations,
                externalURL=alert_payload.externalURL,
                version=alert_payload.version,
                groupKey=alert_payload.groupKey,
                truncatedAlerts=0,  # Reset for individual alert
                orgId=alert_payload.orgId,
                title=alert_payload.title,
                state=alert_payload.state,
                message=alert_payload.message
            )
            individual_payloads.append(individual_payload)
            
        return individual_payloads
 