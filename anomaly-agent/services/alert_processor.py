"""
Simplified alert processing service for Grafana alerts.
Basic validation and conversion to agent prompts.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ValidationError
from models.grafana import GrafanaAlertPayload, GrafanaWebhookPayload

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class AlertProcessor:
    """Simplified alert processor for basic validation."""

    def __init__(self):
        pass
    
    async def process_alert(
        self,
        payload_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple alert processing with basic validation.

        Returns:
            Dict with success/failure status and processed alert data
        """
        processing_start = time.time()

        try:
            import json

            # Try to parse as Grafana alert format
            try:
                alert_payload = GrafanaAlertPayload(**payload_data)
                agent_prompt = alert_payload.to_agent_prompt()
                summary = alert_payload.get_alert_summary()

                logger.info(f"{datetime.now()} - Alert processed: {summary}")

            except ValidationError as e:
                # If validation fails, create a simple prompt from raw data
                logger.warning(f"{datetime.now()} - Alert validation failed, using raw data: {e}")

                alert_json = json.dumps(payload_data, indent=2)
                agent_prompt = f"""
You have received alert data from Grafana. Please analyze this alert.

Alert Data:
{alert_json}

Please analyze this alert and take appropriate action.
"""
                summary = "Raw alert data"

            processing_time = (time.time() - processing_start) * 1000

            return {
                "success": True,
                "agent_prompt": agent_prompt,
                "context_id": context_id,
                "processing_time_ms": processing_time,
                "summary": summary
            }

        except Exception as e:
            processing_time = (time.time() - processing_start) * 1000
            logger.error(f"{datetime.now()} - Error processing alert: {e}")
            return {
                "success": False,
                "error": "Internal processing error",
                "processing_time_ms": processing_time
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Simple health check for the alert processor."""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "processor": "alert_processor"
            }

        except Exception as e:
            logger.error(f"{datetime.now()} - Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }