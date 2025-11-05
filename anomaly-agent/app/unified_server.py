"""
Unified Anomaly Agent Server
FastAPI-based server with A2A capabilities mounted as sub-application.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

import click
import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from app.agent_executor import AnomalyAgentExecutor
from app.agent import AnomalyAgent
from config.mcp_servers import get_mcp_config

load_dotenv()

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simple request models for FastAPI docs
class AlertPayload(BaseModel):
    """Generic alert payload model for FastAPI docs"""
    data: Dict[str, Any] = {}

class RawPayload(BaseModel):
    """Generic raw payload model for FastAPI docs"""
    data: Dict[str, Any] = {}

# Global instances
webhook_agent: Optional[AnomalyAgent] = None


# Authentication removed for testing




async def process_payload_background(payload_data: Dict[str, Any], context_id: str, payload_type: str = "alert"):
    """
    Background task to process payload directly with the anomaly agent.

    Args:
        payload_data: The payload data to process
        context_id: Unique identifier for this request
        payload_type: Type of payload - "alert" or "raw"
    """
    global webhook_agent

    if not webhook_agent:
        logger.error(f"Webhook agent not initialized")
        return

    try:
        logger.info(f"Processing {payload_type} payload: {context_id}")

        # Check if this is the new payload format with metric data
        if "metric" in payload_data and "current_value" in payload_data:
            logger.info(f"Detected new metric payload format")
            agent_result = await webhook_agent.process_alert(payload_data)

            # Log results
            if agent_result.get('success'):
                logger.info(f"Payload processing completed: {context_id}")
                if agent_result.get('incident_id'):
                    logger.info(f"Incident ID: {agent_result.get('incident_id')}")
                if agent_result.get('jira_ticket_id'):
                    logger.info(f"Jira Ticket: {agent_result.get('jira_ticket_id')}")
            else:
                logger.warning(f"Payload processing incomplete: {context_id}")
                if agent_result.get('error'):
                    logger.error(f"Error: {agent_result.get('error')}")
        else:
            # Legacy/standard format processing
            import json
            payload_json = json.dumps(payload_data, indent=2)

            if payload_type == "alert":
                prompt = f"""
You have received alert data from Grafana. Please analyze this alert and determine the appropriate response.

Alert Data:
{payload_json}

Please analyze this alert and take appropriate action.
"""
            else:
                prompt = f"""
You have received metric data corresponding to a running container. Please analyze this data and raise incident which reflects the metric data received.

Data:
{payload_json}

Please analyze this data.
"""

            # Process with agent
            agent_result = await webhook_agent.process_alert(prompt)

            # Log results
            if agent_result.get('is_task_complete') or agent_result.get('success'):
                logger.info(f"Payload processing completed: {context_id}")
            else:
                logger.warning(f"Payload processing incomplete: {context_id}")

    except Exception as e:
        logger.error(f"Error in payload processing {context_id}: {str(e)}")


# FastAPI webhook endpoints
async def _handle_webhook_request(
    request: Request,
    background_tasks: BackgroundTasks,
    payload_type: str = "alert"
):
    """
    Generic webhook handler for processing incoming requests.

    Args:
        request: FastAPI request object
        background_tasks: FastAPI background tasks
        payload_type: Type of payload - "alert" or "raw"
    """
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # Generate context ID for this request
    prefix = "webhook" if payload_type == "alert" else "raw_payload"
    context_id = f"{prefix}_{int(time.time() * 1000)}_{abs(hash(str(request.url)))}"

    try:
        # Parse JSON payload
        try:
            payload_data = await request.json()

            # Log payload for debugging
            import json
            raw_payload = json.dumps(payload_data, indent=2)
            logger.info(f"{payload_type.capitalize()} webhook received from {client_ip}")
            logger.info(f"Payload: {raw_payload}")

        except Exception as e:
            logger.error(f"Invalid JSON payload from {client_ip}: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        logger.info(f"Processing {payload_type} request: {context_id}")

        # Schedule background processing
        background_tasks.add_task(
            process_payload_background,
            payload_data=payload_data,
            context_id=context_id,
            payload_type=payload_type
        )

        processing_time_seconds = time.time() - start_time
        message = f"{payload_type.capitalize()} received and queued for processing"

        return {
            "status": "accepted",
            "message": message,
            "context_id": context_id,
            "processing_time_ms": round(processing_time_seconds * 1000, 2)
        }

    except HTTPException as e:
        raise
    except Exception as e:
        processing_time_seconds = time.time() - start_time
        logger.error(f"Unexpected webhook error from {client_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def grafana_webhook(request: Request, background_tasks: BackgroundTasks):
    """Grafana alert webhook endpoint (HTTP request variant)."""
    return await _handle_webhook_request(request, background_tasks, "alert")


async def raw_payload_webhook(request: Request, background_tasks: BackgroundTasks):
    """Generic raw payload webhook endpoint (HTTP request variant)."""
    return await _handle_webhook_request(request, background_tasks, "raw")


async def _handle_direct_payload(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    payload_type: str = "alert"
):
    """
    Generic direct payload handler that accepts payload dict.

    Args:
        payload: The payload data dictionary
        background_tasks: FastAPI background tasks
        payload_type: Type of payload - "alert" or "raw"
    """
    start_time = time.time()

    # Generate context ID for this request
    prefix = "webhook" if payload_type == "alert" else "raw_payload"
    context_id = f"{prefix}_{int(time.time() * 1000)}_{abs(hash(str(payload)))}"

    try:
        logger.info(f"{payload_type.capitalize()} webhook received")

        # Log payload for debugging
        import json
        raw_payload = json.dumps(payload, indent=2)
        logger.info(f"Payload: {raw_payload}")
        logger.info(f"Processing {payload_type} request: {context_id}")

        # Schedule background processing
        background_tasks.add_task(
            process_payload_background,
            payload_data=payload,
            context_id=context_id,
            payload_type=payload_type
        )

        processing_time_seconds = time.time() - start_time
        message = f"{payload_type.capitalize()} received and queued for processing"

        return {
            "status": "accepted",
            "message": message,
            "context_id": context_id,
            "processing_time_ms": round(processing_time_seconds * 1000, 2)
        }

    except Exception as e:
        processing_time_seconds = time.time() - start_time
        logger.error(f"Unexpected {payload_type} webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def grafana_webhook_direct(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """Direct Grafana webhook handler that accepts payload dict."""
    return await _handle_direct_payload(payload, background_tasks, "alert")


async def raw_payload_webhook_direct(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """Direct raw payload webhook handler that accepts payload dict."""
    return await _handle_direct_payload(payload, background_tasks, "raw")


async def health_check():
    """Simple health check endpoint."""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "anomaly-agent",
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")

        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        # Simple readiness check
        return {
            "ready": True,
            "timestamp": datetime.now().isoformat(),
            "service": "anomaly-agent"
        }

    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")


async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    try:
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat(),
            "service": "anomaly-agent"
        }
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }




async def status_endpoint():
    """Simple status endpoint with basic system information."""
    try:
        return {
            "application": {
                "name": "anomaly-agent",
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            },
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Status endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Status collection error")


def create_unified_app(host: str, port: int) -> FastAPI:
    """Create FastAPI-based application with A2A capabilities mounted."""
    global webhook_agent

    # Initialize webhook-specific agent
    try:
        mcp_config = get_mcp_config()
        webhook_agent = AnomalyAgent()
        logger.info(f"Webhook agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize webhook agent: {str(e)}")

    # Create A2A application
    try:
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skills = [
            AgentSkill(
                id="anomaly_detection",
                name="Anomaly Detection",
                description="Detects new incidents from alerts and triggers orchestrator and Jira incident creation.",
                tags=["anomaly", "incident", "alert", "jira", "orchestrator"],
                examples=["Alert: HighNumberOfUPIRequest"]
            )
        ]
        agent_card = AgentCard(
            name='Anomaly Agent',
            description='Unified agent for anomaly detection with A2A and webhook capabilities.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=AnomalyAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )

        a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        a2a_starlette = a2a_app.build()

        logger.info(f"A2A application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize A2A application: {str(e)}")
        raise

    # Create main FastAPI application
    app = FastAPI(
        title="Anomaly Agent",
        description="FastAPI-based anomaly detection agent with A2A capabilities",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','),
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add webhook routes
    @app.post("/grafana")
    async def grafana_webhook_endpoint(payload: Dict[str, Any], background_tasks: BackgroundTasks):
        """Grafana webhook endpoint - accepts any JSON payload"""
        return await grafana_webhook_direct(payload, background_tasks)

    @app.post("/webhook/grafana")
    async def webhook_grafana_endpoint(payload: Dict[str, Any], background_tasks: BackgroundTasks):
        """Grafana webhook endpoint - accepts any JSON payload"""
        return await grafana_webhook_direct(payload, background_tasks)

    @app.post("/payload")
    async def raw_payload_endpoint(payload: Dict[str, Any], background_tasks: BackgroundTasks):
        """Raw payload endpoint - accepts any JSON payload"""
        return await raw_payload_webhook_direct(payload, background_tasks)

    # Health endpoints
    @app.get("/health")
    async def health_check_endpoint():
        return await health_check()

    @app.get("/health/ready")
    async def readiness_check_endpoint():
        return await readiness_check()

    @app.get("/health/live")
    async def liveness_check_endpoint():
        return await liveness_check()

    @app.get("/status")
    async def status_endpoint_decorator():
        return await status_endpoint()

    # Root info endpoint
    @app.get("/")
    async def root_info():
        return {
            "service": "Anomaly Agent",
            "version": "1.0.0",
            "status": "running",
            "description": "FastAPI-based anomaly detection agent with A2A capabilities",
            "endpoints": {
                "webhook": "/webhook/grafana",
                "raw_payload": "/payload",
                "health": "/health",
                "status": "/status",
                "tasks": "/tasks"
            },
            "timestamp": datetime.now().isoformat()
        }

    # Mount A2A application under /tasks
    app.mount("/tasks", a2a_starlette)

    return app


@click.command()
@click.option('--host', 'host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', 'port', default=8000, help='Port to bind to')
def main(host, port):
    """Starts the FastAPI-based Anomaly Agent server with A2A capabilities."""
    try:
        logger.info(f"Starting Anomaly Agent Server on {host}:{port}")

        # Create unified application
        app = create_unified_app(host, port)

        logger.info(f"Starting FastAPI Anomaly Agent Server")
        logger.info(f"A2A endpoints: http://{host}:{port}/tasks/*")
        logger.info(f"Webhook endpoints: http://{host}:{port}/webhook/*")
        logger.info(f"Health check: http://{host}:{port}/health")

        # Run the server
        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()