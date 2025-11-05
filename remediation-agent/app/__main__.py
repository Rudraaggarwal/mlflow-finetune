"""
Remediation Agent entry point
"""

import logging
import os
import sys
import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from app.agent_executor import RemediationAgentExecutor

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8012)
def main(host, port):
    """Starts the Remediation Agent server."""
    
    # Override with environment variables
    host =  host
    port =  int(port)
    
    # Validate required environment variables


    try:
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skills = [
            AgentSkill(
                id="Remediation Analysis",
                name="Remediation Agent",
                description="Perform Remediation for SRE alerts",
                tags=["remediation", "resolve", "analysis"],
                examples=["Provide remediation  for alert X"]
            )
        ]
        agent_card = AgentCard(
            name='Remediation Agent',
            description='Performs Remediation analysis for SRE incidents',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=RemediationAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        print(f"Starting Remediation Agent on port {port}")
        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f'Error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main() 