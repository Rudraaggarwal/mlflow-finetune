import logging
import os
import sys
import click
import httpx
import uvicorn
import asyncio
import signal
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from app.agent_executor import CorrelationAgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for cleanup
agent_executor = None
httpx_client = None

async def cleanup_resources():
    """Cleanup all resources properly."""
    global agent_executor, httpx_client
    
    logger.info("🔄 Starting cleanup process...")
    
    try:
        # Cleanup agent executor with timeout
        if agent_executor:
            try:
                if hasattr(agent_executor, 'cleanup'):
                    await asyncio.wait_for(agent_executor.cleanup(), timeout=10.0)
                    logger.info("✅ Agent executor cleanup completed")
                elif hasattr(agent_executor, 'agent'):
                    if hasattr(agent_executor.agent, 'cleanup_mcp'):
                        await asyncio.wait_for(agent_executor.agent.cleanup_mcp(), timeout=10.0)
                        logger.info("✅ Agent MCP cleanup completed")
                    if hasattr(agent_executor.agent, 'log_fetch_service'):
                        await asyncio.wait_for(agent_executor.agent.log_fetch_service.cleanup(), timeout=10.0)
                        logger.info("✅ Log fetch service cleanup completed")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Agent cleanup timed out, continuing...")
            except Exception as e:
                logger.error(f"❌ Error during agent cleanup: {e}")
    except Exception as e:
        logger.error(f"❌ Error during agent cleanup: {e}")
    
    try:
        # Cleanup HTTP client with timeout
        if httpx_client:
            await asyncio.wait_for(httpx_client.aclose(), timeout=5.0)
            logger.info("✅ HTTP client cleanup completed")
    except asyncio.TimeoutError:
        logger.warning("⚠️ HTTP client cleanup timed out, continuing...")
    except Exception as e:
        logger.error(f"❌ Error during HTTP client cleanup: {e}")
    
    logger.info("✅ Cleanup process completed")

def cleanup_sync():
    """Synchronous cleanup for when event loop is closed."""
    global agent_executor, httpx_client
    
    logger.info("🔄 Starting synchronous cleanup process...")
    
    try:
        # Basic cleanup without async operations
        if agent_executor:
            logger.info("✅ Agent executor marked for cleanup")
        
        if httpx_client:
            logger.info("✅ HTTP client marked for cleanup")
        
        logger.info("✅ Synchronous cleanup process completed")
    except Exception as e:
        logger.error(f"❌ Error during synchronous cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
    # Don't run cleanup here - let uvicorn handle it gracefully

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8040)
def main(host, port):
    """Starts the Correlation Agent server."""
    global agent_executor, httpx_client
    
    try:
        # Override with environment variables
        host = host
        port = int(port)
        
        logger.info("🚀 Starting Correlation Agent initialization...")
        logger.info(f"📋 Configuration: Host={host}, Port={port}")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Log environment setup
        logger.info("🔧 Loading environment configuration...")
        redis_url = os.getenv('REDIS_URL', '')
        
        


        
        logger.info("🎯 Setting up agent capabilities...")
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skills = [
            AgentSkill(
                id="log_correlation",
                name="Log Correlation",
                description="Correlate logs for SRE alerts",
                tags=["correlation", "logs", "sre", "analysis"],
                examples=["Correlate logs for alert X"]
            )
        ]
        
        logger.info("📝 Creating agent card...")
        agent_card = AgentCard(
            name='Correlation Agent',
            description='Performs log correlation analysis for SRE incidents',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )
        
        logger.info("🔗 Initializing HTTP client...")
        httpx_client = httpx.AsyncClient()
        
        logger.info("⚙️ Setting up request handler...")
        agent_executor = CorrelationAgentExecutor()
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        
        logger.info("🏗️ Building server application...")
        server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        
        logger.info(f"🎉 Starting Correlation Agent server on http://{host}:{port}")
        logger.info("✅ Server initialization completed successfully")
        logger.info("🔄 Ready to process incidents in parallel...")
        
        # Run with cleanup on exit
        try:
            uvicorn.run(server.build(), host=host, port=port)
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received, shutting down gracefully...")
        finally:
            # Cleanup without using event loop (since it's closed)
            logger.info("🔄 Starting synchronous cleanup...")
            cleanup_sync()
        
    except Exception as e:
        logger.error(f'❌ Error during server startup: {e}')
        logger.error(f'❌ Error type: {type(e).__name__}')
        import traceback
        logger.error(f'❌ Full traceback: {traceback.format_exc()}')
        sys.exit(1)

if __name__ == '__main__':
    main() 