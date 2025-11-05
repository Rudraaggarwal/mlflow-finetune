import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError, InvalidParamsError, Part, Task, TaskState, TextPart, UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from .agent import CorrelationAgent
import os

from .models import CorrelationArray, CorrelatedLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrelationAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = CorrelationAgent(
            db_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            mcp_sse_url=os.getenv("MCP_SSE_URL", "placeholder")
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())
        query = context.get_user_input()
        task = context.current_task
        if not task:
            if context.message:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            else:
                raise ServerError(error=InvalidParamsError())
        updater = TaskUpdater(event_queue, task.id, task.contextId)
        try:
            request_data = json.loads(query)
            incident_key = request_data.get('incident_key')
            incident_id = request_data.get('incident_id', 'unknown')
            
            current_trace_id = request_data.get('current_trace_id')
            current_observation_id = request_data.get('current_observation_id')
            session_id = request_data.get('session_id')
            
            langfuse_trace_context = {
                "trace_id" : current_trace_id,
                "parent_span_id" : current_observation_id,
                "session_id": session_id
            }

            if not incident_key:
                raise ValueError("incident_key is required in the request payload")

            logger.info(f"🔄 Starting correlation and metrics analysis for incident {incident_id} with key: {incident_key}")

            # Process the incident using the new analyze_incident method
            result = await self.agent.analyze_incident(incident_key,langfuse_trace_context=langfuse_trace_context)

            logger.info(f"✅ Correlation and metrics analysis completed for incident {incident_id}")

            # Format result as string for A2A framework
            if isinstance(result, dict):
                formatted_result = json.dumps(result, indent=2)
            else:
                formatted_result = str(result)
            
            # The result is already formatted from analyze_incident method
            # Just use the formatted_result created above
            
            await updater.add_artifact([
                Part(root=TextPart(text=formatted_result))
            ], name='correlation_result')
            await updater.complete()
            logger.info("✅ Correlation agent execution completed successfully")
        except Exception as e:
            logger.error(f'❌ Error in correlation agent: {e}')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'❌ Error details: {str(e)}')
            import traceback
            logger.error(f'❌ Full traceback: {traceback.format_exc()}')
            raise ServerError(error=InternalError()) from e

 
    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            if hasattr(self.agent, 'close'):
                await self.agent.close()
                logger.info("✅ Correlation agent executor cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during correlation agent executor cleanup: {e}") 