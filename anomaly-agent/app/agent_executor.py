import logging
import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError, InvalidParamsError, Part, Task, TextPart, UnsupportedOperationError,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError
from dotenv import load_dotenv

from app.agent import AnomalyAgent
from config.grafana_logs import fetch_all_error_logs

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyAgentExecutor(AgentExecutor):
    """Executor for the Anomaly Agent, handling A2A protocol requests."""

    def __init__(self):
        """Initialize the executor with an AnomalyAgent instance."""
        self.agent = AnomalyAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the anomaly agent with the given context.

        Args:
            context: The request context containing user input
            event_queue: Event queue for publishing updates
        """
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
            # Get grafana logs and process with anomaly agent
            chunked_logs = await fetch_all_error_logs()

            for logs in chunked_logs:
                # Construct query with log data
                construct_query = query + json.dumps(logs)
                result = await self.agent.process_alert(construct_query)

            await updater.add_artifact([
                Part(root=TextPart(text=str(result.get('content', 'Processing complete'))))
            ], name='anomaly_result')
            await updater.complete()

        except Exception as e:
            logger.error(f'Error in anomaly agent: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the incoming request (currently always returns False)."""
        return False

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        """Cancel is not supported for this agent."""
        raise ServerError(error=UnsupportedOperationError()) 