import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError, InvalidParamsError, Part, Task, TextPart, UnsupportedOperationError,
)
from a2a.utils import new_task
from a2a.utils.errors import ServerError
from app.agent import AnomalyAgent
# from observability.langfuse import LangfuseObservability
# from langfuse import observe
import json
from datetime import datetime, timedelta, timezone
import os
from config.grafana_logs import fetch_all_error_logs
from dotenv import load_dotenv
# from observability.langfuse import LangfuseObservability
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyAgentExecutor(AgentExecutor):
    # @observe(name="anomaly agent executor (starting server)")
    def __init__(self):
        self.agent = AnomalyAgent()

    # @LangfuseObservability.observe
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
            # get grafana logs
            chunked_logs =await fetch_all_error_logs()
            for logs in chunked_logs:
                constuct_query = query + json.dumps(logs)
                result = await self.agent.ainvoke(constuct_query, task.contextId)
                
            # result = await self.agent.ainvoke(query, task.contextId)
            await updater.add_artifact([
                Part(root=TextPart(text=str(result['content'])))
            ], name='anomaly_result')
            await updater.complete()
        except Exception as e:
            logger.error(f'Error in anomaly agent: {e}')
            raise ServerError(error=InternalError()) from e

    # @LangfuseObservability.observe
    def _validate_request(self, context: RequestContext) -> bool:
        return False

    # @LangfuseObservability.observe
    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError()) 