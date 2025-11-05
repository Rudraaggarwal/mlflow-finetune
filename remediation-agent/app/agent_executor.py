"""
Simple Remediation Agent Executor
"""

import logging
import json
import os
from datetime import datetime
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError, InvalidParamsError, Part, Task, TaskState, TextPart, UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from app.simple_remediation_agent import SimpleRemediationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemediationAgentExecutor(AgentExecutor):
    def __init__(self):
        # Initialize simple remediation agent
        redis_url = os.getenv("REDIS_URL")
        database_url = os.getenv("DATABASE_URL")

        self.agent = SimpleRemediationAgent(redis_url=redis_url, database_url=database_url)
        logger.info("Simple Remediation Agent Executor initialized")

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
            logger.info("Starting remediation processing")

            # Parse query to extract incident data
            query_str = query if isinstance(query, str) else str(query)
            try:
                query_data = json.loads(query_str)
            except json.JSONDecodeError:
                # If not JSON, treat as simple incident ID
                query_data = {"incident_id": query_str.strip()}

            incident_id = query_data.get('incident_id', query_data.get('incident_key', f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

            # Create incident object - accept data from correlation agent
            incident = {
                'incident_id': incident_id,
                'alert_name': query_data.get('alert_name', query_data.get('alertname', 'Remediation Analysis Request')),
                'description': query_data.get('description', query_data.get('message', f'Remediation analysis for incident {incident_id}')),
                'priority': query_data.get('priority', query_data.get('severity', 'medium')),
                'service': query_data.get('service', 'Unknown Service'),
                'instance': query_data.get('instance', 'Unknown')
            }

            logger.info(f"Processing incident: {incident_id}")

            # Fetch RCA analysis and correlation data from Redis
            logger.info("Fetching RCA analysis and correlation data from Redis...")
            rca_analysis, correlation_data = await self.agent.fetch_data_from_redis(incident_id)

            langfuse_trace_context = {
                "trace_id" : query_data.get('current_trace_id'),
                # "parent_span_id" : query_data.get('current_observation_id'),
                "session_id": query_data.get('session_id')
            }

            # Fetch complete incident data from Redis using incident_key if available
            incident_key = query_data.get('incident_key')
            if incident_key:
                logger.info(f"Fetching complete incident data from Redis using key: {incident_key}")
                complete_incident_data = await self.agent.fetch_complete_incident_data_from_redis(incident_key,langfuse_trace_context=langfuse_trace_context)
                if complete_incident_data:
                    # Update incident object with complete data from Redis (including jira_ticket_id)
                    incident.update(complete_incident_data)
                    logger.info("Enhanced incident data with Redis information")
                    logger.info(f"Jira ticket ID: {incident.get('jira_ticket_id', 'Not found')}")
                else:
                    logger.warning("Could not fetch complete incident data from Redis")

            
            current_trace_id = query_data.get('current_trace_id')
            current_observation_id = query_data.get('current_observation_id')
            
            langfuse_trace_context = {
                "trace_id" : current_trace_id,
                "parent_span_id" : current_observation_id
            }

            # Generate remediation analysis with RCA data and correlation data
            logger.info("Generating remediation analysis...")
            remediation_result = await self.agent.analyze_remediation(incident, rca_analysis, correlation_data,langfuse_trace_context=langfuse_trace_context)
            logger.info(f"Remediation analysis completed ({len(remediation_result)} characters)")

            # Create response
            response_data = {
                "incident_id": incident_id,
                "remediation_analysis": remediation_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed" if not remediation_result.startswith("Remediation analysis failed") else "failed"
            }

            # Return the result
            await updater.add_artifact([
                Part(root=TextPart(text=json.dumps(response_data, indent=2)))
            ], name='remediation_result')

            await updater.complete()
            logger.info("Remediation agent execution completed successfully")

        except Exception as e:
            logger.error(f'Error in simple remediation agent: {e}')
            logger.error(f'Error type: {type(e).__name__}')
            logger.error(f'Error details: {str(e)}')
            import traceback
            logger.error(f'Full traceback: {traceback.format_exc()}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())