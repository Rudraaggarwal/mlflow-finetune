"""
Tools Helper for Correlation Agent.
Creates LangChain tools for the agent.
"""

import logging
import json
from langchain_core.tools import tool
from langfuse import get_client

from .logging_utils import get_timestamp

logger = logging.getLogger(__name__)
langfuse = get_client()


class ToolsHelper:
    """Helper for creating LangChain tools."""

    @staticmethod
    def create_service_dependencies_tool():
        """Create service dependencies tool that returns JSON format expected by the node."""
        with langfuse.start_as_current_span(name="create-service-dependencies-tool") as span:
            @tool
            def get_service_dependencies(service_name: str) -> str:
                """
                Get service dependencies recursively for a given service name to check all related services that might be impacted.

                Args:
                    service_name (str): Name of the service to get dependencies for

                Returns:
                    str: JSON string containing service dependencies information with all recursive dependencies
                """
                try:
                    # Import and use the service_dependency tool
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    from service_dependency import execute_memgraph_query

                    # Create Cypher query to get dependencies for the service
                    query = f"""
                    MATCH (s:Service {{name: '{service_name}'}})-[r:DEPENDS_ON]->(t:Service)
                    RETURN s.name AS source, t.name AS dependency, t.namespace AS namespace
                    """

                    logger.info(f"Executing Memgraph query for service: {service_name}")

                    # Execute the query
                    result = execute_memgraph_query(query)
                    logger.info(f"Memgraph query result: {result}")

                    # Parse the response format
                    dependencies = []
                    namespace = "unknown"

                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict) and item.get('source') == service_name:
                                dep = item.get('dependency')
                                if dep and dep not in dependencies:
                                    dependencies.append(dep)
                                namespace = item.get('namespace', namespace)

                        logger.info(f"Parsed dependencies for {service_name}: {dependencies}")

                        # Create the expected JSON response format
                        response_data = {
                            "service": service_name,
                            "direct_dependencies": dependencies,
                            "all_dependencies": dependencies,
                            "namespace": namespace,
                            "all_services_to_check": [service_name] + dependencies,
                            "dependency_count": len(dependencies),
                            "services_to_monitor": len([service_name] + dependencies)
                        }

                        return json.dumps(response_data, indent=2)

                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Memgraph query error: {result['error']}")
                        return json.dumps({"error": f"Memgraph query failed: {result['error']}"})

                    else:
                        logger.warning(f"No dependencies found for service: {service_name}")
                        return json.dumps({
                            "service": service_name,
                            "direct_dependencies": [],
                            "all_dependencies": [],
                            "namespace": "unknown",
                            "all_services_to_check": [service_name],
                            "message": f"No dependencies found for {service_name} in Memgraph"
                        })

                except Exception as e:
                    logger.error(f"Error querying Memgraph for service dependencies: {e}")
                    return json.dumps({"error": f"Failed to query Memgraph: {str(e)}"})

            span.update(
                output={"tool_created": True},
                metadata={"status": "success"}
            )

            return get_service_dependencies
