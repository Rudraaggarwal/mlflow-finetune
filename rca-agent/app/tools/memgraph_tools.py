"""
Memgraph HTTP tools for RCA agent
Handles query_log and insert_log API calls
"""

import os
import logging
import httpx
import asyncio
from typing import Dict, Any, List
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables
MEMGRAPH_URL = os.getenv("MEMGRAPH_URL", "http://192.168.101.147:31998")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, will be multiplied for backoff


async def _post_with_retries(url: str, payload: dict) -> httpx.Response:
    """Helper to POST with retries and exponential backoff."""
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0,verify=False) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
    raise last_exception


@tool
async def query_log_tool(
    query_log: str,
    top_k: int = 2,
    source: str = "",
    alert_type: str = "error"
) -> Dict[str, Any]:
    """
    Query the memgraph database for similar incidents using correlation summary.
    """
    try:
        logger.info(f"Querying memgraph for similar incidents: {MEMGRAPH_URL}/query_log")

        if not query_log or not query_log.strip():
            return {"success": False, "error": "Empty query_log provided", "results": []}

        payload = {
            "query_log": query_log.strip(),
            "top_k": 10,
            "source": "",
            "alert_type": alert_type.strip() if alert_type else ""
        }

        logger.info(f"📋 Query payload: {payload}")

        response = await _post_with_retries(f"{MEMGRAPH_URL}/query_log", payload)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Found {len(result.get('results', []))} results")
            return result
        else:
            error_msg = f"Memgraph query failed {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "results": []}

    except Exception as e:
        logger.error(f"Error querying memgraph: {e}", exc_info=True)
        return {"success": False, "error": str(e), "results": []}


@tool
async def insert_log_tool(
    log: str,
    rca: str,
    source: str = "",
    alert_type: str = "error",
    namespace: str = ""
) -> Dict[str, Any]:
    """
    Insert a new incident log and RCA into the memgraph database.
    """
    try:
        logger.info(f"Inserting incident into memgraph: {MEMGRAPH_URL}/insert_log")

        if not log or not log.strip():
            return {"success": False, "error": "Empty log provided", "node_id": None}
        if not rca or not rca.strip():
            return {"success": False, "error": "Empty rca provided", "node_id": None}

        payload = {
            "log": log.strip(),
            "rca": rca.strip(),
            "source": source.strip() if source else "",
            "alert_type": alert_type.strip() if alert_type else "error",
            "namespace": namespace.strip() if namespace else "",
            "generated_by":"SRE AGENT"
        }

        logger.info(f"📋 Insert payload: {payload}")

        response = await _post_with_retries(f"{MEMGRAPH_URL}/insert_log", payload)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Inserted incident, node_id: {result.get('node_id')}")
            return result
        else:
            error_msg = f"Memgraph insert failed {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "node_id": None}

    except Exception as e:
        logger.error(f"Error inserting into memgraph: {e}", exc_info=True)
        return {"success": False, "error": str(e), "node_id": None}


def get_memgraph_tools():
    return [query_log_tool, insert_log_tool]


def filter_high_similarity_results(results: List[Dict[str, Any]], threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """Filter memgraph results by similarity threshold"""
    if not results:
        return []
    filtered = [r for r in results if r.get("similarity", 0) >= threshold]
    logger.info(f"Filtered {len(filtered)} results (>= {threshold}) from {len(results)}")
    return filtered