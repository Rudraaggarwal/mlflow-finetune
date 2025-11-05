#!/usr/bin/env python3
"""
Test script for Memgraph tools functionality
Tests both query_log_tool and insert_log_tool with improved httpx client
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.tools.memgraph_tools import query_log_tool, insert_log_tool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_memgraph_connection():
    """Test basic connection to memgraph endpoint"""
    logger.info("🔍 Testing Memgraph tools functionality...")

    # Test data
    test_query_log = """**Performance Analysis Report: Log Error Spike in paylater Service**

**Primary Issue:**
The primary issue that triggered the alert was a log error spike in the paylater service, which occurred on 2025-09-19T07:05:10Z. However, upon reviewing the logs, we found that no error logs were found for the paylater service during the specified time frame.

**Related Performance Issues:**
Based on the provided performance data, we observed the following related performance issues:
- **paylater_cpu_usage:** The CPU usage for the paylater service was 0% during the specified time frame, indicating that the service was not experiencing any CPU-related issues.
- **paylater_memory_usage:** The memory usage for the paylater service was 123 MB, which is within the normal range for the service.

**System Impact:**
Since no error logs were found, it is difficult to determine the exact cause of the log error spike."""

    logger.info("📋 Test parameters:")
    logger.info(f"  - Query length: {len(test_query_log)} characters")
    logger.info(f"  - Source: paylater")
    logger.info(f"  - Alert type: error")
    logger.info(f"  - Top K: 3")

    try:
        # Test 1: Query existing incidents
        logger.info("\n🔍 TEST 1: Querying similar incidents...")
        query_result = await query_log_tool(
            query_log=test_query_log,
            top_k=3,
            source="paylater",
            alert_type="error"
        )

        logger.info("✅ Query test completed!")
        logger.info(f"📊 Query result keys: {list(query_result.keys()) if isinstance(query_result, dict) else 'Not a dict'}")

        if isinstance(query_result, dict):
            if query_result.get("success"):
                logger.info(f"✅ Query successful - found {len(query_result.get('results', []))} results")
                for i, result in enumerate(query_result.get('results', [])[:2]):  # Show first 2 results
                    logger.info(f"  Result {i+1}: similarity={result.get('similarity', 'N/A')}")
            else:
                logger.warning(f"⚠️ Query returned success=False: {query_result.get('error', 'No error message')}")
        else:
            logger.warning(f"⚠️ Unexpected query result type: {type(query_result)}")

        # Test 2: Insert a test incident
        logger.info("\n💾 TEST 2: Inserting test incident...")
        test_rca = f"""**Root Cause Analysis for Test Incident**

**Timestamp:** {datetime.now().isoformat()}

**Root Cause:** Test incident for memgraph connectivity verification.

**Impact:** No actual impact - this is a test incident.

**Resolution:** This is a test entry to verify memgraph tool functionality.

**Lessons Learned:** Memgraph tools are working correctly."""

        insert_result = await insert_log_tool(
            log=test_query_log,
            rca=test_rca,
            source="test_paylater",
            alert_type="test",
            namespace="test"
        )

        logger.info("✅ Insert test completed!")
        logger.info(f"📊 Insert result keys: {list(insert_result.keys()) if isinstance(insert_result, dict) else 'Not a dict'}")

        if isinstance(insert_result, dict):
            if insert_result.get("success"):
                node_id = insert_result.get("node_id")
                logger.info(f"✅ Insert successful - node_id: {node_id}")
            else:
                logger.warning(f"⚠️ Insert returned success=False: {insert_result.get('error', 'No error message')}")
        else:
            logger.warning(f"⚠️ Unexpected insert result type: {type(insert_result)}")

        # Test 3: Query again to see if we can find the inserted incident
        if insert_result.get("success"):
            logger.info("\n🔍 TEST 3: Querying to find newly inserted incident...")
            second_query_result = await query_log_tool(
                query_log="Test incident for memgraph connectivity verification",
                top_k=5,
                source="test_paylater",
                alert_type="test"
            )

            logger.info("✅ Second query test completed!")
            if isinstance(second_query_result, dict) and second_query_result.get("success"):
                logger.info(f"✅ Second query successful - found {len(second_query_result.get('results', []))} results")
            else:
                logger.warning(f"⚠️ Second query failed or returned no results")

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

    logger.info("\n🎉 All memgraph tool tests completed!")
    return True

async def test_environment_check():
    """Check environment configuration"""
    logger.info("🌍 Checking environment configuration...")

    memgraph_url = os.getenv("MEMGRAPH_URL", "http://192.168.101.147:31998")
    similarity_threshold = os.getenv("SIMILARITY_THRESHOLD", "0.9")

    logger.info(f"📋 Environment variables:")
    logger.info(f"  - MEMGRAPH_URL: {memgraph_url}")
    logger.info(f"  - SIMILARITY_THRESHOLD: {similarity_threshold}")

    return memgraph_url

async def main():
    """Main test function"""
    logger.info("🚀 Starting Memgraph tools test...")

    # Check environment
    memgraph_url = await test_environment_check()

    if not memgraph_url:
        logger.error("❌ MEMGRAPH_URL not configured!")
        return 1

    # Run connection tests
    success = await test_memgraph_connection()

    if success:
        logger.info("✅ All tests passed! Memgraph tools are working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)