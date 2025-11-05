#!/usr/bin/env python3
"""
Test script for the RCA Agent using A2A JSON-RPC format.
"""

import asyncio
import httpx
import json
import logging
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_redis_data():
    """Setup test correlation and metrics data in Redis for RCA testing."""
    try:
        # Connect to Redis
        r = redis.from_url("redis://:Y7PvFpi4g7@192.168.101.144:32125/1")

        # Test incident ID
        incident_id = "232"

        # Store correlation data (what correlation agent would have stored)
        correlation_data = """
**LOG CORRELATION ANALYSIS:**

Found 15 relevant log entries for alert 'oemconnector' in paylater namespace:

**CRITICAL ERROR PATTERNS:**
- Connection timeouts to database service
- OEM Connector pod restarts detected
- Service mesh connectivity issues
- Memory pressure warnings before failure

**TIMELINE:**
- 06:47:13 UTC: First connection timeout logged
- 06:48:45 UTC: Pod restart initiated
- 06:50:13 UTC: Alert triggered
- Service degradation ongoing

**ROOT CAUSE INDICATORS:**
- Database connection pool exhaustion
- Pod resource limits exceeded
- Network connectivity issues between services

**IMPACT ASSESSMENT:**
- CRITICAL: OEM Connector service unavailable
- Payment processing disrupted in UAT environment
- Downstream services experiencing cascading failures
        """

        correlation_key = f"correlation_data:{incident_id}"
        r.setex(correlation_key, 3600, correlation_data)  # 1 hour TTL
        logger.info(f"🔑 Stored correlation data for incident {incident_id}")

        # Store metrics analysis (what correlation agent would have stored)
        metrics_data = """
**METRICS CORRELATION ANALYSIS:**

**RESOURCE UTILIZATION:**
- CPU: 95% (threshold: 80%) - CRITICAL
- Memory: 87% (threshold: 85%) - WARNING
- Disk I/O: 78% (threshold: 80%) - OK
- Network: 45% (threshold: 70%) - OK

**SERVICE METRICS:**
- Request latency: 8.5s (SLA: 2s) - CRITICAL
- Error rate: 23% (threshold: 5%) - CRITICAL
- Throughput: 45 req/min (normal: 120 req/min) - DEGRADED

**DATABASE METRICS:**
- Connection pool: 98% utilization - CRITICAL
- Active connections: 195/200 - WARNING
- Query response time: 4.2s (normal: 300ms) - CRITICAL

**CONTAINER METRICS:**
- Pod restarts: 3 in last hour - WARNING
- Memory limit reached: 95% of 2Gi - CRITICAL
- OOMKilled events: 2 - CRITICAL

**ALERT CORRELATION:**
Multiple related alerts firing:
- DatabaseConnectionTimeout
- PodRestartExceeded
- HighMemoryUsage
- ServiceLatencyHigh
        """

        metrics_key = f"metrics_analysis:{incident_id}"
        r.setex(metrics_key, 3600, metrics_data)  # 1 hour TTL
        logger.info(f"🔑 Stored metrics analysis for incident {incident_id}")

        logger.info("✅ Test Redis data setup complete for RCA agent!")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to setup Redis test data: {e}")
        return False

async def test_rca_agent():
    """Test the RCA agent with proper A2A format and incident data."""

    # Test incidents that match the payload format from call_rca_agent.py
    test_incidents = [
        {
            "incident_id": "232",
            "incident_key": "incidents:232:main",
            "alert_name": "oemconnector",
            "description": "RCA analysis request",
            "severity": "critical",
            "service": "emipaylater",
            "instance": "172.10.2.15:8080"
        }
    ]

    try:
        logger.info("🚀 Testing RCA agent with A2A format...")

        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            for i, incident in enumerate(test_incidents):
                # Create RCA payload (matches call_rca_agent.py)
                rca_payload = {
                    "incident_id": incident["incident_id"],
                    "incident_key": incident["incident_key"],
                    "alert_name": incident["alert_name"],
                    "description": incident["description"],
                    "severity": incident["severity"],
                    "service": incident["service"],
                    "instance": incident["instance"]
                }

                # Create A2A JSON-RPC request (matches call_rca_agent.py format)
                request_data = {
                    "jsonrpc": "2.0",
                    "id": f"correlation-agent-test-{i+1}",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": f"correlation-rca-test-{i+1}",
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": json.dumps(rca_payload)
                                }
                            ]
                        }
                    }
                }

                logger.info(f"\n📤 Sending RCA request {i+1}: {incident['description']}")
                logger.info(f"🔑 Incident ID: {incident['incident_id']}")
                logger.info(f"🏷️ Alert: {incident['alert_name']} ({incident['severity']})")
                logger.info(f"🎯 Service: {incident['service']}")

                response = await client.post(
                    "http://localhost:8011/",  # RCA agent port
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )

                logger.info(f"📥 Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and "artifacts" in result["result"]:
                        rca_result = result["result"]["artifacts"][0]["parts"][0]["text"]

                        # Parse the RCA result
                        try:
                            parsed_result = json.loads(rca_result)
                            logger.info(f"✅ Success! RCA analysis completed")

                            # Check workflow completion
                            if parsed_result.get("completed"):
                                logger.info(f"🎉 RCA LangGraph workflow completed successfully!")

                                # Log workflow summary
                                workflow_summary = parsed_result.get("workflow_summary", {})
                                logger.info(f"📊 RCA Workflow Summary:")
                                logger.info(f"   - RCA analysis completed: {workflow_summary.get('rca_completed', False)}")
                                logger.info(f"   - Redis data fetched: {workflow_summary.get('redis_data_fetched', False)}")
                                logger.info(f"   - Results stored: {workflow_summary.get('results_stored', False)}")
                                logger.info(f"   - Jira comment added: {workflow_summary.get('jira_comment_added', False)}")
                                logger.info(f"   - Remediation agent called: {workflow_summary.get('remediation_called', False)}")

                                # Check RCA analysis results
                                if parsed_result.get("rca_analysis"):
                                    logger.info(f"🔍 RCA analysis: ✅ Available")
                                    rca_content = parsed_result["rca_analysis"]
                                    if "ROOT CAUSE" in rca_content:
                                        logger.info(f"🎯 ROOT CAUSE identified!")
                                    if "REMEDIATION" in rca_content:
                                        logger.info(f"🔧 REMEDIATION recommendations provided!")

                                # Check if remediation agent was called
                                if parsed_result.get("remediation_status"):
                                    logger.info(f"🚀 Remediation agent status: {parsed_result['remediation_status']}")

                            else:
                                logger.warning(f"⚠️ RCA workflow not completed. Current step: {parsed_result.get('current_step', 'unknown')}")
                                if parsed_result.get("error"):
                                    logger.error(f"❌ Error: {parsed_result['error']}")

                        except json.JSONDecodeError:
                            logger.info(f"📝 Raw result (not JSON): {rca_result[:500]}...")

                    else:
                        logger.error(f"❌ Unexpected response format: {result}")

                elif response.status_code == 202:
                    logger.info(f"✅ RCA agent accepted incident for background processing")

                else:
                    logger.error(f"❌ Request failed with status {response.status_code}")
                    logger.error(f"📝 Error response: {response.text}")

                # Wait between requests
                logger.info("⏳ Waiting before next request...\n")
                await asyncio.sleep(5)

        return True

    except Exception as e:
        logger.error(f"❌ RCA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Setting up test Redis data for RCA agent...")
    # redis_setup = asyncio.run(setup_test_redis_data())
    redis_setup=True

    if redis_setup:
        print("🚀 Starting RCA agent tests...")
        success = asyncio.run(test_rca_agent())
        if success:
            logger.info("🎉 All RCA tests completed!")
        else:
            logger.error("💥 RCA tests failed!")
            exit(1)
    else:
        logger.error("💥 Redis setup failed!")
        exit(1)