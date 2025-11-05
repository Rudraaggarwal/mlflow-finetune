#!/usr/bin/env python3
"""
Simple test script for the new RCA Agent
Tests the basic functionality without complex orchestration
"""

import asyncio
import json
from app.simple_rca_agent import SimpleRCAAgent

async def test_simple_rca_agent():
    """Test the simple RCA agent"""
    print("🚀 Testing Simple RCA Agent...")
    
    # Initialize agent
    agent = SimpleRCAAgent()
    
    # Test incident
    test_incident = {
        "incident_id": "test_incident_001",
        "alert_name": "High CPU Usage Alert",
        "description": "CPU usage has exceeded 90% on production server",
        "priority": "high",
        "service_name": "web-api",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    print(f"📝 Test incident: {test_incident['alert_name']}")
    print(f"📚 Correlation data will be read from PostgreSQL automatically")
    
    # Generate RCA (correlation data read from PostgreSQL)
    try:
        result = await agent.analyze_root_cause(test_incident)
        
        print(f"\n✅ RCA Analysis Generated:")
        print(f"📄 Length: {len(result)} characters")
        print(f"🔍 Preview: {result[:200]}...")
        
        if len(result) > 100 and not result.startswith("RCA analysis failed"):
            print("\n🎉 SUCCESS: Simple RCA Agent is working!")
            return True
        else:
            print(f"\n❌ FAILURE: RCA result seems incomplete or failed")
            print(f"Full result: {result}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_rca_agent())
    if success:
        print("\n✅ All tests passed! The simple RCA agent is ready to use.")
    else:
        print("\n❌ Tests failed! Please check the configuration.")