# RCA (Root Cause Analysis) Agent

A specialized AI agent for performing comprehensive root cause analysis of SRE (Site Reliability Engineering) incidents. This agent analyzes correlation data, system metrics, and incident context to identify the underlying causes of system failures and provide actionable insights.

## 🎯 Purpose

The RCA Agent performs intelligent root cause analysis by:
- **Analyzing correlation data** from the correlation agent
- **Fetching system metrics** and logs around incident timestamps
- **Identifying root causes** using LLM-powered analysis
- **Generating structured RCA reports** with confidence levels
- **Providing actionable recommendations** for incident resolution
- **Storing analysis results** for downstream agents

## 🏗️ Architecture

### Core Components

```
rcaAgent/
├── app/
│   ├── __main__.py              # A2A server entry point
│   ├── agent_executor.py        # A2A protocol executor
│   ├── enhanced_agent.py        # Main RCA logic with ReAct capabilities
│   ├── agent.py                 # Legacy RCA agent implementation
│   ├── agents/
│   │   └── rca_react_agent.py   # ReAct-based RCA agent
│   ├── models/
│   │   └── rca_models.py        # Pydantic models for structured output
│   ├── services/
│   │   ├── metrics_service.py       # System metrics and log fetching
│   │   ├── rca_analysis_service.py  # RCA analysis orchestration
│   │   └── report_service.py        # Report generation and storage
│   └── tools/
│       ├── metrics_fetcher_tool.py  # Tool for fetching metrics
│       ├── rca_analysis_tool.py     # Tool for LLM-based analysis
│       └── rca_redis_tool.py        # Redis storage and retrieval
├── pyproject.toml              # Dependencies and project config
└── README.md                   # This file
```

### Key Features

- **ReAct Agent**: Uses reasoning and action framework for better analysis
- **Multi-source data**: Integrates correlation data, metrics, and logs
- **Structured output**: Generates comprehensive RCA reports
- **Confidence scoring**: Provides confidence levels for analysis quality
- **Parallel processing**: Handles multiple incidents concurrently
- **Redis integration**: Stores and retrieves analysis results
- **Circuit breaker pattern**: Resilient error handling
- **A2A protocol**: Standardized agent communication

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Redis server (for data storage and retrieval)
- Google API key for LLM operations
- MCP tools (optional, for enhanced metrics fetching)
- Correlation agent (for correlation data)

### Installation

```bash
cd rcaAgent
uv sync
```

### Environment Setup

**📋 See [ENVIRONMENT_SETUP.md](../ENVIRONMENT_SETUP.md) for complete configuration guide**

Copy the environment example file and configure your settings:

```bash
# Copy environment template
cp env.example .env

# Edit .env with your Azure OpenAI and MCP settings
# Required variables:
# - AZURE_OPENAI_DEPLOYMENT
# - AZURE_OPENAI_VERSION  
# - AZURE_OPENAI_MODEL
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - MCP_GRAFANA_URL
# - GRAFANA_DATASOURCE_UID
```

### Running the Agent

```bash
# Basic startup (after configuring .env)
uv run -m app --host localhost --port 8011

# With custom configuration
uv run -m app --host 0.0.0.0 --port 8011
```

The agent will start on `http://localhost:8011` and expose:
- A2A protocol endpoint: `/tasks/send`
- Health check: `/health`
- Agent card: `/.well-known/agent.json`

## 📋 API Reference

### A2A Protocol

The agent implements the A2A (Agent-to-Agent) protocol for standardized communication.

#### Agent Card

```json
{
  "name": "RCA Agent",
  "description": "Performs root cause analysis for SRE incidents",
  "version": "2.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true
  },
  "skills": [
    {
      "id": "root_cause_analysis",
      "name": "Root Cause Analysis",
      "description": "Analyze incident root causes using correlation data and metrics",
      "tags": ["rca", "analysis", "sre", "incident"]
    }
  ]
}
```

#### Request Format

**Simple Incident Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "request_1",
  "method": "message/send",
  "params": {
    "message": {
      "parts": [
        {
          "type": "text",
          "text": "Analyze root cause for Database Connection Pool Exhausted incident"
        }
      ]
    }
  }
}
```

**Structured Incident Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "request_2",
  "method": "message/send",
  "params": {
    "message": {
      "parts": [
        {
          "type": "text",
          "text": "{\"incident\": {\"incident_id\": \"INC-001\", \"alert_name\": \"Database Connection Pool Exhausted\", \"description\": \"Connection pool at maximum capacity\", \"priority\": \"high\", \"service_name\": \"user-service\", \"incident_timestamp\": \"2024-01-15T10:30:00Z\"}}"
        }
      ]
    }
  }
}
```

#### Response Format

The agent returns standardized responses in the following format:

```json
{
  "status": "success",
  "agent_type": "rca",
  "incident_id": "INC-001",
  "summary": "Root cause analysis completed",
  "detailed_result": "Primary root cause identified: Database connection pool configuration...",
  "structured_data": {
    "primary_root_cause": {
      "cause_id": "RC-001",
      "description": "Database connection pool configuration too restrictive",
      "category": "configuration",
      "confidence": "high",
      "supporting_evidence": ["Connection pool max size: 10", "Peak connections: 15"]
    },
    "contributing_factors": [
      {
        "factor_id": "CF-001",
        "description": "Increased user load during peak hours",
        "category": "operational",
        "impact_weight": 0.3
      }
    ],
    "impact_analysis": {
      "severity": "high",
      "scope": "service_specific",
      "affected_services": ["user-service"],
      "estimated_duration": "30 minutes"
    },
    "recommended_actions": [
      {
        "action_id": "RA-001",
        "priority": "immediate",
        "description": "Increase connection pool size from 10 to 25",
        "category": "fix"
      }
    ],
    "confidence_level": "high",
    "analysis_quality": "comprehensive"
  },
  "processing_metadata": {
    "processing_time_ms": 2500,
    "data_sources_used": ["correlation_data", "system_metrics", "logs"],
    "analysis_depth": "comprehensive"
  }
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI deployment name | - | Yes |
| `AZURE_OPENAI_VERSION` | Azure OpenAI API version | `2024-08-01-preview` | Yes |
| `AZURE_OPENAI_MODEL` | Azure OpenAI model name | `gpt-4o-mini` | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | - | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - | Yes |
| `MCP_GRAFANA_URL` | Grafana MCP server URL | - | No |
| `MCP_GRAFANA_TRANSPORT` | MCP transport protocol | `sse` | No |
| `MCP_GRAFANA_TIMEOUT` | MCP connection timeout | `30` | No |
| `GRAFANA_DATASOURCE_UID` | Grafana datasource UID | - | No |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Agent Configuration

The agent can be configured through the `EnhancedRCAAgent` class:

```python
from app.enhanced_agent import EnhancedRCAAgent

agent = EnhancedRCAAgent(
    redis_url="redis://localhost:6379",
    use_react_agent=True,
    model_config={
        "temperature": 0.1,
        "max_tokens": 4000
    }
)
```

## 🛠️ Usage Examples

### Basic Usage

```python
import asyncio
from app.enhanced_agent import EnhancedRCAAgent

async def basic_rca():
    agent = EnhancedRCAAgent()
    await agent.initialize_services()
    
    # Simple incident analysis
    result = await agent.analyze_root_cause(
        alert={"alert_name": "Database Connection Pool Exhausted"},
        correlated_logs="Found connection timeout patterns..."
    )
    
    print(result)
    await agent.cleanup_services()

# Run the example
asyncio.run(basic_rca())
```

### Parallel Processing

```python
async def parallel_rca():
    agent = EnhancedRCAAgent()
    await agent.initialize_services()
    
    incidents = [
        {"incident_id": "INC-001", "alert_name": "High CPU Usage"},
        {"incident_id": "INC-002", "alert_name": "Memory Leak Detected"},
        {"incident_id": "INC-003", "alert_name": "Network Timeout"}
    ]
    
    result = await agent.process_incidents_parallel(incidents)
    print(result)
    await agent.cleanup_services()
```

### ReAct Agent Usage

```python
from app.agents.rca_react_agent import RCAReActAgent

react_agent = RCAReActAgent()
result = await react_agent.perform_rca_analysis({
    "incident_id": "INC-001",
    "alert_name": "Database Connection Pool Exhausted",
    "correlation_data": "Connection timeout patterns found...",
    "metrics_data": "CPU usage spiked to 95%..."
})
```

## 🔍 Advanced Features

### Structured RCA Models

The agent uses comprehensive Pydantic models for structured output:

```python
from app.models.rca_models import RCAReport, RootCause, ContributingFactor

# Create structured RCA report
rca_report = RCAReport(
    incident_id="INC-001",
    alert_name="Database Connection Pool Exhausted",
    primary_root_cause=RootCause(
        cause_id="RC-001",
        description="Connection pool configuration issue",
        category="configuration",
        confidence="high"
    ),
    contributing_factors=[
        ContributingFactor(
            factor_id="CF-001",
            description="Increased load",
            category="operational",
            impact_weight=0.3
        )
    ]
)
```

### Metrics Service Integration

Fetch system metrics and logs around incident timestamps:

```python
from app.services.metrics_service import MetricsService

metrics_service = MetricsService()
logs = await metrics_service.fetch_info_logs_around_timestamp(
    incident_timestamp=datetime.now(),
    limit=50
)
```

### Redis Storage

Store and retrieve RCA analysis results:

```python
from app.tools.rca_redis_tool import RCARedisStorageTool

redis_tool = RCARedisStorageTool()
await redis_tool._arun(
    incident_id="INC-001",
    operation="store_rca_report",
    data=rca_report_data
)
```

## 🧪 Testing

### Running Tests

```bash
# Test enhanced agent functionality
python test_enhanced_agent.py

# Test parallel processing
python test_parallel_rca.py
```

### Example Test

```python
import asyncio
from test_enhanced_agent import test_enhanced_rca_agent

async def run_tests():
    await test_enhanced_rca_agent()
    print("✅ All tests passed!")

asyncio.run(run_tests())
```

## 🔧 Troubleshooting

### Common Issues

1. **Missing Correlation Data**
   - Ensure correlation agent is running and accessible
   - Check Redis for stored correlation results
   - Verify incident IDs match between agents

2. **MCP Connection Failed**
   - Ensure MCP tools are properly configured
   - Check `MCP_COMMAND` environment variable
   - Verify MCP tool permissions

3. **Redis Connection Issues**
   - Check Redis server is running
   - Verify `REDIS_URL` configuration
   - Check network connectivity

4. **LLM API Errors**
   - Verify `AZURE_OPENAI_API_KEY` is set correctly
   - Check Azure OpenAI deployment configuration
   - Ensure proper API permissions and quota

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
uv run -m app --host localhost --port 8011
```

### Health Check

```bash
curl http://localhost:8011/health
```

Expected response:
```json
{
  "status": "healthy",
  "agent": "RCA Agent",
  "version": "2.0.0"
}
```

## 🔗 Integration

### With Orchestrator

The RCA agent integrates seamlessly with the orchestrator:

```bash
# Register with orchestrator
curl -X POST http://localhost:8000/agents/register \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "http://localhost:8011"}'
```


---

**Built with A2A Protocol, LangGraph, ReAct Framework, and modular agent architecture** 