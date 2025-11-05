# Anomaly Agent

This agent detects new incidents from alerts, notifies the orchestrator, and creates Jira incidents via a Jira MCP server. It is designed to be used as part of a multi-agent SRE system with an orchestrator and other specialized agents.

## Usage

### Start the Agent Server
```bash
cd anomaly-agent
uv sync
uv run -m app --host localhost --port 8013
```

### Use the Terminal Client
```bash
# Interactive mode (default)
uv run anomaly_client.py --interactive

# Send a single alert
uv run anomaly_client.py --alert "HighNumberOfUPIRequest"

# Use custom agent URL
uv run anomaly_client.py --agent http://localhost:8013 --interactive

# Use custom timeout (default is 300 seconds for orchestrator processing)
uv run anomaly_client.py --alert "HighNumberOfUPIRequest" --timeout 400
```

## Configuration

- `ORCHESTRATOR_URL`: The URL of the orchestrator agent (default: http://localhost:8000)
- `JIRA_MCP_URL`: The URL of the Jira MCP server (default: https://a8ba46d5246e.ngrok-free.app/sse)

---

## Environment Variables (.env)

Create a `.env` file in the project root with the following variables (defaults shown):

```
# Orchestrator and Jira MCP URLs
ORCHESTRATOR_URL=http://localhost:8000
JIRA_MCP_URL=https://a8ba46d5246e.ngrok-free.app/sse

# Grafana Loki MCP Server
GRAFANA_LOKI_MCP_SERVER=http://localhost:8000/sse

# Postgres MCP Server
POSTGRES_MCP_SERVER=http://localhost:8030/sse

# Database connection for Postgres MCP
DB_HOST=localhost
DB_PORT=5433
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres

# AWS Credentials for LLM (Bedrock)
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=ap-south-1

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Anomaly Agent Ingress URL
anomaly-agent_INGRESS=http://localhost:8013
```

**Descriptions:**
- `ORCHESTRATOR_URL`: URL for the orchestrator agent to notify about new incidents.
- `JIRA_MCP_URL`: URL for the Jira MCP server to create Jira tickets.
- `GRAFANA_LOKI_MCP_SERVER`: URL for the Grafana Loki MCP server (log analysis).
- `POSTGRES_MCP_SERVER`: URL for the Postgres MCP server (incident storage).
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: Database connection details for the Postgres MCP server.
- `AWS_ACCESS_KEY`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`: AWS credentials and region for using Bedrock LLM (required for some features).
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: Credentials for Langfuse observability and tracing.
- `LANGFUSE_HOST`: Host URL for Langfuse (default: https://cloud.langfuse.com).
- `anomaly-agent_INGRESS`: Public URL or ingress endpoint for the Anomaly Agent server (used for callbacks or external access).

---

## Example

Send an alert:
```
Alert: HighNumberOfUPIRequest
```
The agent will:
- Notify the orchestrator
- Create a real Jira incident ticket with:
  - Summary: "Alert: HighNumberOfUPIRequest"
  - Priority: High
  - Labels: anomaly, auto-generated, sre
  - Detailed description with timestamp and action required

## Terminal Client Features

- **Interactive Mode**: Continuously prompt for alerts
- **Single Alert Mode**: Send one alert and exit
- **Auto-formatting**: Automatically adds "Alert:" prefix if missing
- **Response Formatting**: Pretty-prints agent responses
- **Error Handling**: Graceful error handling and user feedback 