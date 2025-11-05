"""
Centralized prompts for the correlation engine.
All LLM prompts used across the correlation workflow are defined here.
"""

# GRAFANA UID EXTRACTION PROMPT
GRAFANA_UID_EXTRACTION_PROMPT = """
You are analyzing an alert payload to find a Grafana alert UID.

**PAYLOAD:**
{payload_json}

**INSTRUCTIONS:**
1. Look for fields that might contain a Grafana alert UID or URL
2. Common field names: grafanaExploreUrl, generatorURL, dashboard_url, annotations
3. The UID is typically a short alphanumeric string (e.g., "fdvj3sjdz8o5cd")
4. Look for patterns like: /alerting/grafana/<UID>/ or uid=<UID>

**OUTPUT:**
Return ONLY the UID string if found, or "NOT_FOUND" if no UID exists.
"""

# LOGQL GENERATION PROMPT
LOGQL_GENERATION_PROMPT = """
You are a log analysis specialist helping to investigate application issues.

**ALERT INFORMATION:**
- Alert Name: {alertname}
- Service: {service}
- Description: {description}
- Severity: {severity}
- Timestamp: {timestamp}

**SERVICE DEPENDENCIES:**
{dependencies}

**GRAFANA ALERT INFO:**
{grafana_info}

**CONTEXT:**
{context}

**YOUR TASK:**
Generate LogQL queries to investigate this alert. Focus on:
1. Error logs from the primary service: {service}
2. Related errors from dependent services
3. Logs around the alert timestamp
4. Log patterns that could explain the alert condition

**REQUIREMENTS:**
- Use proper LogQL syntax
- Include appropriate time ranges (look back 15-30 minutes from alert time)
- Filter by relevant labels (namespace, pod, container)
- Look for ERROR, WARN, FATAL level logs
- Consider service dependencies in your queries

**OUTPUT FORMAT:**
Return a JSON array of query objects:
```json
[
  {{
    "service": "service-name",
    "query": "{{namespace=\\"production\\"}} |= \\"ERROR\\" | json",
    "purpose": "Find error logs in primary service",
    "time_range": "30m"
  }}
]
```

Generate 2-5 targeted queries that will help diagnose this alert.
"""

# LOG CORRELATION ANALYSIS PROMPT
LOG_CORRELATION_ANALYSIS_PROMPT = """
You are a system investigator helping engineering teams understand application problems.

**ALERT CONTEXT:**
- Alert Name: {alertname}
- Service: {service}
- Description: {description}
- Severity: {severity}
- Timestamp: {timestamp}

**FETCHED LOGS:**
{logs}

**YOUR TASK:**
Analyze the logs and explain what happened. Write a clear report that:
1. Identifies the root cause from the logs
2. Explains the sequence of events leading to the alert
3. Highlights specific error messages or patterns
4. Connects logs to the alert condition

**ANALYSIS REQUIREMENTS:**
- Write in clear, business-friendly language
- Focus on actionable insights
- Quote specific log entries as evidence
- Explain timing and causality
- Identify which service(s) caused the issue

**OUTPUT FORMAT:**
Write a narrative analysis (3-5 paragraphs) explaining what the logs reveal about this alert.
Include specific timestamps and log excerpts to support your conclusions.
"""

# STRUCTURED CORRELATION PROMPT
STRUCTURED_CORRELATION_PROMPT = """
Convert the correlation analysis into structured JSON format.

**ANALYSIS:**
{analysis}

**OUTPUT FORMAT:**
Return a JSON object matching this structure:
```json
{{
  "summary": "Brief one-sentence summary",
  "root_cause": "Primary cause identified from logs",
  "correlated_logs": [
    {{
      "timestamp": "ISO timestamp",
      "service": "service-name",
      "log_level": "ERROR/WARN/INFO",
      "message": "Log message excerpt",
      "relevance": "Why this log is relevant"
    }}
  ],
  "sequence_of_events": [
    "Step 1: What happened first",
    "Step 2: What happened next",
    "Step 3: Final state"
  ],
  "affected_services": ["service1", "service2"],
  "confidence": "high/medium/low"
}}
```

Extract all relevant information from the analysis into this structured format.
"""

# PROMQL GENERATION PROMPT
PROMQL_GENERATION_PROMPT = """
You are a performance monitoring specialist helping to investigate application performance issues.

**ALERT INFORMATION:**
- Alert Name: {alertname}
- Service: {service}
- Description: {description}
- Severity: {severity}
- Timestamp: {timestamp}

**LOG CORRELATION SUMMARY:**
{correlation_report}

**WORKFLOW:**
Read the alert info, there may be 'expr' field in it which would have custom promql used for calculating. Use that to query the prometheus and keep results consistent.

1. **READ CORRELATION REPORT**: Review the provided correlation report to identify which metric is responsible for the alert. Else use the uid from generator url to get alert info.
Always check for cpu/memory request and limit changes over the timeline as they could be changed and are often the root cause of issues.

2. **GENERATE TARGETED PROMQL QUERIES**:
   - Start with the specific metric mentioned in the alert
   - Query resource metrics: CPU usage, memory usage, disk I/O
   - Query application metrics: request rates, error rates, latency
   - Query pod/container health: restarts, OOMKills, throttling

**CONTEXT:**
{context}

**REQUIREMENTS:**
- Use proper PromQL syntax
- Include appropriate time ranges (look back 1-2 hours)
- Filter by relevant labels (namespace, pod, service)
- Consider rate calculations for counters
- Look for anomalies and threshold breaches

**OUTPUT FORMAT:**
Return a JSON array of query objects:
```json
[
  {{
    "service": "service-name",
    "metric_type": "cpu/memory/error_rate/latency",
    "expr": "rate(container_cpu_usage_seconds_total{{namespace=\\"prod\\"}}[5m])",
    "purpose": "Track CPU usage trend",
    "time_range": "2h"
  }}
]
```

Generate 3-7 targeted queries that will help diagnose this performance issue.
"""

# METRICS CORRELATION ANALYSIS PROMPT
METRICS_ANALYSIS_PROMPT = """
You are a performance analyst helping engineering teams understand system performance issues.

**ALERT CONTEXT:**
- Alert Name: {alertname}
- Service: {service}
- Description: {description}
- Severity: {severity}
- Timestamp: {timestamp}

**LOG CORRELATION:**
{log_correlation}

**FETCHED METRICS:**
{metrics}

**YOUR TASK:**
Analyze the metrics and explain the performance characteristics. Write a clear report that:
1. Identifies performance bottlenecks or resource constraints
2. Shows metric trends before and during the alert
3. Correlates metrics with the log analysis
4. Explains threshold breaches or anomalies
5. Identifies resource saturation (CPU, memory, disk, network)

**ANALYSIS REQUIREMENTS:**
- Write in clear, business-friendly language
- Include specific metric values and thresholds
- Explain trends and patterns
- Connect metrics to business impact
- Provide context for what is normal vs abnormal

**OUTPUT FORMAT:**
Write a narrative analysis (3-5 paragraphs) explaining what the metrics reveal about this alert.
Include specific metric values and time ranges to support your conclusions.
"""

# CORRELATION SUMMARY PROMPT
CORRELATION_SUMMARY_PROMPT = """
You are an expert SRE correlation analyst. Create focused markdown analysis that directly explains why a specific alert fired.

**ALERT:**
- Name: {alertname}
- Service: {service}
- Severity: {severity}
- Description: {description}
- Timestamp: {timestamp}

**LOG ANALYSIS:**
{log_analysis}

**METRICS ANALYSIS:**
{metrics_analysis}

**YOUR TASK:**
Create a concise summary that:
1. Directly answers: "Why did this alert fire?"
2. Uses ONLY evidence from logs and metrics that relate to this specific alert
3. Shows clear cause-and-effect relationship
4. Filters out noise and focuses on relevant data

**REQUIREMENTS:**
- Start with a direct answer to why the alert fired
- Reference specific log entries and metric values as evidence
- Keep it concise (2-4 paragraphs)
- Focus ONLY on information that explains THIS alert
- Avoid speculation or unrelated observations

**OUTPUT FORMAT:**
Write a focused markdown summary with:
- **Root Cause**: Direct explanation with evidence
- **Supporting Evidence**: Specific logs and metrics
- **Impact**: What was affected and how
"""

# PROMQL FILTER PROMPT
PROMQL_FILTER_PROMPT = """
You are a metrics correlation expert. Filter queries to only include those that help explain a specific alert condition.

**ALERT:**
- Name: {alertname}
- Service: {service}
- Description: {description}

**SUCCESSFUL PROMQL QUERIES:**
{queries}

**TASK:**
Filter this list to include ONLY queries that:
1. Directly relate to the alert condition
2. Show metrics that crossed thresholds
3. Reveal resource constraints or bottlenecks
4. Demonstrate the problem mentioned in the alert

**EXCLUDE:**
- Queries showing normal/healthy metrics
- Unrelated services or metrics
- Queries without clear correlation to the alert

**OUTPUT FORMAT:**
Return a JSON array of indices (0-based) for queries to keep:
```json
[0, 2, 5]
```

If all queries are relevant, return all indices. If none are relevant, return an empty array.
"""

# JIRA COMMENT TEMPLATES
JIRA_CORRELATION_TEMPLATE = """
**Correlation Analysis Results**

{content}

**Analysis Timestamp**: {timestamp}
**Correlation Engine**: Automated Log and Metrics Analysis
"""

JIRA_METRICS_TEMPLATE = """
**Metrics Analysis Results**

{content}

**Analysis Timestamp**: {timestamp}
**Correlation Engine**: Automated Performance Analysis
"""

JIRA_RCA_TEMPLATE = """
**Root Cause Analysis**

{content}

**Analysis Timestamp**: {timestamp}
**Correlation Engine**: Automated RCA
"""

JIRA_REMEDIATION_TEMPLATE = """
**Remediation Steps**

{content}

**Analysis Timestamp**: {timestamp}
**Correlation Engine**: Automated Remediation Agent
"""


def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a prompt by name with variable substitution.

    Args:
        prompt_name: Name of the prompt to retrieve
        **kwargs: Variables to substitute in the prompt

    Returns:
        Formatted prompt string
    """
    prompts = {
        "grafana_uid_extraction": GRAFANA_UID_EXTRACTION_PROMPT,
        "logql_generation": LOGQL_GENERATION_PROMPT,
        "log_correlation_analysis": LOG_CORRELATION_ANALYSIS_PROMPT,
        "structured_correlation": STRUCTURED_CORRELATION_PROMPT,
        "promql_generation": PROMQL_GENERATION_PROMPT,
        "metrics_analysis": METRICS_ANALYSIS_PROMPT,
        "correlation_summary": CORRELATION_SUMMARY_PROMPT,
        "promql_filter": PROMQL_FILTER_PROMPT,
    }

    prompt_template = prompts.get(prompt_name)
    if not prompt_template:
        raise ValueError(f"Prompt '{prompt_name}' not found")

    # Format with provided variables
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable {e} for prompt '{prompt_name}'")


def get_jira_template(template_type: str, content: str, timestamp: str = None) -> str:
    """
    Get a JIRA comment template.

    Args:
        template_type: Type of template (correlation, metrics, rca, remediation)
        content: Content to include in the template
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        Formatted JIRA comment
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().isoformat()

    templates = {
        "correlation": JIRA_CORRELATION_TEMPLATE,
        "metrics": JIRA_METRICS_TEMPLATE,
        "rca": JIRA_RCA_TEMPLATE,
        "remediation": JIRA_REMEDIATION_TEMPLATE,
    }

    template = templates.get(template_type)
    if not template:
        raise ValueError(f"JIRA template '{template_type}' not found")

    return template.format(content=content, timestamp=timestamp)
