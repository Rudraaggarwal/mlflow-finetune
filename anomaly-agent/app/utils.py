"""Utility functions for the Anomaly Agent."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def escape_sql_string(value: str) -> str:
    """
    Escape single quotes in SQL string values to prevent SQL injection.

    Args:
        value: The string value to escape

    Returns:
        Escaped string safe for SQL queries
    """
    if not isinstance(value, str):
        return str(value)
    return value.replace("'", "''")


def infer_service_from_metric(payload: Dict[str, Any], fallback: str = "unknown") -> str:
    """
    Infer service/container name from metric-based payload.

    Priority:
    1) usage data[0].metric.container
    2) parse from metric name (suffix after last underscore)
    3) fallback provided

    Args:
        payload: Alert payload containing metric data
        fallback: Default value if service cannot be inferred

    Returns:
        Inferred service name
    """
    try:
        usage = payload.get("usage data")
        if isinstance(usage, list) and usage:
            first = usage[0]
            metric_obj = first.get("metric", {}) if isinstance(first, dict) else {}
            container = metric_obj.get("container")
            if isinstance(container, str) and container.strip():
                return container.strip()

        metric_name = payload.get("metric")
        if isinstance(metric_name, str) and metric_name:
            parts = metric_name.split("_")
            if len(parts) >= 2 and parts[-1]:
                return parts[-1]
    except Exception:
        pass
    return fallback


async def infer_alert_name_from_metric(
    alert_payload: Dict[str, Any],
    llm,
    langfuse_handler,
    predefined_trace_id: str
) -> Tuple[str, str]:
    """
    Infer alert name and service from metric payload using a single LLM call returning JSON.

    Args:
        alert_payload: Alert payload containing metric data
        llm: Language model instance for inference
        langfuse_handler: Langfuse callback handler
        predefined_trace_id: Trace ID for observability

    Returns:
        Tuple of (alert_name, service_name)
    """
    from langfuse import get_client
    langfuse = get_client()

    with langfuse.start_as_current_span(name="alert-name-generation") as span:
        try:
            metric_name_raw = alert_payload.get("metric", "unknown") or "unknown"
            current_value = alert_payload.get("current_value", 0)
            mean_value = alert_payload.get("normal value (mean)", 0)
            upper_band = alert_payload.get("upper_band", 0)
            lower_band = alert_payload.get("lower_band", 0)
            status = str(alert_payload.get("status", "")).lower()
            expr = alert_payload.get("expr", "")
            usage = alert_payload.get("usage data", "")

            # Best-effort local inference for service to help LLM
            seed_service = infer_service_from_metric(alert_payload, fallback="unknown")

            # Import here to avoid dependency issues
            from langchain_core.messages import HumanMessage, SystemMessage

            llm_prompt = f"""
You are an SRE assistant. Infer a concise alert name and the service involved from a metric-based anomaly payload. Return ONLY valid JSON.

Metric Payload (normalized):
- metric: {metric_name_raw}
- current_value: {current_value}
- mean_value: {mean_value}
- upper_band: {upper_band}
- lower_band: {lower_band}
- status: {status}
- expr: {expr}
- seed_service: {seed_service}
- usage_data_preview: {usage}

Rules:
- alert_name: 2-5 words, clear (e.g., "High CPU Usage", "Memory Usage Anomaly").
- service: prefer container from usage data; otherwise infer from metric suffix (e.g., memory_usage_oemconnector -> oemconnector). Lowercase.
- If unsure, use seed_service when reasonable.
- Output strictly as JSON with keys: alert_name, service
"""

            messages = [
                SystemMessage(content="You produce only strict JSON responses for downstream parsing."),
                HumanMessage(content=llm_prompt)
            ]

            response = await llm.ainvoke(
                messages,
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_trace_id": predefined_trace_id,
                        "langfuse_tags": ["anomaly-agent", "alert-naming"],
                        "component": "alert_name_service_generation"
                    }
                }
            )

            raw = (response.content or "").strip()
            import re
            match = re.search(r"\{[\s\S]*\}", raw)
            json_text = match.group(0) if match else raw
            data = json.loads(json_text)
            alert_name = str(data.get("alert_name") or "").strip()
            service_name = str(data.get("service") or "").strip().lower()

            if not alert_name:
                alert_name = f"Anomaly: {metric_name_raw}"
            if not service_name or service_name == "unknown":
                service_name = seed_service

            span.update(
                input={
                    "metric": metric_name_raw,
                    "seed_service": seed_service,
                },
                output={"alert_name": alert_name, "service": service_name},
                metadata={"status": "success"}
            )

            return alert_name, service_name
        except Exception as e:
            span.update(
                output={"error": str(e), "fallback_used": True},
                metadata={"status": "fallback"}
            )
            # Fallback: deterministic
            try:
                service_name = infer_service_from_metric(alert_payload, fallback="unknown")
                metric_name_raw = alert_payload.get('metric', 'Unknown Metric')
                return f"Anomaly: {metric_name_raw}", service_name
            except Exception:
                return "Anomaly Alert", "unknown"


async def generate_alert_description(
    state: Dict[str, Any],
    alert_payload: Dict[str, Any],
    llm,
    langfuse_handler,
    predefined_trace_id: str
) -> None:
    """
    Generate alert description using LLM with comprehensive tracing.

    Args:
        state: Current agent state to update with description
        alert_payload: Alert payload data
        llm: Language model instance
        langfuse_handler: Langfuse callback handler
        predefined_trace_id: Trace ID for observability
    """
    from langfuse import get_client
    langfuse = get_client()

    with langfuse.start_as_current_span(name="alert-description-generation") as span:
        try:
            if "alerts" in alert_payload and alert_payload["alerts"]:
                first_alert = alert_payload["alerts"][0]
                labels = first_alert.get("labels", {})
                annotations = first_alert.get("annotations", {})

                span.update(
                    input={
                        "alertname": labels.get("alertname", "Unknown"),
                        "severity": labels.get("severity", "unknown"),
                        "application": labels.get("application", "unknown"),
                        "namespace": labels.get("namespace", "unknown")
                    },
                    metadata={"component": "llm_description_generation"}
                )

                # Import here to avoid dependency issues
                from langchain_core.messages import HumanMessage, SystemMessage

                # Create prompt for LLM
                llm_prompt = f"""
                Analyze this Grafana alert and create a concise technical description:
                Alert: {labels.get("alertname", "Unknown")}
                Severity: {labels.get("severity", "unknown")}
                Application: {labels.get("application", "unknown")}
                Namespace: {labels.get("namespace", "unknown")}
                Container: {labels.get("container_name", "unknown")}
                Pod: {labels.get("pod_name", "unknown")}
                Node: {labels.get("node", "unknown")}
                Annotations:
                Description: {annotations.get("description", "No description")}
                Summary: {annotations.get("summary", "No summary")}
                Values: {first_alert.get("values", {})}
                Create a clear, technical description of what's happening in 1-2 sentences.
                """

                messages = [
                    SystemMessage(content="You are an SRE expert. Create concise, technical alert descriptions in markdown format"),
                    HumanMessage(content=llm_prompt)
                ]

                response = await llm.ainvoke(
                    messages,
                    config={
                        "callbacks": [langfuse_handler],
                        "metadata": {
                            "langfuse_trace_id": predefined_trace_id,
                            "langfuse_tags": ["anomaly-agent", "description-generation"],
                            "component": "alert_description_generation"
                        }
                    }
                )

                state["description"] = response.content.strip()

                span.update(
                    output={
                        "description_generated": True,
                        "description_length": len(state["description"])
                    },
                    metadata={"status": "success"}
                )

                logger.info(f"Generated LLM description: {state['description']}")

        except Exception as e:
            span.update(
                output={"error": str(e), "fallback_used": True},
                metadata={"status": "fallback"}
            )
            logger.warning(f"Failed to generate LLM description: {e}")
            # Fallback to annotation description
            if "alerts" in alert_payload and alert_payload["alerts"]:
                annotations = alert_payload["alerts"][0].get("annotations", {})
                state["description"] = annotations.get("description", annotations.get("summary", "No description available"))


async def format_jira_description(
    state: Dict[str, Any],
    llm,
    langfuse_handler,
    predefined_trace_id: str
) -> str:
    """
    Generate Jira description using LLM based on alert payload with tracing.

    Args:
        state: Agent state containing alert information
        llm: Language model instance
        langfuse_handler: Langfuse callback handler
        predefined_trace_id: Trace ID for observability

    Returns:
        Formatted Jira description string
    """
    from langfuse import get_client
    langfuse = get_client()

    with langfuse.start_as_current_span(name="format-jira-description") as span:
        alert_payload = state.get("alert_payload", {})

        # Type checking for alert_payload
        if not isinstance(alert_payload, dict):
            logger.warning(f"Alert payload is not dict in format_jira_description: {type(alert_payload)}")
            alert_payload = {}

        span.update(
            input={
                "alertname": state.get('alertname', 'Unknown'),
                "service": state.get('service', 'Unknown'),
                "severity": state.get('severity', 'Unknown'),
                "payload_keys": list(alert_payload.keys()) if alert_payload else []
            },
            metadata={"component": "jira_description_formatter"}
        )

        try:
            # Import here to avoid dependency issues
            from langchain_core.messages import HumanMessage, SystemMessage

            # Create LLM prompt for generating Jira description (summarize metric payloads clearly)
            llm_prompt = f"""Create a concise, factual Jira incident description based solely on the provided alert data. Keep it brief and professional.
                Alert Data:
                - Alert: {state.get('alertname', 'Unknown')}
                - Service: {state.get('service', 'Unknown')}
                - Severity: {state.get('severity', 'Unknown')}
                - Status: {state.get('description', 'No description')}
                - Time: {state.get('timestamp', 'Unknown')}

                Raw Alert Payload:
                {json.dumps(alert_payload, indent=2)}

                Dependencies: {', '.join(state.get('service_dependencies', []))}

                Requirements:
                - Write 3-4 sentences maximum
                - State facts only from the alert data
                - If metric payload is present (e.g., fields like "usage data", "current_value", "normal value (mean)", "upper_band", "lower_band", "expr"), summarize it clearly:
                  • Name the metric and container (if present)
                  • Compare current vs mean with % deviation
                  • Mention bands and whether current breached the upper band
                  • Briefly describe the time window covered by the series (start→end UTC)
                - No emojis or excessive formatting
                - Focus on what happened and when, not recommendations
                - Mention affected service and severity
                Generate a brief, factual incident description."""

            messages = [
                SystemMessage(content="You are an expert SRE creating professional Jira incident descriptions. Be concise but comprehensive, focusing on actionable information."),
                HumanMessage(content=llm_prompt)
            ]

            response = await llm.ainvoke(
                messages,
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_trace_id": predefined_trace_id,
                        "langfuse_tags": ["anomaly-agent", "jira-description"],
                        "component": "jira_description_generation"
                    }
                }
            )

            description = response.content.strip()

            span.update(
                output={
                    "description_generated": True,
                    "description_length": len(description),
                    "method": "llm_generated"
                },
                metadata={"status": "success"}
            )

            logger.info(f"Generated LLM-based Jira description: {description[:200]}...")
            return description

        except Exception as e:
            span.update(
                output={"error": str(e), "fallback_used": True, "method": "template_fallback"},
                metadata={"status": "fallback"}
            )
            logger.warning(f"Failed to generate LLM description: {e}, falling back to template")
            # Fallback to template-based description
            return fallback_jira_description(state, alert_payload)


def fallback_jira_description(state: Dict[str, Any], alert_payload: dict) -> str:
    """
    Fallback template-based Jira description if LLM fails with tracing.

    Args:
        state: Agent state containing alert information
        alert_payload: Alert payload data

    Returns:
        Template-based Jira description string
    """
    from langfuse import get_client
    langfuse = get_client()

    with langfuse.start_as_current_span(name="fallback-jira-description") as span:
        span.update(
            input={"fallback_reason": "llm_generation_failed"},
            metadata={"component": "template_description"}
        )

        description = f"Hi Team,\n\nWe got an alert for {state.get('alertname', 'Unknown')}.\n\n"
        description += f"Service: {state.get('service', 'Unknown')}\n"
        description += f"Severity: {state.get('severity', 'Unknown')}\n"
        description += f"Description: {state.get('description', 'No description')}\n\n"

        # Check for metric data with safe access and summarize clearly
        metric_info_added = False
        if isinstance(alert_payload, dict):
            # Check for new payload format with direct metric fields
            if "metric" in alert_payload and "current_value" in alert_payload:
                current_value = alert_payload.get("current_value", 0)
                normal_value = alert_payload.get("normal value (mean)", 0)
                upper_band = alert_payload.get("upper_band", 0)
                lower_band = alert_payload.get("lower_band", 0)
                metric_name = alert_payload.get("metric", "Unknown metric")
                usage_data = alert_payload.get("usage data", "")
                expr = alert_payload.get("expr", "")

                # Calculate percentage deviation
                if normal_value > 0:
                    difference = current_value - normal_value
                    percentage_deviation = (difference / normal_value) * 100
                else:
                    percentage_deviation = 0

                # Derive series window if usage_data is present
                window_text = ""
                try:
                    series = usage_data[0].get("values", []) if isinstance(usage_data, list) and usage_data else []
                    if series:
                        start_ts = series[0][0]
                        end_ts = series[-1][0]
                        start_iso = datetime.fromtimestamp(int(start_ts), tz=timezone.utc).isoformat()
                        end_iso = datetime.fromtimestamp(int(end_ts), tz=timezone.utc).isoformat()
                        window_text = f" (window: {start_iso} → {end_iso})"
                except Exception:
                    window_text = ""

                breached_text = "breached upper band" if upper_band and current_value >= upper_band else "within bands"

                description += f"Metric Analysis ({metric_name}){window_text}:\n"
                description += f"- Current vs mean: {current_value:.4f} vs {normal_value:.4f} ({percentage_deviation:.2f}% deviation, {breached_text})\n"
                description += f"- Bands: lower={lower_band:.4f}, upper={upper_band:.4f}\n"
                if expr:
                    description += f"- Expr: {expr}\n"
                description += "\n"
                metric_info_added = True

        description += f"Dependencies: {', '.join(state.get('service_dependencies', []))}\n"
        description += f"Timestamp: {state['timestamp']}\n\n"
        description += "Please investigate and take necessary action."

        span.update(
            output={
                "description_length": len(description),
                "metric_info_included": metric_info_added
            },
            metadata={"status": "success"}
        )

        return description
