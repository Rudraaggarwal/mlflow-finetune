"""
Utility functions for Correlation Agent.
"""

import logging
from datetime import datetime
from typing import Any
from langfuse import get_client

from .models import CorrelationArray, CorrelatedLog
from .logging_utils import get_timestamp

logger = logging.getLogger(__name__)
langfuse = get_client()


def sanitize_unicode(text: str) -> str:
    """Sanitize Unicode characters for Jira compatibility."""
    try:
        # Replace problematic Unicode characters
        sanitized = text.encode('ascii', 'ignore').decode('ascii')

        # Replace common Unicode punctuation with safe ASCII equivalents
        replacements = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "–": "-",
            "—": "-",
            "…": "..."
        }

        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        return sanitized

    except Exception as e:
        logger.error(f"Error sanitizing Unicode: {e}")
        return text


def create_fallback_correlation_structure(correlation_result: str) -> CorrelationArray:
    """Create fallback structured correlation from text result."""
    try:
        truncated_message = correlation_result[:500] + "..." if len(correlation_result) > 500 else correlation_result

        fallback_correlation = CorrelationArray(correlated_logs=[
            CorrelatedLog(
                timestamp=datetime.now().isoformat(),
                message=truncated_message,
                level="INFO",
                reasoning="Correlation analysis completed using fallback structure"
            )
        ])

        logger.info(f"Created fallback correlation structure")
        return fallback_correlation

    except Exception as e:
        logger.error(f"Failed to create fallback correlation structure: {e}")
        return CorrelationArray(correlated_logs=[])


def extract_metric_decision_from_correlation(structured_correlation: Any) -> bool:
    """Extract metric-based decision from correlation analysis."""
    try:
        decision = True  # Default decision

        if isinstance(structured_correlation, dict):
            decision = structured_correlation.get('is_metric_based', True)
        elif hasattr(structured_correlation, 'model_dump'):
            data = structured_correlation.model_dump()
            decision = data.get('is_metric_based', True)
        else:
            decision = True

        return decision

    except Exception as e:
        logger.error(f"Error extracting metric decision: {e}")
        return True  # Safe fallback


def get_completion_stats(state: dict) -> dict:
    """Gather workflow completion statistics."""
    return {
        "incident_key": state.get('incident_key'),
        "log_correlation_completed": bool(state.get('log_correlation_result')),
        "metrics_correlation_completed": bool(state.get('metrics_correlation_result')),
        "correlation_summary_generated": bool(state.get('correlation_summary')),
        "redis_stored": bool(state.get('redis_stored')),
        "postgres_stored": bool(state.get('postgres_stored')),
        "jira_updated": bool(state.get('jira_updated')),
        "workflow_completed": True,
        "completion_timestamp": datetime.now().isoformat()
    }
