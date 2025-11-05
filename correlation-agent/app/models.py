from dataclasses import Field
from typing import *
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class CorrelatedLog(BaseModel):
    """Simple structured log entry with fixed fields"""
    timestamp: str = Field(description="Log timestamp")
    pod: Optional[str] = Field(default=None, description="Kubernetes pod name")
    instance: Optional[str] = Field(default=None, description="Instance identifier")
    level: Optional[str] = Field(default=None, description="Log level (ERROR, INFO, etc.)")
    stream: Optional[str] = Field(default=None, description="Log stream (stdout/stderr)")
    job: Optional[str] = Field(default=None, description="Job name")
    node: Optional[str] = Field(default=None, description="Node name")
    namespace: Optional[str] = Field(default=None, description="Namespace")
    message: str = Field(description="Complete log message with all details including transaction_id, error_code, thresholds, etc. separated by new lines")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for log correlation with alert")

class CorrelationArray(BaseModel):
    """Array of correlated logs"""
    correlated_logs: List[CorrelatedLog] = Field(description="Array of correlated log entries")
    is_metric_based: Optional[bool] = Field(default=True, description="Whether the alert is metric-based")


