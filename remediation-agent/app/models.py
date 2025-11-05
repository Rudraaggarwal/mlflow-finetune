"""
Database models for Remediation Agent
Contains essential Alert model for PostgreSQL operations
"""

from typing import List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RemediationRecommendation(BaseModel):
    """Structured remediation recommendation format"""
    stop_service_degradation: List[str] = Field(description="Actions to stop service degradation")
    emergency_containment: List[str] = Field(description="Emergency containment measures")
    user_communication: List[str] = Field(description="User communication requirements")
    system_stabilization: List[str] = Field(description="System stabilization steps")
    service_restart: List[str] = Field(description="Service restart procedures")
    monitoring: List[str] = Field(description="Monitoring adjustments needed")
    code_snippet: str = Field(description="Code examples for technical fixes")
    success_criteria: List[str] = Field(description="Criteria to verify successful resolution")
    rollback_plan: List[str] = Field(description="Rollback procedures if actions fail")

class RemediationStructured(BaseModel):
    """Structured remediation format"""
    recommendation: RemediationRecommendation = Field(description="Remediation recommendations")

class Alert(Base):
    """Simple Alert model for database operations"""
    __tablename__ = 'incidents'

    id = Column(String(255), primary_key=True)  # Use incident_id as primary key
    alert_name = Column(String(255), nullable=False)
    message = Column(Text)
    severity = Column(String(50))
    status = Column(String(50))
    logs = Column(Text)
    remediation = Column(Text)
    rca = Column(Text)
    affected_pods = Column(Text)
    cpu_graph = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)