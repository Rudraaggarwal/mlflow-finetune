"""
Simple database models for RCA Agent
Contains only essential Alert model for PostgreSQL operations
"""

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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
    rca = Column(Text)  # Full RCA analysis text
    affected_pods = Column(Text)
    cpu_graph = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)