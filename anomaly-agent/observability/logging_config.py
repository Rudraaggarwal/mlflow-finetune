"""
Simple logging configuration for the anomaly agent.
Basic timestamp-based logging for terminal output.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict, Optional


def configure_logging(log_level: str = None) -> None:
    """Configure simple timestamp-based logging for the anomaly agent."""

    # Get log level from environment or parameter
    log_level = log_level or os.getenv('LOG_LEVEL', 'INFO').upper()

    # Configure basic logging with timestamp
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Set specific logger levels to reduce noise
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.INFO)

    # Create main logger and log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"{datetime.now()} - Simple logging configured")


def get_logger(name: str = None) -> logging.Logger:
    """Get a simple logger instance."""
    return logging.getLogger(name or __name__)


# Initialize logging when module is imported
if not os.getenv('SKIP_LOGGING_INIT'):
    configure_logging()