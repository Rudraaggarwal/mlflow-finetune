import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse import Langfuse, get_client
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not installed. Falling back to local prompts.")

logger = logging.getLogger(__name__)

class CorrelationLangfusePromptManager:
    """
    Manages fetching prompts from Langfuse at runtime for the correlation agent
    """

    def __init__(self):
        self.client = None
        self.enabled = self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initialize Langfuse client if credentials are available"""
        if not LANGFUSE_AVAILABLE:
            logger.info("Langfuse not available, using local prompts")
            return False

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.info("Langfuse credentials not found, using local prompts")
            return False

        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info("Correlation agent Langfuse client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            return False

    def get_prompt(self, prompt_name: str, variables: Dict[str, Any] = None) -> str:
        """
        Fetch a prompt from Langfuse with variable substitution
        Falls back to local prompts if Langfuse is not available
        """
        if not self.enabled or not self.client:
            return self._get_local_prompt(prompt_name, variables)

        try:
            # Fetch prompt from Langfuse
            prompt_response = self.client.get_prompt(prompt_name)

            if prompt_response and hasattr(prompt_response, 'prompt'):
                prompt_content = prompt_response.prompt

                # Substitute variables if provided
                if variables and isinstance(prompt_content, str):
                    for key, value in variables.items():
                        prompt_content = prompt_content.replace(f"{{{key}}}", str(value))

                logger.info(f"Successfully fetched prompt '{prompt_name}' from Langfuse")
                return prompt_content
            else:
                logger.warning(f"Prompt '{prompt_name}' not found in Langfuse, using local fallback")
                return self._get_local_prompt(prompt_name, variables)

        except Exception as e:
            logger.error(f"Error fetching prompt '{prompt_name}' from Langfuse: {e}")
            return self._get_local_prompt(prompt_name, variables)

    def _get_local_prompt(self, prompt_name: str, variables: Dict[str, Any] = None) -> str:
        """Fallback local prompts when Langfuse is not available"""
        local_prompts = {
            "correlation-agent": """
You are an expert SRE log correlation specialist. Your task is to analyze logs and identify entries relevant to the given alert.

**CONTEXT:**
- Alert Name: {alert_name}
- Service: {service_name}
- Namespace: {namespace}
- Time Range: Focus on logs around the alert time

**YOUR TASK:**
1. Analyze the provided logs for entries that correlate with the alert condition
2. Look for error patterns, service failures, and relevant events
3. Consider timing correlation with the alert
4. Identify root cause indicators

**ANALYSIS REQUIREMENTS:**
- Focus on logs that directly relate to the alert condition
- Include timestamp and service context
- Explain the correlation reasoning
- Identify patterns that led to the alert

Provide detailed correlation analysis with specific log entries and reasoning.
""",
            "metrics-system": """
You are an expert metrics analysis agent for SRE operations.

**CONTEXT:**
- Service: {service_name}
- Namespace: {namespace}
- Time Range: {start} to {end}

**YOUR TASK:**
1. Query relevant Prometheus metrics for the service
2. Analyze metrics that correlate with the alert
3. Identify metric thresholds and anomalies
4. Provide context on service performance

**ANALYSIS REQUIREMENTS:**
- Focus on key metrics: CPU, memory, error rates, latency
- Include actual metric values and thresholds
- Explain metric correlation with the alert
- Assess service health and performance impact

Provide comprehensive metrics analysis with specific values and insights.
"""
        }

        prompt_template = local_prompts.get(prompt_name, f"Prompt '{prompt_name}' not found in local prompts.")

        # Substitute variables if provided
        if variables and isinstance(prompt_template, str):
            for key, value in variables.items():
                prompt_template = prompt_template.replace(f"{{{key}}}", str(value))

        logger.info(f"Using local prompt for '{prompt_name}'")
        return prompt_template


# Global prompt manager instance
_prompt_manager = None

def get_correlation_prompt(prompt_name: str, variables: Dict[str, Any] = None) -> str:
    """
    Global function to get correlation prompts
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = CorrelationLangfusePromptManager()

    return _prompt_manager.get_prompt(prompt_name, variables)