"""
Consolidated prompts for RCA Agent
All prompts used across the application are defined here
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not installed. Falling back to local prompts.")

logger = logging.getLogger(__name__)


# Local prompt templates
LOCAL_PROMPTS = {
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
""",

    "rca-generation": """
You are an expert SRE root cause analysis specialist. Analyze the following incident and provide comprehensive RCA.

**INCIDENT DETAILS:**
- Incident ID: {incident_id}
- Alert Name: {alert_name}
- Description: {description}
- Priority: {priority}

**CORRELATION DATA:**
{correlation_data}

**METRICS ANALYSIS:**
{metrics_analysis}

{reference_rcas_section}

**TASK:** Provide comprehensive root cause analysis with:
1. Incident Summary - what basically happened i.e root cause(should be based on logs and metrics). **IMPORTANT:** If using reference RCAs, mention whether this report was made with assistance from Human or SRE Agent
2. Root Cause Analysis - why it happened (correlate logs with metrics)
3. Log Evidence - supporting evidence from logs
4. Metrics Evidence - supporting evidence from metrics analysis

Use both correlation data and metrics analysis to provide a complete picture.
Format your response in clear sections with bullet points.
""",

    "rca-guided-generation": """
You are an expert SRE root cause analysis specialist. You have found a very similar past incident with high confidence (>90% similarity).{reference_source_note}

**CURRENT INCIDENT DETAILS:**
- Incident ID: {incident_id}
- Alert Name: {alert_name}
- Description: {description}
- Priority: {priority}

**CURRENT INCIDENT CORRELATION SUMMARY:**
{correlation_summary}

**SIMILAR PAST INCIDENT RCA (FOR REFERENCE):**
{existing_rca}

{additional_references_section}

**CRITICAL INSTRUCTIONS:**
1. Generate an RCA for the CURRENT incident that is GUIDED BY the similar past incident's RCA
2. Adapt the past RCA patterns and insights to fit the current incident's specific details
3. Use the correlation summary to understand the current incident's specific context
4. Your RCA should tend towards the same root cause patterns as the reference RCA, but be specific to the current incident
5. If the incidents are truly similar, the root causes should align closely
{human_weightage_note}

**TASK:** Provide comprehensive root cause analysis with:
1. **Incident Summary** - what happened in THIS current incident based on correlation summary
2. **Root Cause Analysis** - why it happened (guided by the similar incident's patterns)
3. **Evidence from Current Incident** - supporting evidence from the current correlation summary
4. **Similarity Assessment** - how this incident relates to the similar past incident
5. **Recommended Actions** - immediate steps to resolve this current incident

**IMPORTANT:** Focus on the current incident but be heavily influenced by the successful RCA patterns from the similar incident.

**ATTRIBUTION REQUIREMENT:** Include this acknowledgment at the end of your RCA:
{attribution_text}

Format your response in clear sections with bullet points.
""",

    "rca-structuring": """
You are a specialized Root Cause Analysis (RCA) Agent for SRE operations.

**GENERATED RCA ANALYSIS:**
{rca_analysis}

**TASK:** Convert the above RCA analysis into a structured format.
You must return a valid JSON output that strictly follows this schema:
- "incident_summary": ["string", "string", "string"] - Extract key summary points. **IMPORTANT:** Include any mention of assistance from Human or SRE Agent if referenced in the analysis
- "root_cause_analysis": ["string", "string"] - Extract root cause points
- "log_evidence": ["string", "string"] - Extract evidence points

**CRITICAL RULES:**
- Each field MUST be an array of strings, never a single string
- Split multi-line content into separate array elements
- Remove bullet points ("-", "*") from the strings
- Each array element should be a complete sentence
- Extract actual content from the RCA analysis provided above
- If the RCA analysis mentions assistance from Human or SRE Agent, include this in the incident_summary

**EXAMPLE FORMAT:**
{{
    "incident_summary": [
        "Service experienced downtime affecting user authentication",
        "Error rates increased significantly during the incident window",
        "Multiple dependent services were impacted"
    ],
    "root_cause_analysis": [
        "Database connection pool exhaustion due to connection leak",
        "Insufficient monitoring of connection pool metrics led to delayed detection"
    ],
    "log_evidence": [
        "ERROR: Connection pool exhausted at 2024-01-15 14:30:00",
        "WARNING: High connection count observed in database logs"
    ]
}}

Parse the RCA analysis above and structure it according to this format.
""",

    "memgraph-insert": """
You need to insert a new incident log and its RCA analysis into the memgraph database.

**TASK:** Call the insert_log_tool with the following information:

**Parameters for insert_log_tool:**
- log: "{correlation_summary}"
- rca: "{rca_analysis}"
- source: "{source}"
- alert_type: "error"
- namespace: "{namespace}"

**IMPORTANT:** You must call the insert_log_tool exactly once with these parameters.
The tool will return a node_id which represents the unique identifier in the memgraph database.
""",

    "jira-formatter": """
You are a Jira comment formatter for SRE operations.

**CONTEXT:**
- Analysis Type: {analysis_type}
- Alert Name: {alert_name}
- Severity: {severity}
- Title: {title}

**ANALYSIS CONTENT:**
{analysis_content}

**TASK:** Create a focused markdown comment showing ONLY the {analysis_type} analysis results based on the provided content and context.

Format your response with clear sections and bullet points suitable for Jira comments.
"""
}


# Attribution templates
ATTRIBUTION_TEMPLATES = {
    "human": "**ATTRIBUTION:** This RCA was generated with the help of human-assisted feedback from previous similar incidents analyzed by human experts.",
    "ai": "**ATTRIBUTION:** This RCA was generated with guidance from previous similar incidents analyzed by AI systems.",
    "sre_agent": "**ATTRIBUTION:** This RCA was generated with guidance from previous similar incidents analyzed by SRE Agent.",
    "default": "**ATTRIBUTION:** This RCA was generated with guidance from previous similar incidents."
}


# Reference source notes
REFERENCE_SOURCE_NOTES = {
    "human": """

**IMPORTANT NOTE:** This RCA is generated with the help of human-assisted feedback from a previous similar incident that was analyzed by human experts. The reference RCA below contains human insights and should be given higher weightage in your analysis.
""",
    "ai": """

**REFERENCE NOTE:** This RCA is guided by a previous similar incident that was analyzed by AI. Use this reference to maintain consistency in analysis patterns.
""",
    "sre_agent": """

**REFERENCE NOTE:** This RCA is guided by a previous similar incident that was analyzed by SRE Agent. Use this reference to maintain consistency in analysis patterns.
"""
}


# System prompts
SYSTEM_PROMPTS = {
    "rca-generation": "You are an expert SRE root cause analysis specialist. Provide comprehensive RCA analysis using both log correlation and metrics analysis.",
    "rca-guided-generation": "You are an expert SRE root cause analysis specialist. Generate RCA guided by similar past incidents to ensure consistency and accuracy.",
    "rca-structuring": "You are an expert at converting RCA analysis text into structured JSON format with arrays of strings.",
    "jira-comment": "You are a Jira comment formatter for SRE operations. Create clear, concise comments suitable for incident tracking."
}


class PromptManager:
    """
    Manages prompts from Langfuse at runtime with local fallbacks
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
            logger.info("Prompt manager Langfuse client initialized successfully")
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
            prompt_response = self.client.get_prompt(prompt_name)

            if prompt_response and hasattr(prompt_response, 'prompt'):
                prompt_content = prompt_response.prompt

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
        prompt_template = LOCAL_PROMPTS.get(
            prompt_name,
            f"Prompt '{prompt_name}' not found in local prompts."
        )

        if variables and isinstance(prompt_template, str):
            for key, value in variables.items():
                prompt_template = prompt_template.replace(f"{{{key}}}", str(value))

        logger.info(f"Using local prompt for '{prompt_name}'")
        return prompt_template

    def get_system_prompt(self, prompt_type: str) -> str:
        """Get system prompt for a specific type"""
        return SYSTEM_PROMPTS.get(prompt_type, "You are an expert SRE assistant.")

    def get_attribution_text(self, generated_by: str) -> str:
        """Get attribution text based on who created the reference RCA"""
        generated_by_key = generated_by.lower().replace(" ", "_")
        return ATTRIBUTION_TEMPLATES.get(generated_by_key, ATTRIBUTION_TEMPLATES["default"])

    def get_reference_source_note(self, generated_by: str) -> str:
        """Get reference source note based on who created the reference RCA"""
        generated_by_key = generated_by.lower().replace(" ", "_")
        return REFERENCE_SOURCE_NOTES.get(generated_by_key, "")


# Global prompt manager instance
_prompt_manager = None


def get_prompt(prompt_name: str, variables: Dict[str, Any] = None) -> str:
    """Global function to get prompts"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager.get_prompt(prompt_name, variables)


def get_system_prompt(prompt_type: str) -> str:
    """Global function to get system prompts"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager.get_system_prompt(prompt_type)


def get_attribution_text(generated_by: str) -> str:
    """Global function to get attribution text"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager.get_attribution_text(generated_by)


def get_reference_source_note(generated_by: str) -> str:
    """Global function to get reference source note"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager.get_reference_source_note(generated_by)


# Maintain backward compatibility with correlation agent
def get_correlation_prompt(prompt_name: str, variables: Dict[str, Any] = None) -> str:
    """Backward compatibility function for correlation agent"""
    return get_prompt(prompt_name, variables)
