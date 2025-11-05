"""
Prompt templates for Remediation Agent
All prompts used in the remediation workflow are defined here
"""


def get_remediation_analysis_prompt(incident: dict, rca_analysis: str, correlation_data: str) -> str:
    """
    Generate prompt for initial remediation analysis.

    Args:
        incident: Incident details dictionary
        rca_analysis: Root cause analysis text
        correlation_data: Correlation analysis text

    Returns:
        Formatted prompt string for remediation analysis
    """
    return f"""
You are an expert SRE remediation specialist. Generate SHORT, focused remediation steps based on RCA findings.

**INCIDENT DETAILS:**
- Incident ID: {incident.get('incident_id', 'Unknown')}
- Alert Name: {incident.get('alert_name', 'Unknown')}
- Description: {incident.get('description', 'No description')}
- Priority: {incident.get('priority', 'medium')}

**RCA ANALYSIS:**
{rca_analysis}

**CORRELATION DATA:**
{correlation_data if correlation_data else "No correlation data available"}

**TASK:** Generate comprehensive remediation recommendations with:
1. Stop Service Degradation - immediate actions to prevent further degradation
2. Emergency Containment - containment measures to isolate the issue
3. User Communication - what to communicate to users/stakeholders
4. System Stabilization - steps to stabilize the affected systems
5. Service Restart - procedures to restart affected services
6. Monitoring - monitoring adjustments needed
7. Code Snippet - technical fixes or configuration changes
8. Success Criteria - how to verify the fix worked
9. Rollback Plan - fallback procedures if remediation fails

Be specific with commands, configuration changes, and technical details.
Focus on actionable steps that can be implemented immediately.
"""


def get_remediation_system_message() -> str:
    """
    Get system message for remediation analysis.

    Returns:
        System message string
    """
    return "You are an expert SRE remediation specialist. Generate SHORT, focused remediation steps based on RCA findings. Be specific with commands and technical details."


def get_structured_remediation_prompt(incident: dict, rca_analysis: str, remediation_steps: str) -> str:
    """
    Generate prompt for structuring remediation output.

    Args:
        incident: Incident details dictionary
        rca_analysis: Root cause analysis text
        remediation_steps: Generated remediation steps text

    Returns:
        Formatted prompt string for structured remediation
    """
    return f"""
You are a specialized Remediation Agent for SRE operations.

**INCIDENT DETAILS:**
- Incident ID: {incident.get('incident_id', 'Unknown')}
- Alert Name: {incident.get('alert_name', 'Unknown')}
- Description: {incident.get('description', 'No description')}
- Priority: {incident.get('priority', 'medium')}

**RCA ANALYSIS:**
{rca_analysis}

**REMEDIATION ANALYSIS:**
{remediation_steps}

**TASK:** Create structured RemediationStructured response based on the remediation analysis.

Extract and organize remediation steps into these categories:
- stop_service_degradation: Actions to prevent further service degradation
- emergency_containment: Emergency containment measures
- user_communication: User communication requirements
- system_stabilization: System stabilization steps
- service_restart: Service restart procedures
- monitoring: Monitoring adjustments needed
- code_snippet: Code examples for technical fixes
- success_criteria: Criteria to verify successful resolution
- rollback_plan: Rollback procedures if actions fail

**IMPORTANT:** Only use information from the remediation analysis provided above.
"""


def get_structured_remediation_system_message() -> str:
    """
    Get system message for structured remediation output.

    Returns:
        System message string
    """
    return "You are an expert SRE remediation specialist. Create structured output from remediation text."


def get_fallback_remediation_text(incident: dict) -> str:
    """
    Generate fallback remediation text when LLM fails.

    Args:
        incident: Incident details dictionary

    Returns:
        Fallback remediation text
    """
    return f"""
**STOP SERVICE DEGRADATION:**
• Incident {incident.get('incident_id', 'Unknown')} requires immediate attention
• Monitor system metrics for further degradation
• Implement circuit breaker if applicable

**EMERGENCY CONTAINMENT:**
• Isolate affected services/pods if possible
• Scale down problematic components temporarily
• Implement traffic throttling if needed

**USER COMMUNICATION:**
• Notify stakeholders of ongoing incident
• Provide ETA for resolution if available
• Update status page with current information

**SYSTEM STABILIZATION:**
• Check system resources (CPU, memory, disk)
• Verify database connections and health
• Review recent deployments for potential rollback

**SERVICE RESTART:**
• Restart affected services using standard procedures
• Verify service health after restart
• Check dependency services for issues

**MONITORING:**
• Increase monitoring frequency for affected systems
• Set up additional alerts for related metrics
• Monitor for cascade failures

**CODE SNIPPET:**
# Check service status
kubectl get pods -n production
# Restart service if needed
kubectl rollout restart deployment/service-name

**SUCCESS CRITERIA:**
• System metrics return to normal ranges
• Error rates drop below threshold
• User-facing functionality restored

**ROLLBACK PLAN:**
• Rollback to previous stable version if fixes fail
• Restore from backup if data corruption occurred
• Escalate to senior engineering team if needed
"""


def get_jira_comment_user_prompt() -> str:
    """
    Get user prompt for Jira comment generation.

    Returns:
        User prompt string for Jira comment
    """
    return "**Task:** Create focused markdown comment showing ONLY the remediation analysis results based on the provided content and context."


def format_jira_comment_footer() -> str:
    """
    Format footer for Jira comments.

    Returns:
        Formatted footer string
    """
    from datetime import datetime
    return f"\n\n---\n*Remediation analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*"
