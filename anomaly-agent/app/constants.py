"""Constants for the Anomaly Agent application."""

# Default Service Dependencies
DEFAULT_SERVICE_DEPENDENCIES = ["oemconnector", "catalogueserv"]

# Jira Custom Field IDs
JIRA_FIELD_PRODUCT = "customfield_10044"
JIRA_FIELD_INCIDENT_TYPE = "customfield_10344"
JIRA_FIELD_FAILURE_TYPE = "customfield_10335"
JIRA_FIELD_SEVERITY = "customfield_10065"
JIRA_FIELD_ASSIGNEE = "customfield_10537"
JIRA_FIELD_IMPACT = "customfield_10085"
JIRA_FIELD_DESCRIPTION = "customfield_10538"
JIRA_FIELD_DETECTION_TIME = "customfield_10237"
JIRA_FIELD_OCCURRENCE_TIME = "customfield_10539"
JIRA_FIELD_MANAGER = "customfield_10069"
JIRA_FIELD_BUSINESS_IMPACT = "customfield_10540"
JIRA_FIELD_ROOT_CAUSE = "customfield_10337"
JIRA_FIELD_RESOLUTION_TYPE = "customfield_10338"

# Jira Field Values
JIRA_PRODUCT_VALUE = "EMI"
JIRA_INCIDENT_TYPE_VALUE = "Alert"
JIRA_FAILURE_TYPE_VALUE = "Single Component Failure"
JIRA_SEVERITY_VALUE = "Sev 1"
JIRA_IMPACT_VALUE = "Moderate / Limited"
JIRA_BUSINESS_IMPACT_VALUE = "Not Applicable"
JIRA_ROOT_CAUSE_VALUE = "Change Failure"
JIRA_RESOLUTION_TYPE_VALUE = "Resolved (Permanently)"

# Jira Account IDs
JIRA_DEFAULT_ASSIGNEE_ID = "712020:e3cefd80-5e61-4a6a-a72c-2e504b08e11c"
JIRA_DEFAULT_MANAGER_ID = "557058:ca5a0eee-2f7d-478f-9358-42de1e3c64fb"
