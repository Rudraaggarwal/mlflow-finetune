import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not installed. Falling back to local prompts.")

logger = logging.getLogger(__name__)

class LangfusePromptManager:
    """
    Manages fetching prompts from Langfuse at runtime with variable substitution
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
            logger.info("Langfuse client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            return False
    
    def get_prompt(self, prompt_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Fetch prompt from Langfuse at runtime with variable substitution.
        Always fetches fresh - no caching.
        
        Args:
            prompt_name: Name of the prompt in Langfuse
            variables: Dictionary of variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        if not self.enabled or not self.client:
            logger.warning(f"Langfuse not enabled, cannot fetch prompt '{prompt_name}'")
            return self._get_fallback_prompt()
            
        try:
            logger.info(f"Fetching FRESH prompt '{prompt_name}' from Langfuse (no caching)")
            
            # Always get fresh prompt from Langfuse - no caching
            prompt = self.client.get_prompt(prompt_name)
            
            if not prompt:
                logger.error(f"Prompt '{prompt_name}' not found in Langfuse")
                return self._get_fallback_prompt()
                
            # Get the prompt content
            prompt_content = prompt.prompt
            logger.info(f"Retrieved prompt content ({len(prompt_content)} chars)")
            
            # Substitute variables if provided
            if variables:
                logger.info(f"Substituting {len(variables)} variables in prompt: {list(variables.keys())}")
                try:
                    prompt_content = prompt.compile(**variables)
                    logger.info("✓ Variable substitution successful")
                except KeyError as e:
                    logger.error(f"Variable substitution failed for prompt '{prompt_name}': missing variable {e}")
                    # Return prompt with placeholder intact if substitution fails
                except Exception as e:
                    logger.error(f"Error during variable substitution: {e}")
            else:
                logger.info("No variables provided for substitution")
                    
            logger.info(f"Successfully fetched and formatted prompt '{prompt_name}' (final length: {len(prompt_content)})")
            return prompt_content
            
        except Exception as e:
            logger.error(f"Failed to fetch prompt '{prompt_name}' from Langfuse: {e}")
            return self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        """Return fallback prompt when Langfuse is unavailable"""
        # Import the local prompt as fallback
        try:
            from app.prompts import anomaly_agent_prompt
            logger.info("Using local fallback prompt")
            return anomaly_agent_prompt
        except ImportError:
            logger.error("Could not import local fallback prompt")
            return "You are an anomaly detection agent. Process the alert and create a JIRA ticket."

# Global instance
prompt_manager = LangfusePromptManager()

def get_prompt(prompt_name: str = "anomaly-agent", variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to get prompt with variables
    
    Args:
        prompt_name: Name of the prompt in Langfuse (default: "anomaly-agent")  
        variables: Dictionary of variables to substitute
        
    Returns:
        Formatted prompt string
    """
    return prompt_manager.get_prompt(prompt_name, variables)