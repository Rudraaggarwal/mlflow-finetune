import os
from dotenv import load_dotenv
load_dotenv()

# Providers
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
import boto3
import logging



def mask_secret(secret, show_start=4, show_end=4):
    if not secret:
        return "Not Set"
    masked_part = '*' * max(0, len(secret) - (show_start + show_end))
    return secret[:show_start] + masked_part + secret[-show_end:]

class LLMConfig:
    """
    Select LLM at runtime via env:
      - LLM_PROVIDER=ollama | azure | bedrock | litellm | groq
    """

    @staticmethod
    def get_llm():
        provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        if provider == "ollama":
            return get_ollama_llm()
        elif provider == "azure":
            return get_azure_llm()
        elif provider == "bedrock":
            return get_bedrock_llm()
        elif provider == "litellm":
            return get_litellm_llm()
        elif provider == "groq":
            return get_groq_llm()
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    


def get_ollama_llm():
    base_url = os.getenv("OLLAMA_GPT_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    return ChatOllama(
        model=model,
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        base_url=base_url,
        # Optional auth headers if needed by a remote Ollama
        # headers={"Authorization": f"Bearer {os.getenv('OLLAMA_TOKEN')}"}
    )
    
def get_azure_llm():

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    print(f"Using Azure OpenAI with API Key: {mask_secret(api_key)}")
    print(f"Endpoint: {endpoint}, Deployment: {deployment}")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Azure config missing: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")

    api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    temperature = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.2"))
   
    print(f"Using Azure OpenAI Model: {model} with API Version: {api_version} and Temperature: {temperature}")
    

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,           
        openai_api_version=api_version,
        azure_deployment=deployment,       
        model=model,
        temperature=temperature,
    )
    




def get_bedrock_llm():
    # AWS Bedrock (optional path) - EXACT copy of working anomaly agent
    aws_bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    # Set bearer token unconditionally like anomaly agent
    os.environ['AWS_BEARER_TOKEN_BEDROCK']=aws_bearer_token
    region = os.getenv("AWS_REGION", "ap-south-1")
    model_id = os.getenv("BEDROCK_MODEL", "apac.anthropic.claude-3-7-sonnet-20250219-v1:0")
    
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )

    return ChatBedrock(
        client=client,
        model_id=model_id,
        temperature=float(os.getenv("BEDROCK_TEMPERATURE", "0.2")),
    )

def get_litellm_llm():
    """
    Configure LiteLLM for unified LLM API access
    Supports 100+ LLM providers through LiteLLM proxy
    """

    model = os.getenv("LITELLM_MODEL", "bedrock-claude-3-7-sonnet")  
    temperature = float(os.getenv("LITELLM_TEMPERATURE", "0.2"))
    
    # LiteLLM server configuration
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://192.168.101.144:32090/v1")  
    api_key = os.getenv("LITELLM_API_KEY", "") 
    
    print(f"Using LiteLLM with model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Proxy URL: {proxy_url}")
    print(f"API Key: {mask_secret(api_key)}")


    llm = ChatOpenAI(
        openai_api_base=proxy_url,
        api_key=api_key,
        model=model,
        temperature=0.1,
        tags=["rca_agent"]
    )

    return llm

def get_groq_llm():
    """
    Configure Groq LLM for fast inference
    """
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

    print(f"Using Groq with model: {model}")
    print(f"Temperature: {temperature}")
    print(f"API Key: {mask_secret(api_key)}")

    if not api_key:
        raise ValueError("Groq config missing: GROQ_API_KEY")

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
    )