from pydantic.v1 import Field
from pydantic_settings import BaseSettings

class ChainlitSettings(BaseSettings):
    """Settings for Chainlit."""
    MODELS_LIST: list[str] = Field(..., env="MODELS_LIST")
    ALLOWED_ORIGINS: list[str] = Field(..., json_schema_extra={"env": "ALLOWED_ORIGINS"})

class OllamaSettings(BaseSettings):
    """Settings for Ollama API."""
    OLLAMA_BASE_URL: str = Field(..., env="OLLAMA_BASE_URL")

class GeminiSettings(BaseSettings):
    """Settings for Gemini API."""
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")

class MistralSettings(BaseSettings):
    """Settings for Mistral API."""
    MISTRAL_API_KEY: str = Field(..., env="MISTRAL_API_KEY")

class PostgreSettings(BaseSettings):
    """Settings for PostgreSQL connection."""
    POSTGRE_CONNECTION_STRING: str = Field(..., env="POSTGRE_CONNECTION_STRING")

class OpenRouterSettings(BaseSettings):
    """Settings for OpenRouter API."""
    OPEN_ROUTER_API_URL: str = Field(..., env="OPEN_ROUTER_API_URL")
    OPEN_ROUTER_API_KEY: str = Field(..., env="OPEN_ROUTER_API_KEY")

class MCPSettings(BaseSettings):
    """Settings for MCP servers."""
    MCP_SERVERS: list = Field(..., env="MCP_SERVERS")
