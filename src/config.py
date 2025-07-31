"""
Configuration management for 12-factor agents.
Follows 12-factor app methodology for configuration.
"""

from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "12-factor-agents"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/12factor"
    redis_url: str = "redis://localhost:6379/0"
    
    # LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # HumanLayer
    humanlayer_api_key: Optional[str] = None
    humanlayer_verbose: bool = True
    
    # Context Management
    max_context_tokens: int = 100_000
    context_compression_threshold: float = 0.8
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    structured_logging: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()