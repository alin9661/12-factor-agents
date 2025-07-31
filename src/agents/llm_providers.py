"""
LLM provider implementations for different AI services.
Supports OpenAI, Anthropic, and other providers.
"""

import json
from typing import Dict, List, Optional, Type, Union

import structlog
from pydantic import BaseModel

from ..config import settings

logger = structlog.get_logger(__name__)


class OpenAIProvider:
    """OpenAI LLM provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider")
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key or settings.openai_api_key
        )
        self.model = model
        self.logger = logger.bind(provider="openai", model=model)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Generate response using OpenAI API with optional structured output."""
        
        self.logger.debug("Generating response", message_count=len(messages))
        
        try:
            if response_model:
                # Use structured outputs with Pydantic model
                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_model,
                    temperature=temperature,
                )
                
                if response.choices[0].parsed:
                    return response.choices[0].parsed
                else:
                    # Fallback to regular response if parsing failed
                    content = response.choices[0].message.content
                    self.logger.warning("Structured parsing failed, falling back to text", content=content)
                    return content
            else:
                # Regular text response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
                
        except Exception as e:
            self.logger.error("OpenAI API error", error=str(e), exc_info=True)
            raise


class AnthropicProvider:
    """Anthropic Claude LLM provider implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic provider")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key
        )
        self.model = model
        self.logger = logger.bind(provider="anthropic", model=model)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Generate response using Anthropic API."""
        
        self.logger.debug("Generating response", message_count=len(messages))
        
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            # Prepare the request
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": 4000,
                "temperature": temperature,
            }
            
            if system_message:
                request_params["system"] = system_message
            
            # Add structured output instructions if model is provided
            if response_model:
                schema = response_model.model_json_schema()
                structured_prompt = f"""
Please respond with a valid JSON object that matches this exact schema:

{json.dumps(schema, indent=2)}

Your response must be valid JSON that can be parsed directly. Do not include any additional text or explanation.
"""
                if anthropic_messages:
                    anthropic_messages[-1]["content"] += "\n\n" + structured_prompt
                else:
                    anthropic_messages.append({"role": "user", "content": structured_prompt})
            
            response = await self.client.messages.create(**request_params)
            content = response.content[0].text
            
            if response_model:
                try:
                    # Try to parse as JSON and create model instance
                    parsed_json = json.loads(content)
                    return response_model(**parsed_json)
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning("Failed to parse structured response", content=content, error=str(e))
                    return content
            
            return content
            
        except Exception as e:
            self.logger.error("Anthropic API error", error=str(e), exc_info=True)
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> Union[OpenAIProvider, AnthropicProvider]:
        """Create an LLM provider based on type."""
        
        if provider_type.lower() == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_type.lower() == "anthropic":
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    @staticmethod
    def create_default_provider() -> Union[OpenAIProvider, AnthropicProvider]:
        """Create a default provider based on available API keys."""
        
        if settings.openai_api_key:
            return OpenAIProvider()
        elif settings.anthropic_api_key:
            return AnthropicProvider()
        else:
            # For development, return OpenAI provider (will fail without API key)
            logger.warning("No API keys configured, using OpenAI provider")
            return OpenAIProvider()