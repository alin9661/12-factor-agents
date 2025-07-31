"""
Mock LLM providers for testing.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from src.agents.llm_providers import OpenAIProvider, AnthropicProvider


class MockLLMProvider:
    """Base mock LLM provider for testing."""
    
    def __init__(self, responses: Optional[List[Union[str, Dict[str, Any]]]] = None):
        self.responses = responses or []
        self.response_index = 0
        self.calls = []
        self.model = "mock-model"
        self.logger = MagicMock()
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Mock response generation."""
        
        # Record the call for later assertions
        self.calls.append({
            "messages": messages,
            "response_model": response_model,
            "temperature": temperature,
        })
        
        # Return pre-configured response or default
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            
            if response_model and isinstance(response, dict):
                return response_model(**response)
            elif response_model and isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    return response_model(**parsed)
                except (json.JSONDecodeError, ValueError):
                    return response
            else:
                return response
        
        # Default response based on response_model
        if response_model:
            # Try to create a valid instance with minimal data
            if hasattr(response_model, '__fields__'):
                # Pydantic v1 style
                required_fields = {
                    name: field.default if field.default is not ... else self._get_default_value(field.type_)
                    for name, field in response_model.__fields__.items()
                    if field.required
                }
            else:
                # Pydantic v2 style
                required_fields = {
                    name: self._get_default_value(field.annotation) 
                    for name, field in response_model.model_fields.items()
                    if field.is_required()
                }
            
            return response_model(**required_fields)
        
        return "Mock response"
    
    def _get_default_value(self, field_type):
        """Get a default value for a field type."""
        if field_type == str:
            return "mock_value"
        elif field_type == int:
            return 0
        elif field_type == float:
            return 0.0
        elif field_type == bool:
            return False
        elif field_type == list:
            return []
        elif field_type == dict:
            return {}
        else:
            return None
    
    def reset(self):
        """Reset the mock state."""
        self.calls.clear()
        self.response_index = 0


class MockOpenAIProvider(MockLLMProvider):
    """Mock OpenAI provider for testing."""
    
    def __init__(self, responses: Optional[List[Union[str, Dict[str, Any]]]] = None):
        super().__init__(responses)
        self.model = "gpt-4o-mini"
        self.client = MagicMock()
        
        # Configure mock client
        self.client.chat.completions.create = AsyncMock()
        self.client.beta.chat.completions.parse = AsyncMock()


class MockAnthropicProvider(MockLLMProvider):
    """Mock Anthropic provider for testing."""
    
    def __init__(self, responses: Optional[List[Union[str, Dict[str, Any]]]] = None):
        super().__init__(responses)
        self.model = "claude-3-haiku-20240307"
        self.client = MagicMock()
        
        # Configure mock client
        self.client.messages.create = AsyncMock()


class FailingLLMProvider(MockLLMProvider):
    """Mock LLM provider that always fails for testing error handling."""
    
    def __init__(self, error_message: str = "Mock LLM API error"):
        super().__init__()
        self.error_message = error_message
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Always raise an error."""
        self.calls.append({
            "messages": messages,
            "response_model": response_model,
            "temperature": temperature,
        })
        raise Exception(self.error_message)


class SlowLLMProvider(MockLLMProvider):
    """Mock LLM provider that simulates slow responses for performance testing."""
    
    def __init__(self, delay_seconds: float = 5.0, responses: Optional[List[Union[str, Dict[str, Any]]]] = None):
        super().__init__(responses)
        self.delay_seconds = delay_seconds
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Simulate slow response."""
        import asyncio
        await asyncio.sleep(self.delay_seconds)
        return await super().generate_response(messages, response_model, temperature)