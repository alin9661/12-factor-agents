"""
Unit tests for LLM provider implementations.

These tests verify the LLM provider abstraction layer and ensure proper
handling of different AI service APIs.
"""

import json
import pytest
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from src.agents.llm_providers import OpenAIProvider, AnthropicProvider, LLMProviderFactory
from src.tools.schemas import ExecuteTool


class TestResponseModel(BaseModel):
    """Test Pydantic model for structured responses."""
    intent: str
    message: str
    confidence: float = 0.9


@pytest.mark.unit
class TestOpenAIProvider:
    """Test cases for OpenAI provider implementation."""
    
    def test_openai_provider_initialization_with_api_key(self):
        """Test OpenAI provider initialization with explicit API key."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = MagicMock()
            
            provider = OpenAIProvider(api_key="test_key", model="gpt-4")
            
            assert provider.model == "gpt-4"
            assert hasattr(provider, 'client')
            assert hasattr(provider, 'logger')
            mock_openai.AsyncOpenAI.assert_called_once_with(api_key="test_key")

    def test_openai_provider_initialization_from_settings(self):
        """Test OpenAI provider initialization using settings."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            with patch('src.agents.llm_providers.settings') as mock_settings:
                mock_settings.openai_api_key = "settings_key"
                mock_openai.AsyncOpenAI.return_value = MagicMock()
                
                provider = OpenAIProvider()
                
                assert provider.model == "gpt-4o-mini"  # default model
                mock_openai.AsyncOpenAI.assert_called_once_with(api_key="settings_key")
    
    def test_openai_provider_missing_import(self):
        """Test behavior when openai package is not installed."""
        with patch('src.agents.llm_providers.openai', None):
            with patch.dict('sys.modules', {'openai': None}):
                with pytest.raises(ImportError, match="openai package is required"):
                    OpenAIProvider()
    
    @pytest.mark.asyncio
    async def test_openai_generate_response_text_only(self):
        """Test text-only response generation."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            # Mock the client
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            # Mock the response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "This is a test response"
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider(api_key="test_key")
            
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Say hello"}
            ]
            
            response = await provider.generate_response(messages, temperature=0.5)
            
            assert response == "This is a test response"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5
            )
    
    @pytest.mark.asyncio
    async def test_openai_generate_response_structured_output(self):
        """Test structured output response generation."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            # Mock the client
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            # Mock the structured response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_parsed = TestResponseModel(intent="test", message="structured response")
            mock_response.choices[0].parsed = mock_parsed
            mock_client.beta.chat.completions.parse.return_value = mock_response
            
            provider = OpenAIProvider(api_key="test_key")
            
            messages = [{"role": "user", "content": "Generate structured response"}]
            
            response = await provider.generate_response(
                messages,
                response_model=TestResponseModel,
                temperature=0.1
            )
            
            assert isinstance(response, TestResponseModel)
            assert response.intent == "test"
            assert response.message == "structured response"
            
            mock_client.beta.chat.completions.parse.assert_called_once_with(
                model="gpt-4o-mini",
                messages=messages,
                response_format=TestResponseModel,
                temperature=0.1
            )
    
    @pytest.mark.asyncio
    async def test_openai_structured_parsing_fallback(self):
        """Test fallback when structured parsing fails."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            # Mock the client
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            # Mock failed parsing (parsed is None)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].parsed = None
            mock_response.choices[0].message.content = "Fallback text response"
            mock_client.beta.chat.completions.parse.return_value = mock_response
            
            provider = OpenAIProvider(api_key="test_key")
            
            response = await provider.generate_response(
                [{"role": "user", "content": "test"}],
                response_model=TestResponseModel
            )
            
            assert response == "Fallback text response"
    
    @pytest.mark.asyncio
    async def test_openai_api_error_handling(self):
        """Test handling of OpenAI API errors."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            # Mock the client to raise an exception
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            provider = OpenAIProvider(api_key="test_key")
            
            with pytest.raises(Exception, match="API Error"):
                await provider.generate_response([{"role": "user", "content": "test"}])


@pytest.mark.unit
class TestAnthropicProvider:
    """Test cases for Anthropic provider implementation."""
    
    def test_anthropic_provider_initialization_with_api_key(self):
        """Test Anthropic provider initialization with explicit API key."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = MagicMock()
            
            provider = AnthropicProvider(api_key="test_key", model="claude-3-opus-20240229")
            
            assert provider.model == "claude-3-opus-20240229"
            assert hasattr(provider, 'client')
            assert hasattr(provider, 'logger')
            mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="test_key")
    
    def test_anthropic_provider_initialization_from_settings(self):
        """Test Anthropic provider initialization using settings."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            with patch('src.agents.llm_providers.settings') as mock_settings:
                mock_settings.anthropic_api_key = "settings_key"
                mock_anthropic.AsyncAnthropic.return_value = MagicMock()
                
                provider = AnthropicProvider()
                
                assert provider.model == "claude-3-haiku-20240307"  # default model
                mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="settings_key")
    
    def test_anthropic_provider_missing_import(self):
        """Test behavior when anthropic package is not installed."""
        with patch('src.agents.llm_providers.anthropic', None):
            with patch.dict('sys.modules', {'anthropic': None}):
                with pytest.raises(ImportError, match="anthropic package is required"):
                    AnthropicProvider()
    
    @pytest.mark.asyncio
    async def test_anthropic_generate_response_text_only(self):
        """Test text-only response generation."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            # Mock the client
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            # Mock the response
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "This is Claude's response"
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Say hello"}
            ]
            
            response = await provider.generate_response(messages, temperature=0.5)
            
            assert response == "This is Claude's response"
            
            # Verify the call was made with proper format
            call_args = mock_client.messages.create.call_args
            assert call_args[1]['model'] == "claude-3-haiku-20240307"
            assert call_args[1]['temperature'] == 0.5
            assert call_args[1]['max_tokens'] == 4000
            assert call_args[1]['system'] == "You are helpful"
            assert len(call_args[1]['messages']) == 1
            assert call_args[1]['messages'][0] == {"role": "user", "content": "Say hello"}
    
    @pytest.mark.asyncio
    async def test_anthropic_message_format_conversion(self):
        """Test conversion of OpenAI-style messages to Anthropic format."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Response"
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            # Messages with system message
            messages = [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "user", "content": "Follow up"}
            ]
            
            await provider.generate_response(messages)
            
            call_args = mock_client.messages.create.call_args[1]
            
            # System message should be separate
            assert call_args['system'] == "System prompt"
            
            # Other messages should be in messages array (excluding system)
            assert len(call_args['messages']) == 3
            assert call_args['messages'][0] == {"role": "user", "content": "User message"}
            assert call_args['messages'][1] == {"role": "assistant", "content": "Assistant response"}
            assert call_args['messages'][2] == {"role": "user", "content": "Follow up"}
    
    @pytest.mark.asyncio
    async def test_anthropic_generate_response_structured_output(self):
        """Test structured output response generation."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            # Mock response with valid JSON
            valid_json = '{"intent": "execute", "message": "structured response", "confidence": 0.95}'
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = valid_json
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            messages = [{"role": "user", "content": "Generate structured response"}]
            
            response = await provider.generate_response(
                messages,
                response_model=TestResponseModel,
                temperature=0.1
            )
            
            assert isinstance(response, TestResponseModel)
            assert response.intent == "execute"
            assert response.message == "structured response"
            assert response.confidence == 0.95
            
            # Verify schema was included in the prompt
            call_args = mock_client.messages.create.call_args[1]
            final_message = call_args['messages'][-1]['content']
            assert "JSON object" in final_message
            assert "schema" in final_message
    
    @pytest.mark.asyncio
    async def test_anthropic_structured_response_parsing_failure(self):
        """Test fallback when JSON parsing fails for structured response."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            # Mock response with invalid JSON
            invalid_json = "This is not JSON at all"
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = invalid_json
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            response = await provider.generate_response(
                [{"role": "user", "content": "test"}],
                response_model=TestResponseModel
            )
            
            # Should fallback to text response
            assert response == invalid_json
    
    @pytest.mark.asyncio
    async def test_anthropic_api_error_handling(self):
        """Test handling of Anthropic API errors."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("Claude API Error")
            
            provider = AnthropicProvider(api_key="test_key")
            
            with pytest.raises(Exception, match="Claude API Error"):
                await provider.generate_response([{"role": "user", "content": "test"}])
    
    @pytest.mark.asyncio
    async def test_anthropic_no_system_message(self):
        """Test Anthropic provider with messages containing no system message."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Response without system"
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            # Messages without system message
            messages = [
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant response"}
            ]
            
            response = await provider.generate_response(messages)
            
            assert response == "Response without system"
            
            call_args = mock_client.messages.create.call_args[1]
            # Should not have system parameter
            assert 'system' not in call_args or call_args['system'] == ""
            assert len(call_args['messages']) == 2


@pytest.mark.unit
class TestLLMProviderFactory:
    """Test cases for LLM provider factory."""
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider via factory."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = MagicMock()
            
            provider = LLMProviderFactory.create_provider("openai", api_key="test_key", model="gpt-4")
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-4"
    
    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider via factory."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = MagicMock()
            
            provider = LLMProviderFactory.create_provider("anthropic", api_key="test_key", model="claude-3-opus-20240229")
            
            assert isinstance(provider, AnthropicProvider)
            assert provider.model == "claude-3-opus-20240229"
    
    def test_create_provider_case_insensitive(self):
        """Test that provider creation is case insensitive."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = MagicMock()
            
            provider1 = LLMProviderFactory.create_provider("OPENAI")
            provider2 = LLMProviderFactory.create_provider("OpenAI")
            provider3 = LLMProviderFactory.create_provider("openai")
            
            assert all(isinstance(p, OpenAIProvider) for p in [provider1, provider2, provider3])
    
    def test_create_provider_unsupported_type(self):
        """Test error when creating unsupported provider type."""
        with pytest.raises(ValueError, match="Unsupported provider type: unsupported"):
            LLMProviderFactory.create_provider("unsupported")
    
    def test_create_default_provider_openai_available(self):
        """Test creating default provider when OpenAI key is available."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            with patch('src.agents.llm_providers.settings') as mock_settings:
                mock_settings.openai_api_key = "openai_key"
                mock_settings.anthropic_api_key = None
                mock_openai.AsyncOpenAI.return_value = MagicMock()
                
                provider = LLMProviderFactory.create_default_provider()
                
                assert isinstance(provider, OpenAIProvider)
    
    def test_create_default_provider_anthropic_fallback(self):
        """Test creating default provider falls back to Anthropic when OpenAI unavailable."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            with patch('src.agents.llm_providers.settings') as mock_settings:
                mock_settings.openai_api_key = None
                mock_settings.anthropic_api_key = "anthropic_key"
                mock_anthropic.AsyncAnthropic.return_value = MagicMock()
                
                provider = LLMProviderFactory.create_default_provider()
                
                assert isinstance(provider, AnthropicProvider)
    
    def test_create_default_provider_no_keys(self):
        """Test creating default provider when no API keys are configured."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            with patch('src.agents.llm_providers.settings') as mock_settings:
                mock_settings.openai_api_key = None
                mock_settings.anthropic_api_key = None
                mock_openai.AsyncOpenAI.return_value = MagicMock()
                
                # Should still create OpenAI provider as fallback (will fail without key)
                provider = LLMProviderFactory.create_default_provider()
                
                assert isinstance(provider, OpenAIProvider)


@pytest.mark.unit
class TestLLMProviderEdgeCases:
    """Test edge cases and error conditions for LLM providers."""
    
    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """Test provider behavior with empty messages list."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Empty messages response"
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider(api_key="test_key")
            
            response = await provider.generate_response([])
            
            assert response == "Empty messages response"
            # Verify it was called with empty messages
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['messages'] == []
    
    @pytest.mark.asyncio
    async def test_very_large_messages_list(self):
        """Test provider behavior with very large messages list."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Large messages response"
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            # Create a very large messages list
            large_messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
                for i in range(1000)
            ]
            
            response = await provider.generate_response(large_messages)
            
            assert response == "Large messages response"
            # Should handle the large message list without errors
            mock_client.messages.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_malformed_structured_response_model(self):
        """Test behavior with malformed response model."""
        class MalformedModel(BaseModel):
            # Missing required field annotations or validation
            pass
        
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            # Mock response parsing failure
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].parsed = None
            mock_response.choices[0].message.content = "Fallback response"
            mock_client.beta.chat.completions.parse.return_value = mock_response
            
            provider = OpenAIProvider(api_key="test_key")
            
            # Should not crash, should fallback to text
            response = await provider.generate_response(
                [{"role": "user", "content": "test"}],
                response_model=MalformedModel
            )
            
            assert response == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self):
        """Test provider behavior during network timeouts."""
        import asyncio
        
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client
            
            # Simulate timeout
            mock_client.chat.completions.create.side_effect = asyncio.TimeoutError("Request timed out")
            
            provider = OpenAIProvider(api_key="test_key")
            
            with pytest.raises(asyncio.TimeoutError):
                await provider.generate_response([{"role": "user", "content": "test"}])
    
    def test_provider_with_invalid_model_name(self):
        """Test provider initialization with invalid model name."""
        with patch('src.agents.llm_providers.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = MagicMock()
            
            # Should still initialize (validation happens at API call time)
            provider = OpenAIProvider(api_key="test_key", model="invalid-model-name-12345")
            
            assert provider.model == "invalid-model-name-12345"
    
    @pytest.mark.asyncio
    async def test_anthropic_structured_response_partial_json(self):
        """Test Anthropic provider with partially valid JSON in structured response."""
        with patch('src.agents.llm_providers.anthropic') as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            
            # Partial JSON that's missing required fields
            partial_json = '{"intent": "execute"}'  # Missing 'message' field
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = partial_json
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider(api_key="test_key")
            
            # Should fallback to text when Pydantic validation fails
            response = await provider.generate_response(
                [{"role": "user", "content": "test"}],
                response_model=TestResponseModel
            )
            
            # Should return the raw JSON text since validation failed
            assert response == partial_json