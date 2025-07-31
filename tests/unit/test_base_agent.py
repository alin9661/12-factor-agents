"""
Unit tests for the base agent implementation.

These tests follow TDD principles and test the core agent execution logic,
stateless reducer behavior, and control flow management.
"""

import json
import pytest
from typing import Any, Dict, List, Type
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.base import BaseAgent
from src.tools.schemas import AgentAction, AgentContext, ToolResult, ExecuteTool, Complete, RequestHumanInput, PauseForApproval, ErrorOccurred
from tests.mocks.llm_providers import MockLLMProvider, FailingLLMProvider


class TestableAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def __init__(self, llm_provider, responses: List[Dict[str, Any]] = None):
        super().__init__(
            llm_provider=llm_provider,
            agent_name="test_agent",
            system_prompt="You are a test agent. Follow instructions carefully.",
            max_iterations=5
        )
        self.tool_results = responses or []
        self.tool_call_count = 0
    
    def get_action_model(self) -> Type[AgentAction]:
        """Return the action model for this agent."""
        # In a real implementation, this would return a proper union type
        # For testing, we'll handle the parsing in parse_action
        return dict  # Placeholder
    
    async def execute_tool(self, action: ExecuteTool, context: AgentContext) -> ToolResult:
        """Mock tool execution for testing."""
        self.tool_call_count += 1
        
        if self.tool_call_count <= len(self.tool_results):
            result_data = self.tool_results[self.tool_call_count - 1]
            return ToolResult(**result_data)
        
        # Default successful result
        return ToolResult(
            success=True,
            result={"status": "executed", "tool": action.tool_name},
            metadata={"tool_name": action.tool_name}
        )


@pytest.mark.unit
class TestBaseAgent:
    """Test cases for BaseAgent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization with required parameters."""
        mock_provider = MockLLMProvider()
        
        agent = TestableAgent(mock_provider)
        
        assert agent.llm_provider == mock_provider
        assert agent.agent_name == "test_agent"
        assert agent.system_prompt == "You are a test agent. Follow instructions carefully."
        assert agent.max_iterations == 5
        assert hasattr(agent, 'logger')
    
    @pytest.mark.asyncio
    async def test_agent_run_immediate_completion(self, sample_agent_context):
        """Test agent execution that completes immediately."""
        mock_provider = MockLLMProvider([
            {
                "intent": "complete",
                "summary": "Task completed successfully",
                "results": {"status": "done"}
            }
        ])
        
        agent = TestableAgent(mock_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "completed"
        assert len(result_context.conversation_history) == 2  # Original message + completion action
        assert result_context.conversation_history[-1]["type"] == "agent_action"
        
        # Verify LLM was called
        assert len(mock_provider.calls) == 1
    
    @pytest.mark.asyncio
    async def test_agent_run_with_tool_execution(self, sample_agent_context):
        """Test agent execution that involves tool calls."""
        mock_provider = MockLLMProvider([
            {
                "intent": "execute_tool",
                "tool_name": "test_tool",
                "parameters": {"message": "test"},
                "requires_approval": False
            },
            {
                "intent": "complete",
                "summary": "Tool executed and task completed",
                "results": {"tool_result": "success"}
            }
        ])
        
        tool_results = [
            {"success": True, "result": {"output": "tool executed"}, "metadata": {"tool_name": "test_tool"}}
        ]
        
        agent = TestableAgent(mock_provider, tool_results)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "completed"
        assert len(result_context.conversation_history) == 5  # Original + 2 actions + 1 tool result + completion
        
        # Check tool execution happened
        tool_result_entry = next(
            entry for entry in result_context.conversation_history 
            if entry["type"] == "tool_result"
        )
        assert tool_result_entry["content"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_agent_run_requests_human_input(self, sample_agent_context):
        """Test agent execution that requests human input."""
        mock_provider = MockLLMProvider([
            {
                "intent": "request_human_input",
                "question": "What should I do next?",
                "context": "I need guidance on the next step",
                "urgency": "medium",
                "format": "free_text"
            }
        ])
        
        agent = TestableAgent(mock_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "waiting_for_human"
        assert len(result_context.conversation_history) == 2
        
        # Check the human input request
        action_entry = result_context.conversation_history[-1]
        assert action_entry["type"] == "agent_action"
        assert action_entry["content"]["intent"] == "request_human_input"
    
    @pytest.mark.asyncio
    async def test_agent_run_requests_approval(self, sample_agent_context):
        """Test agent execution that requests approval."""
        mock_provider = MockLLMProvider([
            {
                "intent": "pause_for_approval",
                "action_description": "Delete production database",
                "risk_level": "high",
                "context": "This action is destructive and irreversible"
            }
        ])
        
        agent = TestableAgent(mock_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "waiting_for_approval"
        
        # Check the approval request
        action_entry = result_context.conversation_history[-1]
        assert action_entry["content"]["intent"] == "pause_for_approval"
        assert action_entry["content"]["risk_level"] == "high"
    
    @pytest.mark.asyncio
    async def test_agent_run_handles_error(self, sample_agent_context):
        """Test agent execution when an error occurs."""
        mock_provider = MockLLMProvider([
            {
                "intent": "error_occurred",
                "error_message": "Something went wrong",
                "error_type": "processing_error",
                "recovery_suggestion": "Retry the operation"
            }
        ])
        
        agent = TestableAgent(mock_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "error"
        
        # Check error was recorded
        action_entry = result_context.conversation_history[-1]
        assert action_entry["content"]["intent"] == "error_occurred"
        assert "something went wrong" in action_entry["content"]["error_message"].lower()
    
    @pytest.mark.asyncio
    async def test_agent_run_max_iterations_reached(self, sample_agent_context):
        """Test agent execution when max iterations is reached."""
        # Mock provider that always returns tool execution (infinite loop)
        mock_provider = MockLLMProvider([
            {
                "intent": "execute_tool",
                "tool_name": "endless_tool",
                "parameters": {},
                "requires_approval": False
            }
        ])
        
        # Set infinite responses to simulate loop
        mock_provider.responses = mock_provider.responses * 10
        
        agent = TestableAgent(mock_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "max_iterations_reached"
        # Should have run max_iterations (5) times
        action_entries = [
            entry for entry in result_context.conversation_history
            if entry["type"] == "agent_action"
        ]
        assert len(action_entries) == 5
    
    @pytest.mark.asyncio
    async def test_agent_run_handles_llm_provider_failure(self, sample_agent_context):
        """Test agent execution when LLM provider fails."""
        failing_provider = FailingLLMProvider("LLM service unavailable")
        
        agent = TestableAgent(failing_provider)
        result_context = await agent.run(sample_agent_context)
        
        assert result_context.execution_state == "error"
        
        # Should have error in conversation history
        error_entries = [
            entry for entry in result_context.conversation_history
            if entry["type"] == "system_error" 
        ]
        assert len(error_entries) == 1
        assert "llm service unavailable" in error_entries[0]["content"]["error"].lower()
    
    @pytest.mark.asyncio
    async def test_agent_run_handles_tool_execution_failure(self, sample_agent_context):
        """Test agent behavior when tool execution fails."""
        mock_provider = MockLLMProvider([
            {
                "intent": "execute_tool",
                "tool_name": "failing_tool",
                "parameters": {"action": "fail"},
                "requires_approval": False
            },
            {
                "intent": "complete",
                "summary": "Completed despite tool failure",
                "results": {"status": "partial_success"}
            }
        ])
        
        # Mock a failing tool result
        tool_results = [
            {
                "success": False,
                "error": "Tool execution failed: Invalid parameters",
                "metadata": {"tool_name": "failing_tool", "error_type": "ValidationError"}
            }
        ]
        
        agent = TestableAgent(mock_provider, tool_results)
        result_context = await agent.run(sample_agent_context)
        
        # Agent should continue execution despite tool failure
        assert result_context.execution_state == "completed"
        
        # Check that tool failure was recorded
        tool_result_entry = next(
            entry for entry in result_context.conversation_history
            if entry["type"] == "tool_result"
        )
        assert tool_result_entry["content"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_agent_context_immutability(self, sample_agent_context):
        """Test that original context is not modified (stateless reducer)."""
        mock_provider = MockLLMProvider([
            {
                "intent": "complete",
                "summary": "Task completed",
                "results": {}
            }
        ])
        
        agent = TestableAgent(mock_provider)
        original_history_length = len(sample_agent_context.conversation_history)
        original_state = sample_agent_context.execution_state
        
        result_context = await agent.run(sample_agent_context)
        
        # Original context should be unchanged
        assert len(sample_agent_context.conversation_history) == original_history_length
        assert sample_agent_context.execution_state == original_state
        
        # Result context should have changes
        assert len(result_context.conversation_history) > original_history_length
        assert result_context.execution_state != original_state
    
    def test_build_messages_basic(self, sample_agent_context):
        """Test building messages for LLM from context."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        messages = agent.build_messages(sample_agent_context)
        
        assert len(messages) >= 2  # System prompt + user message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == agent.system_prompt
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Deploy the user service to staging"
    
    def test_build_messages_with_conversation_history(self):
        """Test building messages with extensive conversation history."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        # Create context with long conversation history
        context = AgentContext(
            thread_id="test_thread",
            conversation_history=[
                {"type": "user_message", "content": "Initial request"},
                {"type": "agent_action", "content": {"intent": "execute_tool", "tool_name": "test"}},
                {"type": "tool_result", "content": {"success": True, "result": "done"}},
                {"type": "user_message", "content": "Follow up question"},
                # Add more entries to test truncation (last 10)
            ] + [
                {"type": "user_message", "content": f"Message {i}"}
                for i in range(15)  # This should trigger truncation
            ]
        )
        
        messages = agent.build_messages(context)
        
        # Should include system message + last 10 conversation entries
        assert len(messages) <= 11  # system + max 10 conversation entries
        assert messages[0]["role"] == "system"
    
    def test_parse_action_execute_tool(self):
        """Test parsing execute_tool action from LLM response."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        action_data = {
            "intent": "execute_tool",
            "tool_name": "deploy_service",
            "parameters": {"service": "user", "version": "1.0"},
            "requires_approval": True
        }
        
        action = agent.parse_action(action_data)
        
        assert isinstance(action, ExecuteTool)
        assert action.intent == "execute_tool"
        assert action.tool_name == "deploy_service"
        assert action.parameters == {"service": "user", "version": "1.0"}
        assert action.requires_approval is True
    
    def test_parse_action_complete(self):
        """Test parsing complete action from LLM response."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        action_data = {
            "intent": "complete",
            "summary": "Deployment completed successfully",
            "results": {"deployment_id": "deploy_123", "status": "success"}
        }
        
        action = agent.parse_action(action_data)
        
        assert isinstance(action, Complete)
        assert action.intent == "complete"
        assert action.summary == "Deployment completed successfully"
        assert action.results["deployment_id"] == "deploy_123"
    
    def test_parse_action_request_human_input(self):
        """Test parsing request_human_input action from LLM response."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        action_data = {
            "intent": "request_human_input",
            "question": "Which environment should I deploy to?",
            "context": "The build is ready for deployment",
            "urgency": "high",
            "format": "multiple_choice",
            "choices": ["staging", "production"]
        }
        
        action = agent.parse_action(action_data)
        
        assert isinstance(action, RequestHumanInput)
        assert action.question == "Which environment should I deploy to?"
        assert action.urgency == "high"
        assert action.format == "multiple_choice"
        assert "staging" in action.choices
    
    def test_parse_action_pause_for_approval(self):
        """Test parsing pause_for_approval action from LLM response."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        action_data = {
            "intent": "pause_for_approval",
            "action_description": "Scale down production instances",
            "risk_level": "high",
            "context": "This will reduce capacity during peak hours"
        }
        
        action = agent.parse_action(action_data)
        
        assert isinstance(action, PauseForApproval)
        assert action.action_description == "Scale down production instances"
        assert action.risk_level == "high"
    
    def test_parse_action_unknown_intent(self):
        """Test parsing action with unknown intent returns error."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        action_data = {
            "intent": "unknown_action",
            "data": "some data"
        }
        
        action = agent.parse_action(action_data)
        
        assert isinstance(action, ErrorOccurred)
        assert action.intent == "error_occurred"
        assert "unknown intent" in action.error_message.lower()
    
    def test_create_error_action(self):
        """Test creating error actions."""
        mock_provider = MockLLMProvider()
        agent = TestableAgent(mock_provider)
        
        error_action = agent.create_error_action("Something went wrong")
        
        assert isinstance(error_action, ErrorOccurred)
        assert error_action.intent == "error_occurred"
        assert error_action.error_message == "Something went wrong"
        assert error_action.error_type == "agent_error"
        assert error_action.recovery_suggestion == "Check logs and retry"
    
    @pytest.mark.asyncio
    async def test_determine_next_step_success(self, sample_agent_context):
        """Test successful determination of next step."""
        mock_response = {
            "intent": "execute_tool",
            "tool_name": "check_status",
            "parameters": {"service": "user-service"},
            "requires_approval": False
        }
        
        mock_provider = MockLLMProvider([mock_response])
        agent = TestableAgent(mock_provider)
        
        action = await agent.determine_next_step(sample_agent_context)
        
        assert isinstance(action, ExecuteTool)
        assert action.tool_name == "check_status"
        assert len(mock_provider.calls) == 1
    
    @pytest.mark.asyncio
    async def test_determine_next_step_json_parsing_fallback(self, sample_agent_context):
        """Test fallback to JSON parsing when structured response fails."""
        # Return a JSON string instead of structured response
        json_response = json.dumps({
            "intent": "complete",
            "summary": "Task finished",
            "results": {}
        })
        
        mock_provider = MockLLMProvider([json_response])
        agent = TestableAgent(mock_provider)
        
        action = await agent.determine_next_step(sample_agent_context)
        
        assert isinstance(action, Complete)
        assert action.summary == "Task finished"
    
    @pytest.mark.asyncio
    async def test_determine_next_step_invalid_json_returns_error(self, sample_agent_context):
        """Test that invalid JSON response returns error action."""
        invalid_response = "This is not valid JSON {{"
        
        mock_provider = MockLLMProvider([invalid_response])
        agent = TestableAgent(mock_provider)
        
        action = await agent.determine_next_step(sample_agent_context)
        
        assert isinstance(action, ErrorOccurred)
        assert "failed to parse" in action.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_determine_next_step_llm_exception_returns_error(self, sample_agent_context):
        """Test that LLM provider exceptions return error actions."""
        failing_provider = FailingLLMProvider("Network timeout")
        agent = TestableAgent(failing_provider)
        
        action = await agent.determine_next_step(sample_agent_context)
        
        assert isinstance(action, ErrorOccurred)
        assert "error determining next step" in action.error_message.lower()


@pytest.mark.unit
class TestBaseAgentEdgeCases:
    """Test edge cases and error conditions in BaseAgent."""
    
    @pytest.mark.asyncio
    async def test_agent_with_empty_context(self):
        """Test agent execution with minimal context."""
        mock_provider = MockLLMProvider([
            {"intent": "complete", "summary": "Nothing to do", "results": {}}
        ])
        
        agent = TestableAgent(mock_provider)
        
        empty_context = AgentContext(
            thread_id="empty_test",
            conversation_history=[],
            execution_state="running"
        )
        
        result_context = await agent.run(empty_context)
        
        assert result_context.execution_state == "completed"
        assert len(result_context.conversation_history) == 1  # Just the completion action
    
    @pytest.mark.asyncio
    async def test_agent_with_zero_max_iterations(self):
        """Test agent behavior with zero max iterations."""
        mock_provider = MockLLMProvider()
        
        agent = TestableAgent(mock_provider)
        agent.max_iterations = 0
        
        context = AgentContext(thread_id="zero_iter", execution_state="running")
        result_context = await agent.run(context)
        
        # Should immediately exit without calling LLM
        assert len(mock_provider.calls) == 0
        assert result_context.execution_state == "running"  # Unchanged
    
    @pytest.mark.asyncio 
    async def test_agent_context_deep_copy_isolation(self, sample_agent_context):
        """Test that context changes don't affect original through deep copy."""
        mock_provider = MockLLMProvider([
            {"intent": "complete", "summary": "Modified context", "results": {}}
        ])
        
        agent = TestableAgent(mock_provider)
        
        # Add nested data that could be affected by shallow copy
        sample_agent_context.session_data = {
            "nested": {"value": "original"},
            "list": [1, 2, 3]
        }
        
        result_context = await agent.run(sample_agent_context)
        
        # Modify the result context's nested data
        result_context.session_data["nested"]["value"] = "modified"
        result_context.session_data["list"].append(4)
        
        # Original should be unchanged
        assert sample_agent_context.session_data["nested"]["value"] == "original"
        assert sample_agent_context.session_data["list"] == [1, 2, 3]
    
    def test_agent_inheritance_requires_abstract_methods(self):
        """Test that BaseAgent requires implementation of abstract methods."""
        mock_provider = MockLLMProvider()
        
        # This should work - TestableAgent implements abstract methods
        agent = TestableAgent(mock_provider)
        assert agent is not None
        
        # Trying to instantiate BaseAgent directly should fail
        with pytest.raises(TypeError):
            BaseAgent(mock_provider, "test", "test prompt")