"""
End-to-end workflow tests for agent interactions.

These tests verify complete agent workflows from start to finish,
testing the integration of all components together.
"""

import asyncio
import json
import pytest
import uuid
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient
from fastapi import status

from src.api.main import agent_sessions
from tests.mocks.llm_providers import MockLLMProvider


@pytest.mark.e2e
class TestDeploymentWorkflow:
    """Test complete deployment agent workflows."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_complete_deployment_workflow_success(self, async_client: AsyncClient):
        """Test a complete successful deployment workflow."""
        
        # Mock LLM responses for the workflow
        llm_responses = [
            # 1. Initial response - execute tool to check service status
            {
                "intent": "execute_tool",
                "tool_name": "check_service_status",
                "parameters": {"service_name": "user-service", "environment": "staging"},
                "requires_approval": False
            },
            # 2. After status check - request approval for deployment
            {
                "intent": "pause_for_approval", 
                "action_description": "Deploy user-service v1.2.3 to staging environment",
                "risk_level": "medium",
                "context": "Service is currently healthy, ready for deployment"
            },
            # 3. After approval - execute deployment
            {
                "intent": "execute_tool",
                "tool_name": "deploy_service",
                "parameters": {
                    "service_name": "user-service",
                    "version": "1.2.3", 
                    "environment": "staging"
                },
                "requires_approval": True
            },
            # 4. Final completion
            {
                "intent": "complete",
                "summary": "Successfully deployed user-service v1.2.3 to staging",
                "results": {
                    "deployment_id": "deploy-20240101-120000",
                    "status": "success",
                    "service": "user-service",
                    "version": "1.2.3",
                    "environment": "staging"
                }
            }
        ]
        
        mock_provider = MockLLMProvider(llm_responses)
        
        with patch('src.api.main.DeploymentAgent') as mock_agent_class:
            with patch('src.api.main.run_agent_background') as mock_background:
                with patch('src.agents.llm_providers.LLMProviderFactory.create_default_provider', return_value=mock_provider):
                    
                    # Mock the agent to use our mock provider
                    async def mock_run_agent(agent, thread_id):
                        context = agent_sessions[thread_id]
                        
                        # Simulate the agent workflow steps
                        for i, response in enumerate(llm_responses):
                            if context.execution_state != "running":
                                break
                            
                            # Add agent action to history
                            context.conversation_history.append({
                                "type": "agent_action",
                                "content": response,
                                "iteration": i
                            })
                            
                            if response["intent"] == "execute_tool":
                                # Mock tool execution result
                                tool_result = {
                                    "success": True,
                                    "result": {"status": "completed", "tool": response["tool_name"]},
                                    "metadata": {"tool_name": response["tool_name"]}
                                }
                                context.conversation_history.append({
                                    "type": "tool_result",
                                    "content": tool_result,
                                    "iteration": i
                                })
                            elif response["intent"] == "pause_for_approval":
                                context.execution_state = "waiting_for_approval"
                                break
                            elif response["intent"] == "complete":
                                context.execution_state = "completed"
                                break
                        
                        agent_sessions[thread_id] = context
                    
                    mock_background.side_effect = mock_run_agent
                    
                    # Step 1: Start the deployment workflow
                    start_request = {
                        "message": "Deploy user-service version 1.2.3 to staging environment",
                        "agent_type": "deployment",
                        "user_id": "test_user",
                        "session_data": {
                            "service": "user-service",
                            "version": "1.2.3",
                            "target_environment": "staging"
                        }
                    }
                    
                    response = await async_client.post("/agents/start", json=start_request)
                    assert response.status_code == status.HTTP_200_OK
                    
                    data = response.json()
                    thread_id = data["thread_id"]
                    assert data["status"] == "started"
                    
                    # Wait for background processing
                    await asyncio.sleep(0.1)
                    
                    # Step 2: Check status - should be waiting for approval
                    status_response = await async_client.get(f"/agents/{thread_id}/status")
                    assert status_response.status_code == status.HTTP_200_OK
                    
                    status_data = status_response.json()
                    assert status_data["execution_state"] == "waiting_for_approval"
                    assert status_data["requires_approval"] is True
                    
                    # Step 3: Resume with approval
                    with patch('src.api.main.run_agent_background', side_effect=mock_run_agent):
                        resume_request = {
                            "approved": True,
                            "human_response": "Approved for staging deployment"
                        }
                        
                        resume_response = await async_client.post(
                            f"/agents/{thread_id}/resume",
                            json=resume_request
                        )
                        assert resume_response.status_code == status.HTTP_200_OK
                        
                        resume_data = resume_response.json()
                        assert resume_data["status"] == "resumed"
                    
                    # Wait for completion
                    await asyncio.sleep(0.1)
                    
                    # Step 4: Check final status
                    final_status = await async_client.get(f"/agents/{thread_id}/status")
                    assert final_status.status_code == status.HTTP_200_OK
                    
                    final_data = final_status.json()
                    assert final_data["execution_state"] == "completed"
                    
                    # Step 5: Get conversation history
                    history_response = await async_client.get(f"/agents/{thread_id}/history")
                    assert history_response.status_code == status.HTTP_200_OK
                    
                    history_data = history_response.json()
                    conversation = history_data["conversation_history"]
                    
                    # Verify workflow steps are recorded
                    assert len(conversation) >= 6  # Initial message + 4 agent actions + tool results
                    
                    # Check for specific workflow steps
                    action_types = [
                        entry["content"].get("intent") for entry in conversation
                        if entry.get("type") == "agent_action"
                    ]
                    
                    assert "execute_tool" in action_types  # Tool execution happened
                    assert "pause_for_approval" in action_types  # Requested approval
                    assert "complete" in action_types  # Workflow completed
    
    @pytest.mark.asyncio
    async def test_deployment_workflow_with_human_input(self, async_client: AsyncClient):
        """Test deployment workflow that requires human input."""
        
        llm_responses = [
            # 1. Request human input for environment selection
            {
                "intent": "request_human_input",
                "question": "Which environment should I deploy to?",
                "context": "The service build is ready for deployment",
                "urgency": "medium",
                "format": "multiple_choice",
                "choices": ["staging", "production"]
            },
            # 2. After human input - proceed with deployment
            {
                "intent": "execute_tool",
                "tool_name": "deploy_service",
                "parameters": {
                    "service_name": "api-service",
                    "version": "2.0.0",
                    "environment": "staging"  # Based on human choice
                }
            },
            # 3. Complete
            {
                "intent": "complete",
                "summary": "Deployment completed based on user selection",
                "results": {"environment": "staging", "status": "deployed"}
            }
        ]
        
        mock_provider = MockLLMProvider(llm_responses)
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.agents.llm_providers.LLMProviderFactory.create_default_provider', return_value=mock_provider):
                
                async def mock_workflow(agent, thread_id):
                    context = agent_sessions[thread_id]
                    
                    # First step - request human input
                    context.conversation_history.append({
                        "type": "agent_action",
                        "content": llm_responses[0]
                    })
                    context.execution_state = "waiting_for_human"
                    agent_sessions[thread_id] = context
                
                with patch('src.api.main.run_agent_background', side_effect=mock_workflow):
                    # Start workflow
                    start_request = {
                        "message": "Deploy api-service version 2.0.0",
                        "agent_type": "deployment"
                    }
                    
                    response = await async_client.post("/agents/start", json=start_request)
                    thread_id = response.json()["thread_id"]
                    
                    await asyncio.sleep(0.1)
                    
                    # Check that human input is required
                    status_response = await async_client.get(f"/agents/{thread_id}/status")
                    status_data = status_response.json()
                    
                    assert status_data["execution_state"] == "waiting_for_human"
                    assert status_data["requires_human_input"] is True
                    
                    # Provide human response
                    async def complete_workflow(agent, thread_id):
                        context = agent_sessions[thread_id]
                        
                        # Execute deployment based on human choice
                        context.conversation_history.append({
                            "type": "agent_action",
                            "content": llm_responses[1]
                        })
                        
                        # Add tool result
                        context.conversation_history.append({
                            "type": "tool_result",
                            "content": {
                                "success": True,
                                "result": {"status": "deployed", "environment": "staging"}
                            }
                        })
                        
                        # Complete workflow
                        context.conversation_history.append({
                            "type": "agent_action",
                            "content": llm_responses[2]
                        })
                        
                        context.execution_state = "completed"
                        agent_sessions[thread_id] = context
                    
                    with patch('src.api.main.run_agent_background', side_effect=complete_workflow):
                        human_response = {
                            "human_response": "staging",
                            "additional_context": {"reason": "safer_testing"}
                        }
                        
                        resume_response = await async_client.post(
                            f"/agents/{thread_id}/resume",
                            json=human_response
                        )
                        
                        assert resume_response.status_code == status.HTTP_200_OK
                    
                    await asyncio.sleep(0.1)
                    
                    # Verify completion
                    final_status = await async_client.get(f"/agents/{thread_id}/status")
                    final_data = final_status.json()
                    
                    assert final_data["execution_state"] == "completed"
    
    @pytest.mark.asyncio
    async def test_deployment_workflow_error_handling(self, async_client: AsyncClient):
        """Test deployment workflow with error conditions."""
        
        llm_responses = [
            # 1. Try to check service status but encounter error
            {
                "intent": "execute_tool",
                "tool_name": "check_service_status",
                "parameters": {"service_name": "nonexistent-service", "environment": "production"}
            },
            # 2. Handle error and try alternative
            {
                "intent": "request_human_input",
                "question": "Service status check failed. Should I proceed anyway?",
                "context": "Unable to verify current service status",
                "urgency": "high",
                "format": "yes_no"
            },
            # 3. Complete with error status
            {
                "intent": "complete",
                "summary": "Deployment aborted due to service status check failure",
                "results": {"status": "aborted", "reason": "service_status_unknown"}
            }
        ]
        
        mock_provider = MockLLMProvider(llm_responses)
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.agents.llm_providers.LLMProviderFactory.create_default_provider', return_value=mock_provider):
                
                async def error_workflow(agent, thread_id):
                    context = agent_sessions[thread_id]
                    
                    # First action - try tool that fails
                    context.conversation_history.append({
                        "type": "agent_action",  
                        "content": llm_responses[0]
                    })
                    
                    # Add failed tool result
                    context.conversation_history.append({
                        "type": "tool_result",
                        "content": {
                            "success": False,
                            "error": "Service not found: nonexistent-service",
                            "metadata": {"tool_name": "check_service_status"}
                        }
                    })
                    
                    # Agent requests human input to handle error
                    context.conversation_history.append({
                        "type": "agent_action",
                        "content": llm_responses[1]
                    })
                    
                    context.execution_state = "waiting_for_human"
                    agent_sessions[thread_id] = context
                
                with patch('src.api.main.run_agent_background', side_effect=error_workflow):
                    # Start workflow
                    start_request = {
                        "message": "Deploy nonexistent-service to production",
                        "agent_type": "deployment"
                    }
                    
                    response = await async_client.post("/agents/start", json=start_request)
                    thread_id = response.json()["thread_id"]
                    
                    await asyncio.sleep(0.1)
                    
                    # Check that human input is requested due to error
                    status_response = await async_client.get(f"/agents/{thread_id}/status")
                    status_data = status_response.json()
                    
                    assert status_data["execution_state"] == "waiting_for_human"
                    
                    # Check conversation history contains error
                    history_response = await async_client.get(f"/agents/{thread_id}/history")
                    history_data = history_response.json()
                    
                    tool_results = [
                        entry for entry in history_data["conversation_history"]
                        if entry.get("type") == "tool_result"
                    ]
                    
                    assert len(tool_results) > 0
                    assert tool_results[0]["content"]["success"] is False
                    assert "not found" in tool_results[0]["content"]["error"].lower()


@pytest.mark.e2e
class TestCustomerSupportWorkflow:
    """Test complete customer support workflows."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_customer_support_ticket_workflow(self, async_client: AsyncClient):
        """Test complete customer support ticket creation workflow."""
        
        llm_responses = [
            # 1. Look up customer first
            {
                "intent": "execute_tool",
                "tool_name": "lookup_customer",
                "parameters": {"customer_identifier": "customer@example.com"}
            },
            # 2. Create support ticket
            {
                "intent": "execute_tool",
                "tool_name": "create_support_ticket",
                "parameters": {
                    "customer_id": "cust_123456",
                    "issue_title": "Login Issues",
                    "issue_description": "Customer cannot access their account",
                    "priority": "medium"
                }
            },
            # 3. Complete workflow
            {
                "intent": "complete",
                "summary": "Support ticket created successfully",
                "results": {
                    "ticket_id": "TICKET-20240101-120000",
                    "customer": "customer@example.com",
                    "status": "created"
                }
            }
        ]
        
        mock_provider = MockLLMProvider(llm_responses)
        
        with patch('src.api.main.DeploymentAgent'):  # Using DeploymentAgent as generic agent
            with patch('src.agents.llm_providers.LLMProviderFactory.create_default_provider', return_value=mock_provider):
                
                async def support_workflow(agent, thread_id):
                    context = agent_sessions[thread_id]
                    
                    # Customer lookup
                    context.conversation_history.append({
                        "type": "agent_action",
                        "content": llm_responses[0]
                    })
                    
                    context.conversation_history.append({
                        "type": "tool_result",
                        "content": {
                            "success": True,
                            "result": {
                                "customer_id": "cust_123456",
                                "email": "customer@example.com",
                                "name": "John Doe",
                                "account_status": "active"
                            }
                        }
                    })
                    
                    # Create ticket
                    context.conversation_history.append({
                        "type": "agent_action",
                        "content": llm_responses[1]
                    })
                    
                    context.conversation_history.append({
                        "type": "tool_result",
                        "content": {
                            "success": True,
                            "result": {
                                "ticket_id": "TICKET-20240101-120000",
                                "customer_id": "cust_123456",
                                "status": "open"
                            }
                        }
                    })
                    
                    # Complete
                    context.conversation_history.append({
                        "type": "agent_action",
                        "content": llm_responses[2]
                    })
                    
                    context.execution_state = "completed"
                    agent_sessions[thread_id] = context
                
                with patch('src.api.main.run_agent_background', side_effect=support_workflow):
                    # Start support workflow
                    start_request = {
                        "message": "Create a support ticket for customer@example.com who cannot login",
                        "agent_type": "deployment",  # Using available agent type
                        "session_data": {
                            "customer_email": "customer@example.com",
                            "issue": "login_problems"
                        }
                    }
                    
                    response = await async_client.post("/agents/start", json=start_request)
                    assert response.status_code == status.HTTP_200_OK
                    
                    thread_id = response.json()["thread_id"]
                    
                    await asyncio.sleep(0.1)
                    
                    # Check completion
                    status_response = await async_client.get(f"/agents/{thread_id}/status")
                    status_data = status_response.json()
                    
                    assert status_data["execution_state"] == "completed"
                    
                    # Verify workflow steps in history
                    history_response = await async_client.get(f"/agents/{thread_id}/history")
                    history_data = history_response.json()
                    
                    # Should have customer lookup and ticket creation
                    tool_calls = [
                        entry["content"] for entry in history_data["conversation_history"]
                        if entry.get("type") == "agent_action" and entry["content"].get("intent") == "execute_tool"
                    ]
                    
                    tool_names = [call["tool_name"] for call in tool_calls]
                    assert "lookup_customer" in tool_names
                    assert "create_support_ticket" in tool_names


@pytest.mark.e2e
class TestWebhookTriggeredWorkflows:
    """Test workflows triggered by webhooks."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_webhook_triggered_deployment(self, async_client: AsyncClient):
        """Test deployment workflow triggered by webhook."""
        
        llm_responses = [
            {
                "intent": "execute_tool",
                "tool_name": "deploy_service",
                "parameters": {
                    "service_name": "api-service",
                    "version": "1.5.0",
                    "environment": "staging"
                }
            },
            {
                "intent": "complete",
                "summary": "Webhook-triggered deployment completed",
                "results": {"status": "deployed", "trigger": "webhook"}
            }
        ]
        
        mock_provider = MockLLMProvider(llm_responses)
        
        with patch('src.api.main.start_agent') as mock_start_agent:
            with patch('src.agents.llm_providers.LLMProviderFactory.create_default_provider', return_value=mock_provider):
                
                # Mock the start_agent function to return a proper response
                mock_thread_id = str(uuid.uuid4())
                mock_start_agent.return_value = {
                    "thread_id": mock_thread_id,
                    "agent_type": "deployment",
                    "status": "started",
                    "execution_state": "running"
                }
                
                # Webhook payload simulating GitHub deployment
                webhook_payload = {
                    "message": "Deploy api-service version 1.5.0 to staging",
                    "agent_type": "deployment",
                    "user_id": "github_actions",
                    "context": {
                        "source": "github",
                        "repository": "api-service",
                        "branch": "main",
                        "commit": "abc123def456",
                        "version": "1.5.0",
                        "target_environment": "staging"
                    }
                }
                
                response = await async_client.post("/webhooks/trigger", json=webhook_payload)
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert data["status"] == "webhook_processed"
                assert "agent_response" in data
                assert data["agent_response"]["thread_id"] == mock_thread_id
                
                # Verify start_agent was called with correct parameters
                mock_start_agent.assert_called_once()
                
                call_args = mock_start_agent.call_args[0][0]  # AgentRequest object
                assert call_args.message == "Deploy api-service version 1.5.0 to staging"
                assert call_args.user_id == "github_actions"
                assert call_args.session_data["source"] == "github"
                assert call_args.session_data["version"] == "1.5.0"


@pytest.mark.e2e
class TestMultiAgentWorkflows:
    """Test workflows involving multiple agent sessions."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_sessions(self, async_client: AsyncClient):
        """Test multiple concurrent agent sessions."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Start multiple agent sessions concurrently
                concurrent_requests = []
                for i in range(5):
                    request_data = {
                        "message": f"Deploy service-{i} to staging",
                        "agent_type": "deployment",
                        "user_id": f"user_{i}",
                        "session_data": {"service_id": i}
                    }
                    concurrent_requests.append(
                        async_client.post("/agents/start", json=request_data)
                    )
                
                # Execute all requests concurrently
                responses = await asyncio.gather(*concurrent_requests)
                
                # Verify all sessions were created successfully
                thread_ids = []
                for response in responses:
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    thread_ids.append(data["thread_id"])
                
                # Verify all sessions are tracked
                assert len(thread_ids) == 5
                assert len(set(thread_ids)) == 5  # All unique
                
                # Verify session isolation
                sessions_response = await async_client.get("/agents")
                sessions_data = sessions_response.json()
                
                assert sessions_data["total_count"] == 5
                
                # Each session should have different user_id
                user_ids = set()
                for session in sessions_data["sessions"]:
                    thread_id = session["thread_id"]
                    context = agent_sessions[thread_id]
                    user_ids.add(context.user_id)
                
                assert len(user_ids) == 5  # All different users
    
    @pytest.mark.asyncio
    async def test_session_cleanup_workflow(self, async_client: AsyncClient):
        """Test session cleanup and management."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                # Create several sessions
                thread_ids = []
                for i in range(3):
                    request_data = {
                        "message": f"Test session {i}",
                        "agent_type": "deployment"
                    }
                    
                    response = await async_client.post("/agents/start", json=request_data)
                    thread_id = response.json()["thread_id"]
                    thread_ids.append(thread_id)
                
                # Verify sessions exist
                sessions_response = await async_client.get("/agents")
                assert sessions_response.json()["total_count"] == 3
                
                # Delete sessions individually
                for thread_id in thread_ids:
                    delete_response = await async_client.delete(f"/agents/{thread_id}")
                    assert delete_response.status_code == status.HTTP_200_OK
                
                # Verify all sessions are deleted
                final_sessions = await async_client.get("/agents")
                assert final_sessions.json()["total_count"] == 0
                assert len(agent_sessions) == 0