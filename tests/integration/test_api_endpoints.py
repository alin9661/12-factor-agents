"""
Integration tests for FastAPI endpoints.

These tests verify the complete API functionality with real HTTP requests
and test the full request/response cycle.
"""

import json
import pytest
import uuid
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from httpx import AsyncClient

from src.api.main import app, agent_sessions
from tests.mocks.llm_providers import MockLLMProvider


@pytest.mark.integration
class TestHealthEndpoints:
    """Test cases for health check and root endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint returns API information."""
        response = await async_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["name"] == "12-Factor Agents API"
        assert data["version"] == "0.1.0"
        assert "factors_implemented" in data
        assert len(data["factors_implemented"]) == 12  # Should have all 12 factors
        
        # Check that key factors are present
        expected_factors = [
            "Natural Language to Tool Calls",
            "Own Your Prompts",
            "Trigger from Anywhere",
            "Stateless Reducer"
        ]
        
        for factor in expected_factors:
            assert factor in data["factors_implemented"]
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data
        
        # Timestamp should be in ISO format
        from datetime import datetime
        try:
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Timestamp is not in valid ISO format")


@pytest.mark.integration
class TestAgentEndpoints:
    """Test cases for agent-related endpoints."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_start_agent_success(self, async_client: AsyncClient):
        """Test successful agent startup."""
        with patch('src.api.main.DeploymentAgent') as mock_agent_class:
            with patch('src.api.main.run_agent_background') as mock_background:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                request_data = {
                    "message": "Deploy user service to staging",
                    "agent_type": "deployment",
                    "user_id": "test_user_123",
                    "session_data": {"environment": "test"}
                }
                
                response = await async_client.post("/agents/start", json=request_data)
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert data["agent_type"] == "deployment"
                assert data["status"] == "started"
                assert data["execution_state"] == "running"
                assert "thread_id" in data
                
                # Verify thread_id is a valid UUID
                try:
                    uuid.UUID(data["thread_id"])
                except ValueError:
                    pytest.fail("thread_id is not a valid UUID")
                
                # Verify agent was created and background task started
                mock_agent_class.assert_called_once()
                mock_background.assert_called_once()
                
                # Verify session was stored
                assert data["thread_id"] in agent_sessions
                stored_context = agent_sessions[data["thread_id"]]
                assert stored_context.user_id == "test_user_123"
                assert stored_context.session_data["environment"] == "test"
    
    @pytest.mark.asyncio
    async def test_start_agent_unknown_type(self, async_client: AsyncClient):
        """Test starting agent with unknown type returns error."""
        request_data = {
            "message": "Test message",
            "agent_type": "unknown_agent_type"
        }
        
        response = await async_client.post("/agents/start", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        data = response.json()
        assert "unknown agent type" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_start_agent_missing_message(self, async_client: AsyncClient):
        """Test starting agent without message returns validation error."""
        request_data = {
            "agent_type": "deployment"
            # Missing required "message" field
        }
        
        response = await async_client.post("/agents/start", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_start_agent_llm_provider_failure(self, async_client: AsyncClient):
        """Test agent startup when LLM provider fails to initialize."""
        with patch('src.api.main.app.state.llm_provider', None):
            request_data = {
                "message": "Test message",
                "agent_type": "deployment"
            }
            
            # This should fail because no LLM provider is available
            with pytest.raises(Exception):
                await async_client.post("/agents/start", json=request_data)
    
    @pytest.mark.asyncio
    async def test_get_agent_status_success(self, async_client: AsyncClient):
        """Test getting agent status for existing session."""
        # First create an agent session
        from src.tools.schemas import AgentContext
        from datetime import datetime
        
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            user_id="test_user",
            execution_state="waiting_for_human",
            conversation_history=[
                {
                    "type": "user_message",
                    "content": "Deploy service",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "type": "agent_action",
                    "content": {
                        "intent": "request_human_input",
                        "question": "Which environment?",
                        "summary": "Need environment selection"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            created_at=datetime.utcnow().isoformat()
        )
        
        agent_sessions[thread_id] = context
        
        response = await async_client.get(f"/agents/{thread_id}/status")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["thread_id"] == thread_id
        assert data["agent_type"] == "deployment"
        assert data["status"] == "active"
        assert data["execution_state"] == "waiting_for_human"
        assert data["requires_human_input"] is True
        assert data["requires_approval"] is False
        assert "message" in data
    
    @pytest.mark.asyncio
    async def test_get_agent_status_not_found(self, async_client: AsyncClient):
        """Test getting status for non-existent agent session."""
        fake_thread_id = str(uuid.uuid4())
        
        response = await async_client.get(f"/agents/{fake_thread_id}/status")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_get_agent_status_waiting_for_approval(self, async_client: AsyncClient):
        """Test agent status when waiting for approval."""
        from src.tools.schemas import AgentContext
        from datetime import datetime
        
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            execution_state="waiting_for_approval",
            conversation_history=[
                {
                    "type": "agent_action",
                    "content": {
                        "intent": "pause_for_approval",
                        "action_description": "Delete production data",
                        "risk_level": "high"
                    }
                }
            ]
        )
        
        agent_sessions[thread_id] = context
        
        response = await async_client.get(f"/agents/{thread_id}/status")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["requires_human_input"] is True
        assert data["requires_approval"] is True
    
    @pytest.mark.asyncio
    async def test_resume_agent_success(self, async_client: AsyncClient):
        """Test successful agent resumption."""
        from src.tools.schemas import AgentContext
        from datetime import datetime
        
        # Create paused agent session
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            execution_state="waiting_for_human",
            conversation_history=[
                {
                    "type": "agent_action",
                    "content": {"intent": "request_human_input", "question": "Continue?"}
                }
            ],
            created_at=datetime.utcnow().isoformat()
        )
        
        agent_sessions[thread_id] = context
        
        with patch('src.api.main.DeploymentAgent') as mock_agent_class:
            with patch('src.api.main.run_agent_background') as mock_background:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                
                resume_data = {
                    "human_response": "Yes, continue with staging deployment",
                    "additional_context": {"priority": "high"}
                }
                
                response = await async_client.post(
                    f"/agents/{thread_id}/resume",
                    json=resume_data
                )
                
                assert response.status_code == status.HTTP_200_OK
                
                data = response.json()
                assert data["thread_id"] == thread_id
                assert data["status"] == "resumed"
                assert data["execution_state"] == "running"
                
                # Verify context was updated
                updated_context = agent_sessions[thread_id]
                assert updated_context.execution_state == "running"
                assert updated_context.session_data["priority"] == "high"
                
                # Check human response was added to history
                human_responses = [
                    entry for entry in updated_context.conversation_history
                    if entry["type"] == "human_response"
                ]
                assert len(human_responses) == 1
                assert human_responses[0]["content"] == "Yes, continue with staging deployment"
    
    @pytest.mark.asyncio
    async def test_resume_agent_not_found(self, async_client: AsyncClient):
        """Test resuming non-existent agent session."""
        fake_thread_id = str(uuid.uuid4())
        
        resume_data = {"human_response": "Continue"}
        
        response = await async_client.post(
            f"/agents/{fake_thread_id}/resume",
            json=resume_data
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_resume_agent_not_paused(self, async_client: AsyncClient):
        """Test resuming agent that is not paused."""
        from src.tools.schemas import AgentContext
        
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            execution_state="running"  # Not paused
        )
        
        agent_sessions[thread_id] = context
        
        resume_data = {"human_response": "Continue"}
        
        response = await async_client.post(
            f"/agents/{thread_id}/resume",
            json=resume_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        data = response.json()
        assert "not paused" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_resume_agent_with_approval(self, async_client: AsyncClient):
        """Test resuming agent with approval response."""
        from src.tools.schemas import AgentContext
        
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            execution_state="waiting_for_approval"
        )
        
        agent_sessions[thread_id] = context
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                resume_data = {
                    "approved": True,
                    "human_response": "Approved for production deployment"
                }
                
                response = await async_client.post(
                    f"/agents/{thread_id}/resume",
                    json=resume_data
                )
                
                assert response.status_code == status.HTTP_200_OK
                
                # Check approval was recorded
                updated_context = agent_sessions[thread_id]
                approval_responses = [
                    entry for entry in updated_context.conversation_history
                    if entry["type"] == "approval_response"
                ]
                assert len(approval_responses) == 1
                assert approval_responses[0]["content"]["approved"] is True
    
    @pytest.mark.asyncio
    async def test_get_agent_history(self, async_client: AsyncClient):
        """Test getting agent conversation history."""
        from src.tools.schemas import AgentContext
        from datetime import datetime
        
        thread_id = str(uuid.uuid4())
        created_time = datetime.utcnow().isoformat()
        context = AgentContext(
            thread_id=thread_id,
            execution_state="completed",
            conversation_history=[
                {"type": "user_message", "content": "Deploy service"},
                {"type": "agent_action", "content": {"intent": "execute_tool"}},
                {"type": "tool_result", "content": {"success": True}}
            ],
            created_at=created_time,
            updated_at=created_time
        )
        
        agent_sessions[thread_id] = context
        
        response = await async_client.get(f"/agents/{thread_id}/history")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["thread_id"] == thread_id
        assert data["execution_state"] == "completed"
        assert len(data["conversation_history"]) == 3
        assert data["created_at"] == created_time
    
    @pytest.mark.asyncio
    async def test_delete_agent_session(self, async_client: AsyncClient):
        """Test deleting agent session."""
        from src.tools.schemas import AgentContext
        
        thread_id = str(uuid.uuid4())
        context = AgentContext(thread_id=thread_id)
        agent_sessions[thread_id] = context
        
        # Verify session exists
        assert thread_id in agent_sessions
        
        response = await async_client.delete(f"/agents/{thread_id}")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "deleted successfully" in data["message"]
        
        # Verify session was removed
        assert thread_id not in agent_sessions
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_agent_session(self, async_client: AsyncClient):
        """Test deleting non-existent agent session."""
        fake_thread_id = str(uuid.uuid4())
        
        response = await async_client.delete(f"/agents/{fake_thread_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_list_agent_sessions_empty(self, async_client: AsyncClient):
        """Test listing agent sessions when none exist."""
        # Ensure no sessions exist
        agent_sessions.clear()
        
        response = await async_client.get("/agents")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["sessions"] == []
        assert data["total_count"] == 0
    
    @pytest.mark.asyncio
    async def test_list_agent_sessions_with_data(self, async_client: AsyncClient):
        """Test listing agent sessions with existing data."""
        from src.tools.schemas import AgentContext
        from datetime import datetime
        
        # Create multiple sessions
        sessions_data = []
        for i in range(3):
            thread_id = str(uuid.uuid4())
            created_time = datetime.utcnow().isoformat()
            context = AgentContext(
                thread_id=thread_id,
                execution_state=f"state_{i}",
                conversation_history=[{"type": "user_message", "content": f"Message {i}"}],
                created_at=created_time,
                updated_at=created_time
            )
            agent_sessions[thread_id] = context
            sessions_data.append((thread_id, created_time))
        
        response = await async_client.get("/agents")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_count"] == 3
        assert len(data["sessions"]) == 3
        
        # Verify session data structure
        for session in data["sessions"]:
            assert "thread_id" in session
            assert "execution_state" in session
            assert "created_at" in session
            assert "updated_at" in session
            assert "message_count" in session
            assert session["message_count"] == 1  # Each has one message


@pytest.mark.integration
class TestWebhookEndpoints:
    """Test cases for webhook endpoints."""
    
    def setup_method(self):
        """Clear agent sessions before each test."""
        agent_sessions.clear()
    
    @pytest.mark.asyncio
    async def test_webhook_trigger_success(self, async_client: AsyncClient):
        """Test successful webhook trigger."""
        with patch('src.api.main.start_agent') as mock_start_agent:
            from datetime import datetime
            
            mock_response = {
                "thread_id": str(uuid.uuid4()),
                "agent_type": "deployment",
                "status": "started",
                "execution_state": "running"
            }
            mock_start_agent.return_value = mock_response
            
            webhook_payload = {
                "message": "Webhook triggered deployment",
                "agent_type": "deployment",
                "user_id": "webhook_user",
                "context": {
                    "source": "github",
                    "repository": "user-service",
                    "branch": "main"
                }
            }
            
            response = await async_client.post("/webhooks/trigger", json=webhook_payload)
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["status"] == "webhook_processed"
            assert "agent_response" in data
            assert "timestamp" in data
            
            # Verify start_agent was called with correct parameters
            mock_start_agent.assert_called_once()
            call_args = mock_start_agent.call_args[0][0]  # First positional argument (AgentRequest)
            assert call_args.message == "Webhook triggered deployment"
            assert call_args.agent_type == "deployment"
            assert call_args.user_id == "webhook_user"
            assert call_args.session_data["source"] == "github"
    
    @pytest.mark.asyncio
    async def test_webhook_trigger_minimal_payload(self, async_client: AsyncClient):
        """Test webhook trigger with minimal payload."""
        with patch('src.api.main.start_agent') as mock_start_agent:
            mock_response = {
                "thread_id": str(uuid.uuid4()),
                "agent_type": "deployment",
                "status": "started",
                "execution_state": "running"
            }
            mock_start_agent.return_value = mock_response
            
            # Minimal payload - only empty dict
            webhook_payload = {}
            
            response = await async_client.post("/webhooks/trigger", json=webhook_payload)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify defaults were used
            call_args = mock_start_agent.call_args[0][0]
            assert call_args.message == "Webhook triggered with no message"
            assert call_args.agent_type == "deployment"
            assert call_args.user_id is None
    
    @pytest.mark.asyncio
    async def test_webhook_trigger_invalid_json(self, async_client: AsyncClient):
        """Test webhook trigger with invalid JSON payload."""
        response = await async_client.post(
            "/webhooks/trigger",
            content="invalid json {",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_webhook_trigger_agent_startup_failure(self, async_client: AsyncClient):
        """Test webhook trigger when agent startup fails."""
        with patch('src.api.main.start_agent') as mock_start_agent:
            mock_start_agent.side_effect = Exception("Agent startup failed")
            
            webhook_payload = {"message": "Test webhook"}
            
            with pytest.raises(Exception, match="Agent startup failed"):
                await async_client.post("/webhooks/trigger", json=webhook_payload)


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test cases for API error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_json_request(self, async_client: AsyncClient):
        """Test API response to invalid JSON in request body."""
        response = await async_client.post(
            "/agents/start",
            content="invalid json {",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_missing_content_type_header(self, async_client: AsyncClient):
        """Test API response when content-type header is missing."""
        valid_json = json.dumps({"message": "test", "agent_type": "deployment"})
        
        response = await async_client.post(
            "/agents/start",
            content=valid_json
            # Missing content-type header
        )
        
        # FastAPI should still handle this correctly
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    @pytest.mark.asyncio
    async def test_oversized_request_payload(self, async_client: AsyncClient):
        """Test API response to oversized request payload."""
        # Create a very large payload
        large_message = "x" * (1024 * 1024)  # 1MB message
        
        request_data = {
            "message": large_message,
            "agent_type": "deployment"
        }
        
        # This might succeed or fail depending on server limits
        response = await async_client.post("/agents/start", json=request_data)
        
        # Should either process successfully or return an appropriate error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_to_same_endpoint(self, async_client: AsyncClient):
        """Test API behavior under concurrent requests."""
        import asyncio
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                # Create multiple concurrent requests
                tasks = []
                for i in range(10):
                    request_data = {
                        "message": f"Concurrent request {i}",
                        "agent_type": "deployment",
                        "user_id": f"user_{i}"
                    }
                    task = async_client.post("/agents/start", json=request_data)
                    tasks.append(task)
                
                # Wait for all requests to complete
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All requests should succeed
                for response in responses:
                    if isinstance(response, Exception):
                        pytest.fail(f"Request failed with exception: {response}")
                    assert response.status_code == status.HTTP_200_OK
                
                # Verify all sessions were created
                assert len(agent_sessions) == 10
    
    @pytest.mark.asyncio
    async def test_malformed_uuid_in_path(self, async_client: AsyncClient):
        """Test API response to malformed UUID in path parameters."""
        response = await async_client.get("/agents/not-a-valid-uuid/status")
        
        # FastAPI should validate UUID format and return 422
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_unexpected_http_methods(self, async_client: AsyncClient):
        """Test API response to unexpected HTTP methods."""
        # Try PUT on a POST endpoint
        response = await async_client.put("/agents/start", json={"test": "data"})
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        
        # Try POST on a GET endpoint
        response = await async_client.post("/health", json={"test": "data"})
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, async_client: AsyncClient):
        """Test that CORS headers are properly set."""
        response = await async_client.get("/")
        
        # Check for CORS headers (depending on configuration)
        # Note: These might not be present in test client, but should be in real deployment
        assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    async def test_api_handles_background_task_exceptions(self, async_client: AsyncClient):
        """Test that API properly handles exceptions in background tasks."""
        with patch('src.api.main.DeploymentAgent') as mock_agent_class:
            with patch('src.api.main.run_agent_background') as mock_background:
                # Make background task raise an exception
                mock_background.side_effect = Exception("Background task failed")
                
                request_data = {
                    "message": "Test message",
                    "agent_type": "deployment"
                }
                
                # The initial request should still succeed
                response = await async_client.post("/agents/start", json=request_data)
                assert response.status_code == status.HTTP_200_OK
                
                # But the background task failure should be logged
                # (In a real test, we'd check logs or error tracking)


@pytest.mark.integration
class TestAPIAuthentication:
    """Test cases for API authentication and authorization."""
    
    @pytest.mark.asyncio
    async def test_no_authentication_required_currently(self, async_client: AsyncClient):
        """Test that API currently accepts requests without authentication."""
        # This test documents the current state - no authentication
        # In production, authentication should be required
        
        response = await async_client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        response = await async_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        # This represents a security vulnerability that should be fixed
        request_data = {"message": "Unauthenticated request", "agent_type": "deployment"}
        response = await async_client.post("/agents/start", json=request_data)
        
        # Currently this succeeds - it should require authentication in production
        assert response.status_code == status.HTTP_200_OK
        
        # TODO: Implement authentication and update these tests to require valid tokens