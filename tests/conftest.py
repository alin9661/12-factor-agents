"""
Global test configuration and fixtures.
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.agents.llm_providers import LLMProviderFactory, OpenAIProvider, AnthropicProvider
from src.api.main import app
from src.tools.registry import ToolRegistry
from src.tools.schemas import AgentContext, ToolResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    provider = MagicMock(spec=OpenAIProvider)
    provider.generate_response = AsyncMock()
    provider.model = "gpt-4o-mini"
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Mock Anthropic provider for testing."""
    provider = MagicMock(spec=AnthropicProvider)
    provider.generate_response = AsyncMock()
    provider.model = "claude-3-haiku-20240307"
    return provider


@pytest.fixture
def sample_agent_context():
    """Create a sample agent context for testing."""
    return AgentContext(
        thread_id=str(uuid.uuid4()),
        user_id="test_user_123",
        session_data={"environment": "test", "user_role": "admin"},
        conversation_history=[
            {
                "type": "user_message",
                "content": "Deploy the user service to staging",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ],
        execution_state="running",
        created_at=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def sample_tool_result():
    """Create a sample tool result for testing."""
    return ToolResult(
        success=True,
        result={"status": "completed", "message": "Tool executed successfully"},
        metadata={"tool_name": "test_tool", "execution_time": 0.5}
    )


@pytest.fixture
def failed_tool_result():
    """Create a failed tool result for testing."""
    return ToolResult(
        success=False,
        error="Tool execution failed: Invalid parameters",
        metadata={"tool_name": "test_tool", "error_type": "ValidationError"}
    )


@pytest.fixture
def test_tool_registry():
    """Create a fresh tool registry for testing."""
    registry = ToolRegistry()
    
    # Register a test tool
    def test_tool(message: str, count: int = 1):
        """A simple test tool."""
        return {"message": message, "count": count, "executed": True}
    
    registry.register_tool(
        name="test_tool",
        description="A simple test tool for testing",
        parameters_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "count": {"type": "integer", "minimum": 1}
            },
            "required": ["message"]
        },
        function=test_tool,
        category="test",
        risk_level="low"
    )
    
    # Register a high-risk tool
    async def risky_tool(action: str, confirm: bool = False):
        """A risky tool that requires approval."""
        if not confirm:
            raise ValueError("Confirmation required for risky actions")
        return {"action": action, "status": "executed", "risk": "high"}
    
    registry.register_tool(
        name="risky_tool", 
        description="A high-risk tool requiring approval",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "confirm": {"type": "boolean"}
            },
            "required": ["action"]
        },
        function=risky_tool,
        category="test",
        risk_level="high",
        requires_approval=True
    )
    
    return registry


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_file_content():
    """Sample file content for testing file operations."""
    return """# Test Configuration File
environment: test
debug: true
api_key: test_key_123
database_url: sqlite:///test.db

# Security settings
allowed_hosts:
  - localhost
  - 127.0.0.1
  - test.example.com

# Feature flags
features:
  new_ui: true
  beta_features: false
"""


@pytest.fixture
def malicious_file_paths():
    """Common malicious file paths for security testing."""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",  
        "../../.env",
        "../config.json",
        "file:///etc/passwd",
        "\\\\network\\share\\secret.txt",
        "%2E%2E%2F%2E%2E%2F%2E%2E%2Fetc%2Fpasswd",  # URL encoded ../../../etc/passwd
    ]


@pytest.fixture
def sql_injection_payloads():
    """Common SQL injection payloads for security testing."""
    return [
        "'; DROP TABLE users; --",
        "' OR 1=1 --",
        "admin'--",
        "' UNION SELECT password FROM users WHERE username='admin'--",
        "'; EXEC xp_cmdshell('rm -rf /'); --",
    ]


@pytest.fixture  
def xss_payloads():
    """Common XSS payloads for security testing."""
    return [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "';alert('XSS');//",
        "<svg onload=alert('XSS')>",
    ]


@pytest.fixture
def mock_llm_responses():
    """Pre-defined mock responses for different LLM scenarios."""
    return {
        "tool_execution": {
            "intent": "execute_tool",
            "tool_name": "deploy_service", 
            "parameters": {
                "service_name": "user-service",
                "version": "1.2.3",
                "environment": "staging"
            },
            "requires_approval": True
        },
        "human_input": {
            "intent": "request_human_input",
            "question": "What version should I deploy to production?",
            "context": "The staging deployment was successful. Ready for production deployment.",
            "urgency": "medium",
            "format": "free_text"
        },
        "completion": {
            "intent": "complete",
            "summary": "Successfully deployed user-service v1.2.3 to staging environment",
            "results": {
                "service": "user-service",
                "version": "1.2.3", 
                "environment": "staging",
                "status": "deployed"
            }
        },
        "error": {
            "intent": "error_occurred",
            "error_message": "Failed to connect to deployment API",
            "error_type": "connection_error",
            "recovery_suggestion": "Check network connectivity and API credentials"
        }
    }


@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    return {
        "concurrent_requests": 50,
        "request_timeout": 30.0,
        "expected_response_time": 2.0,  # seconds
        "sample_payloads": [
            {
                "message": f"Test message {i}",
                "agent_type": "deployment",
                "user_id": f"test_user_{i}",
                "session_data": {"test_id": i}
            }
            for i in range(100)
        ]
    }


# Custom pytest markers for different test categories
pytest_plugins = []

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests with external systems") 
    config.addinivalue_line("markers", "security: Security vulnerability tests")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Mock environment variables to avoid requiring real API keys
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8000")