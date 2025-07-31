"""
Unit tests for built-in tools.

These tests verify the functionality of all built-in tools provided
by the 12-factor agents system.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools import builtin_tools
from src.tools.registry import tool_registry


@pytest.mark.unit
class TestUtilityTools:
    """Test cases for utility tools."""
    
    def test_log_message_basic(self):
        """Test basic log message functionality."""
        result = builtin_tools.log_message("Test message")
        
        assert result["status"] == "logged"
        assert result["message"] == "Test message"
        assert result["level"] == "info"
        assert "timestamp" in result
        
        # Verify timestamp is in ISO format
        try:
            datetime.fromisoformat(result["timestamp"])
        except ValueError:
            pytest.fail("Timestamp is not in valid ISO format")
    
    def test_log_message_different_levels(self):
        """Test log message with different log levels."""
        levels = ["debug", "info", "warning", "error"]
        
        for level in levels:
            result = builtin_tools.log_message(f"Test {level} message", level=level)
            
            assert result["level"] == level
            assert result["message"] == f"Test {level} message"
            assert result["status"] == "logged"
    
    def test_log_message_with_context(self):
        """Test log message with context parameter."""
        context = {
            "metadata": {"user_id": "test_user", "session_id": "session_123"}
        }
        
        result = builtin_tools.log_message("Context test", context=context)
        
        assert result["status"] == "logged"
        assert result["message"] == "Context test"
    
    @pytest.mark.asyncio
    async def test_sleep_tool_basic(self):
        """Test basic sleep tool functionality."""
        start_time = asyncio.get_event_loop().time()
        
        result = await builtin_tools.sleep_tool(1)  # Sleep for 1 second
        
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        assert result["status"] == "completed"
        assert result["slept_seconds"] == 1
        assert "timestamp" in result
        assert elapsed >= 0.9  # Allow some tolerance for timing
    
    @pytest.mark.asyncio
    async def test_sleep_tool_with_context(self):
        """Test sleep tool with context parameter."""
        context = {"metadata": {"reason": "rate_limiting"}}
        
        result = await builtin_tools.sleep_tool(0.1, context=context)
        
        assert result["status"] == "completed"
        assert result["slept_seconds"] == 0.1
    
    @pytest.mark.asyncio
    async def test_sleep_tool_zero_seconds(self):
        """Test sleep tool with zero seconds."""
        result = await builtin_tools.sleep_tool(0)
        
        assert result["status"] == "completed"
        assert result["slept_seconds"] == 0


@pytest.mark.unit
class TestHTTPTools:
    """Test cases for HTTP tools."""
    
    @pytest.mark.asyncio
    async def test_http_get_success(self):
        """Test successful HTTP GET request."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock the async context manager
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '{"message": "success"}'
            mock_response.headers = {"content-type": "application/json"}
            mock_response.url = "https://api.example.com/test"
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            
            result = await builtin_tools.http_get("https://api.example.com/test")
            
            assert result["status_code"] == 200
            assert result["content"] == '{"message": "success"}'
            assert result["url"] == "https://api.example.com/test"
            assert "headers" in result
            
            # Verify the HTTP client was called correctly
            mock_client.get.assert_called_once_with("https://api.example.com/test", headers={})
    
    @pytest.mark.asyncio
    async def test_http_get_with_headers(self):
        """Test HTTP GET request with custom headers."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_response.headers = {}
            mock_response.url = "https://api.example.com/test"
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            
            custom_headers = {"Authorization": "Bearer token123", "User-Agent": "Test Client"}
            
            result = await builtin_tools.http_get("https://api.example.com/test", headers=custom_headers)
            
            assert result["status_code"] == 200
            mock_client.get.assert_called_once_with("https://api.example.com/test", headers=custom_headers)
    
    @pytest.mark.asyncio
    async def test_http_get_error(self):
        """Test HTTP GET request with error."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock HTTP error
            import httpx
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            
            with pytest.raises(httpx.HTTPError, match="Connection failed"):
                await builtin_tools.http_get("https://api.example.com/test")
    
    @pytest.mark.asyncio
    async def test_http_post_with_json_data(self):
        """Test HTTP POST request with JSON data."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.text = '{"id": 123, "status": "created"}'
            mock_response.headers = {"content-type": "application/json"}
            mock_response.url = "https://api.example.com/create"
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            
            json_data = {"name": "test item", "value": 42}
            
            result = await builtin_tools.http_post("https://api.example.com/create", json_data=json_data)
            
            assert result["status_code"] == 201
            assert '"id": 123' in result["content"]
            
            # Verify JSON data was passed correctly
            mock_client.post.assert_called_once_with(
                "https://api.example.com/create",
                json=json_data,
                headers={}
            )
    
    @pytest.mark.asyncio
    async def test_http_post_with_form_data(self):
        """Test HTTP POST request with form data."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "Form submitted"
            mock_response.headers = {}
            mock_response.url = "https://api.example.com/submit"
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            
            form_data = {"field1": "value1", "field2": "value2"}
            
            result = await builtin_tools.http_post("https://api.example.com/submit", data=form_data)
            
            assert result["status_code"] == 200
            assert result["content"] == "Form submitted"
            
            mock_client.post.assert_called_once_with(
                "https://api.example.com/submit",
                data=form_data,
                headers={}
            )
    
    @pytest.mark.asyncio
    async def test_http_post_error(self):
        """Test HTTP POST request with error."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            import httpx
            mock_client.post.side_effect = httpx.HTTPError("Server error")
            
            with pytest.raises(httpx.HTTPError, match="Server error"):
                await builtin_tools.http_post("https://api.example.com/test", json_data={"test": "data"})


@pytest.mark.unit
class TestFileSystemTools:
    """Test cases for filesystem tools."""
    
    def test_read_file_success(self, temp_directory: str, sample_file_content: str):
        """Test successful file reading."""
        import os
        
        # Create a test file
        test_file_path = os.path.join(temp_directory, "test_file.txt")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(sample_file_content)
        
        result = builtin_tools.read_file(test_file_path)
        
        assert result["content"] == sample_file_content
        assert result["file_path"] == test_file_path
        assert result["size_bytes"] == len(sample_file_content.encode('utf-8'))
        assert result["lines"] == len(sample_file_content.splitlines())
    
    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            builtin_tools.read_file("/path/to/nonexistent/file.txt")
    
    def test_read_file_with_context(self, temp_directory: str):
        """Test reading file with context parameter."""
        import os
        
        test_file_path = os.path.join(temp_directory, "context_test.txt")
        test_content = "Context test content"
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        context = {"metadata": {"operation": "read_config"}}
        result = builtin_tools.read_file(test_file_path, context=context)
        
        assert result["content"] == test_content
        assert result["file_path"] == test_file_path
    
    def test_write_file_success(self, temp_directory: str):
        """Test successful file writing."""
        import os
        
        test_file_path = os.path.join(temp_directory, "write_test.txt")
        test_content = "This is test content for writing."
        
        result = builtin_tools.write_file(test_file_path, test_content)
        
        assert result["status"] == "success"
        assert result["file_path"] == test_file_path
        assert result["bytes_written"] == len(test_content.encode('utf-8'))
        assert result["mode"] == "w"
        
        # Verify file was actually written
        with open(test_file_path, 'r', encoding='utf-8') as f:
            written_content = f.read()
        
        assert written_content == test_content
    
    def test_write_file_append_mode(self, temp_directory: str):
        """Test file writing in append mode."""
        import os
        
        test_file_path = os.path.join(temp_directory, "append_test.txt")
        initial_content = "Initial content\n"
        append_content = "Appended content\n"
        
        # Write initial content
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)
        
        # Append additional content
        result = builtin_tools.write_file(test_file_path, append_content, mode="a")
        
        assert result["status"] == "success"
        assert result["mode"] == "a"
        
        # Verify both contents are present
        with open(test_file_path, 'r', encoding='utf-8') as f:
            final_content = f.read()
        
        assert final_content == initial_content + append_content
    
    def test_write_file_error(self):
        """Test file writing to invalid location."""
        with pytest.raises(Exception):  # Could be PermissionError or OSError
            builtin_tools.write_file("/root/protected_file.txt", "test content")
    
    def test_write_file_with_context(self, temp_directory: str):
        """Test writing file with context parameter."""
        import os
        
        test_file_path = os.path.join(temp_directory, "context_write.txt")
        test_content = "Content with context"
        context = {"metadata": {"operation": "config_update"}}
        
        result = builtin_tools.write_file(test_file_path, test_content, context=context)
        
        assert result["status"] == "success"
        assert result["file_path"] == test_file_path


@pytest.mark.unit
class TestCommunicationTools:
    """Test cases for communication tools."""
    
    def test_send_email_basic(self):
        """Test basic email sending functionality."""
        # Note: This tests the interface, not actual email sending
        result = builtin_tools.send_email(
            to_email="test@example.com",
            subject="Test Subject",
            body="Test email body"
        )
        
        assert result["status"] == "sent"
        assert result["to"] == "test@example.com"
        assert result["subject"] == "Test Subject"
        assert "timestamp" in result
    
    def test_send_email_with_from_address(self):
        """Test email sending with custom from address."""
        result = builtin_tools.send_email(
            to_email="recipient@example.com",
            subject="Custom From Test",
            body="Test body",
            from_email="sender@example.com"
        )
        
        assert result["status"] == "sent"
        assert result["to"] == "recipient@example.com"
        assert result["subject"] == "Custom From Test"
    
    def test_send_email_with_smtp_config(self):
        """Test email sending with SMTP configuration."""
        result = builtin_tools.send_email(
            to_email="test@example.com",
            subject="SMTP Config Test",
            body="Test body",
            smtp_server="smtp.example.com",
            smtp_port=587
        )
        
        assert result["status"] == "sent"
    
    def test_send_email_with_context(self):
        """Test email sending with context parameter."""
        context = {"metadata": {"notification_type": "alert"}}
        
        result = builtin_tools.send_email(
            to_email="alert@example.com",
            subject="Alert Notification",
            body="This is an alert",
            context=context
        )
        
        assert result["status"] == "sent"


@pytest.mark.unit
class TestDeploymentTools:
    """Test cases for deployment tools."""
    
    @pytest.mark.asyncio
    async def test_check_service_status_success(self):
        """Test successful service status check."""
        result = await builtin_tools.check_service_status("user-service", "production")
        
        assert result["service"] == "user-service"
        assert result["environment"] == "production"
        assert result["status"] in ["healthy", "degraded", "unhealthy", "unknown"]
        assert "uptime" in result
        assert "last_deployment" in result
        assert "version" in result
        assert "instances" in result
    
    @pytest.mark.asyncio
    async def test_check_service_status_default_environment(self):
        """Test service status check with default environment."""
        result = await builtin_tools.check_service_status("api-service")
        
        assert result["service"] == "api-service"
        assert result["environment"] == "production"  # Default environment
    
    @pytest.mark.asyncio
    async def test_check_service_status_with_context(self):
        """Test service status check with context parameter."""
        context = {"metadata": {"monitoring_system": "prometheus"}}
        
        result = await builtin_tools.check_service_status("web-service", "staging", context=context)
        
        assert result["service"] == "web-service"
        assert result["environment"] == "staging"
    
    @pytest.mark.asyncio
    async def test_deploy_service_success(self):
        """Test successful service deployment."""
        result = await builtin_tools.deploy_service("user-service", "1.2.3", "staging")
        
        assert result["status"] == "deployed"
        assert result["service"] == "user-service"
        assert result["version"] == "1.2.3"
        assert result["environment"] == "staging"
        assert "deployment_id" in result
        assert "timestamp" in result
        assert "estimated_rollout_time" in result
        
        # Verify deployment_id format
        assert result["deployment_id"].startswith("deploy-")
    
    @pytest.mark.asyncio
    async def test_deploy_service_with_context(self):
        """Test service deployment with context parameter."""
        context = {"metadata": {"initiated_by": "github_actions"}}
        
        result = await builtin_tools.deploy_service("api-service", "2.0.0", "production", context=context)
        
        assert result["service"] == "api-service"
        assert result["version"] == "2.0.0"
        assert result["environment"] == "production"
        assert result["status"] == "deployed"


@pytest.mark.unit
class TestCustomerSupportTools:
    """Test cases for customer support tools."""
    
    @pytest.mark.asyncio
    async def test_lookup_customer_by_email(self):
        """Test customer lookup by email."""
        result = await builtin_tools.lookup_customer("customer@example.com")
        
        assert result["customer_id"] == "cust_123456"
        assert result["email"] == "customer@example.com"
        assert result["name"] == "John Doe"
        assert result["account_status"] == "active"
        assert result["tier"] == "premium"
        assert "signup_date" in result
        assert "total_orders" in result
        assert "last_login" in result
    
    @pytest.mark.asyncio
    async def test_lookup_customer_by_id(self):
        """Test customer lookup by customer ID."""
        result = await builtin_tools.lookup_customer("cust_456789")
        
        assert result["customer_id"] == "cust_123456"
        assert result["email"] == "customer@example.com"  # Mock returns default email
        assert result["name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_lookup_customer_with_context(self):
        """Test customer lookup with context parameter."""
        context = {"metadata": {"support_agent": "agent_123"}}
        
        result = await builtin_tools.lookup_customer("test@example.com", context=context)
        
        assert result["email"] == "test@example.com"
        assert result["customer_id"] == "cust_123456"
    
    @pytest.mark.asyncio
    async def test_create_support_ticket_basic(self):
        """Test basic support ticket creation."""
        result = await builtin_tools.create_support_ticket(
            customer_id="cust_123456",
            issue_title="Cannot login",
            issue_description="User reports unable to access account"
        )
        
        assert result["customer_id"] == "cust_123456"
        assert result["title"] == "Cannot login"
        assert result["description"] == "User reports unable to access account"
        assert result["priority"] == "medium"  # Default priority
        assert result["status"] == "open"
        assert result["assigned_to"] == "support-queue"
        assert "ticket_id" in result
        assert "created_at" in result
        
        # Verify ticket_id format
        assert result["ticket_id"].startswith("TICKET-")
    
    @pytest.mark.asyncio
    async def test_create_support_ticket_with_priority(self):
        """Test support ticket creation with custom priority."""
        result = await builtin_tools.create_support_ticket(
            customer_id="cust_789012",
            issue_title="System outage",
            issue_description="Complete service unavailable",
            priority="high"
        )
        
        assert result["priority"] == "high"
        assert result["title"] == "System outage"
    
    @pytest.mark.asyncio
    async def test_create_support_ticket_with_context(self):
        """Test support ticket creation with context parameter."""
        context = {"metadata": {"source": "web_chat", "agent": "support_bot"}}
        
        result = await builtin_tools.create_support_ticket(
            customer_id="cust_345678",
            issue_title="Billing question",
            issue_description="Customer has question about recent charge",
            priority="low",
            context=context
        )
        
        assert result["customer_id"] == "cust_345678"
        assert result["priority"] == "low"


@pytest.mark.unit
class TestToolRegistrationIntegration:
    """Test that all built-in tools are properly registered."""
    
    def test_all_tools_registered(self):
        """Test that all expected built-in tools are registered in the registry."""
        # Import to ensure tools are registered
        import src.tools.builtin_tools
        
        expected_tools = [
            "log_message",
            "sleep",
            "http_get",
            "http_post",
            "read_file",
            "write_file",
            "send_email",
            "check_service_status",
            "deploy_service",
            "lookup_customer",
            "create_support_ticket"
        ]
        
        registered_tools = list(tool_registry.tools.keys())
        
        for expected_tool in expected_tools:
            assert expected_tool in registered_tools, f"Tool '{expected_tool}' is not registered"
    
    def test_tool_categorization(self):
        """Test that tools are properly categorized."""
        import src.tools.builtin_tools
        
        expected_categories = {
            "log_message": "utility",
            "sleep": "utility",
            "http_get": "http",
            "http_post": "http",
            "read_file": "filesystem",
            "write_file": "filesystem",
            "send_email": "communication",
            "check_service_status": "deployment",
            "deploy_service": "deployment",
            "lookup_customer": "customer_support",
            "create_support_ticket": "customer_support"
        }
        
        for tool_name, expected_category in expected_categories.items():
            tool_def = tool_registry.get_tool_definition(tool_name)
            assert tool_def is not None, f"Tool '{tool_name}' not found"
            assert tool_def.category == expected_category, f"Tool '{tool_name}' has wrong category"
    
    def test_tool_risk_levels(self):
        """Test that tools have appropriate risk levels."""
        import src.tools.builtin_tools
        
        expected_risk_levels = {
            "log_message": "low",
            "sleep": "low",
            "http_get": "medium",
            "http_post": "medium",
            "read_file": "low",
            "write_file": "high",
            "send_email": "high",
            "check_service_status": "low",
            "deploy_service": "high",
            "lookup_customer": "medium",
            "create_support_ticket": "low"
        }
        
        for tool_name, expected_risk in expected_risk_levels.items():
            tool_def = tool_registry.get_tool_definition(tool_name)
            assert tool_def is not None, f"Tool '{tool_name}' not found"
            assert tool_def.risk_level == expected_risk, f"Tool '{tool_name}' has wrong risk level"
    
    def test_high_risk_tools_require_approval(self):
        """Test that high-risk tools require approval."""
        import src.tools.builtin_tools
        
        high_risk_tools = ["write_file", "send_email", "deploy_service"]
        
        for tool_name in high_risk_tools:
            tool_def = tool_registry.get_tool_definition(tool_name)
            assert tool_def is not None, f"Tool '{tool_name}' not found"
            assert tool_def.requires_approval is True, f"High-risk tool '{tool_name}' should require approval"
    
    def test_tool_schema_completeness(self):
        """Test that all tools have complete parameter schemas."""
        import src.tools.builtin_tools
        
        for tool_name, tool_def in tool_registry.tools.items():
            assert tool_def.parameters_schema is not None, f"Tool '{tool_name}' missing parameters_schema"
            assert "type" in tool_def.parameters_schema, f"Tool '{tool_name}' schema missing type"
            
            if tool_def.parameters_schema.get("type") == "object":
                assert "properties" in tool_def.parameters_schema, f"Tool '{tool_name}' object schema missing properties"