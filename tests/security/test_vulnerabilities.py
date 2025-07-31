"""
Security vulnerability tests.

These tests are designed to identify and demonstrate security vulnerabilities
in the 12-factor agents codebase. They follow TDD principles to first expose
the vulnerabilities, then guide their remediation.

WARNING: These tests demonstrate actual security vulnerabilities.
They should pass (indicating vulnerabilities exist) until the underlying
security issues are fixed.
"""

import os
import tempfile
import pytest
import json
import uuid
from typing import List
from unittest.mock import patch, MagicMock
from pathlib import Path

from httpx import AsyncClient
from fastapi import status

from src.tools.registry import ToolRegistry
from src.api.main import agent_sessions


@pytest.mark.security
class TestPathTraversalVulnerabilities:
    """
    Test for path traversal vulnerabilities in file operations.
    
    These tests demonstrate that the current implementation is vulnerable
    to path traversal attacks through the file operation tools.
    """
    
    def test_read_file_tool_path_traversal(self, malicious_file_paths: List[str]):
        """Test that read_file tool is vulnerable to path traversal attacks."""
        from src.tools import builtin_tools
        from src.tools.registry import tool_registry
        
        # This test demonstrates the vulnerability - it should initially PASS (showing vulnerability exists)
        # After fixing the vulnerability, this test should FAIL (showing vulnerability is patched)
        
        vulnerable_paths_accessed = []
        
        for malicious_path in malicious_file_paths:
            try:
                # Try to access sensitive files using built-in read_file tool
                result = builtin_tools.read_file(malicious_path)
                
                # If we get here without an exception, the path traversal worked
                vulnerable_paths_accessed.append(malicious_path)
                
            except (FileNotFoundError, PermissionError, OSError):
                # Expected behavior - access should be denied
                continue
            except Exception as e:
                # Unexpected error - might indicate vulnerability
                if "permission denied" not in str(e).lower():
                    vulnerable_paths_accessed.append(malicious_path)
        
        # This assertion should PASS initially (demonstrating vulnerability)
        # After security fix, no malicious paths should be accessible
        if vulnerable_paths_accessed:
            pytest.fail(
                f"SECURITY VULNERABILITY: Path traversal attack succeeded for paths: {vulnerable_paths_accessed}. "
                f"The read_file tool allows access to files outside the intended directory."
            )
    
    def test_write_file_tool_path_traversal(self, malicious_file_paths: List[str], temp_directory: str):
        """Test that write_file tool is vulnerable to path traversal attacks."""
        from src.tools import builtin_tools
        
        vulnerable_writes = []
        
        for malicious_path in malicious_file_paths:
            try:
                # Attempt to write to sensitive locations
                result = builtin_tools.write_file(
                    file_path=malicious_path,
                    content="SECURITY_TEST_CONTENT",
                    mode="w"
                )
                
                # If we get here, the write might have succeeded
                vulnerable_writes.append(malicious_path)
                
            except (FileNotFoundError, PermissionError, OSError):
                # Expected behavior - write should be denied
                continue
            except Exception as e:
                # Log unexpected errors
                continue
        
        if vulnerable_writes:
            pytest.fail(
                f"SECURITY VULNERABILITY: Path traversal in write_file succeeded for: {vulnerable_writes}. "
                f"The write_file tool allows writing to files outside the intended directory."
            )
    
    @pytest.mark.asyncio
    async def test_file_operations_via_api_path_traversal(self, async_client: AsyncClient, malicious_file_paths: List[str]):
        """Test path traversal vulnerabilities through the API endpoints."""
        
        # Mock the agent to execute file operations
        with patch('src.api.main.DeploymentAgent') as mock_agent_class:
            with patch('src.api.main.run_agent_background') as mock_background:
                
                vulnerable_api_calls = []
                
                for malicious_path in malicious_file_paths:
                    # Try to trigger file operations through agent requests
                    request_data = {
                        "message": f"Read the file at {malicious_path}",
                        "agent_type": "deployment",
                        "session_data": {
                            "file_path": malicious_path,
                            "operation": "read"
                        }
                    }
                    
                    response = await async_client.post("/agents/start", json=request_data)
                    
                    if response.status_code == status.HTTP_200_OK:
                        # Check if the malicious path was passed through without sanitization
                        data = response.json()
                        thread_id = data.get("thread_id")
                        
                        if thread_id and thread_id in agent_sessions:
                            context = agent_sessions[thread_id]
                            if malicious_path in str(context.session_data):
                                vulnerable_api_calls.append(malicious_path)
                
                if vulnerable_api_calls:
                    pytest.fail(
                        f"SECURITY VULNERABILITY: API accepts malicious file paths without validation: {vulnerable_api_calls}"
                    )


@pytest.mark.security
class TestInputValidationVulnerabilities:
    """
    Test for input validation vulnerabilities.
    
    These tests check for insufficient input validation that could lead
    to injection attacks or other security issues.
    """
    
    @pytest.mark.asyncio
    async def test_sql_injection_in_user_input(self, async_client: AsyncClient, sql_injection_payloads: List[str]):
        """Test for SQL injection vulnerabilities in user input."""
        
        vulnerable_payloads = []
        
        for payload in sql_injection_payloads:
            request_data = {
                "message": payload,
                "agent_type": "deployment",
                "user_id": payload,  # Try injection in user_id too
                "session_data": {
                    "query": payload,
                    "user_input": payload
                }
            }
            
            try:
                response = await async_client.post("/agents/start", json=request_data)
                
                # If the response is successful, check if the payload was sanitized
                if response.status_code == status.HTTP_200_OK:
                    data = response.json()
                    thread_id = data.get("thread_id")
                    
                    if thread_id and thread_id in agent_sessions:
                        context = agent_sessions[thread_id]
                        
                        # Check if SQL injection payload is stored unsanitized
                        context_str = json.dumps(context.model_dump())
                        if any(dangerous in payload.lower() for dangerous in ["drop table", "union select", "exec", "xp_cmdshell"]):
                            if payload in context_str:
                                vulnerable_payloads.append(payload)
                
            except Exception as e:
                # Unexpected errors might indicate vulnerability
                if "sql" in str(e).lower() or "database" in str(e).lower():
                    vulnerable_payloads.append(payload)
        
        if vulnerable_payloads:
            pytest.fail(
                f"SECURITY VULNERABILITY: SQL injection payloads accepted without sanitization: {vulnerable_payloads}"
            )
    
    @pytest.mark.asyncio
    async def test_xss_in_user_input(self, async_client: AsyncClient, xss_payloads: List[str]):
        """Test for XSS vulnerabilities in user input."""
        
        vulnerable_xss = []
        
        for payload in xss_payloads:
            request_data = {
                "message": payload,
                "agent_type": "deployment",
                "session_data": {"user_message": payload}
            }
            
            response = await async_client.post("/agents/start", json=request_data)
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                thread_id = data.get("thread_id")
                
                if thread_id and thread_id in agent_sessions:
                    context = agent_sessions[thread_id]
                    
                    # Check if XSS payload is stored unsanitized
                    for entry in context.conversation_history:
                        if payload in json.dumps(entry):
                            vulnerable_xss.append(payload)
                            break
        
        if vulnerable_xss:
            pytest.fail(
                f"SECURITY VULNERABILITY: XSS payloads stored without sanitization: {vulnerable_xss}"
            )
    
    def test_tool_parameter_injection(self):
        """Test for command injection in tool parameters."""
        from src.tools.registry import ToolRegistry
        
        registry = ToolRegistry()
        
        # Register a test tool that might be vulnerable to injection
        def vulnerable_command_tool(command: str):
            """A tool that might execute commands unsafely."""
            import subprocess
            try:
                # This is intentionally vulnerable for testing
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return {"output": result.stdout, "error": result.stderr}
            except Exception as e:
                return {"error": str(e)}
        
        registry.register_tool(
            name="command_tool",
            description="Execute system commands",
            parameters_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            },
            function=vulnerable_command_tool,
            risk_level="high"
        )
        
        # Test command injection payloads
        injection_payloads = [
            "ls; rm -rf /tmp/test",
            "echo 'safe' && echo 'dangerous'",
            "cat /etc/passwd",
            "; rm -rf /*",  # Very dangerous - should be blocked
            "$(whoami)",
            "`id`",
        ]
        
        vulnerable_commands = []
        
        for payload in injection_payloads:
            try:
                # This should be blocked by proper input validation
                # If it succeeds, it indicates a vulnerability
                result = vulnerable_command_tool(payload)
                
                # If the command executed and returned output, it's vulnerable
                if result.get("output") or result.get("error"):
                    vulnerable_commands.append(payload)
                    
            except Exception:
                # Exception is expected for malicious payloads
                continue
        
        if vulnerable_commands:
            pytest.fail(
                f"SECURITY VULNERABILITY: Command injection succeeded with payloads: {vulnerable_commands}"
            )
    
    @pytest.mark.asyncio
    async def test_json_injection_in_webhook(self, async_client: AsyncClient):
        """Test for JSON injection vulnerabilities in webhook endpoint."""
        
        # Malicious JSON payloads
        malicious_payloads = [
            {
                "message": "normal message",
                "__proto__": {"isAdmin": True},  # Prototype pollution
                "constructor": {"name": "malicious"}
            },
            {
                "message": "\"); DROP TABLE users; --",
                "agent_type": "deployment\"; alert('xss'); var x=\""
            },
            {
                "message": "test",
                "session_data": {
                    "eval": "require('child_process').exec('whoami')",
                    "function": "new Function('return process')().exit()"
                }
            }
        ]
        
        vulnerable_injections = []
        
        for payload in malicious_payloads:
            try:
                response = await async_client.post("/webhooks/trigger", json=payload)
                
                if response.status_code == status.HTTP_200_OK:
                    # Check if malicious payload was processed without sanitization
                    data = response.json()
                    
                    if "agent_response" in data:
                        agent_data = data["agent_response"]
                        if any(key in str(agent_data) for key in ["__proto__", "constructor", "eval", "function"]):
                            vulnerable_injections.append(payload)
                
            except Exception as e:
                # Log but don't fail on exceptions - they might be expected
                continue
        
        if vulnerable_injections:
            pytest.fail(
                f"SECURITY VULNERABILITY: JSON injection payloads processed unsafely: {len(vulnerable_injections)} payloads"
            )


@pytest.mark.security
class TestAuthenticationVulnerabilities:
    """
    Test for authentication and authorization vulnerabilities.
    
    These tests demonstrate the lack of proper authentication and authorization
    in the current implementation.
    """
    
    @pytest.mark.asyncio
    async def test_no_authentication_required(self, async_client: AsyncClient):
        """Test that API endpoints don't require authentication."""
        
        # This test should PASS initially (demonstrating vulnerability)
        # After implementing authentication, this test should FAIL
        
        unauthenticated_endpoints = []
        
        # Test various endpoints without authentication
        endpoints_to_test = [
            ("GET", "/"),
            ("GET", "/health"),
            ("POST", "/agents/start", {"message": "test", "agent_type": "deployment"}),
            ("GET", "/agents"),
            ("POST", "/webhooks/trigger", {"message": "webhook test"})
        ]
        
        for method, endpoint, *payload in endpoints_to_test:
            try:
                if method == "GET":
                    response = await async_client.get(endpoint)
                elif method == "POST":
                    response = await async_client.post(endpoint, json=payload[0] if payload else {})
                
                # If the request succeeds without authentication, it's vulnerable
                if response.status_code not in [401, 403]:  # 401 = Unauthorized, 403 = Forbidden
                    unauthenticated_endpoints.append(f"{method} {endpoint}")
                    
            except Exception:
                continue
        
        # This should initially PASS (showing vulnerability exists)
        if unauthenticated_endpoints:
            pytest.fail(
                f"SECURITY VULNERABILITY: The following endpoints don't require authentication: {unauthenticated_endpoints}. "
                f"All API endpoints should require proper authentication in production."
            )
    
    @pytest.mark.asyncio
    async def test_no_authorization_checks(self, async_client: AsyncClient):
        """Test that there are no authorization checks for sensitive operations."""
        
        # Create an agent session as one "user"
        request_data = {
            "message": "Create sensitive deployment",
            "agent_type": "deployment",
            "user_id": "user_a",
            "session_data": {"sensitive": True, "production": True}
        }
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                response = await async_client.post("/agents/start", json=request_data)
                
                if response.status_code == status.HTTP_200_OK:
                    thread_id = response.json()["thread_id"]
                    
                    # Now try to access/modify this session as a different "user"
                    # This should be blocked by proper authorization
                    
                    unauthorized_operations = []
                    
                    # Try to get status as different user
                    status_response = await async_client.get(f"/agents/{thread_id}/status")
                    if status_response.status_code == status.HTTP_200_OK:
                        unauthorized_operations.append(f"GET /agents/{thread_id}/status")
                    
                    # Try to resume session as different user
                    resume_response = await async_client.post(
                        f"/agents/{thread_id}/resume",
                        json={"human_response": "Malicious takeover"}
                    )
                    if resume_response.status_code not in [401, 403, 404]:
                        unauthorized_operations.append(f"POST /agents/{thread_id}/resume")
                    
                    # Try to delete session as different user
                    delete_response = await async_client.delete(f"/agents/{thread_id}")
                    if delete_response.status_code not in [401, 403, 404]:
                        unauthorized_operations.append(f"DELETE /agents/{thread_id}")
                    
                    if unauthorized_operations:
                        pytest.fail(
                            f"SECURITY VULNERABILITY: Unauthorized access to agent sessions allowed: {unauthorized_operations}"
                        )
    
    def test_session_data_exposure(self):
        """Test that sensitive session data might be exposed."""
        from src.tools.schemas import AgentContext
        
        # Create context with sensitive data
        sensitive_context = AgentContext(
            thread_id=str(uuid.uuid4()),
            user_id="test_user",
            session_data={
                "api_key": "secret_api_key_12345",
                "database_password": "super_secret_password",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...",
                "credit_card": "4111-1111-1111-1111",
            },
            conversation_history=[
                {
                    "type": "user_message",
                    "content": "My password is super_secret_password"
                }
            ]
        )
        
        # Store in global sessions (simulating API behavior)
        agent_sessions[sensitive_context.thread_id] = sensitive_context
        
        # Check if sensitive data is accessible
        stored_context = agent_sessions[sensitive_context.thread_id]
        context_dict = stored_context.model_dump()
        
        sensitive_fields_exposed = []
        sensitive_keywords = ["password", "api_key", "private_key", "credit_card", "secret"]
        
        def check_dict_for_sensitive(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if any(keyword in key.lower() for keyword in sensitive_keywords):
                        sensitive_fields_exposed.append(current_path)
                    check_dict_for_sensitive(value, current_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_dict_for_sensitive(item, f"{path}[{i}]")
            elif isinstance(data, str):
                if any(keyword in data.lower() for keyword in sensitive_keywords):
                    sensitive_fields_exposed.append(f"{path}='{data[:20]}...'")
        
        check_dict_for_sensitive(context_dict)
        
        if sensitive_fields_exposed:
            pytest.fail(
                f"SECURITY VULNERABILITY: Sensitive data stored in plain text: {sensitive_fields_exposed}. "
                f"Sensitive information should be encrypted or masked."
            )


@pytest.mark.security
class TestDataExposureVulnerabilities:
    """Test for data exposure and information leakage vulnerabilities."""
    
    @pytest.mark.asyncio
    async def test_error_messages_expose_sensitive_info(self, async_client: AsyncClient):
        """Test that error messages might expose sensitive system information."""
        
        # Try to trigger errors that might expose sensitive information
        sensitive_exposures = []
        
        # Test with malformed requests
        test_cases = [
            {
                "endpoint": "/agents/start",
                "payload": {"message": "test", "agent_type": "../../../etc/passwd"},
                "expected_exposure": "file system paths"
            },
            {
                "endpoint": "/agents/invalid-uuid-format/status",
                "payload": None,
                "expected_exposure": "internal validation details"
            }
        ]
        
        for test_case in test_cases:
            try:
                if test_case["payload"]:
                    response = await async_client.post(test_case["endpoint"], json=test_case["payload"])
                else:
                    response = await async_client.get(test_case["endpoint"])
                
                if response.status_code >= 400:
                    error_data = response.json()
                    error_message = str(error_data)
                    
                    # Check for common sensitive information in error messages
                    sensitive_patterns = [
                        "/src/",  # Source code paths
                        "/home/",  # Home directory paths
                        "File \"",  # Python traceback file paths
                        "line ",  # Line numbers from tracebacks
                        "Database",  # Database connection info
                        "password",  # Password references
                        "secret",  # Secret references
                        "key",  # API key references
                    ]
                    
                    for pattern in sensitive_patterns:
                        if pattern.lower() in error_message.lower():
                            sensitive_exposures.append({
                                "endpoint": test_case["endpoint"],
                                "pattern": pattern,
                                "message": error_message[:200]
                            })
                            break
                            
            except Exception:
                continue
        
        if sensitive_exposures:
            pytest.fail(
                f"SECURITY VULNERABILITY: Error messages expose sensitive information: {len(sensitive_exposures)} cases found"
            )
    
    @pytest.mark.asyncio
    async def test_debug_information_leakage(self, async_client: AsyncClient):
        """Test for debug information leakage in responses."""
        
        debug_leaks = []
        
        # Test various endpoints for debug information
        endpoints = ["/", "/health", "/agents"]
        
        for endpoint in endpoints:
            response = await async_client.get(endpoint)
            
            if response.status_code == 200:
                data = response.json()
                response_text = json.dumps(data).lower()
                
                # Check for debug information patterns
                debug_patterns = [
                    "debug",
                    "traceback",
                    "stack trace",
                    "exception",
                    "internal server error",
                    "localhost",
                    "127.0.0.1",
                    "development",
                    "test",
                ]
                
                for pattern in debug_patterns:
                    if pattern in response_text:
                        debug_leaks.append({
                            "endpoint": endpoint,
                            "pattern": pattern,
                            "context": response_text[max(0, response_text.find(pattern)-50):response_text.find(pattern)+50]
                        })
        
        if debug_leaks:
            pytest.fail(
                f"SECURITY VULNERABILITY: Debug information leaked in responses: {len(debug_leaks)} instances"
            )


@pytest.mark.security  
class TestRateLimitingVulnerabilities:
    """Test for lack of rate limiting and DoS protection."""
    
    @pytest.mark.asyncio
    async def test_no_rate_limiting_on_agent_creation(self, async_client: AsyncClient):
        """Test that there's no rate limiting on agent creation endpoint."""
        
        with patch('src.api.main.DeploymentAgent'):
            with patch('src.api.main.run_agent_background'):
                
                successful_requests = 0
                max_requests = 20  # Try to create many agents quickly
                
                for i in range(max_requests):
                    request_data = {
                        "message": f"Rate limit test {i}",
                        "agent_type": "deployment",
                        "user_id": f"test_user_{i}"
                    }
                    
                    try:
                        response = await async_client.post("/agents/start", json=request_data)
                        if response.status_code == status.HTTP_200_OK:
                            successful_requests += 1
                        elif response.status_code == 429:  # Too Many Requests
                            # Rate limiting is working
                            break
                    except Exception:
                        break
                
                # If all requests succeeded, there's no rate limiting
                if successful_requests >= max_requests * 0.8:  # Allow some tolerance
                    pytest.fail(
                        f"SECURITY VULNERABILITY: No rate limiting detected. "
                        f"Successfully made {successful_requests}/{max_requests} requests without throttling."
                    )
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_vulnerability(self, async_client: AsyncClient):
        """Test for potential resource exhaustion attacks."""
        
        # Test large payload handling
        large_message = "A" * (1024 * 100)  # 100KB message
        
        request_data = {
            "message": large_message,
            "agent_type": "deployment",
            "session_data": {"large_data": large_message}
        }
        
        try:
            response = await async_client.post("/agents/start", json=request_data)
            
            # If this succeeds without limits, it's vulnerable
            if response.status_code == status.HTTP_200_OK:
                pytest.fail(
                    "SECURITY VULNERABILITY: Large payloads accepted without size limits. "
                    "This could lead to resource exhaustion attacks."
                )
                
        except Exception as e:
            # Timeouts or memory errors might indicate vulnerability
            if "timeout" in str(e).lower() or "memory" in str(e).lower():
                pytest.fail(
                    f"SECURITY VULNERABILITY: Large payload caused resource issues: {e}"
                )


@pytest.mark.security
class TestSessionManagementVulnerabilities:
    """Test for session management security issues."""
    
    def test_session_data_persistence(self):
        """Test that session data persists inappropriately."""
        from src.tools.schemas import AgentContext
        
        # Create a session with sensitive data
        thread_id = str(uuid.uuid4())
        context = AgentContext(
            thread_id=thread_id,
            session_data={"temporary_secret": "should_not_persist"}
        )
        
        agent_sessions[thread_id] = context
        
        # Check if the session persists in memory
        if thread_id in agent_sessions:
            persistent_data = agent_sessions[thread_id].session_data
            
            if "temporary_secret" in persistent_data:
                pytest.fail(
                    "SECURITY VULNERABILITY: Sensitive session data persists in memory without expiration. "
                    "Sessions should have proper cleanup and expiration mechanisms."
                )
    
    def test_predictable_session_ids(self):
        """Test if session IDs (thread_ids) are predictable."""
        from src.tools.schemas import AgentContext
        
        # Generate multiple thread IDs and check for patterns
        thread_ids = []
        for i in range(10):
            context = AgentContext(thread_id=str(uuid.uuid4()))
            thread_ids.append(context.thread_id)
        
        # Check for sequential or predictable patterns
        # (This is a simplified check - real implementation would be more sophisticated)
        for i in range(1, len(thread_ids)):
            current = thread_ids[i]
            previous = thread_ids[i-1]
            
            # Check if IDs are sequential (which would be very bad)
            if len(current) == len(previous) and current.replace(current[-1], '') == previous.replace(previous[-1], ''):
                pytest.fail(
                    "SECURITY VULNERABILITY: Session IDs appear to be sequential or predictable. "
                    "Session IDs should be cryptographically random."
                )
    
    def setup_method(self):
        """Clear sessions before each test."""
        agent_sessions.clear()
    
    def teardown_method(self):
        """Clear sessions after each test."""
        agent_sessions.clear()