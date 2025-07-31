"""
Mock external services for testing.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock


class MockHTTPClient:
    """Mock HTTP client for testing HTTP requests."""
    
    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.requests = []
        
    def set_response(self, url: str, method: str, response: Dict[str, Any]):
        """Set a mock response for a specific URL and method."""
        key = f"{method.upper()}:{url}"
        self.responses[key] = response
    
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None):
        """Mock GET request."""
        self.requests.append({
            "method": "GET",
            "url": url,
            "headers": headers,
        })
        
        key = f"GET:{url}"
        if key in self.responses:
            response = self.responses[key]
            mock_response = MagicMock()
            mock_response.status_code = response.get("status_code", 200)
            mock_response.text = response.get("text", "")
            mock_response.json.return_value = response.get("json", {})
            mock_response.headers = response.get("headers", {})
            mock_response.url = url
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        # Default successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "success"}'
        mock_response.json.return_value = {"status": "success"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.url = url
        mock_response.raise_for_status = MagicMock()
        return mock_response
    
    async def post(self, url: str, data=None, json_data=None, headers: Optional[Dict[str, str]] = None):
        """Mock POST request."""
        self.requests.append({
            "method": "POST",
            "url": url,
            "data": data,
            "json": json_data,
            "headers": headers,
        })
        
        key = f"POST:{url}"
        if key in self.responses:
            response = self.responses[key]
            mock_response = MagicMock()
            mock_response.status_code = response.get("status_code", 200)
            mock_response.text = response.get("text", "")
            mock_response.json.return_value = response.get("json", {})
            mock_response.headers = response.get("headers", {})
            mock_response.url = url
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        # Default successful response
        mock_response = MagicMock()
        mock_response.status_code = 201  
        mock_response.text = '{"status": "created"}'
        mock_response.json.return_value = {"status": "created"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.url = url
        mock_response.raise_for_status = MagicMock()
        return mock_response


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files = {}
        self.operations = []
    
    def add_file(self, path: str, content: str):
        """Add a file to the mock file system."""
        self.files[path] = content
    
    def read_file(self, path: str) -> str:
        """Mock file reading."""
        self.operations.append({"operation": "read", "path": path})
        
        if path in self.files:
            return self.files[path]
        else:
            raise FileNotFoundError(f"No such file: {path}")
    
    def write_file(self, path: str, content: str, mode: str = "w"):
        """Mock file writing."""
        self.operations.append({
            "operation": "write",
            "path": path,
            "content": content,
            "mode": mode
        })
        
        if mode == "a":
            if path in self.files:
                self.files[path] += content
            else:
                self.files[path] = content
        else:
            self.files[path] = content
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return path in self.files
    
    def list_operations(self) -> List[Dict[str, Any]]:
        """Get list of all file operations."""
        return self.operations.copy()
    
    def reset(self):
        """Reset the mock file system."""
        self.files.clear()
        self.operations.clear()


class MockEmailService:
    """Mock email service for testing email sending."""
    
    def __init__(self):
        self.sent_emails = []
        self.should_fail = False
        self.failure_message = "Mock email service failure"
    
    def set_failure(self, should_fail: bool, message: str = "Mock email service failure"):
        """Configure the service to fail."""
        self.should_fail = should_fail
        self.failure_message = message
    
    def send_email(self, to_email: str, subject: str, body: str, from_email: Optional[str] = None):
        """Mock email sending."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        email = {
            "to": to_email,
            "subject": subject,
            "body": body,
            "from": from_email or "noreply@test.com",
            "timestamp": "2024-01-01T12:00:00Z",
        }
        self.sent_emails.append(email)
        return {"status": "sent", "email_id": f"email_{len(self.sent_emails)}"}
    
    def get_sent_emails(self) -> List[Dict[str, Any]]:
        """Get list of sent emails."""
        return self.sent_emails.copy()
    
    def reset(self):
        """Reset the mock email service."""
        self.sent_emails.clear()
        self.should_fail = False


class MockDeploymentService:
    """Mock deployment service for testing deployment operations."""
    
    def __init__(self):
        self.deployments = []
        self.service_statuses = {}
        self.should_fail = False
        self.failure_message = "Mock deployment service failure"
    
    def set_failure(self, should_fail: bool, message: str = "Mock deployment service failure"):
        """Configure the service to fail."""
        self.should_fail = should_fail
        self.failure_message = message
    
    def set_service_status(self, service_name: str, environment: str, status: str):
        """Set the status of a service."""
        key = f"{service_name}:{environment}"
        self.service_statuses[key] = {
            "service": service_name,
            "environment": environment,
            "status": status,
            "uptime": "99.9%" if status == "healthy" else "85.2%",
            "instances": 3 if status == "healthy" else 1,
            "version": "1.0.0",
        }
    
    async def check_service_status(self, service_name: str, environment: str = "production"):
        """Mock service status check."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        key = f"{service_name}:{environment}"
        if key in self.service_statuses:
            return self.service_statuses[key]
        
        # Default status
        return {
            "service": service_name,
            "environment": environment,
            "status": "unknown",
            "uptime": "N/A",
            "instances": 0,
            "version": "unknown",
        }
    
    async def deploy_service(self, service_name: str, version: str, environment: str):
        """Mock service deployment."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        # Simulate deployment time
        await asyncio.sleep(0.5)
        
        deployment = {
            "service": service_name,
            "version": version,
            "environment": environment,
            "status": "deployed",
            "deployment_id": f"deploy_{len(self.deployments) + 1}",
            "timestamp": "2024-01-01T12:00:00Z",
        }
        
        self.deployments.append(deployment)
        
        # Update service status
        self.set_service_status(service_name, environment, "healthy")
        
        return deployment
    
    def get_deployments(self) -> List[Dict[str, Any]]:
        """Get list of deployments."""
        return self.deployments.copy()
    
    def reset(self):
        """Reset the mock deployment service."""
        self.deployments.clear()
        self.service_statuses.clear()
        self.should_fail = False


class MockDatabase:
    """Mock database for testing database operations."""
    
    def __init__(self):
        self.tables = {}
        self.operations = []
        self.should_fail = False
        self.failure_message = "Mock database failure"
    
    def set_failure(self, should_fail: bool, message: str = "Mock database failure"):
        """Configure the database to fail."""
        self.should_fail = should_fail
        self.failure_message = message
    
    def create_table(self, table_name: str, schema: Dict[str, Any]):
        """Create a table."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        self.tables[table_name] = {"schema": schema, "rows": []}
        self.operations.append({"operation": "create_table", "table": table_name, "schema": schema})
    
    def insert(self, table_name: str, data: Dict[str, Any]):
        """Insert data into a table."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        if table_name not in self.tables:
            raise Exception(f"Table {table_name} does not exist")
        
        self.tables[table_name]["rows"].append(data)
        self.operations.append({"operation": "insert", "table": table_name, "data": data})
        return {"id": len(self.tables[table_name]["rows"])}
    
    def select(self, table_name: str, where: Optional[Dict[str, Any]] = None):
        """Select data from a table."""
        if self.should_fail:
            raise Exception(self.failure_message)
        
        if table_name not in self.tables:
            raise Exception(f"Table {table_name} does not exist")
        
        rows = self.tables[table_name]["rows"]
        
        if where:
            filtered_rows = []
            for row in rows:
                match = True
                for key, value in where.items():
                    if key not in row or row[key] != value:
                        match = False
                        break
                if match:
                    filtered_rows.append(row)
            rows = filtered_rows
        
        self.operations.append({"operation": "select", "table": table_name, "where": where})
        return rows
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get list of database operations."""
        return self.operations.copy()
    
    def reset(self):
        """Reset the mock database."""
        self.tables.clear()
        self.operations.clear()
        self.should_fail = False