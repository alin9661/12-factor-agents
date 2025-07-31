"""
Built-in tools for the 12-factor agents system.
These tools provide basic functionality that most agents will need.
"""

import asyncio
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx
import structlog

from .registry import tool_registry

logger = structlog.get_logger(__name__)


# Utility Tools
@tool_registry.tool(
    name="log_message",
    description="Log a message with specified level",
    category="utility",
    risk_level="low",
)
def log_message(message: str, level: str = "info", context: Optional[Dict[str, Any]] = None):
    """Log a message for debugging and monitoring purposes."""
    
    log_context = {"tool": "log_message"}
    if context:
        log_context.update(context.get("metadata", {}))
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, **log_context)
    
    return {
        "status": "logged",
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat(),
    }


@tool_registry.tool(
    name="sleep",
    description="Sleep for a specified number of seconds",
    category="utility",
    risk_level="low",
)
async def sleep_tool(seconds: int, context: Optional[Dict[str, Any]] = None):
    """Sleep for a specified duration (useful for rate limiting or delays)."""
    
    logger.info("Sleeping", seconds=seconds)
    await asyncio.sleep(seconds)
    
    return {
        "status": "completed",
        "slept_seconds": seconds,
        "timestamp": datetime.utcnow().isoformat(),
    }


# HTTP Tools
@tool_registry.tool(
    name="http_get",
    description="Make an HTTP GET request",
    category="http",
    risk_level="medium",
)
async def http_get(url: str, headers: Optional[Dict[str, str]] = None, context: Optional[Dict[str, Any]] = None):
    """Make an HTTP GET request to the specified URL."""
    
    logger.info("Making HTTP GET request", url=url)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers or {})
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
                "url": str(response.url),
            }
            
    except httpx.HTTPError as e:
        logger.error("HTTP request failed", url=url, error=str(e))
        raise


@tool_registry.tool(
    name="http_post",
    description="Make an HTTP POST request",
    category="http",
    risk_level="medium",
    requires_approval=False,  # Set to True for production environments
)
async def http_post(
    url: str,
    data: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """Make an HTTP POST request to the specified URL."""
    
    logger.info("Making HTTP POST request", url=url)
    
    try:
        async with httpx.AsyncClient() as client:
            if json_data:
                response = await client.post(url, json=json_data, headers=headers or {})
            else:
                response = await client.post(url, data=data, headers=headers or {})
            
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
                "url": str(response.url),
            }
            
    except httpx.HTTPError as e:
        logger.error("HTTP POST request failed", url=url, error=str(e))
        raise


# File System Tools
@tool_registry.tool(
    name="read_file",
    description="Read contents of a file",
    category="filesystem",
    risk_level="low",
)
def read_file(file_path: str, context: Optional[Dict[str, Any]] = None):
    """Read the contents of a file."""
    
    logger.info("Reading file", path=file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "file_path": file_path,
            "size_bytes": len(content.encode('utf-8')),
            "lines": len(content.splitlines()),
        }
        
    except Exception as e:
        logger.error("Failed to read file", path=file_path, error=str(e))
        raise


@tool_registry.tool(
    name="write_file",
    description="Write content to a file",
    category="filesystem",
    risk_level="high",
    requires_approval=True,
)
def write_file(file_path: str, content: str, mode: str = "w", context: Optional[Dict[str, Any]] = None):
    """Write content to a file (requires approval due to potential data loss)."""
    
    logger.info("Writing file", path=file_path, mode=mode)
    
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "file_path": file_path,
            "bytes_written": len(content.encode('utf-8')),
            "mode": mode,
        }
        
    except Exception as e:
        logger.error("Failed to write file", path=file_path, error=str(e))
        raise


# Communication Tools
@tool_registry.tool(
    name="send_email",
    description="Send an email message",
    category="communication",
    risk_level="high",
    requires_approval=True,
)
def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    smtp_server: str = "localhost",
    smtp_port: int = 587,
    context: Optional[Dict[str, Any]] = None,
):
    """Send an email message (requires approval)."""
    
    logger.info("Sending email", to=to_email, subject=subject)
    
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email or "noreply@12factor-agents.local"
        msg['To'] = to_email
        
        # In a real implementation, you'd configure SMTP properly
        # For now, we'll just simulate sending
        logger.info("Email would be sent", msg=msg.as_string())
        
        return {
            "status": "sent",
            "to": to_email,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error("Failed to send email", to=to_email, error=str(e))
        raise


# Deployment Tools (Example)
@tool_registry.tool(
    name="check_service_status",
    description="Check the status of a deployed service",
    category="deployment",
    risk_level="low",
)
async def check_service_status(service_name: str, environment: str = "production", context: Optional[Dict[str, Any]] = None):
    """Check the status of a deployed service."""
    
    logger.info("Checking service status", service=service_name, environment=environment)
    
    # Simulate checking service status
    await asyncio.sleep(1)  # Simulate API call delay
    
    # Mock response
    statuses = ["healthy", "degraded", "unhealthy", "unknown"]
    import random
    status = random.choice(statuses)
    
    return {
        "service": service_name,
        "environment": environment,
        "status": status,
        "uptime": "99.9%" if status == "healthy" else "95.2%",
        "last_deployment": "2024-01-15T10:30:00Z",
        "version": "1.2.3",
        "instances": 3 if status == "healthy" else 1,
    }


@tool_registry.tool(
    name="deploy_service",
    description="Deploy a service to an environment",
    category="deployment",
    risk_level="high",
    requires_approval=True,
)
async def deploy_service(
    service_name: str,
    version: str,
    environment: str,
    context: Optional[Dict[str, Any]] = None,
):
    """Deploy a service to the specified environment (requires approval)."""
    
    logger.info("Deploying service", service=service_name, version=version, environment=environment)
    
    # Simulate deployment process
    await asyncio.sleep(3)  # Simulate deployment time
    
    return {
        "status": "deployed",
        "service": service_name,
        "version": version,
        "environment": environment,
        "deployment_id": f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat(),
        "estimated_rollout_time": "5 minutes",
    }


# Customer Support Tools (Example)
@tool_registry.tool(
    name="lookup_customer",
    description="Look up customer information by email or ID",
    category="customer_support",
    risk_level="medium",
)
async def lookup_customer(customer_identifier: str, context: Optional[Dict[str, Any]] = None):
    """Look up customer information by email or customer ID."""
    
    logger.info("Looking up customer", identifier=customer_identifier)
    
    # Simulate database lookup
    await asyncio.sleep(0.5)
    
    # Mock customer data
    return {
        "customer_id": "cust_123456",
        "email": customer_identifier if "@" in customer_identifier else "customer@example.com",
        "name": "John Doe",
        "account_status": "active",
        "tier": "premium",
        "signup_date": "2023-06-15",
        "total_orders": 12,
        "last_login": "2024-01-20T14:30:00Z",
    }


@tool_registry.tool(
    name="create_support_ticket",
    description="Create a support ticket for customer issues",
    category="customer_support",
    risk_level="low",
)
async def create_support_ticket(
    customer_id: str,
    issue_title: str,
    issue_description: str,
    priority: str = "medium",
    context: Optional[Dict[str, Any]] = None,
):
    """Create a support ticket for a customer issue."""
    
    logger.info("Creating support ticket", customer=customer_id, priority=priority)
    
    # Simulate ticket creation
    await asyncio.sleep(0.3)
    
    ticket_id = f"TICKET-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "title": issue_title,
        "description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
        "assigned_to": "support-queue",
    }