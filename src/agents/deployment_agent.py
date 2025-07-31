"""
Deployment agent implementation.
Example agent that demonstrates the 12-factor methodology in practice.
"""

import json
from typing import Any, Dict, Type

from pydantic import BaseModel

from .base import BaseAgent
from .base import LLMProvider
from ..prompts.manager import prompt_manager
from ..tools.schemas import AgentAction, AgentContext, ToolResult
from ..tools.registry import tool_registry


class DeploymentAction(BaseModel):
    """Deployment-specific action schema."""
    intent: str
    service_name: str = ""
    version: str = ""
    environment: str = ""
    reason: str = ""
    
    # Standard action fields
    question: str = ""
    context: str = ""
    tool_name: str = ""
    parameters: Dict[str, Any] = {}
    summary: str = ""
    results: Dict[str, Any] = {}
    error_message: str = ""


class DeploymentAgent(BaseAgent):
    """
    Deployment management agent.
    
    This agent demonstrates all 12 factors:
    1. Natural Language to Tool Calls - Converts deployment requests to structured actions
    2. Own Your Prompts - Uses deployment-specific prompt templates
    3. Own Your Context Window - Manages deployment context efficiently
    4. Tools are Structured Outputs - Uses structured tool calls for deployments
    5. Unified State - Combines execution and business state in context
    6. Launch/Pause/Resume - Supports interruption during deployments
    7. Contact Humans - Requires approval for production deployments
    8. Own Your Control Flow - Custom logic for deployment safety
    9. Compact Errors - Efficient error handling and reporting
    10. Small Focused Agent - Focused only on deployment tasks
    11. Trigger from Anywhere - Can be triggered via API, webhooks, etc.
    12. Stateless Reducer - Pure function that transforms state
    """

    def __init__(self, llm_provider: LLMProvider):
        system_prompt = self._build_system_prompt()
        super().__init__(
            llm_provider=llm_provider,
            agent_name="DeploymentAgent",
            system_prompt=system_prompt,
            max_iterations=15,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt using prompt manager (Factor 2)."""
        try:
            return prompt_manager.render_prompt(
                template_name="deployment_agent",
                variables={
                    "service_name": "N/A",
                    "environment": "N/A", 
                    "version": "N/A",
                    "current_status": "Unknown",
                },
                validate_variables=False,
            )
        except FileNotFoundError:
            # Fallback prompt if template not found
            return """You are a deployment management assistant that helps with safe and reliable software deployments.

CORE RESPONSIBILITIES:
- Manage deployments for frontend and backend systems
- Ensure deployment safety through proper checks and approvals
- Follow best practices for production deployments
- Provide clear status updates and error handling

DEPLOYMENT SAFETY RULES:
1. ALWAYS check deployment environment (staging vs production)
2. ALWAYS verify the correct tag/version to deploy
3. ALWAYS check current system status before deploying
4. ALWAYS require approval for production deployments
5. NEVER deploy without proper testing in staging first

Respond with structured JSON actions only."""

    def get_action_model(self) -> Type[BaseModel]:
        """Return the action model for this agent."""
        return DeploymentAction

    async def execute_tool(self, action: DeploymentAction, context: AgentContext) -> ToolResult:
        """Execute deployment-specific tools."""
        
        if action.intent == "execute_tool":
            # Map deployment actions to appropriate tools
            tool_name = action.tool_name
            parameters = action.parameters
            
            # Add context for tools that need it
            tool_context = {
                "thread_id": context.thread_id,
                "agent_name": self.agent_name,
                "metadata": {
                    "service": action.service_name,
                    "environment": action.environment,
                    "version": action.version,
                }
            }
            
            return await tool_registry.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                context=tool_context,
            )
        
        return ToolResult(
            success=False,
            error=f"Unknown action intent: {action.intent}"
        )

    async def determine_next_step(self, context: AgentContext) -> AgentAction:
        """
        Enhanced next step determination with deployment-specific logic.
        Implements Factor 8: Own Your Control Flow.
        """
        
        # Check if this is a production deployment request
        is_production = self._check_production_deployment(context)
        has_status_check = self._has_recent_status_check(context)
        
        # Custom control flow for deployment safety
        if is_production and not has_status_check:
            # Force status check before production deployment
            from ..tools.schemas import ExecuteTool
            return ExecuteTool(
                intent="execute_tool",
                tool_name="check_service_status",
                parameters={
                    "service_name": self._extract_service_name(context),
                    "environment": "production",
                }
            )
        
        # Use base implementation for normal flow
        return await super().determine_next_step(context)

    def build_messages(self, context: AgentContext) -> list[Dict[str, str]]:
        """
        Build messages with deployment-specific context.
        Implements Factor 3: Own Your Context Window.
        """
        
        # Extract deployment context
        deployment_info = self._extract_deployment_context(context)
        
        # Build enhanced system prompt with current context
        try:
            enhanced_prompt = prompt_manager.render_prompt(
                template_name="deployment_agent",
                variables={
                    "service_name": deployment_info.get("service_name", "N/A"),
                    "environment": deployment_info.get("environment", "N/A"),
                    "version": deployment_info.get("version", "N/A"),
                    "current_status": deployment_info.get("status", "Unknown"),
                    "deployment_request": deployment_info.get("request", ""),
                    "recent_deployments": deployment_info.get("recent_deployments", []),
                },
                validate_variables=False,
            )
        except:
            enhanced_prompt = self.system_prompt
        
        messages = [
            {"role": "system", "content": enhanced_prompt}
        ]
        
        # Add compressed conversation history (Factor 3: Context Window Management)
        history = self._compress_context_history(context.conversation_history)
        for entry in history:
            if entry["type"] == "user_message":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["type"] == "agent_action":
                messages.append({
                    "role": "assistant",
                    "content": f"Action: {json.dumps(entry['content'])}"
                })
            elif entry["type"] == "tool_result":
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(entry['content'])}"
                })
        
        return messages

    def _check_production_deployment(self, context: AgentContext) -> bool:
        """Check if this involves a production deployment."""
        
        for entry in context.conversation_history:
            content = str(entry.get("content", "")).lower()
            if "production" in content and ("deploy" in content or "release" in content):
                return True
        
        return False

    def _has_recent_status_check(self, context: AgentContext) -> bool:
        """Check if we have a recent service status check."""
        
        recent_entries = context.conversation_history[-5:]  # Last 5 entries
        
        for entry in recent_entries:
            if entry.get("type") == "tool_result":
                tool_content = entry.get("content", {})
                if isinstance(tool_content, dict):
                    metadata = tool_content.get("metadata", {})
                    if metadata.get("tool_name") == "check_service_status":
                        return True
        
        return False

    def _extract_service_name(self, context: AgentContext) -> str:
        """Extract service name from context."""
        
        for entry in context.conversation_history:
            content = str(entry.get("content", ""))
            # Simple extraction - in production, use more sophisticated parsing
            if "service" in content.lower():
                words = content.split()
                for i, word in enumerate(words):
                    if word.lower() == "service" and i + 1 < len(words):
                        return words[i + 1].strip(".,!?")
        
        return "unknown-service"

    def _extract_deployment_context(self, context: AgentContext) -> Dict[str, Any]:
        """Extract deployment-specific context information."""
        
        deployment_context = {
            "service_name": "N/A",
            "environment": "N/A",
            "version": "N/A",
            "status": "Unknown",
            "request": "",
            "recent_deployments": [],
        }
        
        # Extract information from conversation history
        for entry in context.conversation_history:
            content = entry.get("content", "")
            
            if entry.get("type") == "user_message":
                content_str = str(content).lower()
                
                # Extract service name
                if "service" in content_str:
                    deployment_context["service_name"] = self._extract_service_name(context)
                
                # Extract environment
                if "production" in content_str:
                    deployment_context["environment"] = "production"
                elif "staging" in content_str:
                    deployment_context["environment"] = "staging"
                elif "development" in content_str:
                    deployment_context["environment"] = "development"
                
                # Store original request
                if not deployment_context["request"]:
                    deployment_context["request"] = str(content)[:200]
            
            elif entry.get("type") == "tool_result":
                tool_content = entry.get("content", {})
                if isinstance(tool_content, dict) and tool_content.get("success"):
                    result = tool_content.get("result", {})
                    
                    # Extract status from status check
                    if "status" in result:
                        deployment_context["status"] = result["status"]
                    
                    # Extract deployment info
                    if "version" in result:
                        deployment_context["version"] = result["version"]
        
        return deployment_context

    def _compress_context_history(self, history: list) -> list:
        """
        Compress context history to fit in context window.
        Implements Factor 3: Own Your Context Window.
        """
        
        # Keep most recent entries and important deployment events
        compressed = []
        recent_entries = history[-8:]  # Last 8 entries
        
        for entry in recent_entries:
            entry_type = entry.get("type")
            
            # Always keep user messages and tool results
            if entry_type in ["user_message", "tool_result"]:
                compressed.append(entry)
            
            # Keep agent actions but compress content
            elif entry_type == "agent_action":
                content = entry.get("content", {})
                if isinstance(content, dict):
                    # Keep only essential fields
                    compressed_content = {
                        "intent": content.get("intent"),
                        "tool_name": content.get("tool_name"),
                        "summary": content.get("summary", "")[:100],  # Truncate summary
                    }
                    compressed.append({
                        "type": entry_type,
                        "content": compressed_content,
                        "iteration": entry.get("iteration"),
                    })
        
        return compressed