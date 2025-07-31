"""
Base agent implementation.
Implements Factor 1: Natural Language to Tool Calls
Implements Factor 8: Own Your Control Flow
Implements Factor 12: Stateless Reducer
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type, Union

from pydantic import BaseModel
import structlog

from ..tools.schemas import AgentAction, AgentContext, ToolResult
from ..config import settings


logger = structlog.get_logger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
    ) -> Union[str, BaseModel]:
        """Generate a response from the LLM."""
        ...


class BaseAgent(ABC):
    """
    Base agent class implementing 12-factor principles.
    
    This agent is designed as a stateless reducer (Factor 12) that processes
    context through deterministic control flow (Factor 8).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        agent_name: str,
        system_prompt: str,
        max_iterations: int = 10,
    ):
        self.llm_provider = llm_provider
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.logger = logger.bind(agent=agent_name)

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Main agent execution loop.
        Implements Factor 8: Own Your Control Flow.
        
        This is a stateless reducer that transforms the input context
        into an output context through a series of LLM calls and tool executions.
        """
        self.logger.info("Starting agent execution", thread_id=context.thread_id)
        
        iterations = 0
        current_context = context.model_copy(deep=True)
        
        while iterations < self.max_iterations and current_context.execution_state == "running":
            try:
                # Factor 1: Convert natural language to structured tool calls
                next_action = await self.determine_next_step(current_context)
                
                # Log the determined action
                self.logger.info(
                    "Next action determined",
                    action_type=next_action.intent,
                    iteration=iterations,
                )
                
                # Update context with the action
                current_context.conversation_history.append({
                    "type": "agent_action",
                    "content": next_action.model_dump(),
                    "iteration": iterations,
                })
                
                # Process the action based on its type
                if next_action.intent == "complete":
                    current_context.execution_state = "completed"
                    self.logger.info("Agent execution completed")
                    break
                    
                elif next_action.intent == "request_human_input":
                    current_context.execution_state = "waiting_for_human"
                    self.logger.info("Agent paused for human input")
                    break
                    
                elif next_action.intent == "pause_for_approval":
                    current_context.execution_state = "waiting_for_approval"
                    self.logger.info("Agent paused for approval")
                    break
                    
                elif next_action.intent == "execute_tool":
                    # Execute the tool and add result to context
                    tool_result = await self.execute_tool(next_action, current_context)
                    current_context.conversation_history.append({
                        "type": "tool_result",
                        "content": tool_result.model_dump(),
                        "iteration": iterations,
                    })
                    
                elif next_action.intent == "error_occurred":
                    current_context.execution_state = "error"
                    self.logger.error("Agent encountered error", error=next_action.error_message)
                    break
                
                iterations += 1
                
            except Exception as e:
                self.logger.error("Unexpected error in agent loop", error=str(e), exc_info=True)
                current_context.execution_state = "error"
                current_context.conversation_history.append({
                    "type": "system_error",
                    "content": {"error": str(e), "iteration": iterations},
                })
                break
        
        if iterations >= self.max_iterations:
            current_context.execution_state = "max_iterations_reached"
            self.logger.warning("Agent reached maximum iterations")
        
        return current_context

    async def determine_next_step(self, context: AgentContext) -> AgentAction:
        """
        Determine the next action based on context.
        Implements Factor 1: Natural Language to Tool Calls.
        
        This method converts the current context into a structured action
        that the agent should take next.
        """
        # Build the conversation history for the LLM
        messages = self.build_messages(context)
        
        try:
            # Get structured response from LLM
            response = await self.llm_provider.generate_response(
                messages=messages,
                response_model=self.get_action_model(),
                temperature=0.1,  # Low temperature for more deterministic responses
            )
            
            if isinstance(response, BaseModel):
                return response
            
            # Fallback: try to parse JSON response
            try:
                action_data = json.loads(response)
                return self.parse_action(action_data)
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error("Failed to parse LLM response", response=response, error=str(e))
                return self.create_error_action(f"Failed to parse LLM response: {e}")
                
        except Exception as e:
            self.logger.error("Error in determine_next_step", error=str(e), exc_info=True)
            return self.create_error_action(f"Error determining next step: {e}")

    def build_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        """Build message history for LLM."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history
        for entry in context.conversation_history[-10:]:  # Last 10 entries to manage context
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

    @abstractmethod
    def get_action_model(self) -> Type[BaseModel]:
        """Return the Pydantic model class for actions this agent can take."""
        pass

    @abstractmethod
    async def execute_tool(self, action: Any, context: AgentContext) -> ToolResult:
        """Execute a tool based on the action."""
        pass

    def parse_action(self, action_data: Dict[str, Any]) -> AgentAction:
        """Parse action data into the appropriate action type."""
        intent = action_data.get("intent")
        
        if intent == "request_human_input":
            from ..tools.schemas import RequestHumanInput
            return RequestHumanInput(**action_data)
        elif intent == "execute_tool":
            from ..tools.schemas import ExecuteTool
            return ExecuteTool(**action_data)
        elif intent == "pause_for_approval":
            from ..tools.schemas import PauseForApproval
            return PauseForApproval(**action_data)
        elif intent == "complete":
            from ..tools.schemas import Complete
            return Complete(**action_data)
        else:
            return self.create_error_action(f"Unknown intent: {intent}")

    def create_error_action(self, error_message: str) -> "ErrorOccurred":
        """Create an error action."""
        from ..tools.schemas import ErrorOccurred
        return ErrorOccurred(
            intent="error_occurred",
            error_message=error_message,
            error_type="agent_error",
            recovery_suggestion="Check logs and retry"
        )