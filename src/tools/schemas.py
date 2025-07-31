"""
Structured output schemas for tools.
Implements Factor 4: Tools are Structured Outputs.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class ActionType(str, Enum):
    """Enumeration of available action types."""
    COMPLETE = "complete"
    REQUEST_HUMAN_INPUT = "request_human_input"
    EXECUTE_TOOL = "execute_tool"
    PAUSE_FOR_APPROVAL = "pause_for_approval"
    ERROR_OCCURRED = "error_occurred"


class RequestHumanInput(BaseModel):
    """Request input from a human user."""
    intent: Literal["request_human_input"] = "request_human_input"
    question: str = Field(..., description="The question to ask the human")
    context: str = Field(..., description="Context for the question")
    urgency: Literal["low", "medium", "high"] = "medium"
    format: Literal["free_text", "yes_no", "multiple_choice"] = "free_text"
    choices: List[str] = Field(default_factory=list, description="Options for multiple choice")


class ExecuteTool(BaseModel):
    """Execute a specific tool with parameters."""
    intent: Literal["execute_tool"] = "execute_tool"
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool")
    requires_approval: bool = False


class PauseForApproval(BaseModel):
    """Pause execution and wait for human approval."""
    intent: Literal["pause_for_approval"] = "pause_for_approval"
    action_description: str = Field(..., description="Description of the action requiring approval")
    risk_level: Literal["low", "medium", "high"] = "medium"
    context: str = Field(..., description="Additional context for the approval request")


class Complete(BaseModel):
    """Mark the task as complete."""
    intent: Literal["complete"] = "complete"
    summary: str = Field(..., description="Summary of what was accomplished")
    results: Dict[str, Any] = Field(default_factory=dict, description="Results of the task")


class ErrorOccurred(BaseModel):
    """Indicate that an error occurred during execution."""
    intent: Literal["error_occurred"] = "error_occurred"
    error_message: str = Field(..., description="The error message")
    error_type: str = Field(..., description="Type of error")
    recovery_suggestion: Optional[str] = None


# Union type for all possible agent actions
AgentAction = Union[
    RequestHumanInput,
    ExecuteTool,
    PauseForApproval,
    Complete,
    ErrorOccurred
]


class ToolResult(BaseModel):
    """Result of executing a tool."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Context information for agent execution."""
    thread_id: str
    user_id: Optional[str] = None
    session_data: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    execution_state: str = "running"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None