"""
Tool registry and execution framework.
Implements Factor 4: Tools are Structured Outputs.

Tools are treated as structured JSON outputs that trigger deterministic code execution.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

import structlog
from pydantic import BaseModel

from .schemas import ToolResult

logger = structlog.get_logger(__name__)


class ToolDefinition(BaseModel):
    """Definition of a tool with its metadata."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_approval: bool = False
    risk_level: str = "low"  # low, medium, high
    category: str = "general"
    function: Optional[Callable] = None
    
    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """
    Registry for managing and executing tools.
    
    Tools are structured outputs that describe what deterministic code should do.
    This registry manages the mapping between tool descriptions and their implementations.
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.logger = logger.bind(component="tool_registry")

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        function: Callable,
        requires_approval: bool = False,
        risk_level: str = "low",
        category: str = "general",
    ) -> None:
        """Register a new tool."""
        
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            requires_approval=requires_approval,
            risk_level=risk_level,
            category=category,
            function=function,
        )
        
        self.tools[name] = tool_def
        
        self.logger.info(
            "Registered tool",
            name=name,
            category=category,
            risk_level=risk_level,
            requires_approval=requires_approval,
        )

    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
        requires_approval: bool = False,
        risk_level: str = "low",
        category: str = "general",
        parameters_schema: Optional[Dict[str, Any]] = None,
    ):
        """Decorator for registering tools."""
        
        def decorator(func: Callable):
            tool_name = name or func.__name__
            
            # Extract parameters schema from function signature if not provided
            if parameters_schema is None:
                schema = self._extract_parameters_schema(func)
            else:
                schema = parameters_schema
            
            self.register_tool(
                name=tool_name,
                description=description or func.__doc__ or "No description provided",
                parameters_schema=schema,
                function=func,
                requires_approval=requires_approval,
                risk_level=risk_level,
                category=category,
            )
            
            return func
        
        return decorator

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute a tool with the given parameters."""
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
            )
        
        tool_def = self.tools[tool_name]
        
        self.logger.info(
            "Executing tool",
            tool=tool_name,
            parameters=list(parameters.keys()),
            risk_level=tool_def.risk_level,
        )
        
        try:
            # Validate parameters against schema
            validation_result = self._validate_parameters(tool_def, parameters)
            if not validation_result.success:
                return validation_result
            
            # Execute the tool function
            if asyncio.iscoroutinefunction(tool_def.function):
                if context:
                    result = await tool_def.function(context=context, **parameters)
                else:
                    result = await tool_def.function(**parameters)
            else:
                if context:
                    result = tool_def.function(context=context, **parameters)
                else:
                    result = tool_def.function(**parameters)
            
            self.logger.info("Tool execution completed", tool=tool_name)
            
            return ToolResult(
                success=True,
                result=result,
                metadata={
                    "tool_name": tool_name,
                    "risk_level": tool_def.risk_level,
                    "requires_approval": tool_def.requires_approval,
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Tool execution failed",
                tool=tool_name,
                error=str(e),
                exc_info=True,
            )
            
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={"tool_name": tool_name, "error_type": type(e).__name__}
            )

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get the definition of a specific tool."""
        return self.tools.get(tool_name)

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """List all available tools, optionally filtered by category."""
        tools = list(self.tools.values())
        
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        return tools

    def get_tools_schema(self) -> Dict[str, Any]:
        """Get a schema of all available tools for LLM consumption."""
        schema = {
            "tools": {},
            "categories": set(),
        }
        
        for tool_name, tool_def in self.tools.items():
            schema["tools"][tool_name] = {
                "description": tool_def.description,
                "parameters": tool_def.parameters_schema,
                "requires_approval": tool_def.requires_approval,
                "risk_level": tool_def.risk_level,
                "category": tool_def.category,
            }
            schema["categories"].add(tool_def.category)
        
        schema["categories"] = list(schema["categories"])
        return schema

    def _extract_parameters_schema(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter schema from function signature."""
        sig = inspect.signature(func)
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue  # Skip context parameter
            
            param_schema = {"type": "string"}  # Default type
            
            # Extract type hints
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == list:
                    param_schema["type"] = "array"
                elif param.annotation == dict:
                    param_schema["type"] = "object"
            
            schema["properties"][param_name] = param_schema
            
            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
        
        return schema

    def _validate_parameters(
        self,
        tool_def: ToolDefinition,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Validate parameters against the tool's schema."""
        
        schema = tool_def.parameters_schema
        required_params = schema.get("required", [])
        
        # Check required parameters
        missing_params = set(required_params) - set(parameters.keys())
        if missing_params:
            return ToolResult(
                success=False,
                error=f"Missing required parameters: {missing_params}",
            )
        
        # Basic type validation (could be enhanced with jsonschema)
        properties = schema.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                
                if expected_type == "integer" and not isinstance(param_value, int):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' must be an integer",
                    )
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' must be a number",
                    )
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    return ToolResult(
                        success=False,
                        error=f"Parameter '{param_name}' must be a boolean",
                    )
        
        return ToolResult(success=True)


# Global tool registry instance
tool_registry = ToolRegistry()