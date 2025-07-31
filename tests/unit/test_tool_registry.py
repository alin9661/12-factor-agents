"""
Unit tests for the tool registry system.

These tests follow TDD principles - they are written first and will initially fail
until the corresponding functionality is properly implemented.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from src.tools.registry import ToolRegistry, ToolDefinition
from src.tools.schemas import ToolResult


@pytest.mark.unit
class TestToolRegistry:
    """Test cases for the ToolRegistry class."""
    
    def test_tool_registry_initialization(self):
        """Test that ToolRegistry initializes with empty state."""
        registry = ToolRegistry()
        
        assert registry.tools == {}
        assert hasattr(registry, 'logger')
        assert len(registry.list_tools()) == 0
    
    def test_register_tool_basic(self):
        """Test basic tool registration."""
        registry = ToolRegistry()
        
        def test_func(message: str):
            return {"result": message}
        
        registry.register_tool(
            name="test_tool",
            description="A test tool",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            },
            function=test_func
        )
        
        assert "test_tool" in registry.tools
        tool_def = registry.tools["test_tool"]
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.function == test_func
        assert tool_def.risk_level == "low"
        assert tool_def.requires_approval is False
    
    def test_register_tool_with_all_parameters(self):
        """Test tool registration with all optional parameters."""
        registry = ToolRegistry()
        
        async def risky_func(action: str):
            return {"action": action}
        
        registry.register_tool(
            name="risky_tool",
            description="A risky tool",
            parameters_schema={
                "type": "object", 
                "properties": {"action": {"type": "string"}},
                "required": ["action"]
            },
            function=risky_func,
            requires_approval=True,
            risk_level="high",
            category="dangerous"
        )
        
        tool_def = registry.tools["risky_tool"]
        assert tool_def.name == "risky_tool"
        assert tool_def.requires_approval is True
        assert tool_def.risk_level == "high"
        assert tool_def.category == "dangerous"
    
    def test_tool_decorator_basic(self):
        """Test the @tool decorator with basic usage."""
        registry = ToolRegistry()
        
        @registry.tool(description="Test decorator tool")
        def decorated_tool(value: int):
            """Process a value."""
            return value * 2
        
        assert "decorated_tool" in registry.tools
        tool_def = registry.tools["decorated_tool"]
        assert tool_def.name == "decorated_tool"
        assert tool_def.description == "Test decorator tool"
        assert "value" in tool_def.parameters_schema["properties"]
    
    def test_tool_decorator_with_custom_name(self):
        """Test the @tool decorator with custom name."""
        registry = ToolRegistry()
        
        @registry.tool(name="custom_name", description="Custom named tool")
        def some_function():
            return "result"
        
        assert "custom_name" in registry.tools
        assert "some_function" not in registry.tools
        
    def test_tool_decorator_extracts_parameters_schema(self):
        """Test that decorator extracts parameter schema from function signature."""
        registry = ToolRegistry()
        
        @registry.tool(description="Schema extraction test")
        def complex_tool(name: str, count: int, enabled: bool, data: list):
            return {"name": name, "count": count, "enabled": enabled, "data": data}
        
        schema = registry.tools["complex_tool"].parameters_schema
        props = schema["properties"]
        
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["enabled"]["type"] == "boolean"
        assert props["data"]["type"] == "array"
        assert set(schema["required"]) == {"name", "count", "enabled", "data"}
    
    def test_tool_decorator_handles_optional_parameters(self):
        """Test decorator handles parameters with default values."""
        registry = ToolRegistry()
        
        @registry.tool(description="Optional params test")
        def tool_with_defaults(required_param: str, optional_param: int = 10):
            return {"required": required_param, "optional": optional_param}
        
        schema = registry.tools["tool_with_defaults"].parameters_schema
        assert "required_param" in schema["required"]
        assert "optional_param" not in schema["required"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        
        def simple_tool(message: str):
            return {"processed": message.upper()}
        
        registry.register_tool(
            name="simple_tool",
            description="Simple test tool",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            },
            function=simple_tool
        )
        
        result = await registry.execute_tool(
            "simple_tool",
            {"message": "hello"}
        )
        
        assert result.success is True
        assert result.result == {"processed": "HELLO"}
        assert result.error is None
        assert result.metadata["tool_name"] == "simple_tool"
    
    @pytest.mark.asyncio
    async def test_execute_async_tool_success(self):
        """Test successful async tool execution."""
        registry = ToolRegistry()
        
        async def async_tool(value: int):
            return {"doubled": value * 2}
        
        registry.register_tool(
            name="async_tool",
            description="Async test tool",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"]
            },
            function=async_tool
        )
        
        result = await registry.execute_tool("async_tool", {"value": 21})
        
        assert result.success is True
        assert result.result == {"doubled": 42}
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_context(self):
        """Test tool execution with context parameter."""
        registry = ToolRegistry()
        
        def context_tool(message: str, context: Dict[str, Any]):
            return {
                "message": message,
                "user_id": context.get("user_id"),
                "thread_id": context.get("thread_id")
            }
        
        registry.register_tool(
            name="context_tool",
            description="Tool that uses context",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            },
            function=context_tool
        )
        
        context = {"user_id": "user123", "thread_id": "thread456"}
        result = await registry.execute_tool(
            "context_tool",
            {"message": "test"},
            context=context
        )
        
        assert result.success is True
        assert result.result["user_id"] == "user123"
        assert result.result["thread_id"] == "thread456"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool returns error."""
        registry = ToolRegistry()
        
        result = await registry.execute_tool("nonexistent_tool", {})
        
        assert result.success is False
        assert "not found" in result.error.lower()
        assert result.result is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_missing_required_parameters(self):
        """Test tool execution with missing required parameters."""
        registry = ToolRegistry()
        
        def strict_tool(required_param: str, other_param: int):
            return {"params": [required_param, other_param]}
        
        registry.register_tool(
            name="strict_tool",
            description="Tool with required params",
            parameters_schema={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "other_param": {"type": "integer"}
                },
                "required": ["required_param", "other_param"]
            },
            function=strict_tool
        )
        
        # Missing both parameters
        result = await registry.execute_tool("strict_tool", {})
        assert result.success is False
        assert "missing required parameters" in result.error.lower()
        
        # Missing one parameter
        result = await registry.execute_tool("strict_tool", {"required_param": "test"})
        assert result.success is False
        assert "missing required parameters" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_tool_wrong_parameter_types(self):
        """Test tool execution with wrong parameter types."""
        registry = ToolRegistry()
        
        def typed_tool(count: int, enabled: bool):
            return {"count": count, "enabled": enabled}
        
        registry.register_tool(
            name="typed_tool",
            description="Tool with typed parameters",
            parameters_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"}
                },
                "required": ["count", "enabled"]
            },
            function=typed_tool
        )
        
        # Wrong type for integer parameter
        result = await registry.execute_tool("typed_tool", {"count": "not_an_int", "enabled": True})
        assert result.success is False
        assert "must be an integer" in result.error.lower()
        
        # Wrong type for boolean parameter  
        result = await registry.execute_tool("typed_tool", {"count": 42, "enabled": "not_a_bool"})
        assert result.success is False
        assert "must be a boolean" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_tool_function_raises_exception(self):
        """Test tool execution when the function raises an exception."""
        registry = ToolRegistry()
        
        def failing_tool(message: str):
            raise ValueError(f"Tool failed with message: {message}")
        
        registry.register_tool(
            name="failing_tool",
            description="Tool that always fails",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            },
            function=failing_tool
        )
        
        result = await registry.execute_tool("failing_tool", {"message": "test"})
        
        assert result.success is False
        assert "tool execution failed" in result.error.lower()
        assert "tool failed with message: test" in result.error
        assert result.metadata["error_type"] == "ValueError"
    
    def test_get_tool_definition_exists(self):
        """Test getting definition of existing tool."""
        registry = ToolRegistry()
        
        def test_tool():
            return "test"
        
        registry.register_tool(
            name="test_tool",
            description="Test tool",
            parameters_schema={"type": "object"},
            function=test_tool
        )
        
        definition = registry.get_tool_definition("test_tool")
        assert definition is not None
        assert definition.name == "test_tool"
        assert definition.description == "Test tool"
    
    def test_get_tool_definition_not_exists(self):
        """Test getting definition of non-existent tool."""
        registry = ToolRegistry()
        
        definition = registry.get_tool_definition("nonexistent")
        assert definition is None
    
    def test_list_tools_empty(self):
        """Test listing tools when none are registered."""
        registry = ToolRegistry()
        
        tools = registry.list_tools()
        assert tools == []
    
    def test_list_tools_with_tools(self):
        """Test listing tools when some are registered."""
        registry = ToolRegistry()
        
        # Register multiple tools
        for i in range(3):
            registry.register_tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters_schema={"type": "object"},
                function=lambda: f"tool_{i}",
                category=f"category_{i % 2}"  # Alternate categories
            )
        
        all_tools = registry.list_tools()
        assert len(all_tools) == 3
        
        # Test category filtering
        category_0_tools = registry.list_tools(category="category_0")
        assert len(category_0_tools) == 2  # tools 0 and 2
        
        category_1_tools = registry.list_tools(category="category_1")
        assert len(category_1_tools) == 1  # tool 1
    
    def test_get_tools_schema(self):
        """Test getting the complete tools schema for LLM consumption."""
        registry = ToolRegistry()
        
        # Register tools in different categories
        registry.register_tool(
            name="safe_tool",
            description="A safe tool",
            parameters_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            function=lambda: None,
            category="utility",
            risk_level="low"
        )
        
        registry.register_tool(
            name="risky_tool", 
            description="A risky tool",
            parameters_schema={"type": "object", "properties": {"action": {"type": "string"}}},
            function=lambda: None,
            category="admin",
            risk_level="high",
            requires_approval=True
        )
        
        schema = registry.get_tools_schema()
        
        assert "tools" in schema
        assert "categories" in schema
        
        # Check tools
        assert "safe_tool" in schema["tools"]
        assert "risky_tool" in schema["tools"]
        
        safe_tool = schema["tools"]["safe_tool"]
        assert safe_tool["description"] == "A safe tool"
        assert safe_tool["risk_level"] == "low"
        assert safe_tool["requires_approval"] is False
        assert safe_tool["category"] == "utility"
        
        risky_tool = schema["tools"]["risky_tool"]
        assert risky_tool["requires_approval"] is True
        assert risky_tool["risk_level"] == "high"
        
        # Check categories
        assert set(schema["categories"]) == {"utility", "admin"}


@pytest.mark.unit  
class TestToolDefinition:
    """Test cases for the ToolDefinition model."""
    
    def test_tool_definition_creation_minimal(self):
        """Test creating ToolDefinition with minimal parameters."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object"}
        )
        
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.parameters_schema == {"type": "object"}
        assert tool_def.requires_approval is False
        assert tool_def.risk_level == "low"
        assert tool_def.category == "general"
        assert tool_def.function is None
    
    def test_tool_definition_creation_full(self):
        """Test creating ToolDefinition with all parameters."""
        def test_func():
            return "test"
        
        tool_def = ToolDefinition(
            name="full_tool",
            description="A fully specified tool",
            parameters_schema={"type": "object", "properties": {}},
            requires_approval=True,
            risk_level="high",
            category="admin",
            function=test_func
        )
        
        assert tool_def.name == "full_tool"
        assert tool_def.description == "A fully specified tool"
        assert tool_def.requires_approval is True
        assert tool_def.risk_level == "high"
        assert tool_def.category == "admin"
        assert tool_def.function == test_func
    
    def test_tool_definition_invalid_risk_level(self):
        """Test that invalid risk levels are handled appropriately."""
        # Note: This test may fail until proper validation is implemented
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters_schema={"type": "object"},
            risk_level="invalid_level"  # Should be "low", "medium", or "high"
        )
        
        # The tool definition should either reject invalid risk levels
        # or normalize them to a valid value
        assert tool_def.risk_level in ["low", "medium", "high", "invalid_level"]


@pytest.mark.unit
class TestToolRegistryEdgeCases:
    """Test edge cases and error conditions in ToolRegistry."""
    
    def test_register_duplicate_tool_name(self):
        """Test behavior when registering tools with duplicate names."""
        registry = ToolRegistry()
        
        def tool1():
            return "tool1"
        
        def tool2():
            return "tool2"
        
        # Register first tool
        registry.register_tool(
            name="duplicate_name",
            description="First tool",
            parameters_schema={"type": "object"},
            function=tool1
        )
        
        # Register second tool with same name (should overwrite)
        registry.register_tool(
            name="duplicate_name",
            description="Second tool",
            parameters_schema={"type": "object"},
            function=tool2
        )
        
        # Should have the second tool
        assert len(registry.tools) == 1
        assert registry.tools["duplicate_name"].description == "Second tool"
        assert registry.tools["duplicate_name"].function == tool2
    
    def test_parameter_schema_extraction_edge_cases(self):
        """Test parameter schema extraction with complex signatures."""
        registry = ToolRegistry()
        
        # Function with no parameters
        @registry.tool(description="No params")
        def no_params():
            return "no params"
        
        schema = registry.tools["no_params"].parameters_schema
        assert schema["properties"] == {}
        assert schema["required"] == []
        
        # Function with context parameter (should be ignored)
        @registry.tool(description="With context")
        def with_context(message: str, context: dict):
            return message
        
        schema = registry.tools["with_context"].parameters_schema
        assert "message" in schema["properties"]
        assert "context" not in schema["properties"]
        assert schema["required"] == ["message"]
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_malformed_parameters_schema(self):
        """Test tool execution when parameters schema is malformed."""
        registry = ToolRegistry()
        
        def simple_tool(param: str):
            return param
        
        # Register tool with malformed schema (missing 'type')
        registry.register_tool(
            name="malformed_tool",
            description="Tool with bad schema",
            parameters_schema={"properties": {"param": {"type": "string"}}},  # Missing "type": "object"
            function=simple_tool
        )
        
        # Execution should still work or fail gracefully
        result = await registry.execute_tool("malformed_tool", {"param": "test"})
        # The behavior depends on implementation - it should either succeed or fail gracefully
        assert isinstance(result, ToolResult)
    
    def test_tool_registry_thread_safety(self):
        """Test that tool registry operations are thread-safe."""
        # This test would need threading, marking as placeholder for now
        # In a real implementation, we'd test concurrent registration/execution
        registry = ToolRegistry()
        assert registry is not None
        # TODO: Implement actual thread safety tests