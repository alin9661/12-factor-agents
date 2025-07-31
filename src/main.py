"""
Main entry point for 12-factor agents application.
"""

import asyncio
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .api.main import app
from .config import settings

cli = typer.Typer(help="12-Factor Agents CLI")
console = Console()


@cli.command()
def serve(
    host: str = typer.Option(settings.api_host, help="Host to bind to"),
    port: int = typer.Option(settings.api_port, help="Port to bind to"),
    reload: bool = typer.Option(settings.debug, help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    console.print(f"🚀 Starting 12-Factor Agents API on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
def test_agent():
    """Test agent functionality with a simple example."""
    
    async def run_test():
        from .agents.deployment_agent import DeploymentAgent
        from .agents.llm_providers import LLMProviderFactory
        from .tools.schemas import AgentContext
        
        console.print("🤖 Testing deployment agent...")
        
        try:
            # Create LLM provider
            llm_provider = LLMProviderFactory.create_default_provider()
            
            # Create agent
            agent = DeploymentAgent(llm_provider)
            
            # Create test context
            context = AgentContext(
                thread_id="test-123",
                conversation_history=[{
                    "type": "user_message",
                    "content": "Deploy api-backend version 1.2.3 to production",
                }]
            )
            
            console.print("📤 Sending test deployment request...")
            
            # Run agent (this will fail without actual LLM API keys)
            result = await agent.run(context)
            
            console.print(f"✅ Agent completed with state: {result.execution_state}")
            console.print(f"📝 Conversation entries: {len(result.conversation_history)}")
            
        except Exception as e:
            console.print(f"❌ Test failed: {e}")
            console.print("💡 Make sure to set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
    
    asyncio.run(run_test())


@cli.command()
def list_tools():
    """List all available tools."""
    from .tools.registry import tool_registry
    from .tools import builtin_tools  # Import to register tools
    
    tools = tool_registry.list_tools()
    
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Risk Level", style="yellow")
    table.add_column("Requires Approval", style="red")
    table.add_column("Description", style="white")
    
    for tool in tools:
        approval_text = "Yes" if tool.requires_approval else "No"
        table.add_row(
            tool.name,
            tool.category,
            tool.risk_level,
            approval_text,
            tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
        )
    
    console.print(table)
    console.print(f"\n📊 Total tools available: {len(tools)}")


@cli.command()
def info():
    """Show information about the 12-factor agents implementation."""
    
    table = Table(title="12-Factor Agents Implementation Status")
    table.add_column("Factor", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Implementation", style="green")
    
    factors = [
        ("1. Natural Language to Tool Calls", "Convert user input to structured actions", "✅ BaseAgent.determine_next_step()"),
        ("2. Own Your Prompts", "Manage prompts as first-class code", "✅ PromptManager with Jinja2"),
        ("3. Own Your Context Window", "Control context and token usage", "✅ Context compression in agents"),
        ("4. Tools are Structured Outputs", "Tools as JSON descriptions", "✅ ToolRegistry framework"),
        ("5. Unify Execution State", "Single state model", "✅ AgentContext schema"),
        ("6. Launch/Pause/Resume", "Simple APIs for control", "✅ FastAPI endpoints"),
        ("7. Contact Humans with Tools", "Human-in-the-loop workflows", "🔄 HumanLayer integration pending"),
        ("8. Own Your Control Flow", "Custom agent logic", "✅ DeploymentAgent example"),
        ("9. Compact Errors", "Efficient error handling", "✅ Structured error responses"),
        ("10. Small Focused Agents", "Single-responsibility agents", "✅ DeploymentAgent example"),
        ("11. Trigger from Anywhere", "Multi-channel access", "✅ API + webhook endpoints"),
        ("12. Stateless Reducer", "Pure function pattern", "✅ Agent.run() method"),
    ]
    
    for factor, description, implementation in factors:
        table.add_row(factor, description, implementation)
    
    console.print(table)
    
    console.print("\n🎯 Key Features:")
    console.print("• FastAPI server with async support")
    console.print("• Multiple LLM provider support (OpenAI, Anthropic)")
    console.print("• Structured tool execution framework")
    console.print("• Template-based prompt management")
    console.print("• Background agent execution")
    console.print("• Webhook integration for external triggers")


if __name__ == "__main__":
    cli()