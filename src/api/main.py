"""
Main FastAPI application.
Implements Factor 11: Trigger from Anywhere - API endpoints, webhooks.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..agents.deployment_agent import DeploymentAgent
from ..agents.llm_providers import LLMProviderFactory
from ..config import settings
from ..tools.schemas import AgentContext
from ..tools import builtin_tools  # Import to register built-in tools

# Configure structured logging
import logging

log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer() if settings.structured_logging else structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        log_level_map.get(settings.log_level, logging.INFO)
    ),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# In-memory storage for demo (replace with proper database in production)
agent_sessions: Dict[str, AgentContext] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting 12-factor agents API", version="0.1.0")
    
    # Initialize default LLM provider
    try:
        app.state.llm_provider = LLMProviderFactory.create_default_provider()
        logger.info("LLM provider initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize LLM provider", error=str(e))
        raise
    
    yield
    
    logger.info("Shutting down 12-factor agents API")


# Create FastAPI app
app = FastAPI(
    title="12-Factor Agents API",
    description="Production-ready AI agent application following 12-factor methodology",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AgentRequest(BaseModel):
    """Request to start or interact with an agent."""
    message: str
    agent_type: str = "deployment"
    user_id: Optional[str] = None
    session_data: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Response from agent interaction."""
    thread_id: str
    agent_type: str
    status: str
    message: Optional[str] = None
    results: Dict[str, Any] = {}
    execution_state: str
    requires_human_input: bool = False
    requires_approval: bool = False


class ResumeAgentRequest(BaseModel):
    """Request to resume a paused agent."""
    human_response: Optional[str] = None
    approved: Optional[bool] = None
    additional_context: Dict[str, Any] = {}


# Dependency to get LLM provider
def get_llm_provider():
    """Get the LLM provider from app state."""
    return app.state.llm_provider


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "12-Factor Agents API",
        "version": "0.1.0",
        "description": "Production-ready AI agent application",
        "factors_implemented": [
            "Natural Language to Tool Calls",
            "Own Your Prompts", 
            "Own Your Context Window",
            "Tools are Structured Outputs",
            "Unify Execution State",
            "Launch/Pause/Resume",
            "Contact Humans with Tools",
            "Own Your Control Flow",
            "Compact Errors",
            "Small Focused Agents",
            "Trigger from Anywhere",
            "Stateless Reducer",
        ]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
    }


@app.post("/agents/start", response_model=AgentResponse, tags=["Agents"])
async def start_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    llm_provider=Depends(get_llm_provider),
):
    """
    Start a new agent session.
    Implements Factor 11: Trigger from Anywhere.
    """
    
    thread_id = str(uuid.uuid4())
    
    logger.info(
        "Starting agent session",
        thread_id=thread_id,
        agent_type=request.agent_type,
        user_id=request.user_id,
    )
    
    # Create agent context
    context = AgentContext(
        thread_id=thread_id,
        user_id=request.user_id,
        session_data=request.session_data,
        conversation_history=[{
            "type": "user_message",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat(),
        }],
        created_at=datetime.utcnow().isoformat(),
    )
    
    # Store context
    agent_sessions[thread_id] = context
    
    # Create appropriate agent
    if request.agent_type == "deployment":
        agent = DeploymentAgent(llm_provider)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")
    
    # Run agent in background
    background_tasks.add_task(run_agent_background, agent, thread_id)
    
    return AgentResponse(
        thread_id=thread_id,
        agent_type=request.agent_type,
        status="started",
        message="Agent session started successfully",
        execution_state="running",
    )


@app.get("/agents/{thread_id}/status", response_model=AgentResponse, tags=["Agents"])
async def get_agent_status(thread_id: str):
    """
    Get the current status of an agent session.
    Implements Factor 6: Launch/Pause/Resume with Simple APIs.
    """
    
    if thread_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    context = agent_sessions[thread_id]
    
    # Determine if human input is required
    requires_human_input = context.execution_state in ["waiting_for_human", "waiting_for_approval"]
    requires_approval = context.execution_state == "waiting_for_approval"
    
    # Get latest message/result
    latest_message = None
    results = {}
    
    if context.conversation_history:
        latest_entry = context.conversation_history[-1]
        if latest_entry["type"] == "agent_action":
            content = latest_entry.get("content", {})
            latest_message = content.get("summary") or content.get("question") or "Agent action completed"
            results = content
    
    return AgentResponse(
        thread_id=thread_id,
        agent_type="deployment",  # TODO: Store agent type in context
        status="active",
        message=latest_message,
        results=results,
        execution_state=context.execution_state,
        requires_human_input=requires_human_input,
        requires_approval=requires_approval,
    )


@app.post("/agents/{thread_id}/resume", response_model=AgentResponse, tags=["Agents"])
async def resume_agent(
    thread_id: str,
    request: ResumeAgentRequest,
    background_tasks: BackgroundTasks,
    llm_provider=Depends(get_llm_provider),
):
    """
    Resume a paused agent session.
    Implements Factor 6: Launch/Pause/Resume with Simple APIs.
    """
    
    if thread_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    context = agent_sessions[thread_id]
    
    if context.execution_state not in ["waiting_for_human", "waiting_for_approval"]:
        raise HTTPException(
            status_code=400,
            detail=f"Agent is not paused (current state: {context.execution_state})"
        )
    
    logger.info(
        "Resuming agent session",
        thread_id=thread_id,
        human_response=bool(request.human_response),
        approved=request.approved,
    )
    
    # Add human response to context
    if request.human_response:
        context.conversation_history.append({
            "type": "human_response",
            "content": request.human_response,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    if request.approved is not None:
        context.conversation_history.append({
            "type": "approval_response",
            "content": {"approved": request.approved},
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    # Update additional context
    if request.additional_context:
        context.session_data.update(request.additional_context)
    
    # Resume execution state
    context.execution_state = "running"
    context.updated_at = datetime.utcnow().isoformat()
    
    # Create agent and continue execution
    agent = DeploymentAgent(llm_provider)
    background_tasks.add_task(run_agent_background, agent, thread_id)
    
    return AgentResponse(
        thread_id=thread_id,
        agent_type="deployment",
        status="resumed",
        message="Agent session resumed successfully",
        execution_state="running",
    )


@app.get("/agents/{thread_id}/history", tags=["Agents"])
async def get_agent_history(thread_id: str):
    """Get the conversation history for an agent session."""
    
    if thread_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    context = agent_sessions[thread_id]
    
    return {
        "thread_id": thread_id,
        "conversation_history": context.conversation_history,
        "execution_state": context.execution_state,
        "created_at": context.created_at,
        "updated_at": context.updated_at,
    }


@app.delete("/agents/{thread_id}", tags=["Agents"])
async def delete_agent_session(thread_id: str):
    """Delete an agent session."""
    
    if thread_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="Agent session not found")
    
    del agent_sessions[thread_id]
    
    logger.info("Deleted agent session", thread_id=thread_id)
    
    return {"message": "Agent session deleted successfully"}


@app.get("/agents", tags=["Agents"])
async def list_agent_sessions():
    """List all active agent sessions."""
    
    sessions = []
    for thread_id, context in agent_sessions.items():
        sessions.append({
            "thread_id": thread_id,
            "execution_state": context.execution_state,
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "message_count": len(context.conversation_history),
        })
    
    return {
        "sessions": sessions,
        "total_count": len(sessions),
    }


# Webhook endpoint for external integrations
@app.post("/webhooks/trigger", tags=["Webhooks"])
async def webhook_trigger(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    llm_provider=Depends(get_llm_provider),
):
    """
    Webhook endpoint for triggering agents from external systems.
    Implements Factor 11: Trigger from Anywhere.
    """
    
    logger.info("Webhook triggered", payload_keys=list(payload.keys()))
    
    # Extract message and agent type from payload
    message = payload.get("message", "Webhook triggered with no message")
    agent_type = payload.get("agent_type", "deployment")
    user_id = payload.get("user_id")
    
    # Create agent request
    request = AgentRequest(
        message=message,
        agent_type=agent_type,
        user_id=user_id,
        session_data=payload.get("context", {}),
    )
    
    # Start agent session
    response = await start_agent(request, background_tasks, llm_provider)
    
    return {
        "status": "webhook_processed",
        "agent_response": response,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Background task to run agent
async def run_agent_background(agent, thread_id: str):
    """Run agent in background and update session storage."""
    
    try:
        context = agent_sessions[thread_id]
        
        logger.info("Running agent in background", thread_id=thread_id)
        
        # Execute agent
        updated_context = await agent.run(context)
        
        # Update stored context
        updated_context.updated_at = datetime.utcnow().isoformat()
        agent_sessions[thread_id] = updated_context
        
        logger.info(
            "Agent execution completed",
            thread_id=thread_id,
            execution_state=updated_context.execution_state,
            iterations=len(updated_context.conversation_history),
        )
        
    except Exception as e:
        logger.error(
            "Agent execution failed",
            thread_id=thread_id,
            error=str(e),
            exc_info=True,
        )
        
        # Update context with error state
        if thread_id in agent_sessions:
            context = agent_sessions[thread_id]
            context.execution_state = "error"
            context.conversation_history.append({
                "type": "system_error",
                "content": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat(),
            })
            context.updated_at = datetime.utcnow().isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )