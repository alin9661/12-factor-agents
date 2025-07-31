# 12-Factor Agents

Production-ready AI agent application following the [12-factor-agents methodology](https://github.com/humanlayer/12-factor-agents) with HumanLayer integration.

## Overview

This application demonstrates how to build reliable, scalable LLM-powered software using the 12-factor agents methodology. Instead of "prompt + tools + loop until done" patterns, it implements agents as mostly deterministic code with LLM steps strategically placed.

##  12-Factor Implementation Status

| Factor | Description | Status | Implementation |
|--------|-------------|--------|----------------|
| 1. Natural Language to Tool Calls | Convert user input to structured actions |  | `BaseAgent.determine_next_step()` |
| 2. Own Your Prompts | Manage prompts as first-class code |  | `PromptManager` with Jinja2 templates |
| 3. Own Your Context Window | Control context and token usage |  | Context compression in agents |
| 4. Tools are Structured Outputs | Tools as JSON descriptions |  | `ToolRegistry` framework |
| 5. Unify Execution State | Single state model |  | `AgentContext` schema |
| 6. Launch/Pause/Resume | Simple APIs for control |  | FastAPI endpoints |
| 7. Contact Humans with Tools | Human-in-the-loop workflows | = | HumanLayer integration pending |
| 8. Own Your Control Flow | Custom agent logic |  | `DeploymentAgent` example |
| 9. Compact Errors | Efficient error handling |  | Structured error responses |
| 10. Small Focused Agents | Single-responsibility agents |  | `DeploymentAgent` example |
| 11. Trigger from Anywhere | Multi-channel access |  | API + webhook endpoints |
| 12. Stateless Reducer | Pure function pattern |  | `Agent.run()` method |

## =€ Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd 12-factor
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the server**
   ```bash
   uv run python -m src.main serve
   ```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Using the CLI

```bash
# Show available commands
uv run python -m src.main --help

# List available tools
uv run python -m src.main list-tools

# Show implementation status
uv run python -m src.main info

# Start the API server
uv run python -m src.main serve
```

## <× Architecture

### Core Components

- **`src/agents/`** - Agent implementations and LLM providers
- **`src/prompts/`** - Prompt management with Jinja2 templates
- **`src/tools/`** - Structured tool registry and built-in tools
- **`src/api/`** - FastAPI server with REST endpoints
- **`src/state/`** - State management and persistence (pending)
- **`src/context/`** - Context window management (pending)
- **`src/human/`** - HumanLayer integration (pending)
- **`src/errors/`** - Error handling and compression

### Example Agent: DeploymentAgent

The `DeploymentAgent` demonstrates all 12 factors:

```python
from src.agents.deployment_agent import DeploymentAgent
from src.agents.llm_providers import OpenAIProvider

# Create agent
provider = OpenAIProvider(api_key="your-key")
agent = DeploymentAgent(provider)

# Create context
context = AgentContext(
    thread_id="deploy-123",
    conversation_history=[{
        "type": "user_message",
        "content": "Deploy api-backend v1.2.3 to production"
    }]
)

# Run agent
result = await agent.run(context)
```

## =à Available Tools

The system includes 11 built-in tools across different categories:

### Utility Tools
- `log_message` - Log messages for debugging
- `sleep` - Add delays for rate limiting

### HTTP Tools  
- `http_get` - Make HTTP GET requests
- `http_post` - Make HTTP POST requests

### Filesystem Tools
- `read_file` - Read file contents
- `write_file` - Write to files (requires approval)

### Communication Tools
- `send_email` - Send emails (requires approval)

### Deployment Tools
- `check_service_status` - Check service health
- `deploy_service` - Deploy services (requires approval)

### Customer Support Tools
- `lookup_customer` - Find customer information
- `create_support_ticket` - Create support tickets

## < API Endpoints

### Agent Management

- `POST /agents/start` - Start a new agent session
- `GET /agents/{thread_id}/status` - Get agent status  
- `POST /agents/{thread_id}/resume` - Resume paused agent
- `GET /agents/{thread_id}/history` - Get conversation history
- `DELETE /agents/{thread_id}` - Delete agent session
- `GET /agents` - List all sessions

### Webhooks

- `POST /webhooks/trigger` - Trigger agents from external systems

### Example API Usage

```bash
# Start an agent
curl -X POST "http://localhost:8000/agents/start" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Deploy api-backend v1.2.3 to production",
       "agent_type": "deployment",
       "user_id": "user123"
     }'

# Check status
curl "http://localhost:8000/agents/{thread_id}/status"

# Resume if paused
curl -X POST "http://localhost:8000/agents/{thread_id}/resume" \
     -H "Content-Type: application/json" \
     -d '{"human_response": "Approved for deployment"}'
```

## <¯ Key Features

### Production-Ready Design
- **Async/await throughout** for high concurrency
- **Structured logging** with JSON output
- **Error handling** with recovery strategies
- **Background task execution** for long-running agents
- **Health checks** and monitoring endpoints

### LLM Provider Support
- **OpenAI** (GPT-4, GPT-3.5-turbo) with structured outputs
- **Anthropic** (Claude) with JSON parsing
- **Extensible** provider system for additional models

### Prompt Management
- **Template-based prompts** with Jinja2
- **Version control** for prompt iterations
- **Variable validation** and testing
- **Prompt testing framework** for quality assurance

### Tool System
- **Structured tool definitions** with JSON schemas
- **Automatic parameter validation**
- **Risk-based approval workflows**
- **Tool categorization** and metadata
- **Extensible tool registry**

### Safety Features
- **Human approval required** for high-risk operations
- **Production deployment gates** with safety checks
- **Error recovery** and graceful degradation
- **Context window management** to prevent token limits
- **Pause/resume capability** for long operations

## =' Configuration

Configuration is managed through environment variables or `.env` file:

```bash
# Application
APP_NAME=12-factor-agents
DEBUG=false

# API
API_HOST=0.0.0.0  
API_PORT=8000

# LLM Providers (set at least one)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# HumanLayer (for human-in-the-loop)
HUMANLAYER_API_KEY=your_humanlayer_key

# Context Management
MAX_CONTEXT_TOKENS=100000
CONTEXT_COMPRESSION_THRESHOLD=0.8

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
```

## =§ Roadmap

### Phase 2: Enhanced State Management
- [ ] PostgreSQL persistence layer
- [ ] Redis caching for session data
- [ ] Database migrations with Alembic
- [ ] State serialization/deserialization

### Phase 3: Human Integration
- [ ] Complete HumanLayer integration
- [ ] Slack bot interface
- [ ] Email notification system
- [ ] Approval workflow management

### Phase 4: Production Features
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards
- [ ] Comprehensive test suite

## =Ú Learn More

- [12-Factor Agents Methodology](https://github.com/humanlayer/12-factor-agents)
- [HumanLayer Documentation](https://docs.humanlayer.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://pydantic-docs.helpmanual.io/)

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## =Ä License

MIT License - see LICENSE file for details.

---

Built with d following the 12-factor agents methodology for production-ready AI applications.