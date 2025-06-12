# Agent Registry System

A multi-agent system implementing Google's Agent-to-Agent (A2A) protocol with a centralized registry for agent discovery and coordination.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Registry Agent (Port 6060)](#registry-agent-port-6060)
  - [Email Agent (Port 6001)](#email-agent-port-6001)
  - [Chat Agent (Port 6002)](#chat-agent-port-6002)
- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Python Dependencies](#python-dependencies)
  - [AI Service APIs](#ai-service-apis)
- [Configuration](#configuration)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Google MCP Server Setup (Required for Email Agent)](#2-google-mcp-server-setup-required-for-email-agent)
  - [3. Agent Configuration Files](#3-agent-configuration-files)
- [Usage](#usage)
  - [Starting the Agents](#starting-the-agents)
  - [Manual Startup](#manual-startup)
  - [Using the Command Line Client](#using-the-command-line-client)
  - [Stopping the Agents](#stopping-the-agents)
  - [List Agent Cards CLI](#list-agent-cards-cli)
- [API Endpoints](#api-endpoints)
  - [Registry Agent (`localhost:6060`)](#registry-agent-localhost6060)
  - [Chat Agent (`localhost:6002`)](#chat-agent-localhost6002)
  - [Email Agent (`localhost:6001`)](#email-agent-localhost6001)
- [Security](#security)
  - [API Keys](#api-keys)
- [Agent Cards](#agent-cards)
- [Logging](#logging)
- [Example Workflows](#example-workflows)
  - [1. Agent Registration](#1-agent-registration)
  - [2. User Request Processing](#2-user-request-processing)
  - [3. Email Sending](#3-email-sending)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Debug Mode](#debug-mode)
- [Proof of Concept](#proof-of-concept)
- [License](#license)
- [Related Projects](#related-projects)

## Overview

This project implements a distributed agent registry system where:

- **Registry Agent** (`svr_registry_agent.py`) - Acts as a central registry that stores agent cards and provides semantic search functionality
- **Chat Agent** (`svr_chat_agent.py`) - A client agent that can dynamically enlist other agents to fulfill user requests
- **Email Agent** (`svr_email_agent.py`) - A specialized agent that sends emails using Google's MCP server
- **Command Line Client** (`chat_client.py`) - Interactive client for testing the chat agent

The system allows agents to register themselves at startup, discover other agents through semantic search, and coordinate to fulfill complex user requests that require multiple capabilities.

The system also uses a local GSuite MCP server that is automatically started and stopped when the Email Agent starts and stops, demonstrating how to do this in code.

Note: This is currently a proof-of-concept project, not a production ready solution. See the Proof Of Concept section of this document for more details.

## Architecture

### Registry Agent (Port 6060)
- Maintains a registry of trusted agents with API key authentication
- Provides semantic search using LLM to match user queries with agent capabilities
- Supports CRUD operations for agent registration
- Stores agent cards with detailed capability descriptions

### Email Agent (Port 6001)
- Specialized agent for sending emails via Google Workspace
- Uses Google's MCP (Model Context Protocol) server for Gmail integration
- Requires Google OAuth2 setup and credentials

### Chat Agent (Port 6002)  
- Acts as an orchestrating agent that can fulfill user requests
- Searches the registry to find agents with needed capabilities
- Coordinates with multiple agents to complete complex tasks
- Maintains conversation sessions with users

## Requirements

### System Requirements
- MacOS or Linux with bash, jq and curl
- Python 3.12.6+ with pip 
- Node.js (for Google MCP server)
- Google Cloud Project with Gmail API enabled (for email functionality)

### Python Dependencies
Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `fastapi` - Web framework for agent servers
- `google-adk` - Google Agent Development Kit
- `litellm` - Multi-LLM support (OpenAI, Anthropic, etc.)
- `pydantic` - Data validation
- `requests` - HTTP client
- `python-dotenv` - Environment configuration

### AI Service APIs
You'll need API keys for at least one of:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google AI (Gemini models)

## Configuration

### 1. Environment Setup
Copy the example environment file and configure:
```bash
cp env.example .env
```

Edit `.env` with your API keys:
```bash
# AI Keys - Add at least one
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key  
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google MCP Server Path (for email functionality)
MCP_GSUITE_PATH=/path/to/gsuite-mcp/
MCP_GSUITE_RUN_FILE=build/index.js

# Logging Level
LOG_LEVEL=DEBUG
```

### 2. Google MCP Server Setup (Required for Email Agent)
The email agent requires the Google Workspace MCP server:

1. Clone and setup the MCP server:
```bash
git clone https://github.com/rishipradeep-think41/gsuite-mcp
cd gsuite-mcp
npm install
npm run build
```

2. Configure Google OAuth2:
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Create OAuth2 credentials
   - Download `credentials.json` to the MCP server directory

3. Get refresh token:
```bash
node get-refresh-token.js
```

4. Copy `token.json` to the parent directory of the MCP server

**Note:** Refresh tokens expire every 7 days and must be renewed.

### 3. Agent Configuration Files
Each agent has a config file with registry connection details:

**svr_chat_agent_config.json:**
```json
{
  "reg_url": "http://localhost:6060/tasks/send",
  "api_key": "demo_api_key3",
  "agent_id": "55f7e37c-7c0b-413a-a15f-42ee1d2dbe18"
}
```

**svr_email_agent_config.json:**
```json
{
  "reg_url": "http://localhost:6060/tasks/send", 
  "api_key": "demo_api_key1",
  "agent_id": "383445c2-7d90-48fc-99cb-38b1819cebe8"
}
```

## Usage

### Starting the Agents

Use the provided startup script to launch all agents:
```bash
chmod +x startup.sh
./startup.sh
```

This will start:
1. Registry Agent on port 6060
2. Email Agent on port 6001  
3. Chat Agent on port 6002

### Manual Startup
You can also start agents individually:

```bash
# Registry Agent
uvicorn svr_registry_agent:app --host 0.0.0.0 --port 6060

# Email Agent  
uvicorn svr_email_agent:app --host 0.0.0.0 --port 6001

# Chat Agent
uvicorn svr_chat_agent:app --host 0.0.0.0 --port 6002
```

### Using the Command Line Client

Once agents are running, test the system:
```bash
python chat_client.py
```

Example interactions:
```
Enter a query: Send an email to john@example.com saying hello
Enter a query: What agents are available in the registry?
Enter a query: exit
```

### Stopping the Agents
```bash
chmod +x shutdown.sh  
./shutdown.sh
```
### List Agent Cards CLI
This is a simple script that makes a request to the registry
to get a list of agent cards. It does not require an API key.

NOTE: requires jq command to format the output.

```bash
chmod +x list_agent_cards.sh  
./list_agent_cards.sh
```


## API Endpoints

### Registry Agent (`localhost:6060`)
- `GET /.well-known/agent.json` - Agent card
- `POST /tasks/send` - Handle tasks (register, update, remove, search agents)

**Tools:**
- `register_agent` - Register new agent (requires API key)
- `update_agent` - Update existing agent (requires API key)  
- `remove_agent` - Remove agent (requires API key)
- `search_agents` - Semantic search for agents (public)
- `list_agent_cards` - List all registered agents (public)

### Chat Agent (`localhost:6002`)
- `GET /.well-known/agent.json` - Agent card
- `POST /tasks/send` - Process user chat requests

### Email Agent (`localhost:6001`)  
- `GET /.well-known/agent.json` - Agent card
- `POST /tasks/send` - Send email requests

## Security

- **API Key Authentication**: Registry operations require valid API keys
- **Trusted Agents**: Only pre-configured agents can register/modify registry
- **Input Validation**: All inputs validated using Pydantic models
- **No Secret Exposure**: API keys and tokens managed through environment variables

### API Keys
Default API keys for demo (change in production):
```python
TRUSTED_AGENTS = {
    "demo_api_key1": "EmailServerAgent",
    "demo_api_key2": "FetchServerAgent", 
    "demo_api_key3": "ChatServerAgent",
    "abcd1234": "GeospatialRoutingAgent",
    "regadmin42": "RegistryAdminAgent"
}
```

## Agent Cards

Each agent publishes an agent card describing its capabilities:

```json
{
  "name": "ChatAgent",
  "description": "A chat agent that takes input request from the user and dynamically enlists other agents to assist.",
  "url": "http://localhost:6002",
  "version": "0.1",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "query_registry", 
      "name": "Query Register Agent",
      "description": "Queries the registry to create find agents it needs.",
      "tags": ["register", "registry", "agent"]
    }
  ]
}
```

## Logging

- Centralized logging via `shared_logger.py`
- Log level controlled by `LOG_LEVEL` environment variable
- Logs written to `agents.log` with rotation
- Console output level configurable per agent

## Example Workflows

### 1. Agent Registration
When agents start, they automatically register with the registry:
```python
await register(APP_NAME)  # Called during agent startup
```

### 2. User Request Processing
1. User sends request to Chat Agent
2. Chat Agent queries Registry for capable agents
3. Registry returns ranked list of matching agents
4. Chat Agent coordinates with selected agents
5. Response aggregated and returned to user

### 3. Email Sending
```
User: "Send an email to test@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"
Chat Agent → Registry Agent: "find agent that can send email"
Registry Agent → Chat Agent: [EmailAgent card with confidence score]
Chat Agent → Email Agent: {to: "test@example.com", subject: "Meeting", body: "Let's meet tomorrow"}
Email Agent → User: "Email sent successfully"
```

## Troubleshooting

### Common Issues

**Registry Agent fails to start:**
- Check API keys in `.env` file
- Verify model availability (OpenAI/Anthropic/Google)

**Email Agent fails:**
- Ensure Google MCP server is properly configured
- Check `credentials.json` and `token.json` files exist
- Verify refresh token hasn't expired (7-day limit)
- Check `MCP_GSUITE_PATH` environment variable

**Agents can't communicate:**
- Verify all agents are running on correct ports
- Check agent config files have correct registry URL
- Ensure API keys match between config and registry

**Chat Client connection fails:**
- Confirm Chat Agent is running on port 6002
- Check firewall settings
- Verify agent card URL is accessible

### Debug Mode
Set `LOG_LEVEL=DEBUG` in `.env` for detailed logging.

## Proof of Concept

This is a proof-of-concept implementation. For production use:

- Replace in-memory storage with persistent database
- Implement proper authentication and authorization
- Implement TLS communications between agents
- Use a production ready email service / MCP that does not require OAuth refresh tokens.
- Load LLM selection from .env 
- Add monitoring and health checks
- Potentially use remote MCP servers instead of local processes
- Add comprehensive error handling and retry logic

## License

MIT License - See LICENSE file for details.

## Related Projects

- [Google Agent Development Kit](https://google.github.io/adk-docs/)
- [Google Agent2Agent (A2A) Protocol](https://google-a2a.github.io/A2A/#/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Google Workspace MCP Server](https://github.com/rishipradeep-think41/gsuite-mcp)
