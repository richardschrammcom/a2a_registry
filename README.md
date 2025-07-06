# Agent Registry System

A multi-agent system implementing Google's Agent-to-Agent (A2A) protocol with a centralized registry for agent discovery and coordination.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Registry Agent (Port 6060)](#registry-agent-port-6060)
  - [Email Agent (Port 6001)](#email-agent-port-6001)
  - [Chat Agent (Port 6002)](#chat-agent-port-6002)
  - [SMS Agent (Port 6003)](#sms-agent-port-6003)
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
- **SMS Agent** (`svr_sms_agent.py`) - A specialized agent that sends SMS text messages using a third-party SMS service
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

### SMS Agent (Port 6003)
- Specialized agent for sending SMS text messages via third-party SMS service
- Parses unstructured user requests to extract phone numbers and message content
- Handles various phone number formats including E.164 format with country codes
- Requires SMS service URL and API key configuration in environment variables

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

# SMS Service Configuration (for SMS functionality)
SMS_URL=https://your-sms-service.com/api/send
SMS_KEY=your_sms_api_key

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

Use the provided startup script to launch all core agents:
```bash
chmod +x startup.sh
./startup.sh
```

This will start:
1. Registry Agent on port 6060
2. Email Agent on port 6001  
3. Chat Agent on port 6002

**SMS Agent Demonstration Setup:**
For demonstration purposes, the SMS Agent can be started separately to show agent failure/recovery:
```bash
chmod +x startup_sms.sh
./startup_sms.sh
```

This starts only the SMS Agent on port 6003. The separate startup script allows you to:
1. Start the core agents without SMS capability
2. Test Chat Agent SMS requests (which will fail)
3. Start the SMS Agent separately
4. Retry SMS requests (which will now succeed)

### Manual Startup
You can also start agents individually:

```bash
# Registry Agent
uvicorn svr_registry_agent:app --host 0.0.0.0 --port 6060

# Email Agent  
uvicorn svr_email_agent:app --host 0.0.0.0 --port 6001

# Chat Agent
uvicorn svr_chat_agent:app --host 0.0.0.0 --port 6002

# SMS Agent
uvicorn svr_sms_agent:app --host 0.0.0.0 --port 6003
```

### Using the Command Line Client

The chat client supports both traditional request-response mode and real-time streaming mode for enhanced user experience.

#### Basic Usage
```bash
python chat_client.py
```

#### Streaming Mode (Recommended)
For real-time updates during agent processing:
```bash
python chat_client.py --stream
```

#### Interactive Mode Switching
You can switch between modes during a session:
```
Enter a query: toggle
# Switches between streaming and non-streaming modes
```

#### Example Interactions

**Non-streaming mode:**
```
Enter a query: Send an email to john@example.com saying hello
Agent's reply: I've successfully sent the email to john@example.com...

Enter a query: Send an SMS to 555-123-4567 saying "Meeting at 3pm"
Agent's reply: I've successfully sent the SMS message...
```

**Streaming mode (provides real-time updates):**
```
Enter a query: Send an SMS to 555-123-4567 saying "Meeting at 3pm"
🔄 Connected to agent stream...
💭 Thinking about how to help you...
🔍 I'm going to need help with this request. Searching the registry for an agent that can send a text message...
📋 Great! The registry found an agent that can help us.
📞 Passing your request to the SMSAgent...
✅ The SMSAgent responded that it successfully completed your request!

🎯 I've successfully sent the SMS message "Meeting at 3pm" to 555-123-4567. If you need further assistance or have any other requests, feel free to let me know!

What else can I do for you today? (or type 'quit' or 'exit' to quit)
```

#### Additional Commands
```
Enter a query: my name is Rich         # Set conversation context
Enter a query: what's my name?         # Test session continuity  
Enter a query: exit                    # Quit the client
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

### SMS Agent (`localhost:6003`)
- `GET /.well-known/agent.json` - Agent card
- `POST /tasks/send` - Send SMS text message requests

**Tools:**
- `send_sms` - Send SMS to phone number with message content
- `query_registry` - Search for other agents in the registry
- `call_remote_agent` - Call other agents using A2A protocol

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
    "demo_api_key4": "SMSServerAgent",
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

### 4. SMS Demonstration Workflow
This workflow demonstrates the agent failure/recovery pattern:

**Step 1: Clean slate - ensure SMS Agent is not in registry**
```bash
./shutdown.sh  # Stop all agents to ensure clean state
./startup.sh   # Starts Registry, Email, and Chat agents only
```

**Important:** If the SMS Agent was previously registered, you may need to remove it from the registry first. The registry maintains agent registrations in memory, so restarting with `./shutdown.sh` followed by `./startup.sh` ensures a clean state where the SMS Agent is not registered.

**Step 2: Test SMS request (will fail)**
```bash
python chat_client.py
# Enter: "Send SMS to 555-123-4567 saying 'Hello from the agent system'"
# Result: Chat Agent will search registry and report no SMS-capable agents found
```

**Step 3: Start SMS Agent (will auto-register)**
```bash
./startup_sms.sh  # Starts SMS Agent on port 6003
# SMS Agent automatically registers itself with the Registry upon startup
```

**Step 4: Retry SMS request (will succeed)**
```bash
python chat_client.py
# Enter: "Send SMS to 555-123-4567 saying 'Hello from the agent system'"
# Result: Chat Agent now finds SMS Agent in registry and successfully sends message
```

This demonstrates the dynamic nature of the agent system - agents can come online at any time and immediately become available to other agents through the registry.

**SMS Request Flow:**
```
User: "Send SMS to +1-555-123-4567 saying 'Meeting reminder'"
Chat Agent → Registry Agent: "find agent that can send SMS"
Registry Agent → Chat Agent: [SMSAgent card with confidence score]
Chat Agent → SMS Agent: {phone: "+1-555-123-4567", message: "Meeting reminder"}
SMS Agent → Third-party SMS Service: API call with phone and message
SMS Agent → User: "SMS sent successfully"
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

**SMS Agent fails:**
- Check SMS service URL and API key in `.env` file
- Verify `SMS_URL` and `SMS_KEY` environment variables are set
- Ensure SMS service is accessible and API key is valid
- Check phone number format (accepts E.164 format with + prefix)

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
