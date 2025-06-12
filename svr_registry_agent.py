# trusted_agent_registry.py

import uuid
import os
import tempfile
import shutil
import re
from dotenv import load_dotenv, dotenv_values, set_key
from contextlib import asynccontextmanager
import json
from typing import Optional, List
from pydantic import BaseModel

# Google Agent Imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types 
from google.adk.tools import agent_tool
#from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Imports for being an Agent Host
from fastapi import FastAPI, Request, Header, HTTPException, Body
from fastapi.responses import JSONResponse

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Load environment
env_path=os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Set the token path to load the Google token information from token.json
TOKEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "token.json"))

# Set Registry File to be in same directory as this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(BASE_DIR, "registry.json")

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
from shared_logger import SharedLogger
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

SHOW_AGENT_EVENTS = True

# In-memory registry storage
AGENT_REGISTRY = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load the registry from the filesystem.
    # NOTE: This is for demo purposes only and should
    # be replaced with persisting in a redis cache or
    # a persistence layer.
    # TODO: Replace me in production!
    global AGENT_REGISTRY
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(load_registry())

    # Start the agent.
    await create_agent()
    yield
    logger.info("Shutting down registry agent. Cleaning up resources...")
    exit_stack = getattr(app, "exit_stack", None)
    if exit_stack:
        try:
            await exit_stack.aclose()
        except ProcessLookupError:
            logger.warning("Subprocess already terminated; skipping cleanup.")

# Create a simple web server with FastAPI using the lifespan context manager
app = FastAPI(lifespan=lifespan)


# API key whitelist
TRUSTED_AGENTS = {
    "demo_api_key1": "EmailServerAgent",
    "demo_api_key2": "FetchServerAgent",
    "demo_api_key3": "ChatServerAgent",
    "demo_api_key-SMS": "SMSServerAgent",
    "abcd1234": "GeospatialRoutingAgent",
    "regadmin42": "RegistryAdminAgent"
}

APP_NAME = "registry_agent"

initial_agent_instructions = """
You are a semantic router for agent cards stored in a Python dictionary called `AGENT_REGISTRY`.

You have a tool called list_agent_cards which takes no parameters and returns the python dictionary to you to analyze.

Each key in the dictionary is a unique agent ID or alias. Each value is a dictionary with metadata about that agent, including:
- `name`: the human-readable name of the agent
- `description`: a summary of what the agent does
- `skills`: a list of skill dictionaries, each with its own `name` and `description`

Your task is to:
1. Use your "list_agent_cards" tool to retrieve a list of registered agents for you analyze against the user query. 
2. Compare a user's query to the key, `name`, and `description` fields of each agent card.
3. Also consider each skill's `name`, `description`, and `example` fields.
4. Score each agent card from 0 to 100 based on how well it matches the request, with 100 representing a perfect match.
5. Format a payload in your response as a JSON array of objects, ordered by score (highest first)
6. You may add additional information in the response in addition to the payload, such as your reasoning, etc. However, the response
must ALWAYS contain a JSON payload, because that payload will be used by other agents to identify trusted agents. The non-payload text
in the response will not be sent to the caller, only the JSON payload, as described below.

JSON payload format - a JSON array where each object contains exactly these 4 fields:
{
  "name": "Agent Name from the card",
  "confidence_score": 85,
  "explanation": "Brief explanation of why this score was assigned and how the agent matches the query",
  "agent_card": {entire agent card object exactly as stored in registry}
}

Payload Requirements:
- Order results by confidence_score (highest first) 
- Explanation should be 1-2 sentences explaining the match reasoning
- Agent_card must be the complete, unmodified card from the registry
- You MUST return all the cards in the registry in the results, even if all of them have a score of 0.
- You MUST NOT ADD ANYTHING ADDITIONAL TO THE PAYLOAD FORMAT LISTED BELOW.
- You MUST WRAP THE PAYLOAD WITH three backticks, then the string 'json' then the payload, then 3 more backticks.
- You MUST NOT DEVIATE FROM THE PAYLOAD FORMAT LISTED BELOW. Code is expecting to see this explicit format in order to parse the
  payload from your response.

Example payload response:
```json
[
  {
    "name": "GeoSpatial Route Planner Agent",
    "confidence_score": 92,
    "explanation": "Perfect match for routing requests with traffic-aware optimization and custom mapping capabilities.",
    "agent_card": {complete agent card object}
  },
  {
    "name": "Weather Agent", 
    "confidence_score": 67,
    "explanation": "Partially relevant for location-based queries but focuses on weather rather than routing.",
    "agent_card": {complete agent card object}
  }
]
```

Below is an example of a card that could be in the AGENT_REGISTRY:
{
  "name": "GeoSpatial Route Planner Agent",
  "description": "Provides advanced route planning, traffic analysis, and custom map generation services. This agent can calculate optimal routes, estimate travel times considering real-time traffic, and create personalized maps with points of interest.",
  "url": "https://georoute-agent.example.com/a2a/v1",
  "provider": {
    "organization": "Example Geo Services Inc.",
    "url": "https://www.examplegeoservices.com"
  },
  "version": "1.2.0",
  "documentationUrl": "https://docs.examplegeoservices.com/georoute-agent/api",
  "capabilities": {
    "streaming": True,
    "pushNotifications": True,
    "stateTransitionHistory": False
  },
  "authentication": {
    "schemes": ["OAuth2"],
    "credentials": "{\"authorizationUrl\": \"https://auth.examplegeoservices.com/authorize\", \"tokenUrl\": \"https://auth.examplegeoservices.com/token\", \"scopes\": {\"route:plan\": \"Allows planning new routes.\", \"map:custom\": \"Allows creating and managing custom maps.\"}}"
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "image/png"],
  "skills": [
    {
      "id": "route-optimizer-traffic",
      "name": "Traffic-Aware Route Optimizer",
      "description": "Calculates the optimal driving route between two or more locations, taking into account real-time traffic conditions, road closures, and user preferences (e.g., avoid tolls, prefer highways).",
      "tags": ["maps", "routing", "navigation", "directions", "traffic"],
      "examples": [
        "Plan a route from '1600 Amphitheatre Parkway, Mountain View, CA' to 'San Francisco International Airport' avoiding tolls.",
        "{\"origin\": {\"lat\": 37.422, \"lng\": -122.084}, \"destination\": {\"lat\": 37.7749, \"lng\": -122.4194}, \"preferences\": [\"avoid_ferries\"]}"
      ],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
        "application/vnd.geo+json",
        "text/html"
      ]
    },
    {
      "id": "custom-map-generator",
      "name": "Personalized Map Generator",
      "description": "Creates custom map images or interactive map views based on user-defined points of interest, routes, and style preferences. Can overlay data layers.",
      "tags": ["maps", "customization", "visualization", "cartography"],
      "examples": [
        "Generate a map of my upcoming road trip with all planned stops highlighted.",
        "Show me a map visualizing all coffee shops within a 1-mile radius of my current location."
      ],
      "inputModes": ["application/json"],
      "outputModes": [
        "image/png",
        "image/jpeg",
        "application/json",
        "text/html"
      ]
    }
  ]
}
"""

if log_level_name == 'DEBUG':
    logger.debug("Adding LLM debug instructions to the initial message.")
    initial_agent_instructions += """

This program is currently running in debug mode, so it additional information (outside of the payload described previously)
may be added to your response. Please add your reasoning step by step:
1. First state what you're doing: "I'm searching for agents that match your query..."
2. When using tools, explain: "Let me retrieve the current agent registry..."
3. When analyzing, explain: "I found X agents, now comparing them to your query..."

CRITICAL - ALWAYS INCLUDE THE PAYLOAD PREVIOUSLY DESCRIBED in your response in addition to any other information you wish to share.
"""

# --- Define Model Constants for easier use ---
# Note: Specific model names might change. Refer to LiteLLM/Provider documentation.
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"

# Set the active model here once so it can be switched out on just this line.
ACTIVE_MODEL = MODEL_GPT_4O

# Use one of the model constants defined earlier
from google.adk.models.lite_llm import LiteLlm
AGENT_MODEL = ACTIVE_MODEL # Starting with OpenAI

registry_agent = None # Initialize to None
runner = None      # Initialize runner to None
session_service = InMemorySessionService() # Create a dedicated service

AGENT_CARD = {
  "name": "Registry Agent",
  "description": "Provides a registry of trusted agents on the network.",
  "url": "http://localhost:6060/",
  "provider": {
    "organization": "Fairwind Technologies, LLC.",
    "url": "https://www.richardschramm.com"
  },
  "version": "0.0.1",
  "documentationUrl": "https://richardshramm.com/agent_registry/",
  "capabilities": {
    "streaming": False,
    "pushNotifications": False,
    "stateTransitionHistory": False
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "image/png"],
  "skills": [
    {
      "id": "register_agent",
      "name": "Register Agent",
      "description": "Registers an agent in the trusted registry",
      "tags": ["register", "registry", "agent"],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
      ]
    },
    {
      "id": "update_agent",
      "name": "Update Agent",
      "description": "Updates an agent in the trusted registry",
      "tags": ["update", "registry", "agent"],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
      ]
    },
    {
      "id": "remove_agent",
      "name": "Remove Agent",
      "description": "Removes an agent in the trusted registry",
      "tags": ["remove", "delete", "registry", "agent"],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
      ]
    },
    {
      "id": "search_agents",
      "name": "Search Agent",
      "description": "Searches for an agent in the trusted registry that matches the user query",
      "tags": ["search", "registry", "agent"],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
      ]
    },
  ]
}

class Part(BaseModel):
    text: str

class Message(BaseModel):
    parts: List[Part]

class TaskRequest(BaseModel):
    id: str
    agent_id: Optional[str] = None
    tool_name: str
    message: Optional[Message] = None

# App endpoints
# Serve the Agent Card at the well-known URL.
@app.get("/.well-known/agent.json")
def get_agent_card():
    logger.info("Request for agent card received.")
    """Endpoint to provide this agent's metadata (Agent Card)."""
    return JSONResponse(AGENT_CARD)

@app.post("/tasks/send")
async def handle_task(
    task_request: TaskRequest = Body(...),
    x_api_key: Optional[str] = Header(None),
    request: Request = None
):

    logger.debug("*************************************")
    logger.debug('New request received!')
    logger.debug(f'Task request JSON {task_request}')
    logger.debug("*************************************")

    # Extract the task ID and the user's message text from the request.
    tool_name = task_request.tool_name
    logger.debug(f"Tool requested: {tool_name}")

    # SEARCH AGENTS
    if tool_name == "search_agents":
        logger.debug("Searching agents.")
        results = await search_agents(task_request) 
        return results

    # REGISTER AGENT
    elif tool_name == "register_agent":
        if not x_api_key or x_api_key not in TRUSTED_AGENTS:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

        logger.debug("Registering agent.")
        return register_agent(task_request)

    # UPDATE AGENT
    elif tool_name == "update_agent":
        if not x_api_key or x_api_key not in TRUSTED_AGENTS:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

        logger.debug("Updating agent.")
        return update_agent(task_request) 

    # REMOVE AGENT
    elif tool_name == "remove_agent":
        if not x_api_key or x_api_key not in TRUSTED_AGENTS:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

        logger.debug("Removing agent.")
        return remove_agent(task_request)

    # GET AGENT CARD
    elif tool_name == "get_agent_card":
        logger.debug("Getting agent card.")
        return get_agent_card(task_request)

    # LIST ALL AGENTS
    elif tool_name == "list_agent_cards":
        logger.debug("Getting all agent cards.")
        return list_agent_cards()

    else:
        logger.error(f"Unknown tool name: {tool_name}")
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

# ------------------------------
# Functions
# ------------------------------
def register_agent(task_request: TaskRequest):
    logger.debug("In register_agent function.")
    try:
        agent_card = parse_message_json(task_request)
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing input JSON: {e}")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

    if not agent_card:
        raise HTTPException(status_code=422, detail="Missing 'card' for agent registration.")
    
    # Before registering the agent, let's make sure it doesn't already exist.
    # Check if agent_id is provided and not empty
    if task_request.agent_id and task_request.agent_id.strip():
        logger.debug("----------------------------------------------------------------")
        logger.debug(f"Received Agent ID {task_request.agent_id}.")
        logger.debug("----------------------------------------------------------------")
        # Check if agent already exists in registry
        if task_request.agent_id in AGENT_REGISTRY:
            logger.debug("----------------------------------------------------------------")
            logger.debug(f"Agent {task_request.agent_id} already exists. Updating instead.")
            logger.debug("----------------------------------------------------------------")
            # Update existing agent
            return update_agent(task_request)
        else:
            # Agent ID provided but doesn't exist, register with the provided ID
            logger.debug("----------------------------------------------------------------")
            logger.debug(f"Agent {task_request.agent_id} doesn't exist in registry. Registering with provided ID.")
            logger.debug("----------------------------------------------------------------")
            AGENT_REGISTRY[task_request.agent_id] = agent_card
            save_registry(AGENT_REGISTRY)
            logger.info(f"Registered agent with existing id {task_request.agent_id}")
            return {"status": "Agent registered successfully", "id": task_request.agent_id}
    else:
        # No agent_id provided, generate a new GUID and register
        logger.debug("----------------------------------------------------------------")
        logger.debug("No agent_id provided. Generating new GUID for registration.")

        guid = str(uuid.uuid4())
        AGENT_REGISTRY[guid] = agent_card
        save_registry(AGENT_REGISTRY)

        logger.debug(f"Agent added to registry with ID of {guid}.")
        logger.debug("----------------------------------------------------------------")

        logger.info(f"Registered agent with new id of {guid}")
        return {"status": "Agent registered successfully", "id": guid}

def update_agent(task_request: TaskRequest):
    logger.debug("In update_agent function.")
    try:
        agent_card = parse_message_json(task_request)
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing input JSON: {e}")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

    if not agent_card:
        raise HTTPException(status_code=422, detail="Missing 'card' for agent registration.")
     
    # Before updating the agent, let's make sure it is already in the registry.
    # Check if agent_id is provided and not empty
    if task_request.agent_id and task_request.agent_id.strip():
        logger.debug("----------------------------------------------------------------")
        logger.debug(f"Received Agent ID {task_request.agent_id}.")
        logger.debug("----------------------------------------------------------------")
        # Check if agent already exists in registry
        if task_request.agent_id in AGENT_REGISTRY:
            logger.debug("----------------------------------------------------------------")
            logger.debug(f"Agent {task_request.agent_id} exists. Updating registry.")
            logger.debug("----------------------------------------------------------------")
            # Update existing agent
            AGENT_REGISTRY[task_request.agent_id] = agent_card
            save_registry(AGENT_REGISTRY)
            logger.info(f"Updated agent in registry with id of {task_request.agent_id}")
            return {"status": "Agent updated successfully", "id": task_request.agent_id}
        else:
            # Agent ID provided but doesn't exist, register with the provided ID
            logger.debug("----------------------------------------------------------------")
            logger.debug(f"Agent {task_request.agent_id} doesn't exist in registry. Sending to register_agent")
            logger.debug("----------------------------------------------------------------")
            return register_agent(task_request)
    else:
        # No agent_id provided, generate a new GUID and register
        logger.debug("----------------------------------------------------------------")
        logger.debug("No agent_id provided. Sending to register_agent.")
        return register_agent(task_request)

def remove_agent(task_request: TaskRequest):
    logger.debug("In remove_agent function.")
    agent_id = task_request.agent_id
    if not agent_id:
        logger.error(f"Missing agent_id to remove agent.")
        raise HTTPException(status_code=422, detail="Missing 'agent_id' for agent removal.")
    if agent_id in AGENT_REGISTRY:
        del AGENT_REGISTRY[agent_id]
        save_registry(AGENT_REGISTRY)
        logger.info(f"Removed agent with agent_id of {agent_id}")
        return {"status": f"Agent with agent_id '{agent_id}' removed."}
    else:
        raise HTTPException(status_code=404, detail=f"Agent with agent_id '{agent_id}' not found.")

async def search_agents(task_request: TaskRequest) -> str:
    logger.debug("In search_agents function.")
    try:
        user_query = parse_message_text(task_request)
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing input JSON: {e}")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

    if not user_query or not isinstance(user_query, str):
        raise HTTPException(status_code=422, detail="Missing or invalid 'query' in message.")

    logger.debug(f"User Query retrieved: {user_query}")
    logger.info("Search Agents query received.")

    # We could probably add all the logic from process_search_request function 
    # here, but this feels cleaner.
    logger.debug(f"Processing search request for query: {user_query}")
    agent_reply_text = await process_search_request(task_request.id, user_query)
    logger.debug(f"Got agent_reply_text from process_search_request: {agent_reply_text.body}")
    logger.debug(f"Returning results.")
    return agent_reply_text

def get_agent_card(task_request: TaskRequest):
    logger.debug("Getting Agent Card.")
    agent_id = task_request.agent_id
    if not agent_id:
        raise HTTPException(status_code=422, detail="Missing 'agent_id' for get_agent_card.")
    if agent_id in AGENT_REGISTRY:
        logger.info(f"Get agent card request for {agent_id}")
        return AGENT_REGISTRY[agent_id]
    else:
        raise HTTPException(status_code=404, detail=f"Agent with agent_id '{agent_id}' not found.")

# This is decorated as an agent_tool so that LLM can call it when running.
def list_agent_cards():
    logger.info("List agent cards request received.")
    if (log_level_name == "DEBUG"):
        logger.debug("+++++++++++++++++++++++++++++++++++")
        for key, value in AGENT_REGISTRY.items():
            logger.debug(f"Registry Key: {key}: \nValue: {value}")
            logger.debug("+++++++++++++++++++++++++++++++++++")

    return list(AGENT_REGISTRY.values())

def parse_message_json(task_request: TaskRequest) -> dict:    
    try:
        return json.loads(task_request.message.parts[0].text)
    except (IndexError, AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing input JSON: ")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

def parse_message_text(task_request: TaskRequest) -> str:    
    try:
        return task_request.message.parts[0].text
    except (IndexError, AttributeError) as e:
        logger.error(f"Error accessing message content: {e}")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

def extract_json_from_response(text: str):
    # Try fenced JSON block first (```json\n...\n```)
    logger.debug("JSON Matcher attempt 1 starting.")
    match = re.search(r"```json\s+(.*?)```", text, re.DOTALL)
    if match:
        logger.debug("JSON Matcher attempt 1 worked.")
        logger.debug(f"Extracted JSON: {match.group(1)} ")
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}"  )
            pass  # fall back to next method
    else:
        logger.debug("Skipping to method 2.")

    # Fallback: find the first top-level JSON array
    logger.debug("JSON Matcher attempt 2 starting.")
    match = re.search(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    if match:
        logger.debug("JSON Matcher attempt 2 worked.")
        logger.debug(f"Extracted JSON: {match.group(1)} ")
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}"  )
            pass
    else:
        logger.debug("Skipping to method 3.")

    # Fallback: find a top-level JSON object (e.g., `{ "name": ... }`)
    logger.debug("JSON Matcher attempt 3 starting.")
    match = re.search(r"({\s*\".*?}\s*)", text, re.DOTALL)
    if match:
        logger.debug("JSON Matcher attempt 3 worked.")
        logger.debug(f"Extracted JSON: {match.group(1)} ")
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}"  )
            pass

    logger.debug("No JSON Matcher worked.")
    return json.loads(f'{"Output":"JSON Parser failed to parse the JSON from the response. Response: {json.dumps(text)}"}')

async def process_search_request(task_id: str, user_query: str):
    logger.debug("In process_search_request.")

    if not user_query or not isinstance(user_query, str):
        raise HTTPException(status_code=422, detail="Missing or invalid 'query' in message content.")

    # Set up session per request.
    user_id = f"user_{task_id}"  # simple mapping (or another logic)
    session_id = f"session_{task_id}"
    body = ""
    logger.debug(f"UserID: {user_id} / SessionID: {session_id} ")

    # Create the specific session where the conversation will happen
    logger.debug("Creating Request Session.")
    req_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )
    logger.debug(f"Session created: App='{APP_NAME}', User='{user_id}', Session='{session_id}'")

    # For debugging.
    logger.debug('#############################################')
    logger.debug(f"Session ID: {req_session.id} User ID: {req_session.user_id}")
    logger.debug(f"User Message: {user_query}")
    logger.debug('#############################################')

    # This requires nest_asyncio.apply() earlier to allow nested asyncio.run() inside Flask.
    agent_reply_text = await call_agent_async(query = user_query,
                          runner = app.runner,
                          user_id=req_session.user_id,
                          session_id=req_session.id)
 
    logger.debug("///////////////////////////////////////")
    logger.debug(f"Agent Reply Text:\n{agent_reply_text}")
    # Formulate the response in A2A Task format.
    # We'll return a Task object with the final state = 'completed' and the agent's message.

    logger.debug("Creating response task for A2A Caller.")
    json_results = extract_json_from_response(agent_reply_text)
    logger.debug("+++++++++++++++++++++++++++++++++++++")
    logger.debug(f"Got json_results from extraction: {json_results} ")
    logger.debug("+++++++++++++++++++++++++++++++++++++")

    logger.debug("Building response_task.")
    response_task = {
        "id": task_id,
        "status": {"state": "completed"},
        "messages": [
            {
                "role": "agent",                        # the agent's reply
                "parts": [{"text": json_results}]   # agent's message content as a TextPart
            }
        ]
        # We could also include an "artifacts" field if the agent returned files or other data.
    }
    logger.debug("Returning response task for A2A Caller.")
    logger.debug("//////////////////////////////////////////")
    return JSONResponse(response_task)

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  logger.debug('****************************')
  logger.debug(f"In call_agent_async")
  logger.debug(f"User Query: {query}")
  logger.debug('****************************')

  user_query = "Find an agent in the registry that can do the following. " + query
  # Prepare the user's message in ADK format

  logger.debug(f"User query to pass to LLM: {user_query}")
  content = types.Content(role='user', parts=[types.Part(text=user_query)])

  # Default response
  final_response_text = "Agent did not produce a final response." 

  # Iterate through events to find the final answer.
  logger.debug(f"Calling with userid: {user_id}, sessionid: {session_id}")
  sessions = await session_service.list_sessions(app_name=APP_NAME,user_id=user_id)
  logger.debug(f"Current sessions: {sessions}")
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):

      # If you really want to see the nitty gritty
      if SHOW_AGENT_EVENTS:
        logger.debug(f" [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # This is the final message, which is what we care about.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text

          # Handle potential errors/escalations
          elif event.actions and event.actions.escalate: 
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"

          # Stop processing events once the final response is found
          break 

  logger.debug(f"Agent Response: {final_response_text}")
  return final_response_text

async def create_agent():
    logger.debug(f"******************************************")
    logger.debug(f"Creating Agent")
    global registry_agent, runner 
    try:
        registry_agent = Agent(
            name="registry_agent",
            model=LiteLlm(model=ACTIVE_MODEL),
            description="Registry agent that uses send_email send notifications.",
            instruction=initial_agent_instructions,
            tools=[list_agent_cards]
        )

        logger.debug(f"Agent '{registry_agent.name}' created using model '{ACTIVE_MODEL}'.")

        # Create a runner specific to this agent and its session service
        runner = Runner(
            agent=registry_agent,
            app_name=APP_NAME,       # Use the specific app name
            session_service=session_service# Use the specific session service
            )
        logger.debug(f"Runner created for agent '{runner.agent.name}'.")

        # Now store the running runner in the app config.
        app.runner = runner

    except Exception as e:
        logger.error(f"Could not create or run agent '{ACTIVE_MODEL}'. Check API Key and model name. Error: {e}")
    
    logger.debug(f"******************************************")
    return registry_agent, runner

def load_registry():
    logger.debug(f"Loading registry from {REGISTRY_FILE}.")
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"The file {REGISTRY_FILE} could not be found. No registry was loaded.")
    return {}

def save_registry(registry):
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.debug("Saving registry to the filesystem.")
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    temp_fd, temp_path = tempfile.mkstemp(dir=BASE_DIR)
    try:
        with os.fdopen(temp_fd, 'w') as tmp:
            json.dump(registry, tmp, indent=2)
        shutil.move(temp_path, REGISTRY_FILE)  # atomic replace
    except Exception as e:
        print(f"Failed to save registry: {e}")
        os.remove(temp_path)    

# ###################################################################
# Run the FastAPI app (A2A server) if this script is executed directly.
if __name__ == "__main__":
    print("Run with: uvicorn trusted_agent_registry:app --host 0.0.0.0 --port 6060")  # Fixed typo in filename to match current file name