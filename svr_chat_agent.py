# Import necessary libraries
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import json
import uuid
import requests

# Google Agent Imports
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types 
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

# Imports for being an Agent Host
from fastapi import FastAPI, Request, Header, HTTPException, Body
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from typing import Optional, List, Any, Dict, AsyncGenerator
from pydantic import BaseModel
import asyncio
import json
from datetime import datetime

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Load environment
env_path=os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
from shared_logger import SharedLogger
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

APP_NAME = "chat_agent"

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # e.g., "agent_1"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")

# --- Define Model Constants for easier use ---
# Note: Specific model names might change. Refer to LiteLLM/Provider documentation.
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_GPT_41 = "openai/gpt-4.1-mini"
MODEL_GPT_35T = "openai/gpt-3.5-turbo"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"

# Set the active model here once so it can be switched out on just this line.
ACTIVE_MODEL = MODEL_GPT_35T

# Define the Fetch Agent
# Use one of the model constants defined earlier
from google.adk.models.lite_llm import LiteLlm
AGENT_MODEL = ACTIVE_MODEL # Starting with OpenAI

chat_agent = None # Initialize to None
runner = None      # Initialize runner to None
session_service = InMemorySessionService() # Create a dedicated service

# Global event publisher for SSE streaming
event_publishers: Dict[str, List[asyncio.Queue]] = {}

# Context variable to pass task_id to tools
from contextvars import ContextVar
current_task_id: ContextVar[Optional[str]] = ContextVar('current_task_id', default=None)

class StreamEvent:
    def __init__(self, event_type: str, data: Dict[str, Any], task_id: str):
        self.event_type = event_type
        self.data = data
        self.task_id = task_id
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type,
            "data": {
                **self.data,
                "timestamp": self.timestamp,
                "task_id": self.task_id
            }
        }

async def publish_event(task_id: str, event_type: str, data: Dict[str, Any]):
    """Publish an event to all subscribers for a given task_id"""
    if task_id in event_publishers:
        event = StreamEvent(event_type, data, task_id)
        for queue in event_publishers[task_id]:
            try:
                await queue.put(event)
            except:
                # Queue might be closed, ignore
                pass

def add_event_subscriber(task_id: str) -> asyncio.Queue:
    """Add a new event subscriber for a task_id"""
    if task_id not in event_publishers:
        event_publishers[task_id] = []
    
    queue = asyncio.Queue()
    event_publishers[task_id].append(queue)
    return queue

def remove_event_subscriber(task_id: str, queue: asyncio.Queue):
    """Remove an event subscriber"""
    if task_id in event_publishers and queue in event_publishers[task_id]:
        event_publishers[task_id].remove(queue)
        if not event_publishers[task_id]:
            del event_publishers[task_id]

# -------------------------------------------------------------------
# Set up the FastAPI app and use
# modern FastAPI lifespan for startup/shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Register the agent in the registry with APP_NAME
    await register(APP_NAME)

    await create_agent()
    yield
    logger.info("Shutting down chat agent. Cleaning up resources...")
    exit_stack = getattr(app, "exit_stack", None)
    if exit_stack:
        try:
            await exit_stack.aclose()
        except ProcessLookupError:
            logger.warning("Subprocess already terminated; skipping cleanup.")

# Create a simple web server with FastAPI using the lifespan context manager
app = FastAPI(lifespan=lifespan)

AGENT_CARD = {
    "name": "ChatAgent",  
    "description": "A chat agent that takes input request from the user and dynamically enlists other agents to assist.", 
    "url": "http://localhost:6002",  # The base URL where this agent is hosted
    "version": "0.1",
    "capabilities": {
        "streaming": False,           # This agent doesn't support streaming responses
        "pushNotifications": False    # No push notifications in this simple example
    },
     "skills": [
        {
            "id": "query_registry",
            "name": "Query Register Agent",
            "description": "Queries the registry to create find agents it needs.",
            "tags": ["register", "registry", "agent"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": [
                "application/json",
            ]
        },
        {
            "id": "call_remote_agent",
            "name": "Call Agent",
            "description": "Calls another agent.",
            "tags": ["agent"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": [
                "application/json",
            ]
        }
    ]
}

# Set up pydantic model to get task request:
class Part(BaseModel):
    text: str

class Message(BaseModel):
    parts: List[Part]

class TaskRequest(BaseModel):
    id: str
    message: Message

# App endpoints
# Serve the Agent Card at the well-known URL.
@app.get("/.well-known/agent.json")
def get_agent_card():
    # Endpoint to provide this agent's metadata (Agent Card).
    logger.info("Request for agent card received.")
    return JSONResponse(AGENT_CARD)

# Process requests
@app.post("/tasks/send")
async def handle_task(
    task_request: TaskRequest = Body(...),
    x_api_key: Optional[str] = Header(None),
    request: Request = None,
    stream: bool = False
):
    # Endpoint to receive and process task requests.

    logger.debug("*************************************")
    logger.info("Chat request received.")
    logger.debug(f'Task request JSON {task_request}')
    logger.debug("*************************************")

    # Extract the task ID and the user's message text from the request.
    #user_message = task_request.message
    user_message = parse_message_text(task_request)
    logger.debug(f"User message: {user_message}")
    task_id = task_request.id

    # Set up or recall session.
    user_id = f"user_{task_id}"  # simple mapping (or another logic)
    session_id = f"session_{task_id}"
    logger.debug(f"UserID: {user_id} / SessionID: {session_id} ")
    req_session = await get_session(user_id, session_id)

    # ############################################
    # Try to get the session if it exists, 
    # otherwise create a new one.
    # ############################################
    
    # Check if streaming was requested
    if stream:
        # Generate a unique stream task ID for this request
        stream_task_id = str(uuid.uuid4())
        stream_url = f"{request.base_url}tasks/stream/{stream_task_id}"
        
        # Start processing in background
        async def background_processing():
            try:
                await process_user_message(user_message, req_session, stream_task_id)
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                await publish_event(stream_task_id, "error", {
                    "message": f"Error processing request: {str(e)}"
                })
                await publish_event(stream_task_id, "stream_end", {
                    "message": "Stream completed with error"
                })
        
        asyncio.create_task(background_processing())
        
        return JSONResponse({
            "id": task_id,
            "status": {"state": "processing"},
            "stream_url": stream_url,
            "stream_task_id": stream_task_id,
            "message": "Task is being processed. Connect to stream_url for real-time updates."
        })

    # Compose the prompt for the LLM based on new instructions
    logger.debug(f"Session ID: {req_session.id} User ID: {req_session.user_id}")
    logger.debug(f"User Message: {user_message}")

    # This requires nest_asyncio.apply() earlier to allow nested asyncio.run() inside Flask.
    agent_reply_text = await process_user_message(user_message, req_session, task_id)
    logger.debug("+++++++++++++++++++++++++++++++")
    logger.debug(f"The handle_task function received agent reply text from process_user_message: {agent_reply_text}")
    logger.debug("+++++++++++++++++++++++++++++++")
 
    # Formulate the response in A2A Task format.
    # We'll return a Task object with the final state = 'completed' and the agent's message.
    logger.debug("Creating the response_task.")
    response_task = {
        "id": task_id,
        "status": {"state": "completed"},
        "messages": [
            user_message,
            {
                "role": "agent",                        # the agent's reply
                "parts": [{"text": agent_reply_text}]   # agent's message content as a TextPart
            }
        ]
        # We could also include an "artifacts" field if the agent returned files or other data.
    }
    logger.debug(f"Returning the response: {response_task}")
    return JSONResponse(response_task)

# SSE streaming endpoint
@app.get("/tasks/stream/{task_id}")
async def stream_task_events(task_id: str):
    """Stream real-time events for a specific task"""
    
    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        queue = add_event_subscriber(task_id)
        
        try:
            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({
                    "message": "Connected to task stream",
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat()
                })
            }
            
            while True:
                try:
                    # Wait for events with timeout
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event.event_type,
                        "data": json.dumps(event.data)
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({
                            "message": "Connection alive",
                            "timestamp": datetime.now().isoformat()
                        })
                    }
                except Exception as e:
                    logger.error(f"Error in event stream: {e}")
                    break
                    
        finally:
            remove_event_subscriber(task_id, queue)
    
    return EventSourceResponse(event_generator())

# -------------------------------------------------------------------

def before_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"BEFORE_AGENT_CALLBACK FOR: {callback_context.agent_name}")

def after_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"AFTER_AGENT_CALLBACK FOR: {callback_context.agent_name}")

def before_model_cb(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    logger.info(f"BEFORE_MODEL_CALLBACK FOR: {callback_context.agent_name}")

def after_model_cb(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    logger.info(f"AFTER_MODEL_CALLBACK FOR: {callback_context.agent_name}")
async def get_session(user_id: str, session_id: str):
    try:
        logger.debug("Getting Request Session.")
        req_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        if (req_session):
            logger.debug(f"Session restored: App='{APP_NAME}', User='{user_id}', Session='{session_id}'")
            return req_session
        else: 
            raise ValueError("Could not restore session. Creating a new one.")
    except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.debug("Creating a new session.")
            req_session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id
            )
            logger.debug(f"Session created: App='{APP_NAME}', User='{user_id}', Session='{session_id}'")
            return req_session

def parse_message_text(task_request: TaskRequest) -> str:    
    try:
        return task_request.message.parts[0].text
    except (IndexError, AttributeError) as e:
        logger.error(f"Error accessing message content: {e}")
        raise HTTPException(status_code=422, detail="Invalid or malformed message format.")

async def process_user_message(message: str, req_session, task_id: str = None) -> str:
    try:
           logger.debug("In process user message.")
           query = message  # Use the newly composed prompt directly
           logger.debug(f"User to run: {query} ")
           
           if task_id:
               await publish_event(task_id, "processing_start", {
                   "message": "Starting to process user message",
                   "query": query
               })
           
           logger.debug("Making request to call_agent_async function.")
           response = await call_agent_async(query = query,
                          runner = app.runner,
                          user_id=req_session.user_id,
                          session_id=req_session.id,
                          task_id=task_id)
           logger.debug(f'*******************************************')
           logger.debug(f"The call_agent_async function returned the response {response}. \nReturning that.")        
           logger.debug(f'*******************************************')
           return response
    except Exception as e:
            logger.error(f"An error occurred: {e}")

async def call_agent_async(query: str, runner, user_id, session_id, task_id: str = None):
  """Sends a query to the agent and prints the final response."""
  logger.debug(f'*******************************************')
  logger.debug(f"In call_agent_async")
  logger.debug(f"\n>>> User Query: {query}")
  logger.debug(f'*******************************************')

  # Set the task_id in context for tools to access
  if task_id:
      current_task_id.set(task_id)

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  if task_id:
      await publish_event(task_id, "agent_thinking", {
          "message": "Agent is processing your request...",
          "query": query
      })

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  logger.debug(f"Calling Running with userid: {user_id}, sessionid: {session_id}")
  sessions = session_service.list_sessions(app_name=APP_NAME,user_id=user_id)
  logger.debug(f"Current sessions: {sessions}")
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      logger.debug(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Stream ADK events to client
      if task_id:
          event_data = {
              "author": event.author,
              "event_type": type(event).__name__,
              "is_final": event.is_final_response()
          }
          
          # Add content if available
          if event.content and event.content.parts:
              event_data["content"] = event.content.parts[0].text if event.content.parts[0].text else ""
          
          # Check if this is a function call
          if hasattr(event, 'actions') and event.actions:
              if hasattr(event.actions, 'function_calls') and event.actions.function_calls:
                  func_call = event.actions.function_calls[0]
                  event_data["function_call"] = {
                      "name": func_call.name,
                      "arguments": func_call.args if hasattr(func_call, 'args') else {}
                  }
                  
                  # Publish specific events for function calls
                  if func_call.name == "query_registry":
                      await publish_event(task_id, "registry_query", {
                          "message": "Searching for agents to help with your request...",
                          "function": func_call.name
                      })
                  elif func_call.name == "call_remote_agent":
                      await publish_event(task_id, "agent_call", {
                          "message": "Calling another agent to assist...",
                          "function": func_call.name
                      })
          
          await publish_event(task_id, "agent_event", event_data)

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  if task_id:
      await publish_event(task_id, "final_response", {
          "message": "Task completed",
          "response": final_response_text
      })
      # Signal end of stream
      await publish_event(task_id, "stream_end", {
          "message": "Stream completed"
      })

  logger.debug(f"<<< Agent Response: {final_response_text}")
  return final_response_text

async def register(appname: str):
    logger.debug(f"Registering App: {appname}")

    config = load_config()
    agent_id = config.get("agent_id","")
    api_key = config.get("api_key","")

    # If we can't get the registration URL, exit.
    reg_url = config.get("reg_url")
    if not reg_url: 
        logger.error(f"Could not load the registration URL from 'reg_url' key in the configuration file.")
        exit(1)

    request_id = str(uuid.uuid4())

    # Construct the payload
    payload = {
        "id": request_id,
        "agent_id": agent_id,
        "tool_name": "register_agent",
        "message": {
            "parts": [
                {
                    "text": json.dumps(AGENT_CARD)
                }
            ]
        }
    }

    # Print the payload (pretty-printed)
    logger.debug(json.dumps(payload, indent=2))

    # Send the request
    response = requests.post(
        reg_url,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key
        },
        data=json.dumps(payload)
    )

    # Parse the JSON response
    if response.status_code == 200:
        try:
            response_data = response.json()
            agent_id = response_data.get("id")
            logger.info(f"Agent registered with ID: {agent_id}")

            # Save it to your config (if needed)
            config["agent_id"] = agent_id
            save_config(config)

        except ValueError:
            logger.error("Failed to parse JSON response:", response.text)
    else:
        logger.error(f"Registration failed with status {response.status_code}: {response.text}")

def load_config():
    if os.path.exists(CONFIG_FILE):
        logger.debug(f"Loading config file: {CONFIG_FILE}")
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        logger.error(f"Could not find config file {CONFIG_FILE}! Exiting...")
        exit(1)

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

async def query_registry(user_req: str) -> dict:
    """
    The query_registry tool searches the trusted agents registry on the local network by 
    calling the registry agent and passing in a string of text describing the functionality 
    being sought from an agent on the network. For example, if the calling agent needs to 
    send a text message and does not have that capability natively, then this tool can be
    called to find a trusted, registered agent on the network to do that. 
    
    Parameters:
        This tool takes a string input as 'user_req' and that should contain the type of 
        functionality being sought. For example, expected input would be something like 
        'send a text message'. Given that, the registry agent will search for all registered 
        agents that it thinks can send a text message. 

    Returns:
        A dictionary listing the agents the registry believes can solve the request, ordered
        by the most likely to least likely. Each element contains a confidence score and the
        agent card of the agent. The agent card contains all the information the 'call_remote_agent'
        tool needs to call the helper agent.
    """
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.debug(f"Querying Registry with: {user_req}")
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    task_id = current_task_id.get()
    if task_id:
        await publish_event(task_id, "registry_query_start", {
            "message": f"Searching registry for: {user_req}",
            "query": user_req
        })

    config = load_config()
    agent_id = config.get("agent_id","")
    api_key = config.get("api_key","")

    # If we can't get the registration URL, exit.
    reg_url = config.get("reg_url")
    if not reg_url: 
        logger.error(f"Could not load the registration URL from 'reg_url' key in the configuration file.")
        exit(1)

    request_id = str(uuid.uuid4())

    # Construct the payload
    payload = {
        "id": request_id,
        "agent_id": agent_id,
        "tool_name": "search_agents",
        "message": {
            "parts": [
                {
                    "text": user_req
                }
            ]
        }
    }

    # Print the payload (pretty-printed)
    logger.debug(json.dumps(payload, indent=2))

    # Send the request
    logger.info("Querying the registry agent to find someone to help me.")
    response = requests.post(
        reg_url,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key
        },
        data=json.dumps(payload)
    )

    # Parse the JSON response
    logger.info("Received response from the registry.")
    if response.status_code == 200:
        try:
            result = response.json()
            if task_id:
                # The registry response structure is different - need to parse the actual response
                try:
                    # Extract the registry response from the nested structure
                    messages = result.get('messages', [])
                    if messages:
                        agent_response = messages[-1]
                        response_parts = agent_response.get('parts', [])
                        if response_parts and 'text' in response_parts[0]:
                            response_text = response_parts[0]['text']
                            # Try to parse as JSON to get agent count
                            try:
                                registry_data = json.loads(response_text)
                                agent_count = len(registry_data.get('agents', []))
                            except:
                                # If not JSON, count based on presence of agents in text
                                agent_count = 1 if 'agent' in response_text.lower() else 0
                        else:
                            agent_count = 0
                    else:
                        agent_count = 0
                        
                    await publish_event(task_id, "registry_response", {
                        "message": f"Registry found {agent_count} matching agents",
                        "agent_count": agent_count
                    })
                except Exception as e:
                    logger.debug(f"Error parsing registry response: {e}")
                    await publish_event(task_id, "registry_response", {
                        "message": "Registry responded",
                        "agent_count": "unknown"
                    })
            return result
        except ValueError:
            logger.error("Failed to parse JSON response:", response.text)
    else:
        logger.error(f"Query Registry failed with status {response.status_code}: {response.text}")
        if task_id:
            await publish_event(task_id, "registry_error", {
                "message": f"Registry query failed: {response.status_code}",
                "error": response.text
            })
        return {
            "error": True,
            "message": f"Query Registry failed with status {response.status_code}: {response.text}"
        } 

async def call_remote_agent(query: str, card: dict, **kwargs) -> dict:
    """
    The call_remote_agent tool creates a call to a trusted agent on the network using the Google
    A2A protocol. This tool will read the agent card passed in to it, along with the 
    query string, and will structure a request to the agent and await a response. It then
    returns the response to the server chat agent.

    Parameters:
        This tool takes two parameters:
        - query, as a string, containing the specific request to make to the server.
        - card, as a dictonary, containing the agent card of the agent to call.

    Returns:
       A dictionary object with two elements: 
       - 'error' containing a boolean value to indicate whether the resulted in an error or not.
       - 'message' containing a string that is the response from the remote agent. That message be a string, json text, or other formats.
    """
    logger.info(f"Calling agent {card.get('name')} from {card.get('url')}")
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.debug("Calling agent to assist with query:")
    logger.debug(f"Query: {query}")
    logger.debug(f"Agent Card: {card}")
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    task_id = current_task_id.get()
    if task_id:
        await publish_event(task_id, "agent_call_start", {
            "message": f"Calling {card.get('name')} agent",
            "agent_name": card.get('name'),
            "agent_url": card.get('url'),
            "query": query
        })

    # Let's parse the card and make the call.
    call_task_id = str(uuid.uuid4())  # generate a random unique task ID
    task_payload = {
        "id": call_task_id,
        "message": {
            "role": "user",
            "parts": [
                {"text": query}
            ]
        }
    }
    logger.info(f"Sending task {call_task_id} to agent with message: '{query}'")

    tasks_send_url = f"{card.get('url')}/tasks/send"
    result = requests.post(tasks_send_url, json=task_payload)
    logger.info(f"Received a response from {card.get('name')}")
    if result.status_code != 200:
        logger.error(f"Request to agent failed:  {result.status_code}, {result.text}")
        if task_id:
            await publish_event(task_id, "agent_call_error", {
                "message": f"Failed to call {card.get('name')} agent",
                "error": f"{result.status_code}: {result.text}"
            })
        raise RuntimeError(f"Task request failed: {result.status_code}, {result.text}")
    task_response = result.json()

    if task_response.get("status", {}).get("state") == "completed":
        messages = task_response.get("messages", [])
        if messages:
            agent_message = messages[-1]
            agent_reply_text = ""
            for part in agent_message.get("parts", []):
                if "text" in part:
                    agent_reply_text += part["text"]
            logger.debug(f"Agent's reply: {agent_reply_text}")
            
            if task_id:
                await publish_event(task_id, "agent_call_response", {
                    "message": f"Received response from {card.get('name')} agent",
                    "agent_name": card.get('name'),
                    "response": agent_reply_text[:100] + "..." if len(agent_reply_text) > 100 else agent_reply_text
                })
            
            return {
                "error": False,
                "message": agent_reply_text
            } 
        else:
            logger.error("No messages in response!")
            if task_id:
                await publish_event(task_id, "agent_call_error", {
                    "message": f"No response from {card.get('name')} agent",
                    "error": "No messages in response"
                })
            return {
                "error": True,
                "message": task_response.get("status")
            } 
    else:
        logger.error("Task did not complete. Status:", task_response.get("status"))
        if task_id:
            await publish_event(task_id, "agent_call_error", {
                "message": f"Task failed for {card.get('name')} agent",
                "error": str(task_response.get("status"))
            })
        return {
            "error": True,
            "message": f"An error occurred: {task_response.get('status')}"
        } 
    

async def create_agent():
    logger.debug(f"******************************************")
    logger.debug(f"Creating Agent")
    logger.debug(f"******************************************")
    global chat_agent, runner  # Ensure modifying the global variables
    try:
        # Define tools (they will access task_id from context variable)
        tools = [query_registry, call_remote_agent]
        agent_prompt=f"""
        You are a client agent powered by {ACTIVE_MODEL}. Your primary responsibility is to respond to user requests to the best of your ability. 
        You operate within a multi-agent environment where other trusted agents are available to assist with specific capabilities that you may lack.

        Each agent in the network registers itself with the Registry Agent using an AGENT_CARD that describes the agent and its skills. 
        If a user request requires functionality you do not have (e.g., sending a text message), you must:
        1. Use your `query_registry` tool to contact the Registry Agent and ask if any agents can fulfill the needed capability.
        2. Review the returned list of agents (ranked by confidence) and determine if any are capable of performing the task.
        3. If one is suitable, use your `contact_agent` tool to send the subtask and the target agent’s AGENT_CARD.
        4. Wait for the response, then incorporate the results into your final response to the user.
        5. Inform the user that assistance was provided by another agent, including the name(s) from the AGENT_CARD(s).

        Example: If the user asks you to generate a poem and text it to 555-555-5555:
        - You generate the poem yourself.
        - Then query the registry to find an agent that can send text messages.
        - If one is found, you send the request to that agent with the poem.
        - After receiving the response, report the outcome to the user.

        You may need to split a user request into multiple subtasks for multiple agents. 
        Always attempt to fulfill as much of the request as you can yourself, and transparently communicate when and how 
        other agents were used.

        Be collaborative, capable, and clear in your responses.
        
        SPECIAL INSTRUCTIONS FOR EMAIL REQUESTS:
        When the user asks you to send an email, you must carefully extract ALL the email information from their request:
        - recipient email address (to)
        - subject line  
        - body content
        
        If the user provides incomplete information (e.g., missing body content), do not send an incomplete email request.
        Instead, ask the user to provide the missing information first.
        
        When calling the EmailAgent, structure your query as JSON like this:
        {{"to":"recipient@email.com","subject":"Subject Line","body":"Complete message body content"}}
        
        CRITICAL: Never send an email request with an empty body field. Always ensure you have complete information before calling the EmailAgent.
        
        MOST IMPORTANTLY:
        - THINK DEEPLY - SPEED TO REPLY IS NOT AS IMPORTANT AS CORRECT RESPONSE.
        - ALWAYS ATTEMPT TO FIND AN AGENT IN THE AGENT REGISTRY USING THE 'query_registry' TOOL BEFORE RESPONDING TO A USER THAT YOU ARE
        UNABLE TO HANDLE THE REQUEST OR FIND AN AGENT TO HELP.
        - IF YOU STILL ARE NOT ABLE TO FIND AN AGENT OR SERVICE THE REQUEST AFTER USING YOUR AVAILABLE TOOLS, THEN EXPLAIN WHY TO THE USER.

        """
        chat_agent = LlmAgent(
            name="chat_agent",
            # Key change: Wrap the LiteLLM model identifier
            model=LiteLlm(model=ACTIVE_MODEL, stream=True),
            description="An agent that satisfies user chat requests, and dynamically enlists other agents on the network to assist as needed.",
            instruction=agent_prompt,
            tools=tools,
            before_agent_callback=before_agent_cb,
            after_agent_callback=after_agent_cb,
            before_model_callback=before_model_cb,
            after_model_callback=after_model_cb
        )
        logger.debug(f"Agent '{chat_agent.name}' created using model '{ACTIVE_MODEL}'.")

        # Create a runner specific to this agent and its session service
        runner = Runner(
            agent=chat_agent,
            app_name=APP_NAME,       # Use the specific app name
            session_service=session_service# Use the specific session service
            )
        logger.debug(f"Runner created for agent '{runner.agent.name}'.")

        # Now store the running runner in the app config.
        app.runner = runner
        app.exit_stack = None  # No MCP exit stack needed

    except Exception as e:
        logger.error(f"Could not create or run agent '{ACTIVE_MODEL}'. Check API Key and model name. Error: {e}")
    
    return chat_agent, runner

# ###################################################################
# Run the FastAPI app (A2A server) if this script is executed directly.
if __name__ == "__main__":
    print("Run with: uvicorn svr_chat_agent:app --host 0.0.0.0 --port 6002")  # Fixed typo in filename to match current file name