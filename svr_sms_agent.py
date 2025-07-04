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
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types 

# Imports for being an Agent Host
from fastapi import FastAPI, Request, Header, HTTPException, Body
from fastapi.responses import JSONResponse

from typing import Optional, List, Any
from pydantic import BaseModel

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
logger.debug(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

APP_NAME = "sms_agent"

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # e.g., "agent_1"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")

# Set the SMS URL and key from the .env file.
sms_url = os.getenv("SMS_URL")
sms_key = os.getenv("SMS_KEY")

# --- Define Model Constants for easier use ---
# Note: Specific model names might change. Refer to LiteLLM/Provider documentation.
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_GPT_41 = "openai/gpt-4.1-mini"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"

# Set the active model here once so it can be switched out on just this line.
ACTIVE_MODEL = MODEL_GPT_41

# Define the Fetch Agent
# Use one of the model constants defined earlier
from google.adk.models.lite_llm import LiteLlm
AGENT_MODEL = ACTIVE_MODEL # Starting with OpenAI

sms_agent = None # Initialize to None
runner = None      # Initialize runner to None
session_service = InMemorySessionService() # Create a dedicated service

# -------------------------------------------------------------------
# Set up the FastAPI app and use
# modern FastAPI lifespan for startup/shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Register the agent in the registry with APP_NAME
    await register(APP_NAME)

    await create_agent()
    yield
    logger.debug("Shutting down sms agent. Cleaning up resources...")
    exit_stack = getattr(app, "exit_stack", None)
    if exit_stack:
        try:
            await exit_stack.aclose()
        except ProcessLookupError:
            logger.warning("Subprocess already terminated; skipping cleanup.")

# Create a simple web server with FastAPI using the lifespan context manager
app = FastAPI(lifespan=lifespan)

AGENT_CARD = {
    "name": "SMSAgent",  
    "description": "A SMS agent that takes a phone number and message content and uses it's send_sms skill to send SMS text messages to the phone number.", 
    "url": "http://localhost:6003",  # The base URL where this agent is hosted
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
        },
        {
            "id": "send_sms",
            "name": "Send SMS",
            "description": "Receives a message and phone number and Sends an SMS text to provided phone number with the provided content.",
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
    logger.debug("Request for agent card received.")
    return JSONResponse(AGENT_CARD)

# Process requests
@app.post("/tasks/send")
async def handle_task(
    task_request: TaskRequest = Body(...),
    x_api_key: Optional[str] = Header(None),
    request: Request = None
):
    # Endpoint to receive and process task requests.

    logger.debug("*************************************")
    logger.debug("SMS request received.")
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
    

    # Compose the prompt for the LLM based on new instructions
    logger.debug(f"Session ID: {req_session.id} User ID: {req_session.user_id}")
    logger.debug(f"User Message: {user_message}")

    # This requires nest_asyncio.apply() earlier to allow nested asyncio.run() inside Flask.
    agent_reply_text = await process_user_message(user_message, req_session)
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
# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
def before_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"BEFORE_AGENT_CALLBACK FOR: {callback_context.agent_name}")

def after_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"AFTER_AGENT_CALLBACK FOR: {callback_context.agent_name}")

def before_model_cb(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    logger.info(f"BEFORE_MODEL_CALLBACK FOR: {callback_context.agent_name}")
    logger.info(f"Request contents: {llm_request.contents}")

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

async def process_user_message(message: str, req_session) -> str:
    try:
           logger.debug("In process user message.")
           query = message  # Use the newly composed prompt directly
           logger.debug(f"User to run: {query} ")
           logger.debug("Making request to call_agent_async function.")
           response = await call_agent_async(query = query,
                          runner = app.runner,
                          user_id=req_session.user_id,
                          session_id=req_session.id)
           logger.debug(f'*******************************************')
           logger.debug(f"The call_agent_async function returned the response {response}. \nReturning that.")        
           logger.debug(f'*******************************************')
           return response
    except Exception as e:
            logger.error(f"An error occurred: {e}")

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  logger.debug(f'*******************************************')
  logger.debug(f"In call_agent_async")
  logger.debug(f"\n>>> User Query: {query}")
  logger.debug(f'*******************************************')

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  logger.debug(f"Calling Runner with userid: {user_id}, sessionid: {session_id}")
  sessions = session_service.list_sessions(app_name=APP_NAME,user_id=user_id)
  logger.debug(f"Current sessions: {sessions}")
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      logger.debug(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

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
            logger.debug(f"Agent registered with ID: {agent_id}")

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

async def send_sms(msg: str, phone: str) -> dict:
    """
    The send_sms tool will send take the input from the Agent and call a third party
    SMS service to send an SMS text message. The service URL and API key are loaded from the
    environment as sms_url and sms_key.

     Parameters:
        This tool takes 2 parameters:
            msg: the message to send via SMS service.
            phone: the phone number to send the SMS to. The phone number should be 10 digits only.

     Returns:
        A dictionary representing the JSON returned by the SMS service. 
    """

    # Send the request
    logger.debug("Request to SMS server:")
    logger.debug(f"URL: {sms_url}")
    logger.debug(f"Phone: {phone}")
    logger.debug(f"Msg: {msg}")
    logger.debug(f"Key: {sms_key}")

    response = requests.post(sms_url, {
        'phone': phone,
        'message': msg,
        'key': sms_key,
    })
    try:
        logger.debug(f"Response from the SMS server: {response.json()}")
    except Exception as e:
        logger.debug(f"Error getting response.json {e}")

    # Parse the JSON response
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            logger.error("Failed to parse JSON response:", response.text)
    else:
        logger.error(f"Send SMS failed with status {response.status_code}: {response.text}")
        return {
            "error": True,
            "message": f"Send SMS failed with status {response.status_code}: {response.text}"
        } 

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
            return response.json()
        except ValueError:
            logger.error("Failed to parse JSON response:", response.text)
    else:
        logger.error(f"Query Registry failed with status {response.status_code}: {response.text}")
        return {
            "error": True,
            "message": f"Query Registry failed with status {response.status_code}: {response.text}"
        } 

async def call_remote_agent(query: str, card: dict) -> dict:
    """
    The call_remote_agent tool creates a call to a trusted agent on the network using the Google
    A2A protocol. This tool will read the agent card passed in to it, along with the 
    query string, and will structure a request to the agent and await a response. It then
    returns the response to the server SMS agent.

    Parameters:
        This tool takes two parameters:
        - query, as a string, containing the specific request to make to the server.
        - card, as a dictonary, containing the agent card of the agent to call.

    Returns:
       A dictionary object with two elements: 
       - 'error' containing a boolean value to indicate whether the resulted in an error or not.
       - 'message' containing a string that is the response from the remote agent. That message be a string, json text, or other formats.
    """
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.debug("Calling agent to assist with query:")
    logger.debug(f"Query: {query}")
    logger.debug(f"Agent Card: {card}")
    logger.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # Let's parse the card and make the call.
    logger.debug(f"Calling agent {card.get('name')} from {card.get('url')}")
    task_id = str(uuid.uuid4())  # generate a random unique task ID
    task_payload = {
        "id": task_id,
        "message": {
            "role": "user",
            "parts": [
                {"text": query}
            ]
        }
    }
    logger.debug(f"Sending task {task_id} to agent with message: '{query}'")

    tasks_send_url = f"{card.get('url')}/tasks/send"
    result = requests.post(tasks_send_url, json=task_payload)
    if result.status_code != 200:
        logger.error(f"Request to agent failed:  {result.status_code}, {result.text}")
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
            logger.debug("Agent's reply:", agent_reply_text)
            return {
                "error": False,
                "message": agent_reply_text
            } 
        else:
            logger.error("No messages in response!")
            return {
                "error": True,
                "message": task_response.get("status")
            } 
    else:
        logger.error("Task did not complete. Status:", task_response.get("status"))
        return {
            "error": True,
            "message": f"An error occurred: {task_response.get('status')}"
        } 
    

async def create_agent():
    logger.debug(f"******************************************")
    logger.debug(f"Creating Agent")
    logger.debug(f"******************************************")
    global sms_agent, runner  # Ensure modifying the global variables
    try:
        # Define tools as proxy functions
        tools = [send_sms, query_registry, call_remote_agent]
        agent_prompt=f"""
        You are an agent powered by {ACTIVE_MODEL}. Your primary responsibility is to send SMS messages by using your send_sms tool. 
        You operate within a multi-agent environment where other trusted agents are available and they will be connecting to you to
        send SMS messages.

        The user request will be a string of text that should specify the phone number to send the SMS message to, and the body of the
        SMS message to send. However, this format is unstructured, so you will need to use your reasoning skills and NLP to read the 
        user request, understand the context, and extract the phone number and SMS message content from the user request.

        The phone number will likely be in E.164 format, with a + prefix, country code, and subscriber number. Howver, you could also
        receive it as a 10 digit number, in which case you can pass that in directly. 
        
        IT IS IMPORTANT TO NOTE: If the number starts with
        a plus sign (+), it must be transformed to it's url encoded value of %2B and replaced with that value in the number. The SMS system  
        is capable of handling other text characters and spaces. Only the + sign must be url_encoded before being sent to the send_sms tool.

        If the size of the message text is more than 160 characters, the SMS service will break it into multiple messages.
        
        An example user request could be 'Text Hey Mike how are you to 555-555-5555'. In this case, the msg would be 'Hey Mike how are you' and
        the sms_phone would be '555-555-5555'

        Another example uxser request could be 'Send an SMS to +1 (555) 555-555 saying Time for dinner'. In this case, the msg would be 'Time for dinner'
        and the sms_phone would be '%2B1 (555) 555-5555'

        If the user provides incomplete information (e.g., missing mesg content), do not attempt to send the message.
        Instead, ask the user to provide the missing information first.
        
        MOST IMPORTANTLY:
        - THINK DEEPLY - SPEED TO REPLY IS NOT AS IMPORTANT AS CORRECT RESPONSE.

        """
        sms_agent = LlmAgent(
            name="sms_agent",
            # Key change: Wrap the LiteLLM model identifier
            model=LiteLlm(model=ACTIVE_MODEL),
            description="An agent that satisfies user SMS requests, and dynamically enlists other agents on the network to assist as needed.",
            instruction=agent_prompt,
            tools=tools,
            before_agent_callback=before_agent_cb,
            after_agent_callback=after_agent_cb,
            before_model_callback=before_model_cb,
            after_model_callback=after_model_cb
        )
        logger.debug(f"Agent '{sms_agent.name}' created using model '{ACTIVE_MODEL}'.")

        # Create a runner specific to this agent and its session service
        runner = Runner(
            agent=sms_agent,
            app_name=APP_NAME,       # Use the specific app name
            session_service=session_service# Use the specific session service
            )
        logger.debug(f"Runner created for agent '{runner.agent.name}'.")

        # Now store the running runner in the app config.
        app.runner = runner
        app.exit_stack = None  # No MCP exit stack needed

    except Exception as e:
        logger.error(f"Could not create or run agent '{ACTIVE_MODEL}'. Check API Key and model name. Error: {e}")
    
    return sms_agent, runner

# ###################################################################
# Run the FastAPI app (A2A server) if this script is executed directly.
if __name__ == "__main__":
    print("Run with: uvicorn svr_sms_agent:app --host 0.0.0.0 --port 6003")  # Fixed typo in filename to match current file name