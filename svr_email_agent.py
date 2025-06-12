# Import necessary libraries
import os
from dotenv import load_dotenv, set_key
from contextlib import asynccontextmanager
import json
from datetime import timedelta, datetime, timezone
import uuid
import requests

# Google Agent Imports
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types 
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Imports for being an Agent Host
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Load environment
env_path=os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Set the GSuite MCP location and token path to load the Google token information from token.json
# NOTE: this code uses a local MCP server to show the process of starting and stopping a local
# MCP server during the agent creation and server shutdown events.  In a production environment, 
# consider a remotely hosted MCP server such as https://smithery.ai/server/google-workspace-server
MCP_GSUITE_PATH = os.getenv("MCP_GSUITE_PATH")
MCP_GSUITE_RUN_FILE = os.getenv("MCP_GSUITE_RUN_FILE")
MCP_SRVR = f'{MCP_GSUITE_PATH}/{MCP_GSUITE_RUN_FILE}'
MCP_ENV = {}

TOKEN_PATH = os.path.abspath(os.path.join(os.path.dirname(MCP_GSUITE_PATH), "token.json"))
CREDENTIALS_PATH = os.path.abspath(os.path.join(os.path.dirname(MCP_GSUITE_PATH), "credentials.json"))

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
from shared_logger import SharedLogger
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

# Create a subclass of MCPToolset so we can override cleanup and
# catch ProcessLookupError exceptions.
class SafeMCPToolset(MCPToolset):
    def __init__(self, connection_params, **kwargs):
        try:
            super().__init__(connection_params=connection_params, **kwargs)
            self._exit_stack = None
        except Exception as e:
            logger.error(f"Error initializing MCPToolset: {e}")
            raise
    
    def set_exit_stack(self, exit_stack):
        """Store reference to exit stack for cleanup"""
        self._exit_stack = exit_stack
    
    async def cleanup(self):
        """Custom cleanup method that handles ProcessLookupError"""
        try:
            if self._exit_stack:
                await self._exit_stack.aclose()
        except ProcessLookupError:
            logger.warning("Subprocess already terminated inside MCPToolset; skipping cleanup.")

APP_NAME = "email_agent"

# --- Define Model Constants for easier use ---
# Note: Specific model names might change. Refer to LiteLLM/Provider documentation.
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"

# Set the active model here once so it can be switched out on just this line.
ACTIVE_MODEL = MODEL_GPT_4O

# Define the Email Agent
# Use one of the model constants defined earlier
from google.adk.models.lite_llm import LiteLlm
AGENT_MODEL = ACTIVE_MODEL # Starting with OpenAI

email_agent = None # Initialize to None
runner = None      # Initialize runner to None
session_service = InMemorySessionService() # Create a dedicated service
send_email_instruction = ""

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # e.g., "agent_1"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")

# --------------------------------------
# Helper functions needed for  startup.
# --------------------------------------
def load_and_validate_token_at_startup():
    """
    Reads token.json, checks if token is expired.
    If expired, displays instructions and exits.
    If valid, updates .env with the refresh_token and returns expiry.
    """
    logger.debug("App Preflight")
    logger.debug(f"Log level is: {log_level_name}")

    logger.debug("------------------------------")
    logger.debug("Loading data from token.json")
    logger.debug("------------------------------")

    if not os.path.exists(TOKEN_PATH):
        logger.error(f"Error: token.json not found. Please run `node get-refresh-token.js` from")
        logger.error(f"{MCP_GSUITE_PATH}, then copy token.json to {TOKEN_PATH}.")
        exit(1)
    
    if not os.path.exists(CREDENTIALS_PATH):
        logger.error(f"Error: credentials.json not found in {MCP_GSUITE_PATH}. Please generate your OAuth credentials from Google Cloud Console.")
        exit(1)

    try:
        with open(TOKEN_PATH, "r") as f:
            token_data = json.load(f)
        
        # Load credentials.json
        with open(CREDENTIALS_PATH, "r") as f:
            credentials_data = json.load(f)

        # Check Access Token expiry (from the token data)
        now = datetime.now(timezone.utc)
        logger.debug(f"It is now {now}")

        # Get Refresh Token expiry (based on file creation time)
        file_stats = os.stat(TOKEN_PATH)
        if hasattr(file_stats, 'st_birthtime'):  # macOS
            token_created_time = file_stats.st_birthtime
        else:  # Linux, Windows
            token_created_time = file_stats.st_mtime

        refresh_token_created = datetime.fromtimestamp(token_created_time, tz=timezone.utc)
        logger.debug(f"Refresh token created {refresh_token_created}")

        refresh_token_expires_in = token_data.get("refresh_token_expires_in", 0)  # Default ~7 days
        logger.debug(f"The refresh_token_expires_in from token is set to: {refresh_token_expires_in} seconds.")

        refresh_expiry = refresh_token_created + timedelta(seconds=refresh_token_expires_in)
        logger.debug(f"The refresh_expiry is: {refresh_expiry}.")

        refresh_delta = refresh_expiry - now
        logger.debug(f"The refresh_delta is: {refresh_delta}")

        refresh_days = refresh_delta.days
        refresh_hours, refresh_remainder = divmod(refresh_delta.seconds, 3600)
        refresh_minutes, refresh_seconds = divmod(refresh_remainder, 60)
        refresh_formatted = f"{refresh_days} day{'s' if refresh_days != 1 else ''}, {refresh_hours} hour{'s' if refresh_hours != 1 else ''}, {refresh_minutes} minute{'s' if refresh_minutes != 1 else ''}, and {refresh_seconds} second{'s' if refresh_seconds != 1 else ''}"

        logger.debug(f"Your refresh token will expire in {refresh_formatted}")

        # Check refresh token
        refresh_expiry_delta_seconds = refresh_delta.total_seconds()
        if refresh_expiry_delta_seconds <= 0:
            logger.error("Error: Your refresh token has expired.")
            logger.error(f"Please re-run `node get-refresh-token.js` from {MCP_GSUITE_PATH} and copy the file token.json to the {TOKEN_PATH}.")
            exit(1)
        else:
            logger.debug(f"Refresh token is valid. Expires at: {refresh_formatted}")

        # Update .env with refresh token
        global MCP_ENV
        MCP_ENV ["GOOGLE_REFRESH_TOKEN"]= token_data["refresh_token"]
        MCP_ENV ["GOOGLE_CLIENT_ID"]= credentials_data["web"]["client_id"]
        MCP_ENV ["GOOGLE_CLIENT_SECRET"]= credentials_data["web"]["client_secret"]

        return refresh_expiry.timestamp()

    except Exception as e:
        logger.error(f"Error: Failed to validate or load data from token.json or credentials.json: {e}")
        exit(1)

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

async def create_agent():
    logger.debug(f"******************************************")
    logger.debug(f"Creating Agent")
    global email_agent, runner  # Ensure modifying the global variables
    try:
        # First, launch the Google Workspace / GSuite MCP Server
        # Use the SafeMCPToolset subclass so we can deal with exceptions on cleanup
        logger.debug('Starting up gsuite MCP Server.')
        logger.debug(f"Google env: {MCP_ENV}")
        
        # Create the MCPToolset with the new API, filtering to only send_email tool
        try:
            mcp_toolset = SafeMCPToolset(
                connection_params=StdioServerParameters(
                    command='node', # Command to run the server
                    args=[MCP_SRVR],
                    env=MCP_ENV
                ),
                tool_filter=['send_email']  # Only use the send_email tool to avoid schema issues
            )
        except Exception as e:
            logger.error(f"Failed to create MCPToolset with send_email filter: {e}")
            # Try without filter to see all available tools
            logger.debug("Attempting to create MCPToolset without tool filter...")
            try:
                mcp_toolset = SafeMCPToolset(
                    connection_params=StdioServerParameters(
                        command='node', # Command to run the server
                        args=[MCP_SRVR],
                        env=MCP_ENV
                    )
                )
            except Exception as e2:
                logger.error(f"Failed to create MCPToolset without filter: {e2}")
                raise e2
        
        # In the new API, MCPToolset is used directly as a tool in the agent
        # We'll use the entire toolset but can optionally filter later
        send_email_input_schema = "send_email tool from Google Workspace MCP server"
        
        logger.debug(f"-----------------------------------")
        logger.debug(f"MCPToolset Type: {type(mcp_toolset)}")
        logger.debug(f"MCPToolset created successfully for Google Workspace")
        logger.debug(f"-----------------------------------")

        # Use the entire toolset - the agent will discover available tools automatically
        tool_list_only_send_email_tool = [mcp_toolset]

        #You have a tool called send_email with an input schema as follows: {send_email_input_schema} 
        global send_email_instruction
        send_email_instruction=f"""
        You are an email agent powered by GPT-4o. Use your available tools to send emails based on the data you receive. 
        Use the tool:
        {send_email_input_schema} 
        to send emails with.
        """
        logger.debug(f"Instructions type: {type(send_email_instruction)}")
        logger.debug(f"Instructions: {send_email_instruction}")
        email_agent = Agent(
            name="email_agent",
            model=LiteLlm(model=ACTIVE_MODEL),
            description="Email agent that uses send_email send notifications.",
            instruction=send_email_instruction,
            tools=tool_list_only_send_email_tool
        )

        logger.debug(f"Agent '{email_agent.name}' created using model '{ACTIVE_MODEL}'.")

        # Create a runner specific to this agent and its session service
        runner = Runner(
            agent=email_agent,
            app_name=APP_NAME,       # Use the specific app name
            session_service=session_service# Use the specific session service
            )
        logger.debug(f"Runner created for agent '{runner.agent.name}'.")

        # Now store the running runner and mcp_toolset in the app config.
        app.runner = runner
        app.mcp_toolset = mcp_toolset 

    except Exception as e:
        logger.error(f"Could not create or run agent '{ACTIVE_MODEL}'. Check API Key and model name. Error: {e}")
    
    logger.debug(f"******************************************")
    return email_agent, runner

def check_token_expiry_on_request(refresh_expiry_timestamp):
    """
    Validates that the current time is still before token expiry.
    If expired, notifies and exits the server.
    If valid, returns a human-readable time delta.
    """
    now = datetime.now(timezone.utc).timestamp()
    if refresh_expiry_timestamp < now:
        logger.error("Error: Token has expired during request handling.")
        logger.error(f"Please re-run `node get-refresh-token.js` from {MCP_GSUITE_PATH} and copy the file token.json to the parent directory.")
        exit(1)

    # Calculate remaining time (delta is in seconds)
    delta_seconds = refresh_expiry_timestamp - now
    
    # Convert seconds to days, hours, minutes, seconds
    days, remainder = divmod(delta_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)      # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)        # 60 seconds in a minute
    
    # Convert to integers for display
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    formatted = f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}, and {seconds} second{'s' if seconds != 1 else ''}"
    return formatted
# -------------------------------------------------------------------
# End of helper functions
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Set up the FastAPI app and use
# modern FastAPI lifespan for startup/shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate the token and exit if invalid
    token_expiry = load_and_validate_token_at_startup()
    
    # Store token_expiry in app state for later use (e.g., /task/send route)
    app.state.token_expiry = token_expiry

    # Register with APP_NAME
    await register(APP_NAME)

    # Start the agent.
    await create_agent()
    yield
    logger.info("Shutting down Email Agent. Cleaning up resources...")
    mcp_toolset = getattr(app, "mcp_toolset", None)
    if mcp_toolset:
        try:
            await mcp_toolset.cleanup()
        except ProcessLookupError:
            logger.warning("Subprocess already terminated; skipping cleanup.")

# Create a simple web server with FastAPI using the lifespan context manager
app = FastAPI(lifespan=lifespan)

# This card will be sent to the registry at startup.
# The name, description, url, and skills are critical 
AGENT_CARD = {
  "name": "EmailAgent",  
  "description": "An email agent that uses the Google MCP server to send_email .",  
  "url": "http://localhost:6001",  
  "version": "1.0",
  "provider": {
    "organization": "Fairwind Technologies, LLC.",
    "url": "https://www.fairwindtechnologies.com"
  },
  "version": "0.1.0",
  "capabilities": {
    "streaming": False,
    "pushNotifications": False,
    "stateTransitionHistory": False
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "text/plain"],
  "skills": [
    {
      "id": "send_email",
      "name": "Send Email",
      "description": "Send an email to one or more recipients using the Google gsuite MCP server. Requires the To address, a subject line, and the message body.",
      "tags": ["send", "email"],
      "examples": [
        "Send an email to test@test.com with a subject line of 'Touch Base' and a body of 'Hi Test! Just reaching out to say Hi!' "
      ],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": [
        "application/json",
        "text/plain"
      ]
    }
  ]
}

# App endpoints
# Serve the Agent Card at the well-known URL.
@app.get("/.well-known/agent.json")
def get_agent_card():
    logger.info("Request for agent card received.")
    """Endpoint to provide this agent's metadata (Agent Card)."""
    return JSONResponse(AGENT_CARD)

# Process requests
@app.post("/tasks/send")
async def handle_task(request: Request):
    # Validate that the token has not expired and if not, when it will.
    time_left = check_token_expiry_on_request(request.app.state.token_expiry)
    logger.debug(f"The current token will expire in {time_left}")

    """Endpoint for A2A clients to send a new task (with an initial user message)."""
    logger.debug("*************************************")
    logger.debug('New request received!')
    task_request = await request.json()  # parse incoming JSON request
    logger.debug(f'Task request JSON received:\n{task_request}')
    logger.debug("*************************************")
    # Extract the task ID and the user's message text from the request.
    task_id = task_request.get("id")

    # Set up session per request.
    user_id = f"user_{task_id}"  # simple mapping (or another logic)
    session_id = f"session_{task_id}"
    to = ""
    subject = ""
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

    # Extract the email details from the message
    message = task_request.get("message", {})
    parts = message.get("parts", [])
    user_message = ""

    if parts:
        email_text = parts[0].get("text", "")
        logger.debug(f"Received message text: {email_text}")
        
        # Try to parse as structured JSON first
        try:
            email_data = json.loads(email_text)
            to = email_data.get("to", "")
            subject = email_data.get("subject", "")
            body = email_data.get("body", "")

            if to and subject and body:
                user_message = f""" 
                Send an email to {to} 
                with the subject of {subject} 
                and the message body content of {body}
                """
                logger.debug("------------------------------------")
                logger.debug(f"Parsed structured JSON - Email to: {to}")
                logger.debug(f"Email subject: {subject}")
                logger.debug(f"Email body: {body}")
                logger.debug("------------------------------------")
            else:
                logger.debug("Structured JSON missing required fields")
                user_message = email_text  # Use the text as-is for the agent to parse
                
        except json.JSONDecodeError:
            # Not JSON, treat as plain text instruction for the agent to parse
            logger.debug("Not structured JSON, treating as plain text instruction")
            user_message = email_text
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            user_message = email_text  # Fallback to plain text
    
    if not user_message:
        user_message = """No message content received. Please provide email details: recipient address, subject, and body."""

    # For debugging.
    logger.debug('#############################################')
    logger.debug(f'LLM Prompt for this run:\n{user_message} ')
    logger.debug(f"Session ID: {req_session.id} User ID: {req_session.user_id}")
    logger.debug(f"User Message: {user_message}")
    logger.debug('#############################################')

    # This requires nest_asyncio.apply() earlier to allow nested asyncio.run() inside Flask.
    agent_reply_text = await process_user_message(user_message, req_session)
 
    # Formulate the response in A2A Task format.
    # We'll return a Task object with the final state = 'completed' and the agent's message.
    response_task = {
        "id": task_id,
        "status": {"state": "completed"},
        "messages": [
            task_request.get("message", {}),             # include the original user message in history
            {
                "role": "agent",                        # the agent's reply
                "parts": [{"text": agent_reply_text}]   # agent's message content as a TextPart
            }
        ]
        # We could also include an "artifacts" field if the agent returned files or other data.
    }
    return JSONResponse(response_task)
# -------------------------------------------------------------------

async def process_user_message(query: str, req_session) -> str:
    try:
           logger.debug('****************************')
           logger.debug(f"Calling the agent for a request handled by an **agent tool** Starting asyncio run to call_agent_async")
           logger.debug('****************************')

           logger.debug(f"Query to run: {query} ")
           response = await call_agent_async(query = query,
                          runner = app.runner,
                          user_id=req_session.user_id,
                          session_id=req_session.id)
           return response
    except Exception as e:
            logger.error(f"An error occurred: {e}")

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    logger.debug('****************************')
    logger.debug(f"In call_agent_async")
    logger.debug(f"User Query: {query}")
    logger.debug('****************************')

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # We iterate through events to find the final answer.
    logger.debug(f"Calling Running with userid: {user_id}, sessionid: {session_id}")
    sessions = await session_service.list_sessions(app_name=APP_NAME,user_id=user_id)
    logger.debug(f"Current sessions: {sessions}")
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        logger.debug(f" [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")


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



# ###################################################################
# Run the FastAPI app (A2A server) if this script is executed directly.
if __name__ == "__main__":
    print("Run with: uvicorn svr_email_agent:app --host 0.0.0.0 --port 6001")  # Fixed typo in filename to match current file name