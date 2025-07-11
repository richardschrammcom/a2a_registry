import os
import sys
import uuid
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any

import uvicorn
import requests
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directory to path for shared_logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_logger import SharedLogger

# Load environment variables from parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Configuration
WEB_PORT = int(os.getenv("WEB_PORT", "8081"))
LOGO_IMAGE = os.getenv("LOGO_IMAGE", "logo_placeholder.png")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:6002")

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()
logger.info(f"Log level set to: {log_level_name}")

# FastAPI app
app = FastAPI(title="Chat Agent Web Interface")

# Static files and templates
webagent_path = Path(__file__).parent
static_path = webagent_path / "static"
templates_path = webagent_path / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Session storage (in-memory for now)
sessions: Dict[str, Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    message: str
    session_id: str

def check_agent_availability():
    """Check if the chat agent is available"""
    agent_card_url = f"{AGENT_BASE_URL}/.well-known/agent.json"
    try:
        logger.debug(f"Checking agent availability at: {agent_card_url}")
        response = requests.get(agent_card_url, timeout=5)
        is_available = response.status_code == 200
        logger.info(f"Agent availability check: {is_available} (status: {response.status_code})")
        return is_available
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking agent availability: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the main chat interface"""
    # Generate a new session ID for each page load
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [],
        "created_at": asyncio.get_event_loop().time()
    }
    
    logger.info(f"New session created: {session_id}")
    
    # Check agent availability
    agent_available = check_agent_availability()
    logger.debug(f"Rendering chat interface. Agent available: {agent_available}")
    
    # Cache busting parameter
    cache_bust = str(int(time.time()))
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session_id,
        "logo_image": LOGO_IMAGE,
        "agent_available": agent_available,
        "agent_url": AGENT_BASE_URL,
        "cache_bust": cache_bust
    })

@app.post("/send_message")
async def send_message(message: str = Form(...), session_id: str = Form(...)):
    """Send a message to the chat agent"""
    logger.info(f"Received message for session {session_id}: {message}")
    
    if session_id not in sessions:
        logger.error(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=400, detail="Invalid session")
    
    # Check agent availability
    if not check_agent_availability():
        logger.error("Chat agent is not available")
        raise HTTPException(status_code=503, detail="Chat agent is not available")
    
    try:
        # Prepare the task payload
        task_payload = {
            "id": session_id,
            "message": {
                "role": "user",
                "parts": [{"text": message}]
            }
        }
        
        logger.debug(f"Task payload: {json.dumps(task_payload, indent=2)}")
        
        # Send streaming request to the agent
        tasks_send_url = f"{AGENT_BASE_URL}/tasks/send"
        logger.info(f"Sending request to: {tasks_send_url}")
        
        result = requests.post(tasks_send_url, json=task_payload, params={"stream": "true"}, timeout=10)
        
        logger.info(f"Agent response status: {result.status_code}")
        logger.debug(f"Agent response text: {result.text}")
        
        if result.status_code != 200:
            logger.error(f"Request to agent failed: {result.status_code}, {result.text}")
            raise HTTPException(status_code=result.status_code, detail="Failed to send message to agent")
        
        task_response = result.json()
        logger.debug(f"Task response: {json.dumps(task_response, indent=2)}")
        
        if task_response.get("status", {}).get("state") == "processing":
            stream_url = task_response.get("stream_url")
            if stream_url:
                logger.info(f"Stream URL received: {stream_url}")
                return {"status": "success", "stream_url": stream_url}
            else:
                logger.error("No stream URL provided in response")
                raise HTTPException(status_code=500, detail="No stream URL provided")
        else:
            logger.error(f"Task did not start processing. Status: {task_response.get('status')}")
            raise HTTPException(status_code=500, detail="Task did not start processing correctly")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with agent: {e}")
        raise HTTPException(status_code=503, detail="Unable to connect to chat agent")

@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """Stream SSE events from the chat agent"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")
    
    async def event_generator():
        # This endpoint will be called with a stream_url parameter
        # For now, we'll implement a basic event stream
        yield {
            "event": "connected",
            "data": json.dumps({"message": "Connected to chat stream"})
        }
    
    return EventSourceResponse(event_generator())

@app.get("/proxy_stream")
async def proxy_stream(stream_url: str):
    """Proxy SSE events from the chat agent"""
    logger.info(f"Starting SSE proxy for stream: {stream_url}")
    
    try:
        async def event_generator():
            try:
                logger.debug(f"Connecting to stream: {stream_url}")
                # Use requests to get the SSE stream from the agent
                response = requests.get(stream_url, stream=True, headers={'Accept': 'text/event-stream'}, timeout=30)
                
                logger.info(f"Stream connection status: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Failed to connect to agent stream: {response.status_code}"
                    logger.error(error_msg)
                    yield {
                        "event": "error", 
                        "data": json.dumps({"event": "error", "message": error_msg})
                    }
                    return
                
                current_event = None
                line_count = 0
                stream_ended = False
                
                # Forward the SSE events
                for line in response.iter_lines(decode_unicode=True):
                    line_count += 1
                    if line:
                        logger.debug(f"SSE Line {line_count}: {line}")
                        
                        if line.startswith('event: '):
                            current_event = line[7:]  # Remove 'event: ' prefix
                            logger.debug(f"SSE Event type: {current_event}")
                        elif line.startswith('data: '):
                            data_part = line[6:]  # Remove 'data: ' prefix
                            logger.debug(f"SSE Data: {data_part}")
                            
                            try:
                                # Try to parse as JSON to validate
                                parsed_data = json.loads(data_part)
                                yield {
                                    "event": "message",
                                    "data": json.dumps({"event": current_event, "data": parsed_data})
                                }
                            except json.JSONDecodeError:
                                # If not valid JSON, send as raw text
                                logger.debug(f"Non-JSON data received: {data_part}")
                                yield {
                                    "event": "message", 
                                    "data": json.dumps({"event": current_event or "message", "data": data_part})
                                }
                        elif line.strip() == '':
                            # Empty line indicates end of event
                            logger.debug("SSE Event boundary (empty line)")
                            current_event = None
                        else:
                            # Handle other SSE format lines
                            logger.debug(f"SSE Other line: {line}")
                            yield {
                                "event": "message",
                                "data": json.dumps({"event": "raw", "data": line})
                            }
                        
                        # Check if we should stop streaming
                        if current_event == "stream_end":
                            logger.info("Stream end event received, terminating")
                            stream_ended = True
                            break
                
                logger.info(f"SSE stream ended. Total lines processed: {line_count}. Stream ended cleanly: {stream_ended}")
                            
            except requests.exceptions.Timeout:
                error_msg = "Connection to agent timed out"
                logger.error(error_msg)
                yield {
                    "event": "error",
                    "data": json.dumps({"event": "error", "message": error_msg})
                }
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Unable to connect to chat agent: {e}"
                logger.error(error_msg)
                yield {
                    "event": "error", 
                    "data": json.dumps({"event": "error", "message": "Unable to connect to chat agent"})
                }
            except Exception as e:
                error_msg = f"Unexpected error in stream: {e}"
                logger.error(error_msg)
                yield {
                    "event": "error",
                    "data": json.dumps({"event": "error", "message": f"Stream error: {str(e)}"})
                }
        
        return EventSourceResponse(event_generator())
        
    except Exception as e:
        error_msg = f"Error setting up stream proxy: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail="Unable to establish connection to agent stream")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent_available = check_agent_availability()
    return {
        "status": "healthy",
        "agent_available": agent_available,
        "agent_url": AGENT_BASE_URL
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Web Agent server starting on port {WEB_PORT}")
    logger.info(f"Agent URL: {AGENT_BASE_URL}")
    logger.info(f"Logo image: {LOGO_IMAGE}")
    logger.info(f"Static files path: {static_path}")
    logger.info(f"Templates path: {templates_path}")
    
    # Check if required directories exist
    if not static_path.exists():
        logger.error(f"Static files directory does not exist: {static_path}")
    if not templates_path.exists():
        logger.error(f"Templates directory does not exist: {templates_path}")

if __name__ == "__main__":
    logger.info(f"Starting web agent server on port {WEB_PORT}")
    logger.info(f"Agent URL: {AGENT_BASE_URL}")
    logger.info(f"Logo image: {LOGO_IMAGE}")
    
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="info")