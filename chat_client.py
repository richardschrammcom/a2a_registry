import requests
import uuid
import os
import json
import argparse
import sseclient
from shared_logger import SharedLogger

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]

# Only display warning and error in this client, no matter the default.
logger = SharedLogger.get_logger(logger_name, console_level_override="WARNING")

logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

# 1. Discover the agent by fetching its Agent Card (done only once)
AGENT_BASE_URL = "http://localhost:6002"

def check_agent_availability():
    agent_card_url = f"{AGENT_BASE_URL}/.well-known/agent.json"
    try:
        response = requests.get(agent_card_url)
        if response.status_code != 200:
            logger.error(f"Error fetching agent card: {response.status_code}. Exiting.")
            exit(1)
        logger.info("Chat agent appears to be up and running.")
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to chat agent. Make sure the server is running on port 6002.")
        exit(1)

def stream_events(stream_url):
    """Stream events from the SSE endpoint and display them in real-time"""
    try:
        response = requests.get(stream_url, stream=True, headers={'Accept': 'text/event-stream'})
        if response.status_code != 200:
            print(f"Failed to connect to stream: {response.status_code}")
            return None
        
        client = sseclient.SSEClient(response)
        final_response = None
        
        for event in client.events():
            if event.event and event.data:
                try:
                    data = json.loads(event.data)
                    
                    if event.event == "connected":
                        print("🔄 Connected to agent stream...")
                    elif event.event == "processing_start":
                        print("🤖 Agent is processing your request...")
                    elif event.event == "agent_thinking":
                        print("💭 Agent is thinking...")
                    elif event.event == "registry_query_start":
                        print(f"🔍 {data.get('message', 'Searching registry...')}")
                    elif event.event == "registry_response":
                        print(f"📋 {data.get('message', 'Registry responded')}")
                    elif event.event == "agent_call_start":
                        print(f"📞 {data.get('message', 'Calling another agent...')}")
                    elif event.event == "agent_call_response":
                        print(f"✅ {data.get('message', 'Agent responded')}")
                    elif event.event == "final_response":
                        print("\n🎯 Final Response:")
                        final_response = data.get('response', '')
                        print(final_response)
                    elif event.event == "stream_end":
                        print("✅ Stream completed")
                        break
                    elif event.event == "registry_error" or event.event == "agent_call_error":
                        print(f"❌ {data.get('message', 'Error occurred')}")
                    elif event.event == "error":
                        print(f"❌ {data.get('message', 'Error occurred')}")
                        break
                    elif event.event == "keepalive":
                        print(".", end="", flush=True)
                    else:
                        # Handle other events
                        if data.get('message'):
                            print(f"📡 {data.get('message')}")
                except json.JSONDecodeError:
                    print(f"Raw event: {event.event} - {event.data}")
                    
        return final_response
        
    except Exception as e:
        print(f"Error streaming events: {e}")
        return None

def send_non_streaming_request(task_id, user_text, session_id):
    """Send a non-streaming request to the agent"""
    task_payload = {
        "id": session_id,  # Use session_id for conversation history
        "message": {
            "role": "user",
            "parts": [
                {"text": user_text}
            ]
        }
    }
    
    tasks_send_url = f"{AGENT_BASE_URL}/tasks/send"
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
                    agent_reply_text += part["text"] or ""
            print("Agent's reply:", agent_reply_text)
        else:
            print("No messages in response!")
    else:
        print("Task did not complete. Status:", task_response.get("status"))

def send_streaming_request(task_id, user_text, session_id):
    """Send a streaming request to the agent"""
    task_payload = {
        "id": session_id,  # Use session_id for conversation history
        "message": {
            "role": "user",
            "parts": [
                {"text": user_text}
            ]
        }
    }
    
    tasks_send_url = f"{AGENT_BASE_URL}/tasks/send"
    result = requests.post(tasks_send_url, json=task_payload, params={"stream": "true"})
    if result.status_code != 200:
        logger.error(f"Request to agent failed:  {result.status_code}, {result.text}")
        raise RuntimeError(f"Task request failed: {result.status_code}, {result.text}")
    
    task_response = result.json()
    
    if task_response.get("status", {}).get("state") == "processing":
        stream_url = task_response.get("stream_url")
        if stream_url:
            print(f"🔄 Streaming from: {stream_url}")
            return stream_events(stream_url)
        else:
            print("No stream URL provided")
            return None
    else:
        print("Task did not start processing correctly")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chat client with optional streaming support')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    args = parser.parse_args()

    # Check if agent is available
    check_agent_availability()

    # Generate a session ID once to maintain conversation history
    session_id = str(uuid.uuid4())
    print(f"Chat client started in {'streaming' if args.stream else 'non-streaming'} mode")
    print(f"Session ID: {session_id}")
    print("Enter 'exit' or 'quit' to exit, 'toggle' to switch modes")

    while True:
        input_text = input("Enter a query for the chatbot: ").strip()
        if input_text.lower() in ("exit", "quit"):
            logger.info(f"{__name__} is exiting.")
            break
        elif input_text.lower() == "toggle":
            args.stream = not args.stream
            print(f"Switched to {'streaming' if args.stream else 'non-streaming'} mode")
            continue

        # Generate a new task ID for each request, but use the same session_id for history
        task_id = str(uuid.uuid4())
        user_text = input_text
        logger.info(f"Sending task {task_id} to agent with message: '{user_text}'")

        try:
            if args.stream:
                send_streaming_request(task_id, user_text, session_id)
            else:
                send_non_streaming_request(task_id, user_text, session_id)
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Error processing request: {e}")

if __name__ == "__main__":
    main()