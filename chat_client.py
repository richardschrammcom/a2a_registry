import requests
import uuid
import os
from shared_logger import SharedLogger

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]

# Only display warning and error in this client, no matter the default.
logger = SharedLogger.get_logger(logger_name, console_level_override="WARNING")

logger.info(f"{logger_name} has started.")
log_level_name = SharedLogger.get_log_level_name()

# 1. Discover the agent by fetching its Agent Card (done only once)
AGENT_BASE_URL = "http://localhost:6002"
agent_card_url = f"{AGENT_BASE_URL}/.well-known/agent.json"
response = requests.get(agent_card_url)
if response.status_code != 200:
    #raise RuntimeError(f"Failed to get agent card: {response.status_code}")
    logger.error(f"Error fetching agent card: {response.status_code}. Exiting.")
    exit(1)
logger.info("Chat agent appears to be up and running.")

# Setting the unique ID once to maintain history in the session between chat messages.
task_id = str(uuid.uuid4())  # generate a random unique task ID

while True:
    input_text = input("Enter a query for the chatbot, (or type 'exit' to quit): ").strip()
    if input_text.lower() in ("exit", "quit"):
        logger.info(f"{__name__} is exiting.")
        break

    # 2. Prepare the request.
    user_text = input_text
    task_payload = {
        "id": task_id,
        "message": {
            "role": "user",
            "parts": [
                {"text": user_text}
            ]
        }
    }
    logger.info(f"Sending task {task_id} to agent with message: '{user_text}'")

    # 3. Send the task to the agent's tasks/send endpoint
    tasks_send_url = f"{AGENT_BASE_URL}/tasks/send"
    result = requests.post(tasks_send_url, json=task_payload)
    if result.status_code != 200:
        logger.error(f"Request to agent failed:  {result.status_code}, {result.text}")
        raise RuntimeError(f"Task request failed: {result.status_code}, {result.text}")
    task_response = result.json()

    # 4. Process the agent's response
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