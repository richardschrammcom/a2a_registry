#!/bin/bash

  echo "Getting all agent cards from the registry."

  # Read raw JSON content from the file

  # Construct the JSON payload for /tasks/send
  PAYLOAD=$(jq -n \
    --arg tool "list_agent_cards" \
    --arg id "100100100" \
    '{
      id: $id, 
      tool_name: $tool
    }')

  # Print the payload for debugging (optional)
  echo "$PAYLOAD" | jq .

  # Send the request
  curl -s -X POST "http://localhost:6060/tasks/send" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $KEY" \
    -d "$PAYLOAD" | jq

  echo
