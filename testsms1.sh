#!/bin/bash

# Test script for SMS Agent (svr_sms_agent.py)
# Make sure the agent is running on localhost:6003 before running these tests

BASE_URL="http://localhost:6003"

echo "Testing SMS Agent..."
echo "===================="

# Test 1: Agent Card endpoint
echo "1. Testing Agent Card endpoint..."
curl -s -X GET "${BASE_URL}/.well-known/agent.json" | jq '.'
echo -e "\n"

# Test 2: Simple SMS request
echo "2. Testing simple SMS request..."
curl -s -X POST "${BASE_URL}/tasks/send" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-task-001",
    "message": {
      "parts": [
        {
          "text": "Send an SMS to 513.400.7390 saying Hello from the SMS agent test!"
        }
      ]
    }
  }' | jq '.'
echo -e "\n"

