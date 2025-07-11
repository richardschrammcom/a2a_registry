# Web Agent Interface

A FastAPI-based web interface for the chat agent with real-time streaming support.

## Features

- Modern web interface with Bootstrap styling
- Real-time chat with Server-Sent Events (SSE)
- Green user bubbles and blue agent bubbles
- Silver/white theme with blue/green accents
- Logo placeholder support
- Responsive design
- Error handling for agent connectivity

## Setup

1. Make sure the chat agent is running on port 6002:
   ```bash
   python3 svr_chat_agent.py
   ```

2. Start the web interface:
   ```bash
   python3 start_webagent.py
   ```

3. Open your browser to: http://localhost:8081

## Configuration

Configuration is handled through the main `.env` file in the project root:

- `WEB_PORT`: Port for the web interface (default: 8081)
- `LOGO_IMAGE`: Logo image filename in static/images/ (default: logo_placeholder.png)  
- `AGENT_BASE_URL`: URL of the chat agent (default: http://localhost:6002)

## File Structure

```
webagent/
├── .env                    # Environment configuration
├── start_webagent.py      # Startup script
├── webagent_server.py     # FastAPI server
├── README.md              # This file
├── static/
│   ├── css/
│   │   └── chat.css       # Custom styling
│   ├── js/
│   │   └── chat.js        # JavaScript functionality
│   └── images/
│       └── logo_placeholder.png  # Logo image
└── templates/
    └── chat.html          # Main chat template
```

## Usage

1. Type your message in the text input at the bottom
2. Press Enter or click Send to submit
3. Watch the real-time streaming response from the agent
4. Green bubbles = your messages, Blue bubbles = agent responses

## Error Handling

- If the chat agent is offline, the interface will show an error message
- Connection issues are handled gracefully with user feedback
- SSE stream errors are caught and displayed to the user