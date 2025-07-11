#!/usr/bin/env python3
"""
Web Agent Startup Script
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import shared_logger
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Change to the webagent directory
os.chdir(Path(__file__).parent)

# Import and run the server
from webagent_server import app
import uvicorn
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Configuration
WEB_PORT = int(os.getenv("WEB_PORT", "8081"))

if __name__ == "__main__":
    print(f"Starting Web Agent on port {WEB_PORT}")
    print("Open your browser to: http://localhost:8081")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT)