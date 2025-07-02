#!/bin/bash

# Function to check if a server is ready
check_server_ready() {
    local host=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "Checking if server at ${host}:${port} is ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://${host}:${port}/.well-known/agent.json" > /dev/null 2>&1; then
            echo "Server at ${host}:${port} is ready (attempt ${attempt}/${max_attempts})"
            return 0
        fi
        
        echo "Attempt ${attempt}/${max_attempts}: Server at ${host}:${port} not ready yet, waiting..."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Server at ${host}:${port} did not become ready after ${max_attempts} attempts"
    return 1
}

# Show ps for the server processes
show_agent_processes() {
    echo
    echo "Current Server Agent Processes:"
    echo "----------------------------------------"
    
    # Show ps header
    ps -ef | head -1
    
    # Read and display each PID
    for pid_file in *.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if [ -n "$pid" ]; then
                ps -ef | grep -E "^\s*[^ ]+\s+${pid}\s" || echo "# Process $pid ($pid_file) not found"
            fi
        fi
    done
}

echo "Starting REGISTRY (svr_registry_agent) agent (FastAPI/uvicorn) on port 6060..."
uvicorn svr_registry_agent:app --host 0.0.0.0 --port 6060 > registry_agent.log 2>&1 &
echo $! > registry_agent.pid

if check_server_ready "0.0.0.0" "6060"; then
    echo "Server is up, starting next server..."
fi

#echo "Starting EMAIL (svr_email_agent) (FastAPI/uvicorn) on port 6001... "
#uvicorn svr_email_agent:app --host 0.0.0.0 --port 6001 > email_agent.log 2>&1 &
#echo $! > email_agent.pid

#if check_server_ready "0.0.0.0" "6001"; then
#    echo "Server is up, starting next server..."
#fi

echo "Starting CHAT (svr_chat_agent) agent (FastAPI/uvicorn) on port 6002"
uvicorn svr_chat_agent:app --host 0.0.0.0 --port 6002 > chat_agent.log 2>&1 &
echo $! > chat_agent.pid

if check_server_ready "0.0.0.0" "6002"; then
    echo "All agents started successfully."
fi

echo "Starting SMS (svr_sms_agent) agent (FastAPI/uvicorn) on port 6003..."
uvicorn svr_sms_agent:app --host 0.0.0.0 --port 6003 > sms_agent.log 2>&1 &
echo $! > sms_agent.pid

if check_server_ready "0.0.0.0" "6003"; then
    echo "Server is up!"
fi

# Now show the processes.
show_agent_processes