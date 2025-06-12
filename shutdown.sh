#!/bin/bash

echo "Shutting down all agents with .pid files..."

for pid_file in *.pid; do
    # Skip if no .pid files exist
    [ -e "$pid_file" ] || continue

    pid=$(cat "$pid_file")
    if kill "$pid" > /dev/null 2>&1; then
        rm "$pid_file"
        echo "Stopped process with PID $pid from $pid_file"
    else
        echo "Warning: No process found for PID $pid from $pid_file (might already be stopped)"
        rm "$pid_file"
    fi
done

echo "Shutdown complete."