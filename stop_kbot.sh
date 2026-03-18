#!/bin/bash

# Define service root directory (adjust according to your startup script)
SERVICE_ROOT="$(dirname "$0")"

# Function: Safely get PID of specific Python script running in specified directory
get_service_pid() {
    local script_dir="$1"
    local script_name="$2"
    
    # Use pgrep to find processes and pwdx to check if working directory matches
    # This ensures only the specific script running in the specified directory is targeted
    local pids=""
    pgrep -f "python.*${script_name}" | while read pid; do
        # Check if process working directory is under service root directory
        if pwdx "$pid" 2>/dev/null | grep -q "${script_dir}"; then
            echo "$pid"
        fi
    done
}

# Collect PIDs of processes to be terminated
declare -A PID_MAP  # Use associative array to remove duplicates

# Get main program PID (ensure only the one running in current project directory)
MAIN_PIDS=$(get_service_pid "${SERVICE_ROOT}" "kbot_main.py")
if [ -n "$MAIN_PIDS" ]; then
    while read pid; do
        if [ -n "$pid" ]; then
            PID_MAP["$pid"]=1
            echo "Found main program process: $pid"
        fi
    done <<< "$MAIN_PIDS"
fi

# Get PIDs of each microservice (now all in project root directory)
MICROSERVICES=(
    "kbot_app_embedding.py"
    "kbot_app_llm.py" 
    "kbot_app_vlm.py"
    "kbot_app_reranker.py"
    "kbot_app_parser.py"
)

for service_script in "${MICROSERVICES[@]}"; do
    SERVICE_PIDS=$(get_service_pid "${SERVICE_ROOT}" "${service_script}")
    if [ -n "$SERVICE_PIDS" ]; then
        while read pid; do
            if [ -n "$pid" ]; then
                PID_MAP["$pid"]=1
                echo "Found ${service_script} microservice process: $pid"
            fi
        done <<< "$SERVICE_PIDS"
    fi
done

# Check if there are processes to terminate
if [ ${#PID_MAP[@]} -eq 0 ]; then
    echo "No running KBot services found."
    exit 0
fi

# Convert deduplicated PIDs to space-separated string
PIDS="${!PID_MAP[@]}"

echo "About to terminate the following processes: $PIDS"
# read -p "Confirm termination of these services? (y/n): " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Operation cancelled."
#     exit 0
# fi

# Send SIGTERM signal for graceful shutdown
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        kill -SIGTERM $PID
        echo "Sent graceful shutdown signal (SIGTERM) to process $PID."
    else
        echo "Process $PID no longer exists, skipping."
    fi
done

# Wait for processes to exit gracefully (max 10 seconds)
TIMEOUT=10
for PID in $PIDS; do
    COUNT=0
    while kill -0 $PID 2>/dev/null && [ $COUNT -lt $TIMEOUT ]; do
        sleep 1
        COUNT=$((COUNT + 1))
    done
    
    # Check if process is still running
    if kill -0 $PID 2>/dev/null; then
        echo "Process $PID did not exit within ${TIMEOUT} seconds, will force terminate (SIGKILL)."
        kill -SIGKILL $PID 2>/dev/null || echo "Failed to force terminate process $PID."
    else
        echo "Process $PID exited normally."
    fi
done

echo "All KBot services have been shut down completely."