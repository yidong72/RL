#!/bin/bash
# Agentic job loop - runs until agent creates job_is_done file

set -e

# Configuration
INPUT_PROMPT="${1:-}"
LOGS_DIR="./logs"
JOB_DONE_FILE="./job_is_done"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"  # Safety limit
echo INPUT: $INPUT_PROMPT

if [[ -z "$INPUT_PROMPT" ]]; then
    echo "Usage: $0 <prompt>"
    echo "  Or set INPUT_PROMPT environment variable"
    exit 1
fi

# Create logs directory
mkdir -p "$LOGS_DIR"

# Clean up any previous job_is_done file
rm -f "$JOB_DONE_FILE"

iteration=0
echo "Starting agentic loop..."
echo "Prompt: $INPUT_PROMPT"
echo "Logs will be saved to: $LOGS_DIR"
echo "Loop will exit when agent creates: $JOB_DONE_FILE"
echo "-------------------------------------------"

while [[ ! -f "$JOB_DONE_FILE" ]]; do
    iteration=$((iteration + 1))
    
    if [[ $iteration -gt $MAX_ITERATIONS ]]; then
        echo "ERROR: Reached maximum iterations ($MAX_ITERATIONS). Exiting."
        exit 1
    fi
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$LOGS_DIR/session_${iteration}_${timestamp}.log"
    
    echo ""
    echo "=== Iteration $iteration ($(date)) ==="
    echo "Log file: $log_file"
    
    # Run the agent and capture output
    agent --model opus-4.5-thinking -f --approve-mcps -p "$INPUT_PROMPT" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    
    echo "Agent exited with code: $exit_code"
    
    # Check if job_is_done file was created
    if [[ -f "$JOB_DONE_FILE" ]]; then
        echo ""
        echo "=== Job completed! ==="
        echo "Agent signaled completion by creating $JOB_DONE_FILE"
        echo "Total iterations: $iteration"
        break
    fi
    
    # Brief pause between iterations
    sleep 2
done

echo ""
echo "All session logs saved in: $LOGS_DIR"
ls -la "$LOGS_DIR"
