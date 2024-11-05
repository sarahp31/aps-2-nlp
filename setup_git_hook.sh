#!/bin/bash

# Set the path to your project directory (replace with your actual path)
PROJECT_DIR=$(pwd)

# Check if the run_jobs.sh script exists
if [ ! -f "$PROJECT_DIR/run_jobs.sh" ]; then
    echo "Error: run_jobs.sh not found in the project directory."
    exit 1
fi

# Create the post-merge hook
HOOK_FILE="$PROJECT_DIR/.git/hooks/post-merge"

# Write the hook script
echo "#!/bin/bash" > "$HOOK_FILE"
echo "echo 'Running run_jobs.sh after git pull...'" >> "$HOOK_FILE"
echo "$PROJECT_DIR/run_jobs.sh" >> "$HOOK_FILE"

# Make the post-merge hook executable
chmod +x "$HOOK_FILE"

# Confirm setup completion
echo "post-merge hook has been set up successfully to run run_jobs.sh after each git pull."
