#!/bin/bash

# Ensure the script stops on first error
set -e

# Define the local data paths as the current path plus /data
LOCAL_DATA_SEQ_UPSTREAM_PATH="$(pwd)/data/data_sequences_upstream"
LOCAL_DATA_SEQ_PATH="$(pwd)/data/data_sequences"
LOCAL_DATA_EXP_PATH="$(pwd)/data/data_expression"

DRIVE_DATA_PATH="gdrive:ML4RG_shared_with_students/projects/project-01/Supporting Materials"
# Define the remote drive data paths for shared folders
DRIVE_DATA_SEQ_UPSTREAM_PATH="$DRIVE_DATA_PATH/data_sequences_upstream"
DRIVE_DATA_SEQ_PATH="$DRIVE_DATA_PATH/data_sequences"
DRIVE_DATA_EXP_PATH="$DRIVE_DATA_PATH/data_expression"

# Define common exclude parameters as an array
COMMON_EXCLUDES=(--exclude '__pycache__/**' \
                 --exclude '*.DS_Store' \
                 --exclude '.idea/**' \
                 --exclude '.mypy_cache/**' \
                 --exclude '*.ipynb_checkpoints/**' \
                 --exclude '.git/**')

# Function to check and create local directories if they do not exist
ensure_local_dir() {
    if [ ! -d "$1" ]; then
        echo "Creating directory $1"
        mkdir -p "$1"
    fi
}

# Ensure local directories exist
ensure_local_dir "$LOCAL_DATA_SEQ_UPSTREAM_PATH"
ensure_local_dir "$LOCAL_DATA_SEQ_PATH"
ensure_local_dir "$LOCAL_DATA_EXP_PATH"

# Function to check if the remote directory exists
check_remote_dir() {
    rclone lsf --drive-shared-with-me "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: Remote directory $1 not found or not accessible."
        exit 1
    fi
}

# Check if remote directories exist
check_remote_dir "$DRIVE_DATA_SEQ_UPSTREAM_PATH"
check_remote_dir "$DRIVE_DATA_SEQ_PATH"
check_remote_dir "$DRIVE_DATA_EXP_PATH"

# Loop indefinitely until the user chooses to exit
while true; do
    # Display the menu to the user
    echo "Select an operation to perform:"
    echo "1. Send Data from Local to Drive"
    echo "2. Send Data from Drive to Local"
    echo "q. Exit the script"

    # Read the user's choice
    read -p "Enter your choice (1-2, q to quit): " choice

    # Execute the corresponding command based on the user's choice
    case $choice in
        1) 
           rclone sync -v --progress "$LOCAL_DATA_SEQ_UPSTREAM_PATH" "$DRIVE_DATA_SEQ_UPSTREAM_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           rclone sync -v --progress "$LOCAL_DATA_SEQ_PATH" "$DRIVE_DATA_SEQ_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           rclone sync -v --progress "$LOCAL_DATA_EXP_PATH" "$DRIVE_DATA_EXP_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           ;;
        2) 
           rclone sync -v --progress "$DRIVE_DATA_SEQ_UPSTREAM_PATH" "$LOCAL_DATA_SEQ_UPSTREAM_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           rclone sync -v --progress "$DRIVE_DATA_SEQ_PATH" "$LOCAL_DATA_SEQ_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           rclone sync -v --progress "$DRIVE_DATA_EXP_PATH" "$LOCAL_DATA_EXP_PATH" "${COMMON_EXCLUDES[@]}" --exclude '*_temporary/**' --drive-shared-with-me
           ;;
        q) 
           echo "Exiting script."
           exit 0
           ;;
        *) 
           echo "Invalid choice. Please enter 1, 2, or q to quit."
           ;;
    esac
    echo "" # Print a newline for better readability before the menu is shown again
done
