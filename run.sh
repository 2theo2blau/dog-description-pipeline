#!/bin/bash

# Define the directory where the Python scripts are located
SCRIPTS_DIR="app"

# An array of script names in the order they should be executed
SCRIPTS=(preprocess.py encode.py requests.py compliment.py)

# Function to run a single Python script safely
run_script() {
    local script=$1
    echo "Executing $script..."
    python "$SCRIPTS_DIR/$script" || {
        echo "Error: $script failed to execute."
        exit 1
    }
}

# Iterate over the array and run each script in sequence
for script in "${SCRIPTS[@]}"; do
    run_script "$script"
done

echo "All scripts executed successfully."