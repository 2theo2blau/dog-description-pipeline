#!/bin/bash

# Define the directory containing the Python scripts
APP_DIR="app"
SCRIPTS=("preprocess.py" "encode.py" "requests.py" "compliment.py")

# Activate the conda environment first
source activate dog-compliment || {
    echo "Could not activate the conda environment."
    exit 1
}

# Function to run a Python script and check its exit status
run_script() {
    local script=$1
    echo "Running $script..."
    "$APP_DIR/$script" || {
        echo "Error running $script. Exiting."
        exit 1
    }
    echo "$script completed successfully."
}

# Loop through the scripts and run them one by one
for script in "${SCRIPTS[@]}"; do
    if [ -f "$APP_DIR/$script" ]; then
        run_script "$script"
    else
        echo "The script $APP_DIR/$script does not exist."
        exit 1
    fi
done

echo "All scripts have been executed successfully."