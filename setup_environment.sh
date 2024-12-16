#!/bin/bash

# Define the environment name and base Python version
ENV_NAME="dog-compliment"
PYTHON_VERSION="3.11"

# Define paths
CONDA_ENV_PATH="$PWD/$ENV_NAME"
IMAGES_DIR="$PWD/images"
INPUT_IMG_DIR="$IMAGES_DIR/input_photos"
PROCESSED_IMG_DIR="$IMAGES_DIR/processed_photos"
REQUIREMENTS_FILE="requirements.txt"

# Create the environment if it doesn't exist
if conda list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists."
else
    # Install conda if it's not available
    if ! command -v conda &> /dev/null; then
        echo "conda could not be found. Please install conda first."
        exit 1
    fi

    # Create the environment with Python 3.11
    conda create --name "$ENV_NAME" python=$PYTHON_VERSION && conda activate "$ENV_NAME"
fi

# Create the 'images' directory and its subdirectories if they don't exist
if [ ! -d "$IMAGES_DIR" ]; then
    mkdir -p "$IMAGES_DIR"
fi

# Create input-images and processed-images directories within images
if [ ! -d "$INPUT_IMG_DIR" ]; then
    mkdir -p "$INPUT_IMG_DIR"
fi
if [ ! -d "$PROCESSED_IMG_DIR" ]; then
    mkdir -p "$PROCESSED_IMG_DIR"
fi

# Install Python requirements from the requirements.txt file
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "requirements.txt not found."
    exit 1
fi

touch images.json
touch captions.json
touch responses.json

echo "Environment $ENV_NAME initialized with Python 3.11, directories created, and requirements installed."