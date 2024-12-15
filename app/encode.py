import os
import json
import base64
from PIL import Image

# Function to encode an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Function to process the images directory and write the output to a JSON file
def process_images_to_json(directory, output_file):
    images_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add/modify extensions as needed
                image_path = os.path.join(root, file)
                encoded_image = image_to_base64(image_path)
                images_data.append({
                    "path": image_path,
                    "encoded_image": encoded_image
                })
    
    # Write the data to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(images_data, outfile, indent=4)

# Directory containing images and output JSON filename
IMAGES_DIR = 'images/processed_photos'
OUTPUT_JSON_FILE = 'images.json'

# Process the images and write to JSON
process_images_to_json(IMAGES_DIR, OUTPUT_JSON_FILE)