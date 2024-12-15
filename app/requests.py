import json
import os
from mistralai import Mistral

# Function to retrieve the API key from environment variables
def get_api_key():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY environment variable is not set.")
    return api_key

# Initialize the Mistral client
def get_mistral_client(api_key):
    return Mistral(api_key=api_key)

# Function to send a chat request and return the response content
def get_chat_response(client, messages, model):
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

# Function to process images and write captions to a JSON file
def process_images_for_captions(input_json, output_json, model):
    api_key = get_api_key()
    client = get_mistral_client(api_key)
    
    with open(input_json, 'r') as infile:
        images_data = json.load(infile)
        
    processed_images = []
    
    for image in images_data:
        image_path = image['path']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe in detail the physical characteristics of the dog in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image['encoded_image']}" 
                    }
                ]
            }
        ]
        
        # Get the chat response and extract the caption
        try:
            caption = get_chat_response(client, messages, model)
            processed_images.append({
                "path": image_path,
                "caption": caption
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Write the data with captions to a new JSON file
    with open(output_json, 'w') as outfile:
        json.dump(processed_images, outfile, indent=4)

# Input and output JSON filenames
input_json = 'images.json'  # The output from the last code snippet
output_json = 'captions.json'  # The new file with captions

# Specify model
model = "pixtral-12b-2409"

# Process images for captions and write to JSON
process_images_for_captions(input_json, output_json, model)