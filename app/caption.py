import json
import os
import time
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to retrieve and validate the API key from environment variables
def get_api_key():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY environment variable is not set.")
    # if not api_key.startswith("mis"):  # Basic validation for Mistral API key format
    #     raise ValueError("Invalid Mistral API key format. Key should start with 'mis'")
    return api_key

# Initialize the Mistral client with retry logic
def get_mistral_client(api_key):
    try:
        client = Mistral(api_key=api_key)
        # Test the client with a simple request
        client.chat.complete(
            model="mistral-tiny",
            messages=[{"role": "user", "content": "test"}]
        )
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Mistral client: {str(e)}")

# Function to send a chat request with retry logic
def get_chat_response(client, messages, model, max_retries=3):
    retry_count = 0
    base_delay = 1  # Start with 1 second delay

    while retry_count < max_retries:
        try:
            chat_response = client.chat.complete(
                model=model,
                messages=messages,
            )
            return chat_response.choices[0].message.content
        # except Exception as e:
        #     retry_count += 1
        #     if retry_count == max_retries:
        #         raise Exception(f"Failed after {max_retries} retries: {str(e)}")
        #     delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
        #     print(f"Attempt {retry_count} failed. Retrying in {delay} seconds...")
        #     time.sleep(delay)
        except Exception as e:
            raise Exception(f"Unexpected error during API call: {str(e)}")

# Function to process images and write captions to a JSON file
def process_images_for_captions(input_json, output_json, model):
    # Initialize client once for all requests
    api_key = get_api_key()
    client = get_mistral_client(api_key)
    
    with open(input_json, 'r') as infile:
        images_data = json.load(infile)
        
    processed_images = []
    total_images = len(images_data)
    
    for idx, image in enumerate(images_data, 1):
        image_path = image['path']
        print(f"Processing image {idx}/{total_images}: {image_path}")
        
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
        
        try:
            caption = get_chat_response(client, messages, model)
            processed_images.append({
                "path": image_path,
                "caption": caption
            })
            print(f"Successfully processed {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Continue with next image instead of stopping
            continue
        
        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)
    
    # Write the data with captions to a new JSON file
    with open(output_json, 'w') as outfile:
        json.dump(processed_images, outfile, indent=4)
    
    print(f"\nProcessing complete. Successfully processed {len(processed_images)} out of {total_images} images.")

# Input and output JSON filenames
input_json = 'images.json'  # The output from the last code snippet
output_json = 'captions.json'  # The new file with captions

# Specify model
model = "pixtral-12b-2409"

# Process images for captions and write to JSON
process_images_for_captions(input_json, output_json, model)