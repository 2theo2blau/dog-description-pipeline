import os
from mistralai import Mistral
import json

# Load the API key from the environment variable
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def process_captions(input_json, output_json):
    # Read the input JSON file
    with open(input_json, 'r') as f:
        captions = json.load(f)

    # Prepare a list to hold the responses
    responses = []

    # Loop through each entry in the JSON file
    for entry in captions:
        # Extract the caption and filepath
        caption = entry['caption']
        filepath = entry['path']

        # Send the caption as a prompt to the Mistral agent
        chat_response = client.agents.complete(
            agent_id="ag:5bbf3498:20241215:dog-complimentor:0a8573cd",
            messages=[
                {
                    "role": "user",
                    "content": f"Based on the caption and physical characteristics of the dog described, write a creative and highly personalized compliment for the dog: {caption}",
                },
            ],
        )

        # Collect the agent's response
        agent_response = chat_response.choices[0].message.content if chat_response.choices else "No response received."

        # Append the response with the associated file path to the responses list
        responses.append({
            "path": filepath,
            "agent_response": agent_response
        })

    # Write the responses to a new JSON file
    with open(output_json, 'w') as f:
        json.dump(responses, f, indent=2)

# Assuming you have a 'captions.json' file as input and want to write the output to 'responses.json'
process_captions('captions.json', 'responses.json')