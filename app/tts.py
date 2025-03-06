import os
import json
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

def convert_compliment_to_audio(input_json, output_dir, voice_id, model_id, output_format):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_json, 'r') as infile:
        compliments = json.load(infile)

    converted_to_audio = []
    total_compliments = len(compliments)

    for idx, compliment in enumerate(compliments, 1):
        text = compliment['agent_response']
        print(f"Converting compliment {idx}/{total_compliments} to audio: {text}")

        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        output_path = os.path.join(output_dir, f"compliment_{idx}.mp3")
        
        # Handle the generator response by reading chunks and writing to file
        with open(output_path, "wb") as f:
            # If audio is a generator, consume it chunk by chunk
            if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
                for chunk in audio:
                    f.write(chunk)
            else:
                # If it's already bytes, write directly
                f.write(audio)

        print(f"Saved audio to {output_path}")
        
        # Store the compliment info with audio path
        compliment_info = compliment.copy()
        compliment_info['audio_path'] = output_path
        converted_to_audio.append(compliment_info)

    # Save metadata about the converted files
    metadata_path = os.path.join(output_dir, "audio_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(converted_to_audio, f, indent=4)
    
    print(f"Conversion complete. {len(converted_to_audio)} audio files created.")


input_json = "responses.json"
output_dir = "audio"
voice_id = "g45NCggYaclvkMq2KNuc"
model_id = "eleven_flash_v2_5"
output_format = "mp3_44100_128"

convert_compliment_to_audio(input_json, output_dir, voice_id, model_id, output_format)