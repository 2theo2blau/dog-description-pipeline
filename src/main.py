import asyncio
import logging
import signal
import sys
from datetime import datetime
import uuid
import os
from typing import List, Dict, Any

from .models.database import Database
from .models.media import Image, Audio
from .models.text import NeutralCaption, Compliment
from .models.apis import MistralAPI, MistralAgent, ElevenLabsAPI
from .image_processing import ObjectDetection, EncodeToBase64
from .requests import Mistral, ElevenLabs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dog_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
OBJECT_DETECTION_MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
PROCESSED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "../data/processed_images")
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/audio")
BATCH_SIZE = 5  # Number of images to process in each batch
SLEEP_TIME = 60  # Time to sleep between processing cycles in seconds

# Ensure output directories exist
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Initialize API configurations
mistral_api = MistralAPI(
    api_key="",  # Will be loaded from env
    endpoint_url="https://api.mistral.ai/v1",
    model="pixtral-12b-2409",
    temperature=0.2,
    max_tokens=4096,
    top_k=50,
    top_p=0.9,
    min_p=0.1
)

mistral_agent = MistralAgent(
    api_key="",  # Will be loaded from env
    endpoint_url="https://api.mistral.ai/v1",
    agent_id="ag:16aad2f0:20250305:dog-complimentor:6886ab50",
    max_tokens=4096
)

elevenlabs_api = ElevenLabsAPI(
    api_key="",  # Will be loaded from env
    endpoint_url="https://api.elevenlabs.io/v1",
    model_id="eleven_multilingual_v2",
    voice_id="pNInz6obpgDQGcFmaJgB",  # Example voice ID
    output_format="mp3_44100_128"
)

# Flag to control graceful shutdown
running = True

async def process_object_detection(batch_size: int):
    """Process images that need object detection."""
    try:
        images = await Image.get_ready_for_object_detection()
        logger.info(f"Found {len(images)} images for object detection")
        
        for i, image_data in enumerate(images[:batch_size]):
            try:
                image = Image(**image_data)
                logger.info(f"Processing object detection for image {image.id}")
                
                detector = ObjectDetection(
                    model_url=OBJECT_DETECTION_MODEL_URL,
                    image=image,
                    out_dir=PROCESSED_IMAGES_DIR
                )
                
                # Locate the main subject (dog) in the image
                subject_box = await detector.locate_main_subject(image.id)
                
                # Generate output path for the processed image
                out_path = os.path.join(PROCESSED_IMAGES_DIR, f"processed_{image.id}.{image.type}")
                
                # Crop and resize the image around the subject
                await detector.crop_image_around_subject(image.id, out_path)
                
                logger.info(f"Successfully processed object detection for image {image.id}")
            except Exception as e:
                logger.error(f"Error processing object detection for image {image_data.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_object_detection: {str(e)}")

async def process_base64_encoding(batch_size: int):
    """Process images that need base64 encoding."""
    try:
        images = await Image.get_ready_for_base64_conversion()
        logger.info(f"Found {len(images)} images for base64 encoding")
        
        for i, image_data in enumerate(images[:batch_size]):
            try:
                image = Image(**image_data)
                logger.info(f"Processing base64 encoding for image {image.id}")
                
                encoder = EncodeToBase64(image_id=image.id)
                result = await encoder.image_to_base64()
                
                logger.info(f"Successfully encoded image {image.id} to base64")
            except Exception as e:
                logger.error(f"Error encoding image {image_data.get('id')} to base64: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_base64_encoding: {str(e)}")

async def process_captions(batch_size: int):
    """Process images that need captions."""
    try:
        images = await Image.get_ready_for_caption()
        logger.info(f"Found {len(images)} images for captioning")
        
        mistral_client = Mistral(
            api_key="",  # Will be loaded from env
            model=mistral_api.model,
            agent=None
        )
        
        for i, image_data in enumerate(images[:batch_size]):
            try:
                image = Image(**image_data)
                logger.info(f"Generating caption for image {image.id}")
                
                caption = await mistral_client.get_neutral_caption(image.id)
                await caption.save()
                
                logger.info(f"Successfully generated caption for image {image.id}")
            except Exception as e:
                logger.error(f"Error generating caption for image {image_data.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_captions: {str(e)}")

async def process_compliments(batch_size: int):
    """Process images that need compliments."""
    try:
        images = await Image.get_ready_for_compliment()
        logger.info(f"Found {len(images)} images for compliment generation")
        
        mistral_client = Mistral(
            api_key="",  # Will be loaded from env
            model=None,
            agent=mistral_agent
        )
        
        for i, image_data in enumerate(images[:batch_size]):
            try:
                image = Image(**image_data)
                logger.info(f"Generating compliment for image {image.id}")
                
                compliment = await mistral_client.get_compliment(image.id, mistral_agent)
                await compliment.save()
                
                logger.info(f"Successfully generated compliment for image {image.id}")
            except Exception as e:
                logger.error(f"Error generating compliment for image {image_data.get('id')}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_compliments: {str(e)}")

async def process_tts(batch_size: int):
    """Process compliments that need text-to-speech conversion."""
    try:
        images = await Image.get_ready_for_tts()
        logger.info(f"Found {len(images)} compliments for TTS conversion")
        
        elevenlabs_client = ElevenLabs(
            api_key="",  # Will be loaded from env
            model_id=elevenlabs_api.model_id,
            voice_id=elevenlabs_api.voice_id,
            outtype=elevenlabs_api.output_format,
            out_dir=AUDIO_OUTPUT_DIR
        )
        
        for i, image_data in enumerate(images[:batch_size]):
            try:
                image = Image(**image_data)
                if not image.compliment_id:
                    logger.warning(f"Image {image.id} has no compliment_id, skipping TTS")
                    continue
                    
                compliment = await Compliment.get_compliment_by_id(image.compliment_id)
                if not compliment:
                    logger.warning(f"Compliment {image.compliment_id} not found, skipping TTS")
                    continue
                
                logger.info(f"Converting compliment {compliment.id} to audio")
                
                audio = await elevenlabs_client.convert_compliment_to_audio(
                    compliment=compliment,
                    model_id=elevenlabs_api.model_id,
                    voice_id=elevenlabs_api.voice_id,
                    outtype=elevenlabs_api.output_format,
                    out_dir=AUDIO_OUTPUT_DIR
                )
                await audio.save()
                
                # Update the image record
                image.processed_tts = True
                image.audio_id = audio.id
                await image.save()
                
                logger.info(f"Successfully converted compliment {compliment.id} to audio")
            except Exception as e:
                logger.error(f"Error converting compliment for image {image_data.get('id')} to audio: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_tts: {str(e)}")

async def process_pipeline():
    """Process all stages of the pipeline in sequence."""
    try:
        await process_object_detection(BATCH_SIZE)
        await process_base64_encoding(BATCH_SIZE)
        await process_captions(BATCH_SIZE)
        await process_compliments(BATCH_SIZE)
        await process_tts(BATCH_SIZE)
    except Exception as e:
        logger.error(f"Error in process_pipeline: {str(e)}")

def handle_shutdown(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received shutdown signal {sig}, shutting down gracefully...")
    running = False

async def main():
    """Main function to run the processing pipeline continuously."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("Starting dog compliment pipeline")
    
    # Connect to the database
    try:
        await Database.connect(MONGODB_URI)
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return
    
    # Main processing loop
    while running:
        try:
            logger.info("Starting processing cycle")
            await process_pipeline()
            logger.info(f"Processing cycle completed, sleeping for {SLEEP_TIME} seconds")
            
            # Use asyncio.sleep to allow for graceful shutdown during sleep
            for _ in range(SLEEP_TIME):
                if not running:
                    break
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in main processing loop: {str(e)}")
            # Sleep briefly before retrying to avoid tight error loops
            await asyncio.sleep(5)
    
    # Clean up resources
    logger.info("Shutting down...")
    await Database.close()
    logger.info("Database connection closed")
    logger.info("Pipeline shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())



