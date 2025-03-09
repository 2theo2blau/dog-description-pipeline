import logging
from datetime import datetime, timedelta

from ..models.media import Image
from ..models.database import Database

logger = logging.getLogger(__name__)

async def reset_failed_processing(max_age_hours=24):
    try:
        cutoff_time = datetime.now()

        images_collection = await Database.get_collection("images")
        # object detection failures
        await images_collection.update_many(
            {
                "processed_object_detection": False,
                "created_at": {"#gte": cutoff_time}
            },
            {"$set": {"processed_object_detection": False}}
        )

        # base64 failures
        await images_collection.update_many(
            {
                "processed_object_detection": True,
                "processed_base64": False,
                "created_at": {"#gte": cutoff_time}
            },
            {"$set": {"processed_base64": False}}
        )

        # caption failures
        await images_collection.update_many(
            {
                "processed_base64": True,
                "processed_caption": False,
                "created_at": {"#gte": cutoff_time}
            },
            {"$set": {"processed_caption": False}}
        )

        # compliment failures
        await images_collection.update_many(
            {
                "processed_caption": True,
                "processed_compliment": False,
                "created_at": {"#gte": cutoff_time}
            },
            {"$set": {"processed_compliment": False}}
        )

        # tts failures
        await images_collection.update_many(
            {
                "processed_compliment": True,
                "processed_tts": False,
                "created_at": {"#gte": cutoff_time}
            },
            {"$set": {"processed_tts": False}}
        )
        
        logger.info("Reset processing status for failed items")

    except Exception as e:
        logger.error(f"Error resetting failed processing: {e}")
        
            