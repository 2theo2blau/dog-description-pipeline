from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List

from .database import Database

@dataclass
class Image:
    id: str
    url: str
    width: int
    height: int
    type: str
    processed_object_detection: bool = False
    processed_base64: bool = False
    processed_caption: bool = False
    processed_compliment: bool = False
    processed_tts: bool = False
    subject_box: Optional[List[int]] = None
    created_at: datetime
    object_detection_url: Optional[str] = None
    base64: Optional[str] = None
    caption_id: Optional[str] = None
    compliment_id: Optional[str] = None
    audio_id: Optional[str] = None

    @classmethod
    async def get_image_by_id(cls, image_id: str):
        collection = await Database.get_collection("images")
        doc = await collection.find_one({"id": image_id})
        if not doc:
            return None
        return cls(**doc)
    
    async def save(self):
        collection = await Database.get_collection("images")
        doc = asdict(self)
        await collection.update_one(
            {"id": self.id},
            {"$set": doc},
            upsert=True
        )

    async def get_ready_for_object_detection():
        collection = await Database.get_collection("images")
        cursor = collection.find({"processed_object_detection": False})
        return await cursor.to_list(length=None)
    
    async def get_ready_for_base64_conversion():
        collection = await Database.get_collection("images")
        cursor = collection.find({
            "processed_object_detection": True, 
            "processed_base64": False
            })
        return await cursor.to_list(length=None)
    
    async def get_ready_for_caption():
        collection = await Database.get_collection("images")
        cursor = collection.find({
            "processed_object_detection": True,
            "processed_base64": True,
            "processed_caption": False
            })
        return await cursor.to_list(length=None)

    async def get_ready_for_compliment():
        collection = await Database.get_collection("images")
        cursor = collection.find({
            "processed_object_detection": True,
            "processed_base64": True,
            "processed_caption": True,
            "processed_compliment": False
        })
        return await cursor.to_list(length=None)
    
    async def get_ready_for_tts():
        collection = await Database.get_collection("images")
        cursor = collection.find({
            "processed_object_detection": True,
            "processed_base64": True,
            "processed_caption": True,
            "processed_compliment": True,
            "processed_tts": False
        })
        return await cursor.to_list(length=None)
    

@dataclass
class Audio:
    id: str
    compliment_id: str
    url: str
    type: str
    created_at: datetime

    @classmethod
    async def get_audio_by_id(cls, audio_id: str):
        collection = await Database.get_collection("audio")
        doc = await collection.find_one({"id": audio_id})
        if not doc:
            return None
        return cls(**doc)
    
    async def save(self):
        collection = await Database.get_collection("audio")
        doc = asdict(self)
        await collection.update_one(
            {"id": self.id},
            {"$set": doc},
            upsert=True
        )