from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from .database import Database

@dataclass
class NeutralCaption:
    id: str
    image_id: str
    caption: str
    created_at: datetime
    processed: bool

    @classmethod
    async def get_caption_by_id(cls, caption_id: str):
        collection = await Database.get_collection("neutral_captions")
        doc = await collection.find_one({"id": caption_id})
        if not doc:
            return None
        return cls(**doc)
    
    async def save(self):
        collection = await Database.get_collection("neutral_captions")
        doc = asdict(self)
        await collection.update_one(
            {"id": self.id},
            {"$set": doc},
            upsert=True
        )

@dataclass
class Compliment:
    id: str
    image_id: str
    caption_id: str
    compliment: str
    created_at: datetime
    processed: bool
    audio_id: Optional[str] = None

    @classmethod
    async def get_compliment_by_id(cls, compliment_id: str):
        collection = await Database.get_collection("compliments")
        doc = await collection.find_one({"id": compliment_id})
        if not doc:
            return None
        return cls(**doc)
    
    async def save(self):
        collection = await Database.get_collection("compliments")
        doc = asdict(self)
        await collection.update_one(
            {"id": self.id},
            {"$set": doc},
            upsert=True
        )