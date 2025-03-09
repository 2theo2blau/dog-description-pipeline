from mistralai import Mistral
import os
import dotenv
import datetime
from typing import Optional, List, Dict, Any
import uuid

from ..models.apis import MistralAPI, MistralAgent
from ..models.media import Image
from ..models.text import NeutralCaption, Compliment

class Mistral:
    def __init__(self, 
                 api_key: str, 
                 model: Optional[str], 
                 agent: Optional[MistralAgent]
                 ):
        self.api_key = api_key or self.get_api_key()
        self.model = model or None
        self.agent = agent or None

    async def get_api_key(self) -> str:
        dotenv.load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY environment variable is not set")
        return api_key
    
    async def get_mistral_client(self) -> Mistral:
        return Mistral(api_key=await self.get_api_key())
    
    async def get_model_response(self, messages: List[Dict[str, Any]], mistral_model: MistralAPI) -> str:
        api_key = await self.get_api_key()
        client = await self.get_mistral_client(api_key)
        response = client.chat.complete(
            model=mistral_model.model,
            temperature=mistral_model.temperature,
            max_tokens=mistral_model.max_tokens,
            messages=messages
        )
        return response.choices[0].message.content
    
    async def get_agent_response(self, messages: List[Dict[str, Any]], agent: MistralAgent) -> str:
        api_key = await self.get_api_key()
        client = await self.get_mistral_client(api_key)
        response = client.agents.complete(
            agent_id=agent.id,
            messages=messages
        )
        return response.choices[0].message.content
    
    async def get_neutral_caption(self, image_id: str) -> NeutralCaption:
        image = await Image.get_image_by_id(image_id)
        if not image:
            raise ValueError(f"Image with id {image_id} not found")
        image_base64 = await image.get_image_base64()
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
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
        try:
            caption = await self.get_model_response(messages, MistralAPI.model, 4096, 0.2)
            neutral_caption = NeutralCaption(
                id=str(uuid.uuid4()),
                image_id=image_id,
                caption=caption,
                created_at=datetime.now()
            )
            image.caption_id = neutral_caption.id
            image.processed_caption = True
            await image.save()
            return neutral_caption
        except Exception as e:
            image.processed_caption = False
            await image.save()
            raise e
        
    async def get_compliment(self, image_id: str, agent: MistralAgent) -> Compliment:
        image = await Image.get_image_by_id(image_id)
        if not image:
            raise ValueError(f"Image with id {image_id} not found")
        if image.processed_caption == False or image.caption_id is None:
            raise ValueError(f"Image with id {image_id} has not been captioned")
        caption = NeutralCaption.get_caption_by_id(image.caption_id)
        messages = [
            {
                "role": "user",
                "content": f"Based on the caption and physical characteristics of the dog described, write a creative and highly personalized compliment for the dog: {caption}"
            }
        ]
        try:
            compliment = await self.get_agent_response(messages, agent)
            compliment = Compliment(
                id=str(uuid.uuid4()),
                image_id=image_id,
                caption_id=image.caption_id,
                compliment=compliment,
                created_at=datetime.now()
            )
            image.compliment_id = compliment.id
            image.processed_compliment = True
            await image.save()
            caption.processed = True
            caption.compliment_id = compliment.id
            await caption.save()
            return compliment
        except Exception as e:
            image.processed_compliment = False
            await image.save()
            caption.processed = False
            await caption.save()
            raise e