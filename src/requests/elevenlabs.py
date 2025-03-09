from elevenlabs import ElevenLabs
import os
import dotenv
import uuid
from typing import List, Dict, Any, Optional
import datetime

from ..models.text import Compliment
from ..models.media import Audio

class ElevenLabs:
    def __init__(self, 
                 api_key: str, 
                 model_id: str, 
                 voice_id: str, 
                 outtype: Optional[str],
                 out_dir: Optional[str]):
        self.api_key = api_key or self.get_api_key()
        self.model_id = model_id
        self.voice_id = voice_id
        self.outtype = outtype or "mp3_44100_128"
        self.out_dir = out_dir or "../../data/audio"

    async def get_api_key(self) -> str:
        dotenv.load_dotenv()
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY environment variable not set")
        return api_key
    
    async def get_elevenlabs_client(self):
        return ElevenLabs(
            api_key=await self.get_api_key()
        )
    
    async def convert_compliment_to_audio(self, compliment: Compliment, model_id: str, voice_id: str, outtype: str, out_dir: str) -> Audio:
        client = await self.get_elevenlabs_client()
        audio = client.text_to_speech.convert(
            text=compliment.compliment,
            voice_id=voice_id,
            model_id=model_id,
            output_format=outtype
        )
        audio_id = str(uuid.uuid4())
        out_path = os.path.join(out_dir, f"compliment_{audio_id}.mp3")
        with open(out_path, "wb") as f:
            if hasattr(audio, "__iter__") and not isinstance(audio, bytes):
                for chunk in audio:
                    f.write(chunk)
            else:
                f.write(audio)
        tts = Audio(
            id=audio_id,
            compliment_id=compliment.id,
            url=out_path,
            type=outtype,
            created_at=datetime.now()
        )
        compliment.audio_id = audio_id
        compliment.processed = True
        compliment.save()
        return tts
