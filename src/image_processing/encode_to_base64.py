import os
import base64
from PIL import Image

from ..models.media import Image
from ..utils.image import get_image_by_id

class EncodeToBase64:
    def __init__(self, image_id: str):
        self.image_id = image_id

    async def image_to_base64(self) -> str:
        image = await get_image_by_id(self.image_id)
        if image.base64:
            return f"image {self.image_id} already converted to base64"
        image_path = image.object_detection_url or image.url
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode()
        image.base64 = image_base64
        image.processed_base64 = True
        await image.save()
        return f"image {self.image_id} converted to base64"