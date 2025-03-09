from ..models.media import Image

@staticmethod
async def get_image_by_id(image_id: str) -> Image:
    image = await Image.objects.get(id=image_id)
    return image

@staticmethod
async def get_image_base64(image_id: str) -> str:
    image = await get_image_by_id(image_id)
    if not image.processed_base64:
        raise ValueError(f"Image {image_id} has not been encoded to base64")
    return image.base64