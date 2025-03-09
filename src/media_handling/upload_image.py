import os
import uuid
from datetime import datetime
from PIL import Image as PILImage
from typing import Optional

from ..models.media import Image

async def upload_image(image_path: str, out_dir: str) -> Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            img_format = img.format.lower()

    except Exception as e:
        raise ValueError(f"Error uploading image: {e}")
    
    image_id = str(uuid.uuid4())

    if out_dir:
        out_path = os.path.join(out_dir, f"{image_id}.{img_format}")
        with open(image_path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        url = out_path
    else:
        url = image_path

    image = Image(
        id=image_id,
        url=url,
        width=width,
        height=height,
        type=img_format,
        created_at=datetime.now(),
        processed_object_detection=False,
        processed_base64=False,
        processed_caption=False,
        processed_compliment=False,
        processed_tts=False,
        subject_box=None,
    )

    await image.save()
    return image