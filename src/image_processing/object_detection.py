import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

from typing import List

from ..models.media import Image
from ..utils.image import get_image_by_id

class ObjectDetection:
    def __init__(self,
                 model_url: str,
                 image: Image,
                 out_dir: str):
        self.model_url = model_url
        self.image = image
        self.out_dir = out_dir

    async def load_image(self, image_id: str):
        path = await get_image_by_id(image_id).url
        img = Image.open(path).convert('RGB')
        return img

    async def load_model(self, model_url: str):
        model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        model = hub.load(model_url).signatures['serving_default']
        print("TensorFlow Hub model loaded successfully.")
        return model

    async def locate_main_subject(self, image_id: str, detector_threshold: float = 0.5) -> List[int]:
        image = await get_image_by_id(image_id)
        if image.processed_object_detection == True:
            return f"Object detection already performed for image {image_id}"
        img = await self.load_image(image.url)
        detector = await self.load_model()
        img_tensor = tf.convert_to_tensor(np.array(img))
        img_tensor = tf.expand_dims(img_tensor, 0)

        results = detector(img_tensor)
        scores = results['detection_scores'].numpy()[0]
        boxes = results['detection_boxes'].numpy()[0]

        indices = np.where(scores >= detector_threshold)[0]

        if len(indices) == 0:
            return f"No objects detected in image {image_id}"
        
        top_index = indices[0]
        top_box = boxes[top_index]

        width, height = img.size
        ymin, xmin, ymax, xmax = top_box
        
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        image.subject_box = [left, top, right, bottom]
        return [left, top, right, bottom]

    async def crop_image_around_subject(self,image_id: str, out_url: str, target_width: int = 800, target_height: int = 1200) -> str:
        image = await get_image_by_id(image_id)
        if image.subject_box is None:
            return f"Image {image_id} has not been processed with object detection"
        left, top, right, bottom = image.subject_box
        img = await self.load_image(image.url)
        cropped_image = img.crop((left, top, right, bottom))
        target_size = (target_width, target_height)
        resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        resized_image.save(out_url)
        image.processed_object_detection = True
        image.object_detection_url = out_url
        return f"Image {image_id} cropped and resized around subject box"