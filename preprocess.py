import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define input and output directories
INPUT_DIR = 'images/input_photos'      # Replace with your input directory path
OUTPUT_DIR = 'images/processed_photos' # Replace with your desired output directory path

# Define target resolution
TARGET_WIDTH = 800
TARGET_HEIGHT = 800
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the pre-trained object detection model from TensorFlow Hub
print("Loading the TensorFlow Hub model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2").signatures['serving_default']
print("Model loaded successfully.")

# Function to load and preprocess image
def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

# Function to perform object detection and get the most probable box
def get_main_subject_box(img, detector, threshold=0.5):
    # Convert image to tensor
    img_tensor = tf.convert_to_tensor(np.array(img))
    img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension

    # Run the detector
    results = detector(img_tensor)

    # Extract detection scores and boxes
    scores = results['detection_scores'].numpy()[0]
    boxes = results['detection_boxes'].numpy()[0]

    # Filter out detections below the threshold
    indices = np.where(scores >= threshold)[0]

    if len(indices) == 0:
        return None  # No detections above the threshold

    # Select the detection with the highest score
    top_index = indices[0]
    top_box = boxes[top_index]  # Box coordinates are in ymin, xmin, ymax, xmax format

    # Convert normalized coordinates to pixel coordinates
    width, height = img.size
    ymin, xmin, ymax, xmax = top_box
    left = int(xmin * width)
    right = int(xmax * width)
    top = int(ymin * height)
    bottom = int(ymax * height)

    return (left, top, right, bottom)

# Process each image in the input directory
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing {filename}...")

        # Load image
        image = load_image(input_path)

        # Get the bounding box of the main subject
        box = get_main_subject_box(image, detector)

        if box:
            # Crop the image to the bounding box
            cropped_image = image.crop(box)
            print(f" - Cropped to box: {box}")
        else:
            # If no subject detected, use the original image
            cropped_image = image
            print(" - No subject detected. Using the original image.")

        # Resize the image to the target resolution
        resized_image = cropped_image.resize(TARGET_SIZE, Image.ANTIALIAS)

        # Save the processed image
        output_path = os.path.join(OUTPUT_DIR, filename)
        resized_image.save(output_path)
        print(f" - Saved processed image to {output_path}\n")

print("Processing completed.")