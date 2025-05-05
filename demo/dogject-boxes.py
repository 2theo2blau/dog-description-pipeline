import cv2
import numpy as np
import time
import os
import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image # Import PIL

# --- Configuration ---
INPUT_VIDEO_PATH = 'demo/dogpark.mp4'  # Replace with your input video file path
OUTPUT_VIDEO_PATH = 'output_detected_pytorch.mp4' # Path for the output video using PyTorch
# MODEL_WEIGHTS = 'frozen_inference_graph.pb' # Path to TensorFlow model weights (.pb) # Removed
# MODEL_CONFIG = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # Path to model config (.pbtxt) # Removed
# CLASSES_FILE = 'coco_labels.txt' # Path to file with class names (one per line) # Removed
CONFIDENCE_THRESHOLD = 0.5 # Minimum probability to filter weak detections
NMS_THRESHOLD = 0.4 # Non-maximum suppression threshold (Note: PyTorch model might handle NMS internally or differently)
# TARGET_WIDTH = 320 # Input width expected by the model (Handled by PyTorch model transforms)
# TARGET_HEIGHT = 320 # Input height expected by the model (Handled by PyTorch model transforms)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
BOX_COLOR = (0, 255, 0) # Green for bounding box
TEXT_COLOR = (0, 0, 0) # Black for text background
TEXT_COLOR_FG = (255, 255, 255) # White for text foreground
TEXT_THICKNESS = 1
BOX_THICKNESS = 2

# --- Helper Functions ---

def load_model():
    """Loads the pre-trained object detection model using PyTorch Hub."""
    print("[INFO] Determining device (CUDA or CPU)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading model from PyTorch Hub (SSDLite MobileNetV3 Large)...")
    try:
        # Load the model and weights
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT # Use default weights (COCO)
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        model.eval() # Set model to evaluation mode
        model.to(device) # Move model to the determined device
        print("[INFO] Model loaded successfully.")

        # Get preprocessing transforms and class names from weights
        preprocess = weights.transforms()
        class_names = weights.meta["categories"]
        print(f"[INFO] Loaded {len(class_names)} classes.")

        return model, class_names, preprocess, device
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("Please ensure you have torchvision installed and internet connectivity.")
        return None, None, None, None

# def load_classes(classes_path): # No longer needed, classes come from model weights
#     """Loads class names from a file."""
#     print("[INFO] Loading class names...")
#     if not os.path.exists(classes_path):
#         print(f"[ERROR] Classes file not found: {classes_path}")
#         return None
#     try:
#         with open(classes_path, 'rt') as f:
#             class_names = f.read().rstrip('\n').split('\n')
#         print(f"[INFO] Loaded {len(class_names)} classes.")
#         return class_names
#     except Exception as e:
#         print(f"[ERROR] Failed to load classes: {e}")
#         return None

def process_frame(frame, model, class_names, preprocess, device, frame_height, frame_width):
    """Processes a single video frame for object detection using PyTorch."""
    start_time = time.time()

    # Preprocess the frame
    # 1. Convert BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 1.5 Convert NumPy array to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    # 2. Apply model-specific preprocessing (e.g., ToTensor, normalization)
    input_tensor = preprocess(pil_image) # Pass PIL image
    # 3. Add batch dimension and move to device
    input_batch = [input_tensor.to(device)]

    # Perform inference
    with torch.no_grad(): # No need to track gradients
        predictions = model(input_batch)

    # Post-process detections
    # Predictions are a list (for batch size 1, it's a list with one element)
    # Each element is a dict with 'boxes', 'labels', 'scores'
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy() # Move to CPU and convert to numpy
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    processing_time = time.time() - start_time

    # --- Draw results on the frame ---
    # Filter detections based on confidence threshold
    detections_to_draw = []
    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            # Convert box format if necessary (PyTorch gives [xmin, ymin, xmax, ymax])
            (startX, startY, endX, endY) = boxes[i].astype("int")

            # Ensure the bounding box coordinates are within the frame dimensions (already scaled correctly by model)
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(frame_width, endX)
            endY = min(frame_height, endY)

            detections_to_draw.append({
                'box': (startX, startY, endX - startX, endY - startY), # Convert to (x, y, w, h)
                'score': scores[i],
                'label_idx': labels[i]
            })

    # Note: NMS is typically handled by the torchvision model internally,
    # but you could apply cv2.dnn.NMSBoxes here again if needed,
    # using the 'box' (in x,y,w,h format) and 'score' from detections_to_draw.
    # We will rely on the model's internal NMS for now.

    if len(detections_to_draw) > 0:
        for det in detections_to_draw:
            # Get box coordinates (x, y, w, h)
            (x, y, w, h) = det['box']

            # Get class name and confidence
            class_id = det['label_idx']
            confidence = det['score']
            # PyTorch class indices might be 1-based or have a background class, adjust if needed
            # For COCO loaded via torchvision, indices usually align well, but check model docs.
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"


            # --- Customizable Text ---
            label = f"{class_name}: {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)

            # Draw background rectangle for text (slightly above the box)
            text_y = max(y - 5, text_height + 5) # Position text above box, ensure it's within frame top
            cv2.rectangle(frame, (x, text_y - text_height - baseline), (x + text_width, text_y), TEXT_COLOR, -1) # -1 fills the rectangle

            # Draw text
            cv2.putText(frame, label, (x, text_y - baseline // 2), FONT, FONT_SCALE, TEXT_COLOR_FG, TEXT_THICKNESS, cv2.LINE_AA)

    # Display processing time on frame (optional)
    fps_text = f"FPS: {1 / processing_time:.2f}"
    cv2.putText(frame, fps_text, (10, 30), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

# --- Main Execution ---
if __name__ == "__main__":
    # Load model and classes using PyTorch Hub
    model, class_names, preprocess, device = load_model()
    # class_names = load_classes(CLASSES_FILE) # Removed

    if model is None or class_names is None or preprocess is None:
        print("[ERROR] Exiting due to loading errors.")
        exit()

    # Open video capture
    print(f"[INFO] Opening video file: {INPUT_VIDEO_PATH}")
    vs = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not vs.isOpened():
        print(f"[ERROR] Could not open video file: {INPUT_VIDEO_PATH}")
        exit()

    # Get video properties
    try:
        prop_fps = vs.get(cv2.CAP_PROP_FPS)
        prop_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        prop_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prop_frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video properties: {prop_width}x{prop_height} @ {prop_fps:.2f} FPS, {prop_frame_count} frames")
    except Exception as e:
        print(f"[WARN] Could not get all video properties: {e}")
        prop_fps = 30 # Default FPS if reading fails
        prop_width = 640 # Default width
        prop_height = 480 # Default height

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, prop_fps, (prop_width, prop_height), True)
    if not writer.isOpened():
        print(f"[ERROR] Could not open video writer for path: {OUTPUT_VIDEO_PATH}")
        vs.release()
        exit()

    print(f"[INFO] Processing video and saving to: {OUTPUT_VIDEO_PATH}")
    frame_count = 0
    start_total_time = time.time()

    # Loop over frames from the video file stream
    while True:
        # Read the next frame from the file
        (grabbed, frame) = vs.read()

        # If the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        frame_count += 1
        # Process the frame using the PyTorch model
        processed_frame = process_frame(frame, model, class_names, preprocess, device, prop_height, prop_width)

        # Write the output frame to disk
        writer.write(processed_frame)

        # Display progress (optional)
        if frame_count % 50 == 0:
             elapsed_time = time.time() - start_total_time
             avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
             print(f"[INFO] Processed {frame_count} frames... (Avg FPS: {avg_fps:.2f})")


        # --- Optional: Display the output frame ---
        # cv2.imshow("Output", processed_frame)
        # key = cv2.waitKey(1) & 0xFF
        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
        # --- End Optional Display ---


    # Release resources
    end_total_time = time.time()
    print(f"[INFO] Finished processing {frame_count} frames.")
    print(f"[INFO] Total processing time: {end_total_time - start_total_time:.2f} seconds")
    print("[INFO] Releasing video pointers...")
    vs.release()
    writer.release()
    # cv2.destroyAllWindows() # Uncomment if using cv2.imshow

    print(f"[INFO] Output video saved to: {OUTPUT_VIDEO_PATH}")

