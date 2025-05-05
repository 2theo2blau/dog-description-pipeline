import cv2
import numpy as np
import time
import os
import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image # Import PIL
from deep_sort_realtime.deepsort_tracker import DeepSort # Import DeepSORT

# --- Configuration ---
INPUT_VIDEO_PATH = 'demo/dogpark.mp4'  # Replace with your input video file path
OUTPUT_VIDEO_PATH = 'output_tracked_pytorch.mp4' # Path for the output video with tracking
# MODEL_WEIGHTS = 'frozen_inference_graph.pb' # Path to TensorFlow model weights (.pb) # Removed
# MODEL_CONFIG = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # Path to model config (.pbtxt) # Removed
# CLASSES_FILE = 'coco_labels.txt' # Path to file with class names (one per line) # Removed
CONFIDENCE_THRESHOLD = 0.5 # Minimum probability to filter weak detections
# NMS_THRESHOLD = 0.4 # Non-maximum suppression threshold (Handled by detector/tracker)
# TARGET_WIDTH = 320 # Input width expected by the model (Handled by PyTorch model transforms)
# TARGET_HEIGHT = 320 # Input height expected by the model (Handled by PyTorch model transforms)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
# BOX_COLOR = (0, 255, 0) # Green for bounding box (Will be dynamic for tracks)
TEXT_COLOR = (0, 0, 0) # Black for text background
TEXT_COLOR_FG = (255, 255, 255) # White for text foreground
TEXT_THICKNESS = 1
BOX_THICKNESS = 2

# --- DeepSORT Configuration ---
# Refer to deep-sort-realtime documentation for more options
# https://pypi.org/project/deep-sort-realtime/
# https://github.com/levan92/deep_sort_realtime
DEEPSORT_MAX_AGE = 30 # Number of frames to keep a track alive without new detections

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

def process_frame_and_track(frame, model, class_names, preprocess, device, frame_height, frame_width, tracker):
    """Processes a single video frame for object detection and tracking."""
    detect_start_time = time.time()

    # Preprocess the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(pil_image)
    input_batch = [input_tensor.to(device)]

    # Perform inference
    with torch.no_grad():
        predictions = model(input_batch)

    # Post-process detections
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    detect_processing_time = time.time() - detect_start_time

    # --- Format detections for DeepSORT ---
    # DeepSORT expects detections in the format:
    # [ [left,top,w,h], confidence, class_id ]
    detections_for_tracker = []
    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            # Convert box format [xmin, ymin, xmax, ymax] to [left, top, w, h]
            (startX, startY, endX, endY) = boxes[i].astype("int")

            # Ensure coordinates are within frame bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(frame_width, endX)
            endY = min(frame_height, endY)

            width = endX - startX
            height = endY - startY

            # Skip zero-area boxes
            if width <= 0 or height <= 0:
                continue

            # Append formatted detection
            detections_for_tracker.append(
                ([startX, startY, width, height], scores[i], labels[i])
            )

    # --- Update DeepSORT Tracker ---
    track_start_time = time.time()
    # The 'frame' argument is used by DeepSORT's ReID model to extract appearance features
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
    track_processing_time = time.time() - track_start_time

    # --- Draw results on the frame ---
    for track in tracks:
        if not track.is_confirmed():
            continue # Skip tracks that are not yet confirmed

        track_id = track.track_id
        ltrb = track.to_ltrb() # Get bounding box in (left, top, right, bottom) format
        class_id = track.get_det_class() # Get the class ID associated with the track
        score = track.get_det_conf() # Get the confidence score

        # Get class name (handle potential index issues)
        class_name = class_names[class_id] if class_id is not None and class_id < len(class_names) else f"Class_{class_id}"

        # Convert to integer coordinates
        l, t, r, b = map(int, ltrb)

        # --- Customizable Text and Color ---
        label = f"ID:{track_id} {class_name}"
        # Generate a unique color for each track ID
        color = (int(track_id) * 23 % 255, int(track_id) * 57 % 255, int(track_id) * 83 % 255) # Simple hash for color

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), color, BOX_THICKNESS)

        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)

        # Draw background rectangle for text (slightly above the box)
        text_y = max(t - 5, text_height + 5) # Position text above box, ensure it's within frame top
        cv2.rectangle(frame, (l, text_y - text_height - baseline), (l + text_width, text_y), color, -1) # -1 fills the rectangle

        # Draw text
        cv2.putText(frame, label, (l, text_y - baseline // 2), FONT, FONT_SCALE, TEXT_COLOR_FG, TEXT_THICKNESS, cv2.LINE_AA)

    # Display processing times on frame (optional)
    total_time = detect_processing_time + track_processing_time
    fps_text = f"FPS: {1 / total_time:.2f} (Det: {detect_processing_time*1000:.0f}ms, Track: {track_processing_time*1000:.0f}ms)"
    cv2.putText(frame, fps_text, (10, 30), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

# --- Main Execution ---
if __name__ == "__main__":
    # Load detection model and classes using PyTorch Hub
    model, class_names, preprocess, device = load_model()

    if model is None or class_names is None or preprocess is None:
        print("[ERROR] Exiting due to detection model loading errors.")
        exit()

    # Initialize DeepSORT tracker
    print("[INFO] Initializing DeepSORT tracker...")
    try:
        # You can configure the tracker here (e.g., max_age, n_init, embedder model)
        # Check deep-sort-realtime docs for options: https://github.com/levan92/deep_sort_realtime
        tracker = DeepSort(max_age=DEEPSORT_MAX_AGE)
        print("[INFO] DeepSORT tracker initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize DeepSORT tracker: {e}")
        print("Please ensure 'deep-sort-realtime' is installed (`pip install deep-sort-realtime`)")
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
        # Process the frame using the detector and tracker
        tracked_frame = process_frame_and_track(frame, model, class_names, preprocess, device, prop_height, prop_width, tracker)

        # Write the output frame to disk
        writer.write(tracked_frame)

        # Display progress (optional)
        if frame_count % 50 == 0:
             elapsed_time = time.time() - start_total_time
             avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
             print(f"[INFO] Processed {frame_count} frames... (Avg FPS: {avg_fps:.2f})")


        # --- Optional: Display the output frame ---
        # cv2.imshow("Output", tracked_frame)
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