import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import csv

# File paths
# Adjust these paths as necessary
model_path = r"C:\Users\aayus\OneDrive\Documents\R6S data\converted_keras\keras_model.h5"
label_path = r"C:\Users\aayus\OneDrive\Documents\R6S data\converted_keras\labels.txt"
video_path = r"C:\Users\aayus\Downloads\Video\The Denari 1v5 - Rainbow Six Siege - Macie Jay (1080p, h264, youtube).mp4"


# Configuration
MIN_DURATION = 15.0  # Only track operators appearing for 15+ seconds (filters out noise)
HIGHLIGHT_DURATION = 25.0  # Highlight segments this duration or longer
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
VISUALIZE = True  # Show live visualization window
SAVE_OUTPUT_VIDEO = False  # Save annotated video to file
OUTPUT_VIDEO_PATH = r"C:\Users\aayus\Downloads\R6S_detected_output.mp4"


# Load model - Teachable Machine compatible approach
print("Loading model...")

# Use legacy Keras 2 behavior for Teachable Machine models
try:
    # Try loading with TF's Keras (more compatible with Teachable Machine)
    import tf_keras
    operator_model = tf_keras.models.load_model(model_path, compile=False)
    print("Loaded with tf_keras")
except ImportError:
    # Fallback: Use TensorFlow's built-in loading
    try:
        operator_model = tf.saved_model.load(model_path)
        print("Loaded as SavedModel")
    except:
        # Last resort: try standard keras with legacy mode
        operator_model = keras.models.load_model(model_path, compile=False)
        print("Loaded with keras")

with open(label_path, "r") as f:
    class_names = f.readlines()



# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {total_frames/fps:.2f}s | {fps:.2f} FPS | {w}x{h}")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames\n")

# ROI coordinates- for operator's TOP ROI
roi_left = 740
roi_right = 800
roi_top = 0
roi_bottom = 50

# Setup video writer if saving output
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
    print(f"Saving output to: {OUTPUT_VIDEO_PATH}")

# Storage
predictions = []
frame_count = 0
current_operator = "Detecting..."
current_confidence = 0.0
OPERATOR_NAME_FIX = {
    "CAPITAO": "CAPITÃO",
    "FUSE": "Fuze",
    "JAGER": "Jäger",
    "NOKK": "NØKK",
    "PLUSE": "Pulse",
    "SKOPOS": "Skopós",
    "STRIKR": "Striker",
    "TRATCHER": "Thatcher",
    "TUBARAO": "Tubarão",
    "VALKRIE": "Valkyrie"
}


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        current_time = frame_count / fps
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        if roi.size > 0:
            # Preprocess
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(roi_rgb)
            pil_image_resized = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(pil_image_resized)
            normalized = (image_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized, axis=0)

            # Predict
            prediction = operator_model.predict(data, verbose=0)
            index = np.argmax(prediction)
            class_name = class_names[index][2:].strip()
            confidence = float(prediction[0][index])

            # Update current detection for visualization
            current_operator = class_name
            current_confidence = confidence

            predictions.append({
                'time': current_time,
                'frame': frame_count,
                'operator': class_name,
                'confidence': confidence
            })

    # === VISUALIZATION ===
    if VISUALIZE or SAVE_OUTPUT_VIDEO:
        display_frame = frame.copy()

        # Draw ROI rectangle (green box around detection region)
        cv2.rectangle(display_frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

        # Draw label background
        label_text = f"{current_operator} ({current_confidence*100:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Position label below ROI
        label_x = roi_left
        label_y = roi_bottom + 25

        # Draw black background for text
        cv2.rectangle(display_frame, (label_x - 5, label_y - text_h - 5),
                      (label_x + text_w + 5, label_y + 5), (0, 0, 0), -1)

        # Draw operator name text (color based on confidence)
        if current_confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif current_confidence > 0.5:
            color = (0, 255, 255)  # Yellow for medium
        else:
            color = (0, 0, 255)  # Red for low

        cv2.putText(display_frame, label_text, (label_x, label_y), font, font_scale, color, thickness)

        # Draw timestamp
        time_text = f"Time: {frame_count/fps:.2f}s | Frame: {frame_count}/{total_frames}"
        cv2.putText(display_frame, time_text, (10, h - 20), font, 0.6, (255, 255, 255), 1)

        # Draw ROI label
        cv2.putText(display_frame, "ROI", (roi_left, roi_top - 5), font, 0.5, (0, 255, 0), 1)

        # Save to output video
        if SAVE_OUTPUT_VIDEO:
            out_writer.write(display_frame)

        # Show live preview
        if VISUALIZE:
            # Resize for display if too large
            display_h = 720
            scale = display_h / h
            display_w = int(w * scale)
            display_resized = cv2.resize(display_frame, (display_w, display_h))

            cv2.imshow('Operator Detection', display_resized)

            # Press 'q' to quit, 'space' to pause
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause until any key pressed

    frame_count += 1

cap.release()
if SAVE_OUTPUT_VIDEO:
    out_writer.release()
    print(f"\nOutput video saved to: {OUTPUT_VIDEO_PATH}")
if VISUALIZE:
    cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Analyzing operator appearances...")
print("=" * 60)

# Build segments
segments = []
current_op = None
start_time = None
last_time = None

for pred in predictions:
    if pred['operator'] != current_op:
        # Save previous segment
        if current_op and start_time is not None:
            duration = last_time - start_time
            segments.append({
                'operator': current_op,
                'start': start_time,
                'end': last_time,
                'duration': duration
            })
        # Start new segment
        current_op = pred['operator']
        start_time = pred['time']

    last_time = pred['time']

# Save final segment
if current_op and start_time is not None:
    duration = last_time - start_time
    segments.append({
        'operator': current_op,
        'start': start_time,
        'end': last_time,
        'duration': duration
    })

# Filter segments >= 2 seconds
valid_segments = [s for s in segments if s['duration'] >= MIN_DURATION]

# Build result dictionary with durations
operator_timestamps = {}
for seg in valid_segments:
    op = seg['operator']
    timestamp = f"{seg['start']:.2f}s - {seg['end']:.2f}s"

    if op not in operator_timestamps:
        operator_timestamps[op] = []
    operator_timestamps[op].append((timestamp, seg['duration']))


if operator_timestamps:
    for operator, timestamp_data in sorted(operator_timestamps.items()):
        print(f"\n{operator}:")
        for ts, duration in timestamp_data:
            if duration >= HIGHLIGHT_DURATION:
                print(f"  └─ {ts} ⭐ ({duration:.1f}s)")
            else:
                print(f"  └─ {ts} ({duration:.1f}s)")
else:
    print("No operators detected with  second duration")



