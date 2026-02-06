import cv2

from ultralytics import YOLO

import logging

import numpy as np

from collections import defaultdict

import re

from ocr_processor import initialize_reader as init_ocr

import config

import subprocess

import io

from config import LABEL_GROUPS

from tensorflow import keras


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



# Globals

model_cross = None

model_bomb = None

ocr_reader = None

operator_model = None
operator_labels = None

MODELS_LOADED = False


# Custom layer for Keras model compatibility
class CustomDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' parameter if present (not supported in DepthwiseConv2D)
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

    def get_config(self):
        # Required for model serialization
        config = super().get_config()
        return config


class TextMatcher:

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text: remove non-ASCII, special chars, normalize spaces"""
        if not text:
            return ""
        # Remove non-ASCII characters
        text = text.encode("ascii", "ignore").decode()
        # Remove special characters and underscores
        text = re.sub(r'[\W_]', ' ', text)
        # Normalize spaces and uppercase
        text = ' '.join(text.upper().split())
        return text

    @staticmethod
    def match_to_stat(text: str, region_name: str = None) -> str:
        """Match OCR text to a stat label using exact + substring matching"""
        if not text:
            return None

        normalized = TextMatcher.normalize_text(text)
        if not normalized:
            return None

        # Exact match first
        if normalized in LABEL_GROUPS:
            matched = LABEL_GROUPS[normalized]
            # Skip Death detection from non-Death ROI
            if matched == "Death" and region_name and region_name != "elimination":
                return None
            return matched

        # Substring matching (key in text OR text in key)
        for key, value in LABEL_GROUPS.items():
            if key in normalized or normalized in key:
                if len(normalized) >= 4:  # Only consider if text is reasonably long
                    # Skip Death detection from non-Death ROI
                    if value == "Death" and region_name and region_name != "elimination":
                        return None
                    return value

        return None



class ActiveNotificationTracker:

    """Tracks active notifications on screen to prevent double-counting"""



    def __init__(self, cooldown_period: float = None):

        self.cooldown_period = cooldown_period or config.COOLDOWN_PERIOD

        self.active_notifications = defaultdict(list)



    def cleanup_expired(self, current_time: float):

        """Remove notifications that have expired based on cooldown period"""

        for label in list(self.active_notifications.keys()):

            self.active_notifications[label] = [

                t for t in self.active_notifications[label]

                if current_time - t < self.cooldown_period

            ]

            if not self.active_notifications[label]:

                del self.active_notifications[label]



    def process_detections(self, detections: dict, current_time: float) -> dict:

        """

        Compare current detections against active notifications.

        Only count NEW events (detection_count > active_count).

        Returns dict of newly counted events.

        """

        self.cleanup_expired(current_time)

        newly_counted = defaultdict(int)



        for label, detection_count in detections.items():

            active_count = len(self.active_notifications[label])

            new_events = max(0, detection_count - active_count)

            for _ in range(new_events):

                self.active_notifications[label].append(current_time)

                newly_counted[label] += 1



        return newly_counted



def count_debounced_events(timestamps, debounce_sec):

    """

    Counts events but ignores any that happen within 'debounce_sec' of the previous one.

    """

    if not timestamps: return 0

    sorted_ts = sorted(set(timestamps))

    count = 0

    last_ts = -999.0

    

    for ts in sorted_ts:

        if (ts - last_ts) > debounce_sec:

            count += 1

            last_ts = ts

    return count



# --- WORKER FUNCTIONS ---

def init_worker():

    global model_cross, model_bomb, operator_model, operator_labels, ocr_reader, MODELS_LOADED

    if not MODELS_LOADED:

        logger.info("Loading models...")

        try:
            # Load YOLO models
            model_cross = YOLO(config.YOLO_CROSS_MODEL_PATH)
            model_bomb = YOLO(config.YOLO_BOMB_MODEL_PATH)
            logger.info("✓ YOLO models loaded")

            # Load Keras operator recognition model
            try:
                try:
                    operator_model = keras.models.load_model(
                        config.OPERATOR_KERAS_MODEL_PATH,
                        compile=False,
                        custom_objects={
                            'DepthwiseConv2D': CustomDepthwiseConv2D,
                            'CustomDepthwiseConv2D': CustomDepthwiseConv2D
                        },
                        safe_mode=False
                    )
                except TypeError:
                    operator_model = keras.models.load_model(
                        config.OPERATOR_KERAS_MODEL_PATH,
                        compile=False,
                        custom_objects={
                            'DepthwiseConv2D': CustomDepthwiseConv2D,
                            'CustomDepthwiseConv2D': CustomDepthwiseConv2D
                        }
                    )

                with open(config.OPERATOR_LABELS_PATH, "r") as f:
                    operator_labels = [line.strip() for line in f.readlines()]
                logger.info(f"✓ Operator model loaded ({len(operator_labels)} classes)")

            except Exception as op_error:
                logger.error(f"✗ Operator model failed: {op_error}")
                operator_model = None
                operator_labels = None

            # Load OCR
            ocr_reader = init_ocr()
            logger.info("✓ OCR reader loaded")

            MODELS_LOADED = True
            logger.info("✓ All models ready")

        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            raise



def get_roi_crop(frame, cfg_type):

    h, w = frame.shape[:2]

    if cfg_type == 'rightmost':
        cfg = config.RIGHTMOST_ROI_CONFIG

    elif cfg_type == 'operator':
        # Operator ROI using absolute pixel coordinates
        left = config.OPERATOR_ROI_LEFT
        right = config.OPERATOR_ROI_RIGHT
        top = config.OPERATOR_ROI_TOP
        bottom = config.OPERATOR_ROI_BOTTOM

        if left >= right or top >= bottom:
            return None

        return frame[top:bottom, left:right]

    else:
        cfg = config.ELIMINATION_ROI_CONFIG



    # Use proportional coordinates

    top = int(h * cfg['top'])

    bottom = int(h * cfg['bottom'])

    left = int(w * cfg['left'])

    right = int(w * cfg['right'])



    if left >= right or top >= bottom: return None

    return frame[top:bottom, left:right]



def preprocess_operator_image(roi, target_size=(224, 224)):
    """
    Preprocess ROI for Keras operator model.
    Resize and normalize the image to match training preprocessing.
    """
    if roi is None:
        return None

    # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to model's expected input size using LANCZOS interpolation
    resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Normalize to [-1, 1] range (matching Teachable Machine preprocessing)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    batch = np.expand_dims(normalized, axis=0)

    return batch


def clean_operator_label(label):
    """
    Clean operator label by removing number prefixes (e.g., '47 NOKK' -> 'NOKK').
    Labels from Teachable Machine often have format: 'NUMBER NAME'
    """
    if not label:
        return label

    # Split by space and remove leading numbers
    parts = label.strip().split()
    if len(parts) > 1 and parts[0].isdigit():
        # Remove the number prefix, join remaining parts
        return ' '.join(parts[1:])
    return label.strip()


def predict_operator(roi, confidence_threshold=None):
    """
    Predict operator from ROI using Keras model.
    Returns operator name (no confidence filtering to match test_operator.py behavior).
    """
    global operator_model, operator_labels

    if roi is None or operator_model is None or operator_labels is None:
        return None

    # Preprocess image
    preprocessed = preprocess_operator_image(roi)
    if preprocessed is None:
        return None

    # Make prediction
    predictions = operator_model.predict(preprocessed, verbose=0)
    confidence = np.max(predictions[0])
    class_index = np.argmax(predictions[0])

    # Get raw operator name and clean it (remove number prefix like "47 NOKK" -> "NOKK")
    raw_label = operator_labels[class_index].strip()
    operator_name = clean_operator_label(raw_label)

    # Apply name corrections (e.g., NOKK -> NØKK, JAGER -> Jäger)
    operator_name_upper = operator_name.upper()
    if operator_name_upper in config.OPERATOR_NAME_FIX:
        operator_name = config.OPERATOR_NAME_FIX[operator_name_upper]

    return operator_name


def get_operator_details(operator_names):
    """
    Extract weapons and abilities for detected operators from config.Weapons_details.

    Args:
        operator_names: List of operator names (e.g., ['Ash', 'Jäger', 'Sledge'])

    Returns:
        Dict with 'unique_abilities', 'primary_weapons', 'secondary_weapons'
    """
    if not operator_names:
        return {
            'unique_abilities': [],
            'primary_weapons': [],
            'secondary_weapons': []
        }

    # Create operator lookup map with normalized keys (uppercase without special chars)
    # and also keep original name mapping for exact matches
    operator_map = {}
    operator_map_normalized = {}

    def normalize_name(name):
        """Normalize operator name for case-insensitive comparison"""
        import unicodedata
        # Convert to uppercase
        normalized = name.upper()
        # Remove accents/diacritics but keep base characters
        # e.g., "Jäger" -> "JAGER", "CAPITÃO" -> "CAPITAO", "NØKK" -> "NOKK"
        normalized = ''.join(
            c for c in unicodedata.normalize('NFD', normalized)
            if unicodedata.category(c) != 'Mn'
        )
        # Handle special case for Ø -> O
        normalized = normalized.replace('Ø', 'O')
        return normalized

    for op_data in config.Weapons_details:
        name = op_data.get('Name', '')
        if name:
            # Store with original name as key (exact match priority)
            operator_map[name] = op_data
            # Also store with normalized key for fuzzy matching
            normalized_key = normalize_name(name)
            operator_map_normalized[normalized_key] = op_data

    unique_abilities = []
    primary_weapons = set()
    secondary_weapons = set()

    for operator_name in operator_names:
        # Try exact match first (preserves special characters)
        op_data = operator_map.get(operator_name)

        # If no exact match, try normalized lookup
        if not op_data:
            normalized_query = normalize_name(operator_name)
            op_data = operator_map_normalized.get(normalized_query)

        if not op_data:
            continue

        # Extract unique ability
        ability = op_data.get('Unique Ability')
        if ability and ability is not None:
            unique_abilities.append(ability)

        # Extract primary weapons
        primary = op_data.get('Primary Weapon', '')
        if primary and isinstance(primary, str):
            weapons = [w.strip() for w in primary.split(',') if w.strip()]
            primary_weapons.update(weapons)

        # Extract secondary weapons
        secondary = op_data.get('Secondary Weapon')
        if secondary and isinstance(secondary, str):
            weapons = [w.strip() for w in secondary.split(',') if w.strip()]
            secondary_weapons.update(weapons)

    return {
        'unique_abilities': unique_abilities,
        'primary_weapons': sorted(list(primary_weapons)),
        'secondary_weapons': sorted(list(secondary_weapons))
    }


def process_batch(batch_data):

    global model_cross, model_bomb, ocr_reader

    if not batch_data: return []

    

    frame_indices, frames, timestamps = zip(*batch_data)

    

    # YOLO Inference

    preds_cross = model_cross.predict(list(frames), conf=config.CONF_THRESHOLD, verbose=False)

    preds_bomb = model_bomb.predict(list(frames), conf=config.CONF_THRESHOLD, verbose=False)



    batch_results = []

    

    for i in range(len(frames)):

        frame = frames[i]

        f_idx = frame_indices[i]

        ts = timestamps[i]

        

        detections = {'events': [], 'operators': []}

        raw_ocr_texts = []

        operator_prediction = None



        # YOLO (Still running for Gadgets/Operators, but Kills ignored later)

        for box in preds_cross[i].boxes:

            detections['events'].append(model_cross.names[int(box.cls[0])].lower())

        for box in preds_bomb[i].boxes:

            cls_name = model_bomb.names[int(box.cls[0])].lower()

            detections['events'].append(cls_name)

            if cls_name in config.OPERATOR_LIST:

                detections['operators'].append(cls_name)



        # OCR Sampling - now storing (text, confidence, region) tuples

        if f_idx % config.PROCESS_EVERY_N_FRAMES == 0:

            roi = get_roi_crop(frame, 'rightmost')

            if roi is not None:

                try:

                    res = ocr_reader.readtext(roi, detail=1)

                    for _, text, conf in res:
                        raw_ocr_texts.append((text, conf, 'rightmost'))

                except: pass



        if f_idx % config.PROCESS_ELIM_FRAMES == 0:

            roi = get_roi_crop(frame, 'elimination')

            if roi is not None:

                try:

                    res = ocr_reader.readtext(roi, detail=1)

                    for _, text, conf in res:
                        raw_ocr_texts.append((text, conf, 'elimination'))

                except: pass



        # Operator Detection using Keras Model (process every N frames for accuracy)
        if f_idx % config.OPERATOR_PROCESS_EVERY_N_FRAMES == 0:
            operator_roi = get_roi_crop(frame, 'operator')
            if operator_roi is not None:
                try:
                    operator_prediction = predict_operator(operator_roi)
                except Exception as e:
                    logger.error(f"Operator detection error at {ts:.1f}s: {e}")



        batch_results.append({

            "timestamp": ts,

            "detections": detections,

            "raw_ocr": raw_ocr_texts,  # Now list of (text, conf, region) tuples

            "operator": operator_prediction  # Add operator prediction

        })

        

    return batch_results



def analyze_video(video_path):
    logger.info(f"Analyzing: {video_path}")

    init_worker()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened(): raise Exception("Could not open video file.")

    # Get video info for logging
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    logger.info(f"Video: {duration:.1f}s ({total_frames} frames @ {fps:.1f} FPS)")

    

    # --- MEMORY FIX: Stream processing instead of loading all frames ---

    results_flat = []

    current_batch = []

    

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret: break

        

        # Check skip interval

        if frame_idx % config.FRAME_SKIP_INTERVAL == 0:

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            current_batch.append((frame_idx, frame, ts))

            

            # If batch is full, process immediately and clear memory

            if len(current_batch) >= config.BATCH_SIZE:

                batch_results = process_batch(current_batch)

                results_flat.extend(batch_results)

                current_batch = [] # Critical: Free up RAM

        

        frame_idx += 1

        

    # Process any remaining frames in the last partial batch

    if current_batch:

        batch_results = process_batch(current_batch)

        results_flat.extend(batch_results)

        current_batch = []



    cap.release()

    logger.info(f"Processing {len(results_flat)} frames...")

    if not results_flat: return {}



    # --- NEW AGGREGATION LOGIC WITH ACTIVE NOTIFICATION TRACKING ---



    notification_tracker = ActiveNotificationTracker(cooldown_period=config.COOLDOWN_PERIOD)

    final_counts = defaultdict(int)

    yolo_gadget_timestamps = []

    detected_ops = set()

    operator_predictions = []  # Store operator predictions with timestamps



    for res in results_flat:

        ts = res['timestamp']



        # Operators (from YOLO bomb model)

        for op in res['detections']['operators']: detected_ops.add(op)



        # Operator Recognition (from Keras model)
        if res.get('operator'):
            operator_predictions.append({
                'time': ts,
                'operator': res['operator']
            })



        # Process OCR Events with Active Notification Tracking

        current_frame_detections = defaultdict(int)



        for ocr_item in res['raw_ocr']:

            # Handle both old format (just text) and new format (text, conf, region)
            if isinstance(ocr_item, tuple):
                text, conf, region = ocr_item
            else:
                text, conf, region = ocr_item, 1.0, None

            # Filter low confidence OCR results (matching test.py)
            if conf < 0.5:
                continue

            stat = TextMatcher.match_to_stat(text, region)

            if stat:
                current_frame_detections[stat] += 1



        # Get newly counted events (compares current detections vs active notifications)

        newly_counted = notification_tracker.process_detections(current_frame_detections, ts)



        for stat, count in newly_counted.items():
            final_counts[stat] += count



        # YOLO Gadgets (Still use timestamp debouncing for non-OCR events)

        if 'deploy_icon' in res['detections']['events']:

            yolo_gadget_timestamps.append(ts)



    # Gadgets (YOLO) - Keep existing debouncing for YOLO detections

    yolo_gadgets = count_debounced_events(yolo_gadget_timestamps, 4.0)



    # Get final kill count

    final_kills = final_counts.get('Kill', 0)



    # --- OPERATOR TIMELINE ANALYSIS ---
    logger.info("Processing operator predictions...")

    # Build segments from operator predictions
    operator_segments = []
    if operator_predictions:
        current_op = None
        start_time = None
        last_time = None

        for pred in operator_predictions:
            if pred['operator'] != current_op:
                if current_op and start_time is not None:
                    operator_segments.append({
                        'operator': current_op,
                        'start': start_time,
                        'end': last_time,
                        'duration': last_time - start_time
                    })
                current_op = pred['operator']
                start_time = pred['time']
            last_time = pred['time']

        # Save final segment
        if current_op and start_time is not None:
            operator_segments.append({
                'operator': current_op,
                'start': start_time,
                'end': last_time,
                'duration': last_time - start_time
            })

    # Filter segments by minimum duration
    valid_segments = [s for s in operator_segments if s['duration'] >= config.OPERATOR_MIN_DURATION]

    # Log all detected operators (before filtering)
    all_detected_operators = set(seg['operator'] for seg in operator_segments)
    logger.info(f"Detected operators (all): {all_detected_operators if all_detected_operators else 'None'}")

    # Build operator timestamps dictionary with durations
    operator_timestamps = {}
    for seg in valid_segments:
        op = seg['operator']
        timestamp = f"{seg['start']:.2f}s - {seg['end']:.2f}s"
        if op not in operator_timestamps:
            operator_timestamps[op] = []
        operator_timestamps[op].append({
            'timestamp': timestamp,
            'duration': seg['duration'],
            'start': seg['start'],
            'end': seg['end']
        })

    # Get list of detected operator names (only those with at least one starred/highlighted segment)
    # Starred = segment duration >= OPERATOR_HIGHLIGHT_DURATION (25s)
    keras_operators_list = [
        op for op, segments in operator_timestamps.items()
        if any(seg['duration'] >= config.OPERATOR_HIGHLIGHT_DURATION for seg in segments)
    ] if operator_timestamps else []

    # Log filtered operators (starred) - these are the final operators used
    logger.info(f"Filtered operators (stares): {set(keras_operators_list) if keras_operators_list else 'None'}")

    if keras_operators_list:
        logger.info(f"✓ Operators detected (starred): {', '.join(keras_operators_list)}")
    else:
        logger.warning("✗ No operators detected with starred segments (≥25s duration)")

    # Extract weapons and abilities for detected operators
    operator_details = get_operator_details(keras_operators_list)
    unique_abilities = operator_details['unique_abilities']
    primary_weapons = operator_details['primary_weapons']
    secondary_weapons = operator_details['secondary_weapons']
    all_weapons = primary_weapons + secondary_weapons

    if keras_operators_list:
        logger.info(f"✓ Abilities: {unique_abilities}")
        logger.info(f"✓ Weapons: {all_weapons}")

    # Calculate additional stats for LLM
    kda_ratio = round(final_kills / max(final_counts.get('Death', 1), 1), 2)
    headshot_percentage = round((final_counts.get('Headshot', 0) / max(final_kills, 1)) * 100, 1) if final_kills > 0 else 0.0

    metrics = {

        'kills': final_kills,

        'deaths': final_counts.get('Death', 0),

        'headshots': final_counts.get('Headshot', 0),

        'assists': final_counts.get('Kill Assist', 0),

        'plants': final_counts.get('Defuser Planted', 0),

        'bombs_found': final_counts.get('Bomb Found', 0),

        'drones_destroyed': final_counts.get('Drone Destroyed', 0),

        'cameras_destroyed': final_counts.get('Camera Destroyed', 0),

        'identified_enemies': final_counts.get('Enemy Identified', 0),

        'carrier_denied': final_counts.get('Carrier Denied', 0),

        'penetration': final_counts.get('Penetration', 0),

        'reinforced': final_counts.get('Reinforced', 0),



        'gadgets_deployed': yolo_gadgets,

        'operators_detected': list(detected_ops),  # YOLO detected operators

        'operators': {  # Structured operator data
            'names': keras_operators_list,
            'timestamps': operator_timestamps
        },

        'match_result': "WIN" if final_counts.get('Round Win', 0) > 0 or final_counts.get('Match Victory', 0) > 0 else "Unknown",



        'performance_score': 0.0,
        'aim_stability': 0.0,
        'kda_ratio': kda_ratio,
        'headshot_percentage': headshot_percentage,
        'duration_seconds': duration,  # Video duration in seconds

        'weapons': all_weapons,
        'unique_abilities': unique_abilities,
        'primary_weapons': primary_weapons,
        'secondary_weapons': secondary_weapons,
        'in_abilities': []

    }

    logger.info(f"✓ Results: K:{metrics['kills']} D:{metrics['deaths']} A:{metrics['assists']} HS:{metrics['headshots']}")
    logger.info("✓ Analysis complete")

    return metrics
