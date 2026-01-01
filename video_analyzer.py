import cv2

from ultralytics import YOLO

import logging

import numpy as np

from difflib import SequenceMatcher

from collections import defaultdict

from ocr_processor import initialize_reader as init_ocr

import config

import subprocess

import io



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



# Globals

model_cross = None

model_bomb = None

ocr_reader = None

MODELS_LOADED = False



# --- HELPER CLASSES ---

class TextMatcher:

    @staticmethod

    def calculate_similarity(text_a: str, text_b: str) -> float:

        return SequenceMatcher(None, text_a, text_b).ratio()



    @staticmethod

    def match_to_stat(text: str) -> str:

        if not text: return None

        text_upper = text.strip().upper()


        best_score = 0.0

        best_stat = None

        for stat, patterns in config.STATS_MAPPING.items():

            for pattern in patterns:

                score = TextMatcher.calculate_similarity(text_upper, pattern.upper())

                if score > best_score:

                    best_score = score

                    best_stat = stat



        if best_score >= config.SIMILARITY_THRESHOLD:

            return best_stat

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

    global model_cross, model_bomb, ocr_reader, MODELS_LOADED

    if not MODELS_LOADED:

        logger.info("Initializing R6 models & OCR...")

        try:

            model_cross = YOLO(config.YOLO_CROSS_MODEL_PATH)

            model_bomb = YOLO(config.YOLO_BOMB_MODEL_PATH)

            ocr_reader = init_ocr()

            MODELS_LOADED = True

        except Exception as e:

            logger.error(f"Failed to load R6 models: {e}")

            raise e



def get_roi_crop(frame, cfg_type):

    h, w = frame.shape[:2]

    if cfg_type == 'rightmost':

        cfg = config.RIGHTMOST_ROI_CONFIG

    else:

        cfg = config.ELIMINATION_ROI_CONFIG



    # Use proportional coordinates

    top = int(h * cfg['top'])

    bottom = int(h * cfg['bottom'])

    left = int(w * cfg['left'])

    right = int(w * cfg['right'])



    if left >= right or top >= bottom: return None

    return frame[top:bottom, left:right]



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



        # YOLO (Still running for Gadgets/Operators, but Kills ignored later)

        for box in preds_cross[i].boxes:

            detections['events'].append(model_cross.names[int(box.cls[0])].lower())

        for box in preds_bomb[i].boxes:

            cls_name = model_bomb.names[int(box.cls[0])].lower()

            detections['events'].append(cls_name)

            if cls_name in config.OPERATOR_LIST:

                detections['operators'].append(cls_name)



        # OCR Sampling

        if f_idx % config.PROCESS_EVERY_N_FRAMES == 0:

            roi = get_roi_crop(frame, 'rightmost')

            if roi is not None:

                try:

                    res = ocr_reader.readtext(roi, detail=1)

                    for _, text, conf in res: raw_ocr_texts.append(text)

                except: pass



        if f_idx % config.PROCESS_ELIM_FRAMES == 0:

            roi = get_roi_crop(frame, 'elimination')

            if roi is not None:

                try:

                    res = ocr_reader.readtext(roi, detail=1)

                    for _, text, conf in res: raw_ocr_texts.append(text)

                except: pass



        batch_results.append({

            "timestamp": ts,

            "detections": detections,

            "raw_ocr": raw_ocr_texts

        })

        

    return batch_results



def analyze_video(video_path):

    init_worker()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened(): raise Exception("Could not open video file.")

    

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



    if not results_flat: return {}



    # --- NEW AGGREGATION LOGIC WITH ACTIVE NOTIFICATION TRACKING ---



    notification_tracker = ActiveNotificationTracker(cooldown_period=config.COOLDOWN_PERIOD)

    final_counts = defaultdict(int)

    yolo_gadget_timestamps = []

    detected_ops = set()



    for res in results_flat:

        ts = res['timestamp']



        # Operators

        for op in res['detections']['operators']: detected_ops.add(op)



        # Process OCR Events with Active Notification Tracking

        current_frame_detections = defaultdict(int)



        for text in res['raw_ocr']:

            stat = TextMatcher.match_to_stat(text)

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

    logger.info(f"Using Active Notification Tracking - Kills: {final_kills}")



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

        'operators_detected': list(detected_ops),

        'match_result': "WIN" if final_counts.get('Round Win', 0) > 0 or final_counts.get('Match Victory', 0) > 0 else "Unknown",

        

        'performance_score': 0.0, 'aim_stability': 0.0,

        'weapons': [], 'unique_abilities': [], 'in_abilities': []

    }



    logger.info(f"Final Metrics: {metrics}")

    return metrics