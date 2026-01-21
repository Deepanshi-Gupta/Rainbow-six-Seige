<<<<<<< HEAD
import os







# --- Model Paths ---



YOLO_CROSS_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/Yolo11_cross.pt'



YOLO_BOMB_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/yolo11_bomb.pt'







# --- Analysis Parameters ---



CONF_THRESHOLD = 0.25



IOU_THRESHOLD = 0.45



COOLDOWN_PERIOD = 8.0







# --- LOCAL SCRIPT SETTINGS ---



PROCESS_EVERY_N_FRAMES = 5



PROCESS_ELIM_FRAMES = 10



SIMILARITY_THRESHOLD = 0.60  # From proven working script

MAX_GAP_MULTIPLIER = 2  # For continuity logic



# Master labels for event detection (from working script)

MASTER_LABELS = [

    "Kill",

    "Round Win",

    "Assist",

    "ID Enemy",

    "head shot",

    "Reinforced",

    "Carrier Denied",

    "Opponent Identified",

    "Enemy Identified",

    "Bombs Found",

    "Penetration",

    "Entry Denial Device Bonus",

    "Entry Denial Device",

    "ELIMINATED BY",

    "Barbed Wie Destroyed",

    "camera Destroyed",

    "Hidden Two Objectives",

    "Opponent Identified",

    "Drone Destroyed",

    "Bomb Found"

]



MASTER_LABELS_UP = [m.upper() for m in MASTER_LABELS]



# ROI Coordinates
# Right ROI: frame[int(h * 0.35):int(h * 0.75), int(w * 0.70):w]
# Left ROI: frame[int(h * 0.79):int(h * 0.96), int(w * 0.05):int(w * 0.25)]

RIGHTMOST_ROI_CONFIG = {
    'top': 0.35,
    'bottom': 0.75,
    'left': 0.70,
    'right': 1.0
}

ELIMINATION_ROI_CONFIG = {
    'top': 0.79,
    'bottom': 0.96,
    'left': 0.05,
    'right': 0.25
}







# --- STATS MAPPING ---



STATS_MAPPING = {



    'Death': ['ELIMINATED BY', 'KILLED BY'],



    'Kill Assist': ['kill Assist', 'KILL ASSIST'],



    'Kill': ['Kill', 'KIII','KIL'],



    'Enemy Identified': ['Opponent Identified', 'Enemy Identified', 'ID Enemy', 'Enemy Scan', 'Opponent Detected'],



    'Bomb Found': ['Bombs Found', 'Bomb Found'],



    'Drone Destroyed': ['Drone Destroyed'],



    'Camera Destroyed': ['camera Destroyed', 'Camera Destroyed'],



    'Defuser Planted': ['Defuser Planted'],



    'Headshot': ['head shot', 'headshot', 'Head Shot', 'HEADSHOT'],



    'Round Win': ['Round Win', 'ROUND WIN'],



    'Match Victory': ['MATCH VICTORY'],



    'Carrier Denied': ['CARRIER DENIED'],



    'Penetration': ['PENETRATION'],



    'Reinforced': ['REINFORCED']



}







# --- DEBOUNCE TIMER CONFIG (FIXED FOR KILLS) ---



DEBOUNCE_SECONDS = {



    'Defuser Planted': 30.0,



    'Bomb Found': 10.0,



    'Round Win': 30.0,



    'ID Enemy': 4.0,



    'Drone Destroyed': 4.0,



    'Camera Destroyed': 4.0,



    # INCREASED TO 5.0s to stop lingering text from double counting



    'Kill': 5.0,             



    'Headshot': 2.0,



    'default': 2.0



}







# --- Standard Logic Constants ---



FRAME_SKIP_INTERVAL = 2 



BATCH_SIZE = 16







OPERATOR_LIST = [



    'sledge', 'thatcher', 'ash', 'thermite', 'twitch', 'montagne', 'glaz', 'fuze', 'blitz', 'iq',



    'buck', 'blackbeard', 'capitao', 'hibana', 'jackal', 'ying', 'zofia', 'dokkaebi', 'lion', 'finka',



    'maverick', 'nomad', 'gridlock', 'nokk', 'amaru', 'kali', 'iana', 'ace', 'zero', 'flores', 'osa',



    'sens', 'grim', 'brava', 'ram', 'deimos', 'striker',



    'smoke', 'mute', 'castle', 'pulse', 'doc', 'rook', 'kapkan', 'tachanka', 'jager', 'bandit',



    'frost', 'valkyrie', 'caveira', 'echo', 'mira', 'lesion', 'ela', 'vigil', 'maestro', 'alibi',



    'clash', 'kaid', 'mozzie', 'warden', 'goyo', 'wamai', 'oryx', 'melusi', 'aruni', 'thunderbird',



    'thorn', 'azami', 'solis', 'fenrir', 'tubarao', 'sentry', 'skopos'



]

=======
import os







# --- Model Paths ---



YOLO_CROSS_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/Yolo11_cross.pt'



YOLO_BOMB_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/yolo11_bomb.pt'







# --- Analysis Parameters ---



CONF_THRESHOLD = 0.25



IOU_THRESHOLD = 0.45



COOLDOWN_PERIOD = 8.0







# --- LOCAL SCRIPT SETTINGS ---



PROCESS_EVERY_N_FRAMES = 5



PROCESS_ELIM_FRAMES = 10



SIMILARITY_THRESHOLD = 0.60  # From proven working script

MAX_GAP_MULTIPLIER = 2  # For continuity logic



# Master labels for event detection (from working script)

MASTER_LABELS = [

    "Kill",

    "Round Win",

    "Assist",

    "ID Enemy",

    "head shot",

    "Reinforced",

    "Carrier Denied",

    "Opponent Identified",

    "Enemy Identified",

    "Bombs Found",

    "Penetration",

    "Entry Denial Device Bonus",

    "Entry Denial Device",

    "ELIMINATED BY",

    "Barbed Wie Destroyed",

    "camera Destroyed",

    "Hidden Two Objectives",

    "Opponent Identified",

    "Drone Destroyed",

    "Bomb Found"

]



MASTER_LABELS_UP = [m.upper() for m in MASTER_LABELS]



# ROI Coordinates
# Right ROI: frame[int(h * 0.35):int(h * 0.75), int(w * 0.70):w]
# Left ROI: frame[int(h * 0.79):int(h * 0.96), int(w * 0.05):int(w * 0.25)]

RIGHTMOST_ROI_CONFIG = {
    'top': 0.35,
    'bottom': 0.75,
    'left': 0.70,
    'right': 1.0
}

ELIMINATION_ROI_CONFIG = {
    'top': 0.79,
    'bottom': 0.96,
    'left': 0.05,
    'right': 0.25
}







# --- STATS MAPPING ---



STATS_MAPPING = {



    'Death': ['ELIMINATED BY', 'KILLED BY'],



    'Kill Assist': ['kill Assist', 'KILL ASSIST'],



    'Kill': ['Kill', 'KIII','KIL'],



    'Enemy Identified': ['Opponent Identified', 'Enemy Identified', 'ID Enemy', 'Enemy Scan', 'Opponent Detected'],



    'Bomb Found': ['Bombs Found', 'Bomb Found'],



    'Drone Destroyed': ['Drone Destroyed'],



    'Camera Destroyed': ['camera Destroyed', 'Camera Destroyed'],



    'Defuser Planted': ['Defuser Planted'],



    'Headshot': ['head shot', 'headshot', 'Head Shot', 'HEADSHOT'],



    'Round Win': ['Round Win', 'ROUND WIN'],



    'Match Victory': ['MATCH VICTORY'],



    'Carrier Denied': ['CARRIER DENIED'],



    'Penetration': ['PENETRATION'],



    'Reinforced': ['REINFORCED']



}







# --- DEBOUNCE TIMER CONFIG (FIXED FOR KILLS) ---



DEBOUNCE_SECONDS = {



    'Defuser Planted': 30.0,



    'Bomb Found': 10.0,



    'Round Win': 30.0,



    'ID Enemy': 4.0,



    'Drone Destroyed': 4.0,



    'Camera Destroyed': 4.0,



    # INCREASED TO 5.0s to stop lingering text from double counting



    'Kill': 5.0,             



    'Headshot': 2.0,



    'default': 2.0



}







# --- Standard Logic Constants ---



FRAME_SKIP_INTERVAL = 2 



BATCH_SIZE = 16







OPERATOR_LIST = [



    'sledge', 'thatcher', 'ash', 'thermite', 'twitch', 'montagne', 'glaz', 'fuze', 'blitz', 'iq',



    'buck', 'blackbeard', 'capitao', 'hibana', 'jackal', 'ying', 'zofia', 'dokkaebi', 'lion', 'finka',



    'maverick', 'nomad', 'gridlock', 'nokk', 'amaru', 'kali', 'iana', 'ace', 'zero', 'flores', 'osa',



    'sens', 'grim', 'brava', 'ram', 'deimos', 'striker',



    'smoke', 'mute', 'castle', 'pulse', 'doc', 'rook', 'kapkan', 'tachanka', 'jager', 'bandit',



    'frost', 'valkyrie', 'caveira', 'echo', 'mira', 'lesion', 'ela', 'vigil', 'maestro', 'alibi',



    'clash', 'kaid', 'mozzie', 'warden', 'goyo', 'wamai', 'oryx', 'melusi', 'aruni', 'thunderbird',



    'thorn', 'azami', 'solis', 'fenrir', 'tubarao', 'sentry', 'skopos'



]

>>>>>>> 1208943 (add)
# yet to add mapping - which operators fall under attackers and which under defenders