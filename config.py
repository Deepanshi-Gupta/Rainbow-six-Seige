import os

# --- Chunking Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
CHUNK_EXPIRY = 3600  # Chunks expire after 1 hour
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk


# --- Model Paths ---

YOLO_CROSS_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/Yolo11_cross.pt'
YOLO_BOMB_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/yolo11_bomb.pt'

##Operator Recognition Model Path
OPERATOR_KERAS_MODEL_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/models/operator_model.keras'
OPERATOR_LABELS_PATH = '/home/ubuntu/SlowfastProject/Playformance/r6s/labels.txt'


# --- Operator Recognition Parameters ---

# ROI coordinates for operator's TOP ROI (absolute pixel values)
OPERATOR_ROI_LEFT = 740
OPERATOR_ROI_RIGHT = 800
OPERATOR_ROI_TOP = 0
OPERATOR_ROI_BOTTOM = 50


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

OPERATOR_MIN_DURATION = 10.0   
OPERATOR_HIGHLIGHT_DURATION = 25.0   

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
COOLDOWN_PERIOD = 8.0
PROCESS_EVERY_N_FRAMES = 10
PROCESS_ELIM_FRAMES = 10

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


# --- HELPER CLASSES ---
LABEL_GROUPS = {
    "OPPONENT IDENTIFIED": "Enemy Identified",
    "ENEMY IDENTIFIED": "Enemy Identified",
    "ID ENEMY": "Enemy Identified",
    "OPPONENT DETECTED": "Enemy Identified",
    "ELIMINATED BY": "Death",
    "KILLED BY": "Death",
    "KIL": "Kill",
    "KIII": "Kill",
    "KILL": "Kill",
    "HEADSHOT": "Headshot",
    "HEAD SHOT": "Headshot",
    "KILL ASSIST": "Kill Assist",
    "BOMBS FOUND": "Bomb Found",
    "DRONE DESTROYED": "Drone Destroyed",
    "CAMERA DESTROYED": "Camera Destroyed",
    "DEFUSER PLANTED": "Defuser Planted",
    "ROUND WIN": "Round Win",
    "MATCH VICTORY": "Match Victory",
    "CARRIER DENIED": "Carrier Denied",
    "PENETRATION": "Penetration",
    "REINFORCED": "Reinforced"
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

Weapons_details = [
    {
        "Name": "Denari",
        "Unique Ability": "T.R.I.P. Connector",
        "Primary Weapon": "SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Fenrir",
        "Unique Ability": "F-Natt Dread Mine",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Solis",
        "Unique Ability": "Spec-IO Electro-Sensor",
        "Primary Weapon": "SHOTGUN, SUBMACHINE GUN",
        "Secondary Weapon": "Machine Pistol"
    },
    {
        "Name": "Azami",
        "Unique Ability": "KIBA BARRIER",
        "Primary Weapon": "SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Thorn",
        "Unique Ability": "RAZORBLOOM SHELL",
        "Primary Weapon": "SHOTGUN",
        "Secondary Weapon": "HANDGUN, Machine Pistol"
    },
    {
        "Name": "Thunderbird",
        "Unique Ability": "KONA STATION",
        "Primary Weapon": "SHOTGUN, ASSAULT RIFLE",
        "Secondary Weapon": "Machine Pistol, HANDGUN, SHOTGUN"
    },
    {
        "Name": "Aruni",
        "Unique Ability": "Surya Gate",
        "Primary Weapon": "SUBMACHINE GUN, Marksman Rifle",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Melusi",
        "Unique Ability": "BANSHEE SONIC DEFENSE",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "SHOTGUN, HANDGUN"
    },
    {
        "Name": "Oryx",
        "Unique Ability": "REMAH DASH",
        "Primary Weapon": "SHOTGUN,SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Wamai",
        "Unique Ability": "Mag-net System",
        "Primary Weapon": "ASSAULT RIFLE,SUBMACHINE GUN",
        "Secondary Weapon": "SHOTGUN, HANDGUN"
    },
    {
        "Name": "Goyo",
        "Unique Ability": "Volcan Canister",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Warden",
        "Unique Ability": "Glance Smart Glasses",
        "Primary Weapon": "SHOTGUN, SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN,Machine Pistol"
    },
    {
        "Name": "Mozzie",
        "Unique Ability": "Pest Launcher",
        "Primary Weapon": "ASSAULT RIFLE, SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Kaid",
        "Unique Ability": "\"Rtila\" Electroclaw",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Clash",
        "Unique Ability": "CCE SHIELD MK2",
        "Primary Weapon": "CCE SHIELD MK2",
        "Secondary Weapon": "HANDGUN,Machine Pistol,SHOTGUN"
    },
    {
        "Name": "Maestro",
        "Unique Ability": "Evil Eye",
        "Primary Weapon": "Light Machine Gun, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Alibi",
        "Unique Ability": "Prisma",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Vigil",
        "Unique Ability": "ERC-7",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "Machine Pistol"
    },
    {
        "Name": "Ela",
        "Unique Ability": "Grzmot Mine",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Lesion",
        "Unique Ability": "GU",
        "Primary Weapon": " SHOTGUN, SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN,SHOTGUN"
    },
    {
        "Name": "Mira",
        "Unique Ability": "Black mirror",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN, SHOTGUN"
    },
    {
        "Name": "Echo",
        "Unique Ability": "Yokai",
        "Primary Weapon": "SUBMACHINE GUN,SHOTGUN",
        "Secondary Weapon": "Machine Pistol, HANDGUN"
    },
    {
        "Name": "Caveira",
        "Unique Ability": "Silent Step",
        "Primary Weapon": "SUBMACHINE GUN,SHOTGUN",
        "Secondary Weapon": "Luison"
    },
    {
        "Name": "Valkyrie",
        "Unique Ability": "Black Eye",
        "Primary Weapon": "SUBMACHINE GUN,SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Frost",
        "Unique Ability": "Welcome Mat",
        "Primary Weapon": "SHOTGUN, SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN, SHOTGUN"
    },
    {
        "Name": "Mute",
        "Unique Ability": "Signal Disruptor",
        "Primary Weapon": " SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN, Machine Pistol"
    },
    {
        "Name": "Smoke",
        "Unique Ability": "Remote Gas Grenade",
        "Primary Weapon": " SHOTGUN,SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN, Machine Pistol"
    },
    {
        "Name": "Castle",
        "Unique Ability": "Armor Panel",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "SHOTGUN, HANDGUN"
    },
    {
        "Name": "Pulse",
        "Unique Ability": "Cardiac Sensor",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Doc",
        "Unique Ability": "Stim Pistol",
        "Primary Weapon": "SUBMACHINE GUN,SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Rook",
        "Unique Ability": "Armor Pack",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Bandit",
        "Unique Ability": "Shock Wire",
        "Primary Weapon": "SUBMACHINE GUN,SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Tachanka",
        "Unique Ability": "SHUMIKHA LAUNCHER",
        "Primary Weapon": "Light Machine Gun,SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN,Machine Pistol"
    },
    {
        "Name": "Kapkan",
        "Unique Ability": "Entry Denial Device",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Skopos",
        "Unique Ability": "V10 Pantheon Shells",
        "Primary Weapon": "ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Jäger",
        "Unique Ability": "ACTIVE DEFENSE",
        "Primary Weapon": "SHOTGUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Tubarão",
        "Unique Ability": "Zoto Canister",
        "Primary Weapon": "SUBMACHINE GUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Sentry",
        "Unique Ability": None,
        "Primary Weapon": "ASSAULT RIFLE,SHOTGUN",
        "Secondary Weapon": "Machine Pistol, SHOTGUN"
    },
    {
        "Name": "Rauora",
        "Unique Ability": "D.O.M. Panel Launcher",
        "Primary Weapon": "Marksman Rifle, Light Machine Gun",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Ram",
        "Unique Ability": "BU-GI AUTO BREACHER",
        "Primary Weapon": "Light Machine Gun, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Brava",
        "Unique Ability": "Kludge Drone",
        "Primary Weapon": "ASSAULT RIFLE,Marksman Rifle",
        "Secondary Weapon": "HANDGUN,SHOTGUN"
    },
    {
        "Name": "Grim",
        "Unique Ability": "Kawan Hive Launcher",
        "Primary Weapon": "ASSAULT RIFLE, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Sens",
        "Unique Ability": "R.O.U. Projector System",
        "Primary Weapon": "Marksman Rifle,ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Osa",
        "Unique Ability": "TALON-8 CLEAR SHIELD",
        "Primary Weapon": "ASSAULT RIFLE, SUBMACHINE GUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Flores",
        "Unique Ability": "RCE-RATERO CHARGE",
        "Primary Weapon": "ASSAULT RIFLE,Marksman Rifle",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Zero",
        "Unique Ability": "ARGUS LAUNCHER",
        "Primary Weapon": "SUBMACHINE GUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN, Hand Cannon"
    },
    {
        "Name": "Ace",
        "Unique Ability": "S.E.L.M.A. AQUA BREACHER",
        "Primary Weapon": "ASSAULT RIFLE, SHOTGUN",
        "Secondary Weapon": " HANDGUN"
    },
    {
        "Name": "Iana",
        "Unique Ability": "GEMINI REPLICATOR",
        "Primary Weapon": "ASSAULT RIFLE",
        "Secondary Weapon": "Hand Cannon, HANDGUN"
    },
    {
        "Name": "Kali",
        "Unique Ability": "LV Explosive Lance",
        "Primary Weapon": "Marksman Rifle",
        "Secondary Weapon": "Machine Pistol, HANDGUN"
    },
    {
        "Name": "Amaru",
        "Unique Ability": "Garra Hook",
        "Primary Weapon": "Light Machine Gun,SHOTGUN",
        "Secondary Weapon": "Hand Cannon, Machine Pistol, SHOTGUN"
    },
    {
        "Name": "Gridlock",
        "Unique Ability": "Trax Stingers",
        "Primary Weapon": "ASSAULT RIFLE SAW, Light Machine Gun",
        "Secondary Weapon": "SHOTGUN, HANDGUN"
    },
    {
        "Name": "Nomad",
        "Unique Ability": "Airjab Launcher",
        "Primary Weapon": " ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Maverick",
        "Unique Ability": "Breaching Torch",
        "Primary Weapon": "ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Lion",
        "Unique Ability": "EE-ONE-D",
        "Primary Weapon": "Marksman Rifle, SHOTGUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Finka",
        "Unique Ability": "Adrenal Surge",
        "Primary Weapon": "Light Machine Gun, SHOTGUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Dokkaebi",
        "Unique Ability": "Logic Bomb",
        "Primary Weapon": "Marksman Rifle, SHOTGUN",
        "Secondary Weapon": "Hand Cannon,Machine Pistol"
    },
    {
        "Name": "Zofia",
        "Unique Ability": "KS79 Lifeline",
        "Primary Weapon": "Light Machine Gun,ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Ying",
        "Unique Ability": "Candela",
        "Primary Weapon": "Light Machine Gun",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Jackal",
        "Unique Ability": "Eyenox Model III",
        "Primary Weapon": "ASSAULT RIFLE, SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN, SHOTGUN"
    },
    {
        "Name": "Hibana",
        "Unique Ability": "X-Kairos",
        "Primary Weapon": "ASSAULT RIFLE,SHOTGUN",
        "Secondary Weapon": "HANDGUN, Machine Pistol"
    },
    {
        "Name": "Blackbeard",
        "Unique Ability": "H.U.L.L. ADAPTABLE SHIELD",
        "Primary Weapon": "ASSAULT RIFLE,Marksman Rifle",
        "Secondary Weapon": None
    },
    {
        "Name": "Buck",
        "Unique Ability": "Skeleton Key",
        "Primary Weapon": "ASSAULT RIFLE,Marksman Rifle",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Sledge",
        "Unique Ability": "BREACHING HAMMER",
        "Primary Weapon": "ASSAULT RIFLE,  SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Thatcher",
        "Unique Ability": "EMP Grenade",
        "Primary Weapon": "ASSAULT RIFLE,  SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Ash",
        "Unique Ability": "BREACHING ROUND",
        "Primary Weapon": "ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Thermite",
        "Unique Ability": "Exothermic Charge",
        "Primary Weapon": "SHOTGUN, ASSAULT RIFLE",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Montagne",
        "Unique Ability": "LE ROC SHIELD",
        "Primary Weapon": "LE ROC SHIELD",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Twitch",
        "Unique Ability": "SHOCK DRONE",
        "Primary Weapon": "ASSAULT RIFLE, Marksman Rifle, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Blitz",
        "Unique Ability": "G52-TACTICAL SHIELD",
        "Primary Weapon": "G52-Tactical Shield",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "IQ",
        "Unique Ability": "Electronics Detector",
        "Primary Weapon": "ASSAULT RIFLE, Light Machine Gun",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Fuze",
        "Unique Ability": "Cluster Charge",
        "Primary Weapon": "ASSAULT RIFLE, Light Machine Gun, Ballistic Shield",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Glaz",
        "Unique Ability": "Flip Sight",
        "Primary Weapon": "Marksman Rifle",
        "Secondary Weapon": "HANDGUN, Hand Cannon, Machine Pistol"
    },
    {
        "Name": "CAPITÃO",
        "Unique Ability": "Tactical Crossbow",
        "Primary Weapon":"Light Machine Gun, ASSAULT RIFLE",
        "Secondary Weapon": "Hand Cannon, HANDGUN"
    },
    {
        "Name": "NØKK",
        "Unique Ability": "HEL Presence Reduction",
        "Primary Weapon": "SUBMACHINE GUN, SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Deimos",
        "Unique Ability": "DEATHMARK TRACKER",
        "Primary Weapon": "ASSAULT RIFLE,  SHOTGUN",
        "Secondary Weapon": "HANDGUN"
    },
    {
        "Name": "Striker",
        "Unique Ability": None,
        "Primary Weapon": "ASSAULT RIFLE, Light Machine Gun,Marksman Rifle",
        "Secondary Weapon": "HANDGUN, SHOTGUN"
    }
]

side = {'Denari': 'Defender', 'Fenrir': 'Defender', 'Solis': 'Defender', 'Azami': 'Defender', 'Thorn': 'Defender', 'Thunderbird': 'Defender',
         'Aruni': 'Defender', 'Melusi': 'Defender', 'Oryx': 'Defender', 'Wamai': 'Defender', 'Goyo': 'Defender', 'Warden': 'Defender',
         'Mozzie': 'Defender', 'Kaid': 'Defender', 'Clash': 'Defender', 'Maestro': 'Defender', 'Alibi': 'Defender', 'Vigil': 'Defender',
         'Ela': 'Defender', 'Lesion': 'Defender', 'Mira': 'Defender', 'Echo': 'Defender', 'Caveira': 'Defender', 'Valkyrie': 'Defender',
         'Frost': 'Defender', 'Mute': 'Defender', 'Smoke': 'Defender', 'Castle': 'Defender', 'Pulse': 'Defender', 'Doc': 'Defender',
        'Rook': 'Defender', 'Bandit': 'Defender', 'Tachanka': 'Defender', 'Kapkan': 'Defender', 'Skopós': 'Defender', 'Jäger': 'Defender',
        'Tubarão': 'Defender', 'Sentry': 'Defender', 'Rauora': 'Attacker', 'Ram': 'Attacker', 'Brava': 'Attacker', 'Grim': 'Attacker',
        'Sens': 'Attacker', 'Osa': 'Attacker', 'Flores': 'Attacker', 'Zero': 'Attacker', 'Ace': 'Attacker', 'Iana': 'Attacker',
        'Kali': 'Attacker', 'Amaru': 'Attacker', 'Gridlock': 'Attacker', 'Nomad': 'Attacker', 'Maverick': 'Attacker',
        'Lion': 'Attacker', 'Finka': 'Attacker', 'Dokkaebi': 'Attacker', 'Zofia': 'Attacker', 'Ying': 'Attacker', 'Jackal': 'Attacker',
        'Hibana': 'Attacker', 'Blackbeard': 'Attacker', 'Buck': 'Attacker', 'Sledge': 'Attacker', 'Thatcher': 'Attacker', 'Ash': 'Attacker',
        'Thermite': 'Attacker', 'Montagne': 'Attacker', 'Twitch': 'Attacker', 'Blitz': 'Attacker', 'IQ': 'Attacker', 'Fuze': 'Attacker',
        'Glaz': 'Attacker', 'CAPITÃO': 'Attacker', 'NØKK': 'Attacker', 'Deimos': 'Attacker', 'Striker': 'Attacker'}


