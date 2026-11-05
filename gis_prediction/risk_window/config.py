"""SATARK Risk Window: settings shared by every script."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SATARK_DATA", ROOT / "data"))       # raw downloads (fire points, weather)
OUT = Path(os.environ.get("SATARK_OUT", ROOT / "out"))          # daily outputs (brief, dashboard, scoreboard)
MODELS = Path(os.environ.get("SATARK_MODELS", ROOT / "models")) # trained models + evaluation reports

# Grid cell size in degrees. 0.05 deg is about 5.5 km: small enough to send a
# team to, big enough that one cell holds a few villages.
CELL = 0.05

# Each region: bounding box (west, south, east, north), the burning season
# (month, day) start and end, and which states to keep when naming places.
REGIONS = {
    "punjab_haryana": {
        "label": "Punjab and Haryana",
        "bbox": (73.8, 27.6, 77.6, 32.6),
        "season": ((9, 15), (12, 10)),      # kharif paddy harvest to wheat sowing
        "states": ["Punjab", "Haryana"],
        "lang": "pa",
    },
    "purba_medinipur": {
        "label": "Purba Medinipur (Kolaghat pilot)",
        "bbox": (86.9, 21.5, 88.3, 22.7),
        "season": ((10, 25), (1, 31)),      # aman harvest, runs into January
        "states": ["West Bengal"],
        "lang": "bn",
    },
}

FIRST_YEAR = 2020   # NOAA-20 VIIRS starts in 2020, so counts are comparable from here
# VIIRS sensors used for training. NOAA-21 is added from 2024 in the raw data but
# left out of the target by default so every year is counted the same way.
SENSORS = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"]   # MODIS: Terra (~10:30 am) and Aqua (~1:30 pm)
TARGET_SENSORS = ["N", "N20", "1"]   # FIRMS 'satellite' codes: N = S-NPP, N20 (or 1) = NOAA-20. NOAA-21 (N21) is left out
DAYTIME_ONLY = True                   # farm fires are lit by day; night detections are mostly factories and flares
# A cell is treated as a factory, refinery or power plant (not a farm) if a third or more of its
# detections are at night, or it shows up on half or more of all season days.
STATIC_NIGHT_SHARE, STATIC_MIN_N, STATIC_DAY_SHARE = 0.3, 20, 0.5

# Weather points every 0.5 deg across each region (Open-Meteo, free, no key)
WX_STEP = 0.5
WX_DAILY = ["precipitation_sum", "temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant"]
