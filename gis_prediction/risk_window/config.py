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
    # Big states: only cells that have burned before (and their neighbours) are modelled, and weather is
    # read every 1 deg, so the free weather service and GitHub's machines can keep up.
    "uttar_pradesh": {
        "label": "Uttar Pradesh",
        "bbox": (77.0, 23.8, 84.7, 30.5),
        "season": ((9, 20), (12, 15)),
        "states": ["Uttar Pradesh"],
        "lang": "hi",
        "active_only": True, "wx_step": 1.0,
    },
    "madhya_pradesh": {
        "label": "Madhya Pradesh",
        "bbox": (74.0, 21.0, 82.9, 26.9),
        "season": ((9, 25), (12, 20)),      # paddy only; the April wheat-straw season is not covered yet
        "states": ["Madhya Pradesh"],
        "lang": "hi",
        "active_only": True, "wx_step": 1.0,
    },
    "rajasthan": {
        "label": "Rajasthan",
        "bbox": (69.4, 23.0, 78.3, 30.2),
        "season": ((9, 20), (12, 10)),
        "states": ["Rajasthan"],
        "lang": "hi",
        "active_only": True, "wx_step": 1.0,
    },
}
# smallest first, so the pilot regions are ready first if a run is cut short
ORDER = ["purba_medinipur", "punjab_haryana", "rajasthan", "uttar_pradesh", "madhya_pradesh"]
SHOW = ["punjab_haryana", "purba_medinipur", "uttar_pradesh", "madhya_pradesh", "rajasthan"]   # tab order on the dashboard
ACTIVE_MIN = 2      # a cell is modelled if it had at least this many farm-fire detections in past seasons

FIRST_YEAR = 2020   # NOAA-20 VIIRS starts in 2020, so counts are comparable from here
# VIIRS sensors used for training. NOAA-21 is added from 2024 in the raw data but
# left out of the target by default so every year is counted the same way.
SENSORS = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"]   # MODIS: Terra (~10:30 am) and Aqua (~1:30 pm)
TARGET_SENSORS = ["N", "N20", "1"]   # FIRMS 'satellite' codes: N = S-NPP, N20 (or 1) = NOAA-20. NOAA-21 (N21) is left out
DAYTIME_ONLY = True                   # farm fires are lit by day; night detections are mostly factories and flares
# A cell is treated as a factory, refinery or power plant (not a farm) if a third or more of its
# detections are at night, or it shows up on half or more of all season days.
STATIC_NIGHT_SHARE, STATIC_MIN_N, STATIC_DAY_SHARE = 0.3, 20, 0.5

# Weather points every 0.5 deg across each region (Open-Meteo, free, no key); big states override with wx_step
WX_STEP = 0.5


def wx_step(region):
    return REGIONS[region].get("wx_step", WX_STEP)
WX_DAILY = ["precipitation_sum", "temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant"]
