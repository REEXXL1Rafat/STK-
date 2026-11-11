# SATARK Risk Window

Every morning at 6 am, Risk Window answers one question: **where are farm fires most likely today, tomorrow and the day after, and who should go where?**

- **Anyone** can type their village or PIN code (or tap the map, or press "Near me") and see the chance of a farm fire around it for the next three days, what to do instead of burning, and share it on WhatsApp. The page switches between English, Punjabi, Hindi and Bengali.
- **Field teams and officials** get a plan for however many teams they have, with stops named by the nearest village, and a short brief (English plus Punjabi or Bengali) with Google Maps links.
- **Everyone** can see how it did: the next morning it checks itself against what the satellites saw. If the forecast was wrong, the scoreboard says so.
- **Fire log:** every day of the season, how many fire detections it expected that morning (and a day ahead), and how many the satellites then saw. Each morning's numbers are saved as they are published and never changed. Download it from the dashboard, or find it in `out/<region>/fire_log_<year>.csv`.
- **Ask SATARK:** tap and talk in Punjabi, Hindi, Bengali or English. An AI voice answers from the same forecast and opens the village on the map (optional, paid, see `SETUP.md`).
- **Where today's smoke goes:** paths from the fields expected to burn, the towns they reach and when, and tomorrow's PM2.5 forecast for those towns.
- **Green villages:** places that burned by this date last season and show no fires this season, and every district against last year. Something to celebrate, and to learn from.
- **People in the red zone:** how many people live in today's high-risk areas (needs Earth Engine).
- **Season replay:** the 2025 season day by day, what the model flagged each morning against what the satellites then saw. Add `#kiosk` to the link for a full-screen loop for events.

Regions:

| Region | Why |
|---|---|
| Punjab and Haryana | The biggest stubble-burning area in India |
| Purba Medinipur | SATARK's Kolaghat pilot (BDO Kolaghat mandate) |
| Uttar Pradesh | Large paddy belt, west UP smoke reaches Delhi |
| Madhya Pradesh | Fast-growing paddy-straw burning (the April wheat-straw season is not covered yet) |
| Rajasthan | Paddy in Sri Ganganagar, Hanumangarh and the Kota belt |

The three big states only model the 5 km areas that have burned before (and their neighbours), and read weather every 1° instead of 0.5°, so the free services and GitHub's machines can keep up. Their numbers appear on the dashboard once the setup run has downloaded and tested them.

**Support first, enforcement last.** A high-risk cell means a farmer who needs a machine or decomposer today, not a fine.

## How it works

1. **Data.** NASA FIRMS VIIRS fire detections (375 m, S-NPP and NOAA-20), plus daily weather from Open-Meteo for points 0.5° apart. NASA stopped S-NPP fire data on 2 Nov 2026, so from 3 Nov NOAA-21 counts in its place. Over 2023 to 2025 it saw almost exactly what S-NPP did (2 to 6% more in every region), so the numbers stay comparable (`SNPP_LAST_DAY` in `config.py`).
2. **Features, per cell per day.** Fires in the cell and its neighbours over the last 1 to 7 days, district and region trends, where fires usually are at this point in the season (from past years), how much of the cell's usual burning has already happened, and today's rain, humidity, heat and wind.
3. **Model.** Gradient-boosted trees (scikit-learn): one for "will there be a fire" and one for "how many, if it burns". Expected fires = chance × how many. The chances are then adjusted on a season the model never trained on, so that "20%" really means about one day in five.
4. **Plan.** Picks the top places (never two next to each other, never below a 5% chance), splits them into teams by area, and orders each team's route. Places are ranked by chance, or by expected fires where that caught more fires on the check season (see below).
5. **Check.** Each morning, compares the last three days' forecasts, exactly as published (`out/<region>/pred_<date>.csv`), with the satellite detections, so late satellite data still counts. The results go to `scoreboard.csv` and the fire log.

## How good is it

`train.py` holds back the most recent full season, trains on the earlier ones, and tests on the held-back season. The results go to `models/<region>_report.json` and onto the dashboard. The model is compared with two guesses anyone could make without AI:

- **Persistence:** "fires will be where they were yesterday"
- **Climatology:** "fires will be where they usually are this week"

If the model can't beat both, don't use it to send people anywhere.

### Results on real NASA data (trained 2020 to 2024, tested blind on 2025)

**Punjab and Haryana**: 3,535 cells, 9,654 fire detections in the test season.

| | Flagged places that burned (top 50) | Fires caught in top 50 | Fires caught in top 100 |
|---|---|---|---|
| **Risk Window** | **16%** | **11%** | **19%** |
| Where it burned yesterday | 12% | 9% | 15% |
| Where it usually burns | 12% | 9% | 14% |

- **This season (2026), which the model never saw:** over 47 days with fires, its 50 places held 19% of all fires seen. The "yesterday" guess held 13%.
- **What 50 places means:** that's 1.4% of the map, so the flagged places hold far more than their share of fires.
- **Calibration:** the chances are adjusted on the season before and checked on 2025. When it says 8%, fires came about 10% of the time; at 14%, about 18%; at 24%, about 27%; at 34%, about 36%. Without the adjustment it was too sure of itself at the top: 37% meant about 27%, and 56% about 37%.

**Purba Medinipur**: 500 cells, 2,742 detections in the test season.

| | Flagged places that burned (top 50) | Fires caught in top 50 | Fires caught in top 100 |
|---|---|---|---|
| **Risk Window** | **19%** | **61%** | **82%** |
| Where it burned yesterday | 18% | 59% | 76% |
| Where it usually burns | 18% | 61% | 78% |

In Purba Medinipur the gain is small (it ties "where it usually burns" in the top 50), so treat it as a planning aid, not a big upgrade. The morning brief says so. Burning there peaks from late November into December.

**Rajasthan**: 4,356 cells, 8,614 detections in the test season. Burning here comes back to the same fields every year, so places are ranked by expected fires and the bigger settings are used (both chosen on the 2024 check).

| | Flagged places that burned (top 50) | Fires caught in top 50 | Fires caught in top 100 |
|---|---|---|---|
| **Risk Window** | **26%** | **44%** | **61%** |
| Where it burned yesterday | 20% | 22% | 37% |
| Where it usually burns | 18% | 38% | 51% |

Before this, ranked by chance, it caught 33% in the top 50 and lost to "where it usually burns".

### Looking ahead (tested blind on 2025)

| Punjab and Haryana | Flagged places that burned (top 50) | "Yesterday" guess |
|---|---|---|
| Today | 16% | 12% |
| Tomorrow | 15% | 11% |
| Day after | 14% | 11% |

The gap over the simple guess holds, and in Purba Medinipur it grows, for the days ahead, which is where the lead time helps machine operators most.

### Extra signals: tested, kept only if they help

`train.py --select` tests each extra signal on the season before the test season and keeps it only if it helps. The same check, on the same season, decides two more things per region: bigger, slower model settings or the standard ones, and whether places are ranked by chance or by expected fires. Each is kept only if it catches at least half a point more fires in the top 50 without losing more than 3 points of flagged places that burned. `train.py` also prints a warning, and writes `beats_both_guesses: false` in the report, if a region's model doesn't beat both simple guesses on the test season.

- **Punjab and Haryana:** the third VIIRS satellite (NOAA-21), fire intensity and Diwali timing were tested and none helped, so none are used.
- **Purba Medinipur:** NOAA-21 and fire intensity helped on the 2024 check and are used.

**Two more signals switch on by themselves:**

1. **MODIS and wind direction.** The one-time setup workflow downloads them, and `train.py --select` tests them.
2. **Harvest from Sentinel-2.** This marks fields already cut but not yet cleared. With an Earth Engine service account (see `SETUP.md`), `harvest.py` reads it every morning, then spends up to 25 minutes filling past seasons (2022 onwards, newest first), carrying on the next morning until all are in. The setup run fills them all in one go. Progress is kept in `data/harvest_done.json`. Once past seasons are in, the Sunday retrain tests the signal. Without an account, it is skipped. The manual route still works: run `gee/harvest_signal.js` in the Earth Engine Code Editor, then `python import_harvest.py harvest_*.csv`.

**Factories, flares and power plants removed.** 120 cells in Punjab and Haryana and 10 in Purba Medinipur were dropped because they fire up at night or nearly every day. They are industrial heat sources, not farms. In Purba Medinipur they made up about half of all raw detections. Night detections are ignored everywhere.

## Run it

```bash
pip install -r requirements.txt
python fetch_history.py YOUR_FIRMS_MAP_KEY   # once, 10 to 20 min
python train.py punjab_haryana
python train.py purba_medinipur
FIRMS_MAP_KEY=YOUR_KEY python daily.py      # today's forecast, brief and dashboard
python daily.py --backfill                  # also score every day of this season so far
```

Open `out/index.html`. The brief for each region is in `out/<region>/brief_<date>.txt`.

You can get a free FIRMS key at https://firms.modaps.eosdis.nasa.gov/api/map_key/

## Run it every day, for free

It runs itself on GitHub. Follow `SETUP.md` once (about 10 minutes of clicking, then 1 to 3 hours of the robot working). After that, every morning at 6:00 IST it:

- downloads yesterday's VIIRS and MODIS fires and the weather forecast,
- reads the Sentinel-2 harvest signal (if Earth Engine is set up),
- forecasts today, tomorrow and the day after, plans team routes, writes the brief,
- checks yesterday's forecast against the satellites and adds it to the scoreboard,
- publishes the dashboard on GitHub Pages and, if set up, posts the brief on Telegram.

On Sundays during the season it retrains and re-tests every extra signal, so only the ones that help are used.

| Workflow | When |
|---|---|
| `.github/workflows/risk-window-setup.yml` | Once, by hand |
| `.github/workflows/risk-window.yml` | Every day at 6:00 IST |

## Limits, said plainly

- **Official counts are higher.** Punjab's official count (used in the news) adds MODIS satellites, whose afternoon pass is now later in the day. On Diwali 2026 it reported 1,152 farm fires in Punjab; Risk Window's VIIRS count for Punjab and Haryana was 638. Risk Window counts the same way every day, so its days compare fairly with each other.
- **Missed fires.** The VIIRS satellites pass at about 1:30 pm and 1:30 am. Fires lit later in the afternoon are mostly never detected, so "seen" undercounts real burning. The model learns from detections, so it inherits that blind spot.
- **Cell size.** A 5 km cell holds several villages. The Maps link gives the cell centre, not a field.
- **Weather.** Rain matters. In live runs the weather is a forecast, which can be wrong.
- **District names.** These come from 2011 census boundaries, so districts created later (like Malerkotla) carry their old parent's name.
- **Ground truth.** Guardian field reports ("burned / not burned / helped") would make it better. That's the next step.
- **Smoke paths.** They follow the forecast wind from 2 pm for 36 hours. That shows where smoke drifts, not how thick it is. The PM2.5 numbers are CAMS forecasts for all sources.
- **Burn scars.** The Sentinel-2 burn-scar threshold is a first guess. `train.py --select` keeps the signal only if it improves the forecast.
- **The voice** is an AI. It is told to use only the forecast's numbers and to say when it doesn't know, but it can still mishear or slip. The page says it is an AI.
- **Translations.** The Punjabi, Hindi and Bengali text in `brief.py` and the dashboard (the `STR` table in `dashboard_template.html`) should be checked by a native speaker before wide use.

## Files

| File | What it does |
|---|---|
| `config.py` | Regions, seasons, grid size |
| `sources.py` | Downloads from NASA FIRMS and Open-Meteo (standard library only) |
| `fetch_history.py` | One-time pull of past seasons |
| `grid.py` | The 5 km grid and district lookup |
| `features.py` | Builds the per cell, per day features |
| `train.py` | Trains the model and runs the honest test |
| `daily.py` | Morning run: forecast, plan, brief, scoreboard, dashboard |
| `dispatch.py` | Turns risk into team routes |
| `brief.py` | The morning message |
| `geo/places_*.csv` | Villages and towns (India Post PIN directory) for search and stop names |
| `gee/harvest_signal.js`, `import_harvest.py` | Optional Sentinel-2 harvest signal |
| `harvest.py` | Sentinel-2 harvest and burn-scar signals, and population, through Earth Engine (optional) |
| `smoke.py` | Where today's smoke drifts (wind paths) and PM2.5 for towns (CAMS) |
| `voice/worker.js` | The Cloudflare Worker that keeps the OpenAI key safe for Ask SATARK |
| `out/<region>/fire_log_<year>.csv` | The fire log: expected vs seen, every day of that season. `issued.csv` keeps each morning's totals as published |
| `dashboard.py`, `dashboard_template.html` | The dashboard. The pilot regions are inside the page (works offline); big states load from `out/data/` |
| `assets/` | SATARK logo, riso field photo and DIVA drawing used by the dashboard |
| `SETUP.md` | One-time setup on GitHub |

## Credits

- **Fire data:** NASA FIRMS, LANCE (https://www.earthdata.nasa.gov/data/tools/firms). NASA promotes full and open sharing of FIRMS data. Please keep this credit when you share the outputs.
- **Weather:** Open-Meteo (https://open-meteo.com), CC BY 4.0.
- **District boundaries:** DataMeet India community (https://github.com/datameet/maps), CC BY 4.0.
- **Village names:** India Post All India Pincode Directory via data.gov.in (GODL-India). Some post office coordinates are rough, so treat the name as "near", not exact.
- **Air quality forecast:** Copernicus Atmosphere Monitoring Service (CAMS), via Open-Meteo.
- **Population:** GHS-POP R2023A, European Commission Joint Research Centre. **Land cover:** ESA WorldCover. **Imagery:** Copernicus Sentinel-2, via Google Earth Engine.
- **Voice:** OpenAI realtime API, through your own key.

Licensed GPL-3.0, same as SATARK PRIME.
