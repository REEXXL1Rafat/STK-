"""Sentinel-2 signals and population, straight from Google Earth Engine (no clicking).

For each 5 km area:
  harv  the share of this season's paddy (green at its peak) that is now cut. Cut but not cleared is what gets burned.
  scar  the share of that paddy showing a fresh burn scar (dark, low NBR). It catches fires lit after the
        1:30 pm satellite pass, which VIIRS misses. The threshold is a first guess; train.py --select keeps it only if it helps.
And once per region: how many people live in each area (GHS-POP 2025, JRC), for "people in the red zone".

    python harvest.py                  # daily run: today's reading for every region, then fills past seasons
                                       # for up to HARVEST_BUDGET_MIN minutes (default 25); the next run carries on
    python harvest.py --backfill       # fill every past season in one go, no time limit (setup run, an hour or two)

Past seasons are read every 5 days, newest season first. What has been read is kept in data/harvest_done.json,
so a run that stops (time limit, Earth Engine busy) picks up where it left off. Once 2022 to last season are in,
the Sunday retrain (train.py --select) tests the harvest and burn-scar signals and keeps them only if they help.

Needs an Earth Engine service account key, as JSON text in EE_SERVICE_ACCOUNT_KEY (see SETUP.md).
Without it, this script says so and stops; the forecast keeps working without the harvest signal.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
from config import DATA, REGIONS, ORDER, CELL, FIRST_YEAR
import grid, features as F

STEP, LOOKBACK, SCALE, CHUNK, MIN_PIXELS = 5, 10, 60, 600, 20
BACKFILL_FROM = int(os.environ.get("HARVEST_FROM", 2022))   # earlier seasons add little and cost hours
THREADS = int(os.environ.get("EE_THREADS", 3))
BUDGET_MIN = float(os.environ.get("HARVEST_BUDGET_MIN", 25))   # daily run: minutes for filling past seasons (the step stops at 40)
PROGRESS = DATA / "harvest_done.json"                           # {region: {season: last date read, or "done"}}
PEAK = {"punjab_haryana": ((8, 20), (9, 25)), "purba_medinipur": ((9, 20), (10, 30)), "uttar_pradesh": ((8, 25), (9, 30)),
        "madhya_pradesh": ((8, 25), (9, 30)), "rajasthan": ((8, 20), (9, 25))}
SCAR_NBR = -0.05     # NBR below this on a cut field: charred, most likely burnt
POP = ("JRC/GHSL/P2023A/GHS_POP/2025", "population_count", 100)


def region_cells(region):
    """Big states: only the cells that have burned before (same rule as train.py)."""
    cells = grid.build_cells(region)
    if REGIONS[region].get("active_only") and (DATA / f"fires_{region}.csv.gz").exists():
        fires = F.load_fires(DATA / f"fires_{region}.csv.gz")
        years = list(range(FIRST_YEAR, date.today().year + 1))
        cells = F.keep_active(cells, F.active_cells(fires, region, years, F.static_cells(fires, region)))
    return cells


def cell_chunks(ee, cells):
    feats = [ee.Feature(ee.Geometry.Rectangle([r.lon - CELL / 2, r.lat - CELL / 2, r.lon + CELL / 2, r.lat + CELL / 2], None, False),
                        {"iy": int(r.iy), "ix": int(r.ix)}) for r in cells.itertuples()]
    return [ee.FeatureCollection(feats[i:i + CHUNK]) for i in range(0, len(feats), CHUNK)]


def reduce_chunks(img, chunks, reducer, scale, label):
    out = []
    for fc in chunks:
        for attempt in range(4):
            try:
                got = img.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=4).getInfo()
                break
            except Exception as ex:   # busy or timed out: wait and retry
                if attempt == 3:
                    print(f"  {label}: gave up on a chunk ({ex})"); got = {"features": []}
                time.sleep(10 * (attempt + 1))
        out += [f["properties"] for f in got["features"]]
    return out


def people(ee, region):
    """People per 5 km area, saved once to data/people_<region>.json."""
    path = DATA / f"people_{region}.json"
    if path.exists():
        return
    asset, band, scale = POP
    props = reduce_chunks(ee.Image(asset).select(band), cell_chunks(ee, region_cells(region)), ee.Reducer.sum(), scale, f"{region} people")
    cells = [[p["iy"], p["ix"], int(round(p.get("sum") or 0))] for p in props if (p.get("sum") or 0) >= 1]
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps({"source": "GHS-POP 2025, European Commission JRC", "cells": cells}, separators=(",", ":")))
    print(f"{region}: population for {len(cells):,} areas, {sum(c[2] for c in cells):,} people")


def connect():
    raw = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    if not raw:
        print("No EE_SERVICE_ACCOUNT_KEY set: skipping the harvest signal."); sys.exit(0)
    import ee
    key = json.loads(raw)
    creds = ee.ServiceAccountCredentials(key["client_email"], key_data=raw)
    ee.Initialize(creds, project=os.environ.get("EE_PROJECT", key.get("project_id")))
    return ee


def season_window(region, year):
    (m0, d0), (m1, d1) = REGIONS[region]["season"]
    return date(year, m0, d0), date(year + (1 if m1 < m0 else 0), m1, d1)


def reader(ee, region, year):
    """Returns a function date -> list of (iy, ix, date, share_cut, share_scarred) for every cell."""
    chunks = cell_chunks(ee, region_cells(region))
    w, s, e, n = REGIONS[region]["bbox"]
    geom = ee.Geometry.Rectangle([w, s, e, n], None, False)
    crop = ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def s2(a, b):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(str(a), str(b))
                .linkCollection(cs, ["cs_cdf"])
                .map(lambda img: ee.Image.cat(img.normalizedDifference(["B8", "B4"]).rename("ndvi"),
                                              img.normalizedDifference(["B8", "B12"]).rename("nbr"))
                     .updateMask(img.select("cs_cdf").gte(0.6))))

    (pm0, pd0), (pm1, pd1) = PEAK[region]
    paddy = crop.And(s2(date(year, pm0, pd0), date(year, pm1, pd1)).select("ndvi").max().gt(0.55))
    red = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)

    def read(d):
        now = s2(d - timedelta(days=LOOKBACK), d).median()
        cut = now.select("ndvi").lt(0.35)
        img = ee.Image.cat(cut.rename("cut"), cut.And(now.select("nbr").lt(SCAR_NBR)).rename("scar")).updateMask(paddy)
        rows = []
        for p in reduce_chunks(img, chunks, red, SCALE, str(d)):
            n = p.get("cut_count", p.get("count", 0)) or 0
            c = p.get("cut_mean", p.get("mean"))
            if n >= MIN_PIXELS and c is not None:
                sc = p.get("scar_mean")
                rows.append((p["iy"], p["ix"], str(d), round(c, 4), None if sc is None else round(sc, 4)))
        return rows
    return read


def save(region, rows):
    path = DATA / f"harvest_{region}.csv.gz"
    new = pd.DataFrame(rows, columns=["iy", "ix", "date", "harv", "scar"])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(["iy", "ix", "date"], keep="last").sort_values(["date", "iy", "ix"])
    DATA.mkdir(exist_ok=True)
    new.to_csv(path, index=False, compression="gzip")
    return len(new)


def load_progress():
    try:
        return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    except ValueError:
        return {}


def save_progress(prog):
    DATA.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(prog, indent=1, sort_keys=True))


def season_steps(region, year, today):
    """Every 5th day of a season, up to yesterday (today's reading is taken separately)."""
    a, b = season_window(region, year)
    out, d = [], a
    while d <= min(b, today - timedelta(days=1)):
        out.append(d); d += timedelta(days=STEP)
    return out


def todo(region, year, today, prog):
    """Dates of this season not read yet. A finished past season is marked "done"."""
    mark = prog.get(region, {}).get(str(year), "")
    if mark == "done":
        return []
    return [d for d in season_steps(region, year, today) if str(d) > mark]


def read_today(ee, region, today):
    for year in (today.year - 1, today.year):
        a, b = season_window(region, year)
        if a <= today <= b:
            rows = reader(ee, region, year)(today)
            print(f"{region}: today's reading, {len(rows):,} areas" if rows else f"{region}: no clear view today", flush=True)
            if rows:
                save(region, rows)
            return
    print(f"{region}: not in season today")


def fill_past(ee, region, today, prog, deadline, read_for=None):
    """Past seasons, newest first, a few dates at a time, saving as it goes. Returns False when time is up."""
    for year in range(today.year, BACKFILL_FROM - 1, -1):
        dates = todo(region, year, today, prog)
        if not dates:
            continue
        read = (read_for or reader)(ee, region, year)
        for i in range(0, len(dates), THREADS):
            if deadline and time.time() > deadline:
                return False
            batch = dates[i:i + THREADS]
            with ThreadPoolExecutor(THREADS) as pool:
                rows = [r for got in pool.map(read, batch) for r in got]
            if rows:
                save(region, rows)
            finished = season_window(region, year)[1] < today and i + THREADS >= len(dates)
            prog.setdefault(region, {})[str(year)] = "done" if finished else str(batch[-1])
            save_progress(prog)
            print(f"{region} {year}: read up to {batch[-1]} ({len(rows):,} readings)", flush=True)
    return True


def main():
    ee = connect()
    full = "--backfill" in sys.argv
    today = date.today()
    deadline = None if full else time.time() + 60 * BUDGET_MIN
    regions = [r for r in ORDER if r in REGIONS and r in PEAK]
    for region in regions:               # quick part first: population (once) and today's reading
        try:
            people(ee, region)
        except Exception as ex:
            print(f"{region}: population skipped ({ex})")
        try:
            read_today(ee, region, today)
        except Exception as ex:
            print(f"{region}: today's reading failed ({ex})")
    prog = load_progress()
    for region in regions:               # then history, smallest region first so one becomes usable soon
        try:
            if not fill_past(ee, region, today, prog, deadline):
                print(f"Time limit reached ({BUDGET_MIN:g} min). The next run carries on from here."); return
        except Exception as ex:
            print(f"{region}: past seasons paused ({ex}); the next run tries again")
    left = sum(len(todo(r, y, today, prog)) for r in regions for y in range(BACKFILL_FROM, today.year + 1))
    print("All past seasons read." if not left else f"{left} past dates still to read.")


if __name__ == "__main__":
    main()
