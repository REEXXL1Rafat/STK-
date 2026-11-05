"""Harvest signal from Sentinel-2, straight from Google Earth Engine (no clicking).

For each 5 km area: the share of this season's paddy (green at its peak) that is now cut.
Cut but not yet cleared paddy is what gets burned.

    python harvest.py                  # today's reading for every region (daily run)
    python harvest.py --backfill       # every 5 days of every past season (one-time, can take an hour or two)

Needs an Earth Engine service account key, as JSON text in EE_SERVICE_ACCOUNT_KEY (see SETUP.md).
Without it, this script says so and stops; the forecast keeps working without the harvest signal.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
from config import DATA, REGIONS, CELL
import grid

STEP, LOOKBACK, SCALE, CHUNK, MIN_PIXELS = 5, 10, 60, 600, 20
BACKFILL_FROM = int(os.environ.get("HARVEST_FROM", 2022))   # earlier seasons add little and cost hours
THREADS = int(os.environ.get("EE_THREADS", 3))
PEAK = {"punjab_haryana": ((8, 20), (9, 25)), "purba_medinipur": ((9, 20), (10, 30))}


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
    """Returns a function date -> list of (iy, ix, share_cut, pixels) for every cell."""
    cells = grid.build_cells(region)
    feats = [ee.Feature(ee.Geometry.Rectangle([r.lon - CELL / 2, r.lat - CELL / 2, r.lon + CELL / 2, r.lat + CELL / 2], None, False),
                        {"iy": int(r.iy), "ix": int(r.ix)}) for r in cells.itertuples()]
    chunks = [ee.FeatureCollection(feats[i:i + CHUNK]) for i in range(0, len(feats), CHUNK)]
    w, s, e, n = REGIONS[region]["bbox"]
    geom = ee.Geometry.Rectangle([w, s, e, n], None, False)
    crop = ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def ndvi(a, b):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(str(a), str(b))
                .linkCollection(cs, ["cs_cdf"])
                .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(img.select("cs_cdf").gte(0.6))))

    (pm0, pd0), (pm1, pd1) = PEAK[region]
    paddy = crop.And(ndvi(date(year, pm0, pd0), date(year, pm1, pd1)).max().gt(0.55))
    red = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)

    def read(d):
        now = ndvi(d - timedelta(days=LOOKBACK), d).median()
        cut = now.lt(0.35).rename("cut").updateMask(paddy)
        rows = []
        for fc in chunks:
            for attempt in range(4):
                try:
                    got = cut.reduceRegions(collection=fc, reducer=red, scale=SCALE, tileScale=4).getInfo()
                    break
                except Exception as ex:   # busy or timed out: wait and retry
                    if attempt == 3:
                        print(f"  {d}: gave up on a chunk ({ex})"); got = {"features": []}
                    time.sleep(10 * (attempt + 1))
            for f in got["features"]:
                p = f["properties"]
                if p.get("count", 0) >= MIN_PIXELS and p.get("mean") is not None:
                    rows.append((p["iy"], p["ix"], str(d), round(p["mean"], 4)))
        return rows
    return read


def save(region, rows):
    path = DATA / f"harvest_{region}.csv.gz"
    new = pd.DataFrame(rows, columns=["iy", "ix", "date", "harv"])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(["iy", "ix", "date"], keep="last").sort_values(["date", "iy", "ix"])
    DATA.mkdir(exist_ok=True)
    new.to_csv(path, index=False, compression="gzip")
    return len(new)


def main():
    ee = connect()
    backfill = "--backfill" in sys.argv
    today = date.today()
    for region in REGIONS:
        jobs = []
        for year in range(BACKFILL_FROM if backfill else today.year - 1, today.year + 1):
            a, b = season_window(region, year)
            b = min(b, today)
            if a > today:
                continue
            if backfill:
                d = a
                while d <= b:
                    jobs.append((year, d)); d += timedelta(days=STEP)
            elif a <= today <= season_window(region, year)[1]:
                jobs.append((year, today))
        if not jobs:
            print(f"{region}: not in season, nothing to read"); continue
        rows = []
        for y in sorted({j[0] for j in jobs}):   # build each season's reader once, then read its dates a few at a time
            read = reader(ee, region, y)
            with ThreadPoolExecutor(THREADS) as pool:
                for got in pool.map(read, [d for yy, d in jobs if yy == y]):
                    rows += got
            if rows:
                save(region, rows); rows_saved = len(rows)
            print(f"{region} {y}: {len(rows):,} readings so far", flush=True)
        print(f"{region}: saved, {save(region, rows):,} readings in total" if rows else f"{region}: no clear readings")


if __name__ == "__main__":
    main()
