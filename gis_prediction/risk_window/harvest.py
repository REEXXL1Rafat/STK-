"""Sentinel-2 signals and population, straight from Google Earth Engine (no clicking).

For each 5 km area:
  harv  the share of this season's paddy (green at its peak) that is now cut. Cut but not cleared is what gets burned.
  scar  the share of that paddy showing a fresh burn scar (dark, low NBR). It catches fires lit after the
        1:30 pm satellite pass, which VIIRS misses. The threshold is a first guess; train.py --select keeps it only if it helps.
And once per region: how many people live in each area (GHS-POP 2025, JRC), for "people in the red zone".

    python harvest.py                  # today's reading for every region (daily run), plus population if missing
    python harvest.py --backfill       # every 5 days of every past season (one-time, can take an hour or two)

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


def main():
    ee = connect()
    backfill = "--backfill" in sys.argv
    today = date.today()
    for region in [r for r in ORDER if r in REGIONS]:
        try:
            people(ee, region)
        except Exception as ex:
            print(f"{region}: population skipped ({ex})")
        if region not in PEAK:
            continue
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
                save(region, rows)
            print(f"{region} {y}: {len(rows):,} readings so far", flush=True)
        print(f"{region}: saved, {save(region, rows):,} readings in total" if rows else f"{region}: no clear readings")


if __name__ == "__main__":
    main()
