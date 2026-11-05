"""Turn the Earth Engine export (harvest_<region>.csv from Google Drive) into data/harvest_<region>.csv.gz

    python import_harvest.py harvest_punjab_haryana.csv harvest_purba_medinipur.csv
Then: python train.py punjab_haryana --select   (it keeps the harvest signal only if it helps)
"""
import sys
from pathlib import Path
import pandas as pd
from config import DATA, REGIONS

MIN_PIXELS = 20   # fewer clear paddy pixels than this in a 5 km area: too little to trust

for path in sys.argv[1:]:
    p = Path(path)
    region = next((r for r in REGIONS if r in p.name), None)
    if not region:
        print(f"skip {p.name}: can't tell the region from the file name"); continue
    h = pd.read_csv(p)
    h = h[h["count"] >= MIN_PIXELS].rename(columns={"mean": "harv"})
    h = h[["iy", "ix", "date", "harv"]].dropna()
    DATA.mkdir(exist_ok=True)
    out = DATA / f"harvest_{region}.csv.gz"
    h.to_csv(out, index=False, compression="gzip")
    print(f"{region}: {len(h):,} readings from {h.date.min()} to {h.date.max()} -> {out}")
