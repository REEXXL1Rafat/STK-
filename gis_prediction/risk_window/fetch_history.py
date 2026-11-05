"""Pull past burning seasons once, so the model has something to learn from.

    python fetch_history.py YOUR_MAP_KEY                 # every region, 2020 to now
    python fetch_history.py YOUR_MAP_KEY punjab_haryana  # one region
    python fetch_history.py --sensors=MODIS              # add one sensor to what's already there
    (the key can also come from key.txt or the FIRMS_MAP_KEY environment variable)

Get a free MAP_KEY in a minute at https://firms.modaps.eosdis.nasa.gov/api/map_key/
It resumes where it stopped if the connection drops. The first run takes 30 to 60 minutes;
running it again after an update only downloads what's new.
Writes data/fires_<region>.csv.gz and data/weather_<region>.csv.gz
"""
import csv, gzip, json, os, sys, time
from datetime import date, timedelta
from config import DATA, REGIONS, FIRST_YEAR, SENSORS, WX_STEP, WX_DAILY
import sources


def season_window(region, year, today):
    (m0, d0), (m1, d1) = region["season"]
    start = date(year, m0, d0)
    end = date(year + (1 if m1 < m0 else 0), m1, d1)
    return start, min(end, today - timedelta(days=1))


def main():
    keyfile = DATA.parent / "key.txt"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--sensors=")), None)
    if args and args[0] not in REGIONS:
        key, names = args[0], args[1:]
    elif os.environ.get("FIRMS_MAP_KEY"):
        key, names = os.environ["FIRMS_MAP_KEY"], args
    elif keyfile.exists():
        key, names = keyfile.read_text().strip(), args
    else:
        print(__doc__); sys.exit(1)
    names = names or list(REGIONS)
    sensors = [s for s in SENSORS if not only or s in only]
    today = date.today()
    DATA.mkdir(exist_ok=True)
    cache = DATA / "cache"; cache.mkdir(exist_ok=True)
    avail = sources.firms_availability(key)
    print("FIRMS products:", ", ".join(f"{k} {v[0]}..{v[1]}" for k, v in sorted(avail.items()) if k.startswith("VIIRS")))

    for name in names:
        reg = REGIONS[name]
        rows, wx = [], []
        for year in range(FIRST_YEAR, today.year + 1):
            a, b = season_window(reg, year, today)
            if a > b:
                continue
            print(f"\n{reg['label']} {year}: {a} to {b}")
            for sensor in sensors:
                for start, n in sources.daterange(a, b, 5):
                    prod = sources.pick_product(sensor, start, avail)
                    if not prod:
                        continue
                    f = cache / f"{name}_{prod}_{start}_{n}.json"
                    if f.exists():
                        got = json.loads(f.read_text())
                    else:
                        got = sources.firms_area(key, prod, reg["bbox"], start, n)
                        f.write_text(json.dumps(got))
                        time.sleep(0.4)
                    rows += got
                print(f"  {sensor}: {sum(1 for r in rows if r['product'].startswith(sensor) and r['acq_date'][:4] in (str(year), str(year + 1)))} points so far", flush=True)
            wf = cache / f"{name}_wx{len(WX_DAILY)}_{year}.json"   # new weather fields mean a fresh download
            if wf.exists():
                wx += json.loads(wf.read_text())
            else:
                got = sources.weather_history(sources.wx_points(reg["bbox"], WX_STEP), a, b, WX_DAILY)
                wf.write_text(json.dumps(got)); wx += got
            print(f"  weather: {len(wx)} point-days so far")

        out = DATA / f"fires_{name}.csv.gz"
        if only and out.exists():   # adding a sensor: keep what's already downloaded
            with gzip.open(out, "rt", newline="") as fh:
                rows = list(csv.DictReader(fh)) + rows
        seen = set(); uniq = []
        for r in rows:   # SP and NRT can overlap at the edges
            k = (r["latitude"], r["longitude"], r["acq_date"], r["acq_time"], r["satellite"])
            if k not in seen:
                seen.add(k); uniq.append(r)
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(uniq[0].keys()) if uniq else sources.KEEP + ["product"])
            w.writeheader(); w.writerows(uniq)
        wout = DATA / f"weather_{name}.csv.gz"
        with gzip.open(wout, "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(wx[0].keys()))
            w.writeheader(); w.writerows(wx)
        print(f"\nSaved {len(uniq):,} fire points to {out.name} and {len(wx):,} weather rows to {wout.name}")
    import zipfile
    bundle = DATA.parent / "SEND_THIS_TO_CLAUDE.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        for f in sorted(DATA.glob("*_*.csv.gz")):
            z.write(f, f.name)
    print(f"\nDone. Send this one file back: {bundle}")


if __name__ == "__main__":
    main()
