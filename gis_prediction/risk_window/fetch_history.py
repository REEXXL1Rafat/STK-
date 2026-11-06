"""Pull past burning seasons, so the model has something to learn from.

    python fetch_history.py                      # every region: download only what's missing
    python fetch_history.py uttar_pradesh        # one region
    python fetch_history.py --sensors=MODIS      # only this sensor
    python fetch_history.py --force              # download everything again
    (the NASA key comes from the FIRMS_MAP_KEY environment variable, key.txt, or the first argument)

Get a free MAP_KEY in a minute at https://firms.modaps.eosdis.nasa.gov/api/map_key/
What has been downloaded is listed in data/fetched.json, so a run that stops halfway
picks up where it left off. Writes data/fires_<region>.csv.gz and data/weather_<region>.csv.gz
"""
import csv, gzip, json, os, sys, time
from datetime import date, timedelta
from config import DATA, REGIONS, ORDER, FIRST_YEAR, SENSORS, WX_DAILY, wx_step
import sources

MANIFEST = DATA / "fetched.json"


def season_window(region, year, today):
    (m0, d0), (m1, d1) = region["season"]
    start = date(year, m0, d0)
    end = date(year + (1 if m1 < m0 else 0), m1, d1)
    return start, min(end, today - timedelta(days=1))


def read_csv_gz(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_gz(path, rows, fields=None):
    fields = fields or sorted({k for r in rows for k in r}, key=lambda k: (k not in ("wlat", "wlon", "date", *sources.KEEP), k))
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def wx_key(name):
    return f"wx{len(WX_DAILY)}@{wx_step(name)}"


def load_manifest(today):
    """What's already here. For data from before this file existed, work it out from the files."""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    m = {"fires": {}, "weather": {}}
    for name, reg in REGIONS.items():
        rows = read_csv_gz(DATA / f"fires_{name}.csv.gz")
        if rows:
            got = {}
            for r in rows:
                fam = next((s for s in SENSORS if r.get("product", "").startswith(s)), None)
                if fam:
                    got.setdefault(fam, set()).add(int(r["acq_date"][:4]))
            seasons = {}
            for fam, ys in got.items():   # a season can run into January, so count it by its start year
                seasons[fam] = sorted({y for y in range(FIRST_YEAR, today.year + 1) if y in ys or y + 1 in ys})
            m["fires"][name] = seasons
        wx = read_csv_gz(DATA / f"weather_{name}.csv.gz")
        if wx and all(k in wx[0] for k in WX_DAILY):
            m["weather"][name] = {wx_key(name): sorted({int(r["date"][:4]) for r in wx})}
    return m


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True))


def weather_points(name, fires_rows):
    """Every weather point in the box, or for big states only those near cells that have burned before."""
    reg = REGIONS[name]
    if not reg.get("active_only"):
        return sources.wx_points(reg["bbox"], wx_step(name))
    import pandas as pd
    import grid, features as F
    fires = F.load_fires_frame(pd.DataFrame(fires_rows))
    years = list(range(FIRST_YEAR, date.today().year + 1))
    static = F.static_cells(fires, name)
    cells = F.keep_active(grid.build_cells(name), F.active_cells(fires, name, years, static))
    pts = grid.wx_points_for(cells)
    print(f"  {len(cells):,} cells have burned before; {len(pts)} weather points")
    return pts


def main():
    keyfile = DATA.parent / "key.txt"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--sensors=")), None)
    force = "--force" in sys.argv
    if args and args[0] not in REGIONS:
        key, names = args[0], args[1:]
    elif os.environ.get("FIRMS_MAP_KEY"):
        key, names = os.environ["FIRMS_MAP_KEY"], args
    elif keyfile.exists():
        key, names = keyfile.read_text().strip(), args
    else:
        print(__doc__); sys.exit(1)
    names = names or [r for r in ORDER if r in REGIONS]
    sensors = [s for s in SENSORS if not only or s in only]
    today = date.today()
    DATA.mkdir(exist_ok=True)
    cache = DATA / "cache"; cache.mkdir(exist_ok=True)
    man = load_manifest(today)
    avail = sources.firms_availability(key)
    print("FIRMS products:", ", ".join(f"{k} {v[0]}..{v[1]}" for k, v in sorted(avail.items()) if k.startswith(("VIIRS", "MODIS"))))

    for name in names:
        reg = REGIONS[name]
        years = [y for y in range(FIRST_YEAR, today.year + 1) if season_window(reg, y, today)[0] <= season_window(reg, y, today)[1]]
        have = man["fires"].setdefault(name, {})
        todo = [(s, y) for s in sensors for y in years if force or y not in have.get(s, [])]
        out = DATA / f"fires_{name}.csv.gz"
        rows = read_csv_gz(out)
        if todo:
            print(f"\n{reg['label']}: {len(todo)} sensor-seasons to download")
            new = []
            for sensor, year in todo:
                a, b = season_window(reg, year, today)
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
                    new += got
                print(f"  {sensor} {year}: {len(new):,} new points so far", flush=True)
            seen, uniq = set(), []
            for r in rows + new:   # SP and NRT can overlap at the edges
                k = (r["latitude"], r["longitude"], r["acq_date"], r["acq_time"], r["satellite"])
                if k not in seen:
                    seen.add(k); uniq.append(r)
            rows = uniq
            write_csv_gz(out, rows, sources.KEEP + ["product"])
            for sensor, year in todo:
                have[sensor] = sorted(set(have.get(sensor, [])) | {year})
            save_manifest(man)
            print(f"  saved {len(rows):,} fire points to {out.name}")

        wk = wx_key(name)
        whave = man["weather"].setdefault(name, {})
        wtodo = [y for y in years if force or y not in whave.get(wk, [])]
        if wtodo:
            pts = weather_points(name, rows)
            wout = DATA / f"weather_{name}.csv.gz"
            old = [r for r in read_csv_gz(wout) if wk in whave]   # a new field list or grid means starting over
            new = []
            for year in wtodo:
                a, b = season_window(reg, year, today)
                wf = cache / f"{name}_{wk}_{year}_{len(pts)}.json"
                if wf.exists():
                    got = json.loads(wf.read_text())
                else:
                    got = sources.weather_history(pts, a, b, WX_DAILY)
                    wf.write_text(json.dumps(got))
                new += got
                print(f"  weather {year}: {len(got):,} point-days", flush=True)
            merged = {(str(r["wlat"]), str(r["wlon"]), r["date"]): r for r in old + new}
            write_csv_gz(wout, list(merged.values()))
            prev = set(whave.get(wk, [])) if old else set()
            whave.clear(); whave[wk] = sorted(prev | set(wtodo))
            save_manifest(man)
            print(f"  saved {len(merged):,} weather rows to {wout.name}")
        if not todo and not wtodo:
            print(f"{reg['label']}: nothing new to download")
    print("\nDone.")


if __name__ == "__main__":
    main()
