"""Downloads: NASA FIRMS fire points and Open-Meteo weather. Standard library only."""
import csv, io, json, time, urllib.request, urllib.error
from datetime import date, timedelta

FIRMS = "https://firms.modaps.eosdis.nasa.gov/api"
KEEP = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight"]


def _get(url, tries=4, timeout=90):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "satark-risk-window/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"failed after {tries} tries: {url.split('/api/')[0]}... ({last})")


def firms_availability(key):
    """{'VIIRS_SNPP_SP': (min_date, max_date), ...}"""
    txt = _get(f"{FIRMS}/data_availability/csv/{key}/ALL")
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            out[row["data_id"]] = (date.fromisoformat(row["min_date"]), date.fromisoformat(row["max_date"]))
        except (KeyError, ValueError):
            continue
    if not out:
        raise RuntimeError("FIRMS did not return availability. Check the MAP_KEY: " + txt[:200])
    return out


def pick_product(sensor, day, avail):
    """Science-quality (SP) when it covers the day, otherwise near real time (NRT)."""
    for suffix in ("SP", "NRT"):
        k = f"{sensor}_{suffix}"
        if k in avail and avail[k][0] <= day <= avail[k][1]:
            return k
    return None


def firms_area(key, product, bbox, start, days):
    """Fire points in bbox for start .. start + days - 1 (days <= 5)."""
    area = ",".join(f"{v:g}" for v in bbox)
    txt = _get(f"{FIRMS}/area/csv/{key}/{product}/{area}/{days}/{start.isoformat()}")
    if txt.lstrip().lower().startswith(("invalid", "error", "<")):
        raise RuntimeError(f"FIRMS error for {product} {start}: {txt[:160]}")
    rows = []
    for r in csv.DictReader(io.StringIO(txt)):
        rows.append({k: r.get(k, "") for k in KEEP} | {"product": product})
    return rows


def daterange(a, b, step):
    d = a
    while d <= b:
        yield d, min(step, (b - d).days + 1)
        d += timedelta(days=step)


def wx_points(bbox, step):
    w, s, e, n = bbox
    pts = []
    lat = s + step / 2
    while lat < n:
        lon = w + step / 2
        while lon < e:
            pts.append((round(lat, 3), round(lon, 3)))
            lon += step
        lat += step
    return pts


def _meteo(base, pts, params):
    out = []
    for i in range(0, len(pts), 50):            # 50 locations per call keeps URLs short
        chunk = pts[i:i + 50]
        q = (f"latitude={','.join(str(p[0]) for p in chunk)}&longitude={','.join(str(p[1]) for p in chunk)}"
             f"&timezone=Asia%2FKolkata&{params}")
        js = json.loads(_get(f"{base}?{q}"))
        if isinstance(js, dict):
            js = [js]
        for p, loc in zip(chunk, js):
            d = loc["daily"]
            for j, day in enumerate(d["time"]):
                out.append({"wlat": p[0], "wlon": p[1], "date": day} | {k: d[k][j] for k in d if k != "time"})
        time.sleep(1)
    return out


def weather_history(pts, start, end, daily):
    return _meteo("https://archive-api.open-meteo.com/v1/archive", pts,
                  f"start_date={start}&end_date={end}&daily={','.join(daily)}")


def weather_forecast(pts, daily, past_days=7, forecast_days=3):
    return _meteo("https://api.open-meteo.com/v1/forecast", pts,
                  f"daily={','.join(daily)}&past_days={past_days}&forecast_days={forecast_days}")
