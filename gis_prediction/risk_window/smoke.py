"""Where will today's smoke go? A rough answer from the wind forecast.

From the cells where fires are expected today, it follows the hourly wind (an average of the
10 m and 925 hPa winds, roughly the layer the smoke sits in) for 36 hours from 2 pm, and lists the
towns the smoke passes. It also reads the CAMS air quality forecast (PM2.5, all sources, not only
farm fires) for those towns. Both come from Open-Meteo, free and without a key.
This is a guide to where smoke drifts, not a pollution model.
"""
import json
from datetime import datetime, timedelta
import numpy as np
from config import REGIONS
import sources

TOWNS = {
    "punjab_haryana": [
        ("Delhi", 28.61, 77.21), ("Chandigarh", 30.73, 76.78), ("Ludhiana", 30.90, 75.85), ("Amritsar", 31.63, 74.87),
        ("Jalandhar", 31.33, 75.58), ("Patiala", 30.34, 76.39), ("Bathinda", 30.21, 74.95), ("Sangrur", 30.25, 75.84),
        ("Firozpur", 30.93, 74.61), ("Moga", 30.82, 75.17), ("Karnal", 29.69, 76.99), ("Panipat", 29.39, 76.97),
        ("Kurukshetra", 29.97, 76.88), ("Ambala", 30.38, 76.78), ("Hisar", 29.15, 75.72), ("Sirsa", 29.53, 75.03),
        ("Rohtak", 28.90, 76.61), ("Sonipat", 28.99, 77.02), ("Gurugram", 28.46, 77.03), ("Kaithal", 29.80, 76.40),
        ("Jind", 29.32, 76.31), ("Fatehabad", 29.52, 75.45)],
    "purba_medinipur": [
        ("Kolkata", 22.57, 88.36), ("Howrah", 22.59, 88.31), ("Haldia", 22.06, 88.07), ("Tamluk", 22.30, 87.92),
        ("Kolaghat", 22.43, 87.87), ("Panskura", 22.42, 87.70), ("Kharagpur", 22.34, 87.32), ("Medinipur", 22.42, 87.32),
        ("Contai", 21.78, 87.75), ("Egra", 21.90, 87.53), ("Digha", 21.63, 87.51), ("Uluberia", 22.47, 88.11),
        ("Ghatal", 22.66, 87.73)],
    "uttar_pradesh": [
        ("Delhi", 28.61, 77.21), ("Noida", 28.54, 77.39), ("Ghaziabad", 28.67, 77.45), ("Meerut", 28.98, 77.71),
        ("Muzaffarnagar", 29.47, 77.70), ("Saharanpur", 29.96, 77.55), ("Moradabad", 28.84, 78.77), ("Rampur", 28.80, 79.03),
        ("Bareilly", 28.37, 79.43), ("Pilibhit", 28.63, 79.80), ("Shahjahanpur", 27.88, 79.91), ("Lakhimpur", 27.95, 80.78),
        ("Sitapur", 27.57, 80.68), ("Hardoi", 27.40, 80.13), ("Aligarh", 27.88, 78.08), ("Mathura", 27.49, 77.67),
        ("Agra", 27.18, 78.01), ("Firozabad", 27.15, 78.40), ("Etawah", 26.78, 79.02), ("Kanpur", 26.45, 80.33),
        ("Lucknow", 26.85, 80.95), ("Ayodhya", 26.80, 82.20), ("Gorakhpur", 26.76, 83.37), ("Prayagraj", 25.44, 81.85),
        ("Varanasi", 25.32, 82.97), ("Jhansi", 25.45, 78.57)],
    "madhya_pradesh": [
        ("Bhopal", 23.26, 77.41), ("Indore", 22.72, 75.86), ("Jabalpur", 23.18, 79.99), ("Gwalior", 26.22, 78.18),
        ("Ujjain", 23.18, 75.78), ("Sagar", 23.84, 78.74), ("Rewa", 24.53, 81.30), ("Satna", 24.60, 80.83),
        ("Narmadapuram", 22.75, 77.73), ("Vidisha", 23.52, 77.81), ("Sehore", 23.20, 77.08), ("Raisen", 23.33, 77.79),
        ("Harda", 22.34, 77.10), ("Dewas", 22.97, 76.05), ("Morena", 26.50, 78.00), ("Bhind", 26.56, 78.78),
        ("Shivpuri", 25.42, 77.66), ("Guna", 24.65, 77.31), ("Katni", 23.83, 80.39), ("Chhindwara", 22.06, 78.94),
        ("Seoni", 22.09, 79.55), ("Balaghat", 21.81, 80.18), ("Narsinghpur", 22.95, 79.19), ("Datia", 25.67, 78.46)],
    "rajasthan": [
        ("Jaipur", 26.91, 75.79), ("Sri Ganganagar", 29.90, 73.88), ("Hanumangarh", 29.58, 74.32), ("Bikaner", 28.02, 73.31),
        ("Jodhpur", 26.24, 73.02), ("Kota", 25.21, 75.86), ("Bundi", 25.44, 75.64), ("Baran", 25.10, 76.51),
        ("Alwar", 27.55, 76.60), ("Bharatpur", 27.22, 77.49), ("Ajmer", 26.45, 74.64), ("Sikar", 27.61, 75.14),
        ("Jhunjhunu", 28.13, 75.40), ("Churu", 28.30, 74.95), ("Dholpur", 26.70, 77.89), ("Sawai Madhopur", 26.02, 76.35),
        ("Tonk", 26.17, 75.79), ("Jhalawar", 24.60, 76.16), ("Delhi", 28.61, 77.21)],
}
WIND_VARS = ["wind_speed_10m", "wind_direction_10m", "wind_speed_925hPa", "wind_direction_925hPa"]
START_HOUR, HOURS, HIT_KM = 14, 36, 20


def pm25_band(v):
    """Indian national AQI bands for 24-hour PM2.5 (µg/m³)."""
    for lim, name in ((30, "good"), (60, "satisfactory"), (90, "moderate"), (120, "poor"), (250, "very poor")):
        if v <= lim:
            return name
    return "severe"


def _hourly(base, pts, params):
    out = []
    for i in range(0, len(pts), 50):
        chunk = pts[i:i + 50]
        q = (f"latitude={','.join(str(p[0]) for p in chunk)}&longitude={','.join(str(p[1]) for p in chunk)}"
             f"&timezone=Asia%2FKolkata&{params}")
        js = json.loads(sources._get(f"{base}?{q}"))
        out += js if isinstance(js, list) else [js]
    return out


def fetch(region, today):
    """Hourly winds on a 1 degree grid around the region, and PM2.5 forecasts for its towns."""
    w, s, e, n = REGIONS[region]["bbox"]
    grid = sources.wx_points((w - 1.5, s - 1.5, e + 1.5, n + 1.5), 1.0)
    wind = _hourly("https://api.open-meteo.com/v1/forecast", grid,
                   f"hourly={','.join(WIND_VARS)}&wind_speed_unit=ms&start_date={today}&end_date={today + timedelta(days=2)}")
    towns = TOWNS.get(region, [])
    aq = _hourly("https://air-quality-api.open-meteo.com/v1/air-quality", [(t[1], t[2]) for t in towns],
                 f"hourly=pm2_5&start_date={today}&end_date={today + timedelta(days=1)}") if towns else []
    return {"grid": grid, "wind": wind, "aq": aq}


def _wind_field(raw):
    """u, v (m/s, towards) per hour per grid point, averaging the surface and 925 hPa winds."""
    g = np.array(raw["grid"], float)
    times = raw["wind"][0]["hourly"]["time"]
    U = np.zeros((len(times), len(g))); V = np.zeros_like(U)
    for j, loc in enumerate(raw["wind"]):
        h = loc["hourly"]
        for sp, dr, wt in (("wind_speed_10m", "wind_direction_10m", .5), ("wind_speed_925hPa", "wind_direction_925hPa", .5)):
            s = np.array([np.nan if x is None else x for x in h[sp]], float)
            d = np.deg2rad(np.array([np.nan if x is None else x for x in h[dr]], float))
            U[:, j] += wt * np.nan_to_num(-s * np.sin(d)); V[:, j] += wt * np.nan_to_num(-s * np.cos(d))
    return g, times, U, V


def _interp(g, vals, lat, lon):
    """Inverse-distance weighted value from the 4 nearest grid points."""
    d2 = (g[:, 0] - lat) ** 2 + ((g[:, 1] - lon) * np.cos(np.radians(lat))) ** 2
    idx = np.argsort(d2)[:4]
    wts = 1 / (d2[idx] + 1e-4)
    return float((vals[idx] * wts).sum() / wts.sum())


def sources_from(pred, cell=0.05, max_n=8):
    """Clusters of expected fires (sum of expected detections per 0.5 degree square)."""
    df = pred.assign(bx=(pred.lat // 0.5), by=(pred.lon // 0.5))
    agg = df.groupby(["bx", "by"]).apply(lambda g: (float(g.mu.sum()), float((g.lat * g.mu).sum() / max(g.mu.sum(), 1e-9)),
                                                    float((g.lon * g.mu).sum() / max(g.mu.sum(), 1e-9)),
                                                    g.groupby("district").mu.sum().idxmax()), include_groups=False)
    rows = sorted(agg.tolist(), reverse=True)
    total = sum(r[0] for r in rows)
    keep = [r for r in rows if r[0] >= max(1.5, 0.03 * total)][:max_n]
    return keep


def trajectories(raw, srcs, today):
    g, times, U, V = _wind_field(raw)
    t0 = f"{today}T{START_HOUR:02d}:00"
    i0 = times.index(t0) if t0 in times else 0
    paths = []
    for mu, lat, lon, dist in srcs:
        pts = [(lat, lon, 0)]
        for k in range(min(HOURS, len(times) - i0 - 1)):
            u, v = _interp(g, U[i0 + k], lat, lon), _interp(g, V[i0 + k], lat, lon)
            lat += v * 3600 / 111320
            lon += u * 3600 / (111320 * np.cos(np.radians(lat)))
            pts.append((lat, lon, k + 1))
        paths.append({"from": dist, "fires": round(mu, 1), "pts": [[round(float(a), 3), round(float(b), 3), h] for a, b, h in pts]})
    return paths, times[i0] if times else t0


def towns_hit(region, paths, start):
    t0 = datetime.fromisoformat(start)
    hits = []
    for name, la, lo in TOWNS.get(region, []):
        best = None
        for p in paths:
            for a, b, h in p["pts"]:
                km = 111.32 * np.hypot(a - la, (b - lo) * np.cos(np.radians(la)))
                if km <= HIT_KM and (best is None or h < best[0]):
                    best = (h, p["from"])
                    break
        if best:
            hits.append({"town": name, "lat": la, "lon": lo, "eta": (t0 + timedelta(hours=best[0])).strftime("%Y-%m-%dT%H:00"),
                         "hours": best[0], "from": best[1]})
    return sorted(hits, key=lambda x: x["hours"])


def pm25(region, raw, today):
    out = {}
    for (name, la, lo), loc in zip(TOWNS.get(region, []), raw.get("aq", [])):
        h = loc.get("hourly", {})
        vals = {}
        for t, v in zip(h.get("time", []), h.get("pm2_5", [])):
            if v is not None:
                vals.setdefault(t[:10], []).append(v)
        day = {d: round(float(np.mean(v))) for d, v in vals.items() if len(v) >= 12}
        if day:
            out[name] = day
    return out


def forecast(region, today, pred, raw=None, live=True):
    """Everything the dashboard needs for the smoke layer, or None if the forecast can't be fetched."""
    if raw is None:
        if not live:
            return None
        try:
            raw = fetch(region, today)
        except Exception as e:
            print(f"{region}: smoke forecast skipped ({e})")
            return None
    try:
        srcs = sources_from(pred[pred.mu > 0])
        paths, start = trajectories(raw, srcs, today)
        hit = towns_hit(region, paths, start)
        pm = pm25(region, raw, today)
        tom = str(today + timedelta(days=1))
        towns = []
        for name, la, lo in TOWNS.get(region, []):
            h = next((x for x in hit if x["town"] == name), None)
            v = pm.get(name, {})
            towns.append({"town": name, "lat": la, "lon": lo, "eta": h["eta"] if h else None, "from": h["from"] if h else None,
                          "pm_today": v.get(str(today)), "pm_tomorrow": v.get(tom),
                          "band": pm25_band(v[tom]) if tom in v else None})
        towns.sort(key=lambda t: (t["eta"] is None, t["eta"] or "", -(t["pm_tomorrow"] or 0)))
        return {"start": start, "hours": HOURS, "paths": paths, "towns": towns}
    except Exception as e:
        print(f"{region}: smoke forecast failed ({e})")
        return None
