"""Turn fire points + weather into one row per (cell, day) with things the model can learn from.

Timing rule: a forecast for day D is made at 6 am on day D. So every feature uses
fires up to D-1 (satellite data arrives within ~3 h of each pass) and weather for
D itself (in live runs that is the forecast; in training it is the observed value).
"""
from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from config import CELL, REGIONS, TARGET_SENSORS, DAYTIME_ONLY, STATIC_NIGHT_SHARE, STATIC_MIN_N, STATIC_DAY_SHARE
import grid

BASE = [
    "lag1", "lag2", "lag3", "sum7", "season_so_far",
    "nb1_lag1", "nb1_sum3", "nb2_sum3", "nb2_sum7",
    "dist_lag1", "dist_sum3", "reg_lag1", "reg_sum3", "reg_trend",
    "clim", "clim_total", "last_year", "depletion",
    "day", "dow",
    "rain", "rain_prev3", "rh", "tmax", "wind",
]
# extra signals, each kept only if it helps on the validation season (see train.py)
EXTRA = {
    "noaa21": ["n21_lag1", "n21_nb_lag1"],               # third VIIRS satellite, 2024 onwards
    "intensity": ["frp_sum3", "frp_nb_sum3"],             # fire radiative power: how big the fires were
    "festival": ["diwali"],                                # days from Diwali
    "modis": ["mod_lag1", "mod_nb_lag1"],                  # MODIS morning (Terra) and afternoon (Aqua) passes
    "wind_dir": ["wind_u", "wind_v"],                      # wind direction, as east and north parts
    "harvest": ["harv", "harv_change7", "harv_nb"],        # share of cropland already cut (Sentinel-2)
}
FEATURES = BASE  # what a model uses is stored with the model
DIWALI = {2019: date(2019, 10, 27), 2020: date(2020, 11, 14), 2021: date(2021, 11, 4), 2022: date(2022, 10, 24),
          2023: date(2023, 11, 12), 2024: date(2024, 10, 31), 2025: date(2025, 10, 20), 2026: date(2026, 11, 8),
          2027: date(2027, 10, 29)}


def season_dates(region, year):
    (m0, d0), (m1, d1) = REGIONS[region]["season"]
    a = date(year, m0, d0)
    b = date(year + (1 if m1 < m0 else 0), m1, d1)
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def load_fires(path):
    f = pd.read_csv(path, dtype={"acq_time": str, "satellite": str, "confidence": str})
    f["acq_date"] = pd.to_datetime(f["acq_date"]).dt.date
    return f


def static_cells(fires, region):
    """Cells that burn like a chimney, not a farm: night fires or near-daily detections."""
    iy, ix = grid.cell_of(fires.latitude, fires.longitude, REGIONS[region]["bbox"])
    f = pd.DataFrame({"iy": iy, "ix": ix, "d": fires.acq_date.to_numpy(), "night": (fires.daynight.astype(str) == "N").to_numpy()})
    ndays = max(1, f.d.nunique())
    g = f.groupby(["iy", "ix"]).agg(n=("d", "size"), days=("d", "nunique"), night=("night", "mean"))
    bad = g[((g.night >= STATIC_NIGHT_SHARE) & (g.n >= STATIC_MIN_N)) | (g.days / ndays >= STATIC_DAY_SHARE)]
    return set(map(tuple, bad.index.to_list()))


def fire_cube(fires, region, year, shape_yx, sensors=TARGET_SENSORS, static=frozenset(), value=None):
    """counts[day, iy, ix] for one season: chosen sensors, daytime, not low confidence, not a static source.
    value='frp' sums fire radiative power instead of counting."""
    days = season_dates(region, year)
    idx = {d: i for i, d in enumerate(days)}
    f = fires[fires.acq_date.isin(idx.keys()) & fires.satellite.astype(str).isin(sensors)]
    conf = f.confidence.astype(str).str.lower()
    num = pd.to_numeric(conf, errors="coerce")
    f = f[(conf != "l") & ~(num < 30)]          # VIIRS 'l', or MODIS confidence under 30
    if DAYTIME_ONLY and "daynight" in f:
        f = f[f.daynight.astype(str) != "N"]
    cube = np.zeros((len(days),) + shape_yx, dtype=np.float32)
    if len(f):
        iy, ix = grid.cell_of(f.latitude, f.longitude, REGIONS[region]["bbox"])
        di = f.acq_date.map(idx).to_numpy()
        ok = (iy >= 0) & (iy < shape_yx[0]) & (ix >= 0) & (ix < shape_yx[1])
        if static:
            ok &= np.array([(a, b) not in static for a, b in zip(iy, ix)])
        w = pd.to_numeric(f["frp"], errors="coerce").fillna(0).to_numpy(np.float32) if value == "frp" else 1
        np.add.at(cube, (di[ok], iy[ok], ix[ok]), w[ok] if value == "frp" else 1)
    return days, cube


def _ring(a, k):
    """sum over a (2k+1)^2 window minus the centre, per 2D slice"""
    n = 2 * k + 1
    return np.maximum(uniform_filter(a, size=(1, n, n), mode="constant") * n * n - a, 0)


def _lagsum(c, start, n):
    """sum of c[d-start-n+1 .. d-start] for every d, zero before the season"""
    out = np.zeros_like(c)
    cs = np.cumsum(np.concatenate([np.zeros((1,) + c.shape[1:], c.dtype), c]), axis=0)
    for d in range(c.shape[0]):
        hi = d - start + 1
        lo = max(0, hi - n)
        if hi > 0:
            out[d] = cs[hi] - cs[lo]
    return out


def weather_cube(wx, cells, days):
    """per (day, cell) arrays of weather, nearest 0.5 deg point"""
    if wx is None or len(wx) == 0:
        z = np.full((len(days), len(cells)), np.nan, np.float32)
        return {k: z.copy() for k in ("rain", "rh", "tmax", "wind", "wu", "wv")}
    w = wx.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    w = w.set_index(["wlat", "wlon", "date"])
    cols = {"rain": "precipitation_sum", "rh": "relative_humidity_2m_mean", "tmax": "temperature_2m_max", "wind": "wind_speed_10m_max",
            "wdir": "wind_direction_10m_dominant"}
    out = {}
    keys = list(zip(cells.wlat.round(3), cells.wlon.round(3)))
    for k, col in cols.items():
        tab = w[col].unstack("date") if col in w else None
        arr = np.full((len(days), len(cells)), np.nan, np.float32)
        if tab is not None:
            tab.index = pd.MultiIndex.from_tuples([(round(a, 3), round(b, 3)) for a, b in tab.index])
            sub = tab.reindex(keys)
            for i, d in enumerate(days):
                if d in sub.columns:
                    arr[i] = sub[d].to_numpy(np.float32)
        out[k] = arr
    rad = np.deg2rad(out.pop("wdir"))
    out["wu"], out["wv"] = -np.sin(rad) * out["wind"], -np.cos(rad) * out["wind"]   # direction the wind blows towards
    return out


def climatology(cubes, year, pool):
    """mean of a +/-5 day window in other seasons (from pool), aligned by calendar day"""
    others = [cubes[y] for y in pool if y != year and y in cubes]
    L = cubes[year].shape[0] if year in cubes else max(c.shape[0] for c in cubes.values())
    if not others:
        z = np.zeros((L,) + next(iter(cubes.values())).shape[1:], np.float32)
        return z, z[0]
    stack = []
    for c in others:
        c = c[:L] if c.shape[0] >= L else np.concatenate([c, np.zeros((L - c.shape[0],) + c.shape[1:], c.dtype)])
        stack.append(uniform_filter(c, size=(11, 1, 1), mode="constant"))
    clim = np.mean(stack, axis=0)
    total = np.mean([c.sum(0) for c in others], axis=0)
    return clim, total


def season_rows(region, year, cube, days, cells, cubes, pool, wx, last_day=None, h=1, aux=None, first_day=None):
    """One row per cell per target day t. Everything known at 6 am on the issue day t-h+1:
    fires up to t-h, weather (observed or forecast) for t, history from past seasons.
    h=1 is today, h=2 tomorrow, h=3 the day after."""
    aux = aux or {}
    iy, ix = cells.iy.to_numpy(), cells.ix.to_numpy()
    L = len(days)
    lag1, lag2, lag3 = _lagsum(cube, h, 1), _lagsum(cube, h + 1, 1), _lagsum(cube, h + 2, 1)
    sum7 = _lagsum(cube, h, 7)
    so_far = _lagsum(cube, h, 10_000)
    r1, r2 = _ring(cube, 1), _ring(cube, 2) + _ring(cube, 1)
    nb1_lag1, nb1_sum3 = _lagsum(r1, h, 1), _lagsum(r1, h, 3)
    nb2_sum3, nb2_sum7 = _lagsum(r2, h, 3), _lagsum(r2, h, 7)
    clim, clim_total = climatology(cubes, year, pool)
    clim = clim[:L] if clim.shape[0] >= L else np.concatenate([clim, np.repeat(clim[-1:], L - clim.shape[0], 0)])
    ly = cubes.get(year - 1)
    if ly is not None:
        ly = uniform_filter(ly, size=(7, 1, 1), mode="constant") * 7
        ly = ly[:L] if ly.shape[0] >= L else np.concatenate([ly, np.zeros((L - ly.shape[0],) + ly.shape[1:], ly.dtype)])
    reg_daily = cube[:, iy, ix].sum(1)
    rs = lambda a, b: reg_daily[max(0, a):max(0, b)].sum()
    reg_lag1 = np.array([rs(t - h, t - h + 1) for t in range(L)], np.float32)
    reg_sum3 = np.array([rs(t - h - 2, t - h + 1) for t in range(L)], np.float32)
    reg_prev3 = np.array([rs(t - h - 5, t - h - 2) for t in range(L)], np.float32)
    dist_codes, dist_idx = np.unique(cells.district.to_numpy(), return_inverse=True)
    dist_daily = np.zeros((L, len(dist_codes)), np.float32)
    for k in range(len(dist_codes)):
        dist_daily[:, k] = cube[:, iy[dist_idx == k], ix[dist_idx == k]].sum(1)
    W = weather_cube(wx, cells, days)
    rain_prev3 = np.zeros_like(W["rain"])
    for d in range(L):
        if d:
            rain_prev3[d] = np.nansum(W["rain"][max(0, d - 3):d], axis=0)
    X = {}
    if "n21" in aux:
        X["n21_lag1"] = _lagsum(aux["n21"], h, 1); X["n21_nb_lag1"] = _lagsum(_ring(aux["n21"], 1) + aux["n21"], h, 1)
    if "frp" in aux:
        X["frp_sum3"] = _lagsum(aux["frp"], h, 3); X["frp_nb_sum3"] = _lagsum(_ring(aux["frp"], 1), h, 3)
    if "modis" in aux:
        X["mod_lag1"] = _lagsum(aux["modis"], h, 1); X["mod_nb_lag1"] = _lagsum(_ring(aux["modis"], 1) + aux["modis"], h, 1)
    if "harvest" in aux:   # share of cropland cut, as known on each day (NaN where no clear image yet)
        hv = aux["harvest"]
        sh = lambda a, k: np.concatenate([np.full((k,) + a.shape[1:], np.nan, a.dtype), a[:-k]]) if k else a
        X["harv"] = sh(hv, h); X["harv_change7"] = sh(hv, h) - sh(hv, h + 7)
        X["harv_nb"] = sh(uniform_filter(np.nan_to_num(hv), size=(1, 3, 3), mode="constant"), h)
    dw = DIWALI.get(year)

    lo = max(h, 1) if first_day is None else days.index(first_day)
    upto = L if last_day is None else days.index(last_day) + 1
    frames = []
    for t in range(lo, upto):
        f = {
            "lag1": lag1[t, iy, ix], "lag2": lag2[t, iy, ix], "lag3": lag3[t, iy, ix], "sum7": sum7[t, iy, ix],
            "season_so_far": so_far[t, iy, ix],
            "nb1_lag1": nb1_lag1[t, iy, ix], "nb1_sum3": nb1_sum3[t, iy, ix],
            "nb2_sum3": nb2_sum3[t, iy, ix], "nb2_sum7": nb2_sum7[t, iy, ix],
            "dist_lag1": dist_daily[t - h, dist_idx] if t - h >= 0 else np.zeros(len(cells), np.float32),
            "dist_sum3": dist_daily[max(0, t - h - 2):max(0, t - h + 1), :].sum(0)[dist_idx],
            "reg_lag1": np.full(len(cells), np.log1p(reg_lag1[t])), "reg_sum3": np.full(len(cells), np.log1p(reg_sum3[t])),
            "reg_trend": np.full(len(cells), np.log1p(reg_sum3[t]) - np.log1p(reg_prev3[t])),
            "clim": clim[t, iy, ix], "clim_total": clim_total[iy, ix],
            "last_year": ly[t, iy, ix] if ly is not None else np.full(len(cells), np.nan, np.float32),
            "depletion": so_far[t, iy, ix] / (clim_total[iy, ix] + 1.0),
            "day": np.full(len(cells), t, np.float32), "dow": np.full(len(cells), days[t].weekday(), np.float32),
            "rain": W["rain"][t], "rain_prev3": rain_prev3[t], "rh": W["rh"][t], "tmax": W["tmax"][t], "wind": W["wind"][t],
            "wind_u": W["wu"][t], "wind_v": W["wv"][t],
            "diwali": np.full(len(cells), float(np.clip((days[t] - dw).days, -40, 40)) if dw else np.nan, np.float32),
        }
        for k, arr in X.items():
            f[k] = arr[t, iy, ix]
        df = pd.DataFrame(f)
        df["y"] = cube[t, iy, ix]
        df["date"] = days[t]
        df["year"] = year
        df["iy"], df["ix"] = iy, ix
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def grid_shape(region):
    w, s, e, n = REGIONS[region]["bbox"]
    return int(np.ceil((n - s) / CELL)), int(np.ceil((e - w) / CELL))


def aux_cubes(fires, region, year, static, harvest=None):
    """Extra per-day grids for the optional signals. Only builds what the data has."""
    shp = grid_shape(region)
    out = {}
    sats = set(fires.satellite.astype(str).unique())
    def maybe(c):   # a sensor that saw nothing all season wasn't flying yet: unknown, not zero
        return c if c.sum() > 0 else np.full_like(c, np.nan)
    if sats & {"N21", "2"}:
        out["n21"] = maybe(fire_cube(fires, region, year, shp, sensors=["N21", "2"], static=static)[1])
    if "frp" in fires:
        out["frp"] = fire_cube(fires, region, year, shp, static=static, value="frp")[1]
    if sats & {"Terra", "Aqua", "T", "A"}:
        out["modis"] = maybe(fire_cube(fires, region, year, shp, sensors=["Terra", "Aqua", "T", "A"], static=static)[1])
    if harvest is not None:
        hv = harvest_cube(harvest, region, year, shp)
        if hv is not None:
            out["harvest"] = hv
    return out


def harvest_cube(harvest, region, year, shp):
    """harvest table (iy, ix, date, harv) -> [day, iy, ix], carrying the last clear value forward."""
    days = season_dates(region, year)
    h = harvest[(harvest.date >= days[0] - timedelta(days=30)) & (harvest.date <= days[-1])]
    if not len(h):
        return None
    cube = np.full((len(days),) + shp, np.nan, np.float32)
    last = np.full(shp, np.nan, np.float32)
    byday = {d: g for d, g in h.groupby("date")}
    pre = h[h.date < days[0]].sort_values("date")
    for r in pre.itertuples():
        if 0 <= r.iy < shp[0] and 0 <= r.ix < shp[1]:
            last[r.iy, r.ix] = r.harv
    for i, d in enumerate(days):
        g = byday.get(d)
        if g is not None:
            ok = (g.iy >= 0) & (g.iy < shp[0]) & (g.ix >= 0) & (g.ix < shp[1])
            last[g.iy[ok].to_numpy(), g.ix[ok].to_numpy()] = g.harv[ok].to_numpy()
        cube[i] = last
    return cube


def all_cubes(fires, region, years, static=frozenset()):
    shp = grid_shape(region)
    cubes, dayss = {}, {}
    for y in years:
        dayss[y], cubes[y] = fire_cube(fires, region, y, shp, static=static)
    return cubes, dayss
