"""Run every morning. Pulls the newest fires and weather, predicts today, plans visits,
checks yesterday's forecast against the satellites, and rebuilds the dashboard.

    FIRMS_MAP_KEY=xxxx python daily.py                    # every region, today
    python daily.py --region punjab_haryana --date 2026-11-20
    python daily.py --backfill                            # score every day of this season so far
    python daily.py --teams 8 --stops 5
Optional: TELEGRAM_TOKEN plus TELEGRAM_CHAT (or TELEGRAM_CHAT_PUNJAB_HARYANA,
TELEGRAM_CHAT_PURBA_MEDINIPUR for one group per region) send the brief to Telegram.
"""
import argparse, gzip, csv, json, os, pickle, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
from config import ROOT, DATA, OUT, MODELS, REGIONS, ORDER, CELL, SENSORS, WX_DAILY
import grid, features as F, dispatch, brief, sources, smoke

IST = timezone(timedelta(hours=5, minutes=30))


def season_of(region, d):
    for y in (d.year, d.year - 1):
        ds = F.season_dates(region, y)
        if ds[0] <= d <= ds[-1]:
            return y, ds
    return None, None


def fetch_live(region, key):
    """Last 5 days of near-real-time points, merged into data/live_<region>.csv.gz"""
    reg = REGIONS[region]
    avail = sources.firms_availability(key)
    rows = []
    for s in SENSORS:
        prod = f"{s}_NRT"
        if prod in avail:
            rows += sources.firms_area(key, prod, reg["bbox"], avail[prod][1] - timedelta(days=4), 5)
    path = DATA / f"live_{region}.csv.gz"
    old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame(rows, dtype=str)], ignore_index=True)
    if len(new):
        new = new.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "satellite"])
        new.to_csv(path, index=False, compression="gzip")
    return len(rows)


def fetch_weather(region, cells):
    pts = grid.wx_points_for(cells)
    wx = pd.DataFrame(sources.weather_forecast(pts, WX_DAILY))
    path = DATA / f"wxlive_{region}.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    allw = pd.concat([old, wx], ignore_index=True).drop_duplicates(["wlat", "wlon", "date"], keep="last")
    allw.to_csv(path, index=False, compression="gzip")


def load_all(region):
    parts = [F.load_fires(p) for p in (DATA / f"fires_{region}.csv.gz", DATA / f"live_{region}.csv.gz") if p.exists()]
    fires = pd.concat(parts, ignore_index=True).drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    wparts = [pd.read_csv(p) for p in (DATA / f"weather_{region}.csv.gz", DATA / f"wxlive_{region}.csv.gz") if p.exists()]
    wx = pd.concat(wparts, ignore_index=True).drop_duplicates(["wlat", "wlon", "date"], keep="last") if wparts else None
    return fires, wx


def load_places(region):
    p = ROOT / "geo" / f"places_{region}.csv"
    return pd.read_csv(p, dtype={"pin": str}) if p.exists() else None


def name_cells(cells, places):
    """A recognisable name per cell: the village (post office) nearest the cell centre, if one is within ~6 km."""
    if places is None or not len(places):
        return {}
    P = places[["lat", "lon"]].to_numpy()
    out = {}
    for r in cells.itertuples():
        d2 = (P[:, 0] - r.lat) ** 2 + ((P[:, 1] - r.lon) * np.cos(np.radians(r.lat))) ** 2
        i = int(d2.argmin())
        if d2[i] <= 0.055 ** 2:
            out[(r.iy, r.ix)] = places.name.iat[i]
    return out


def predict_season(region, today, fires, wx, cells, model, harvest=None, backfill=False):
    """Forecasts made this morning for today, tomorrow and the day after.
    With backfill, also today's-horizon forecasts for every earlier day of the season (for the scoreboard)."""
    year, days = season_of(region, today)
    if year is None:
        return None, None, None
    pool = model["seasons"]
    shp = F.grid_shape(region)
    static = set(map(tuple, model.get("static", [])))
    cubes = {y: F.fire_cube(fires, region, y, shp, static=static)[1] for y in pool}
    _, cubes[year] = F.fire_cube(fires, region, year, shp, static=static)
    aux = F.aux_cubes(fires, region, year, static, harvest)
    feats = model["features"]
    others = [y for y in pool if y != year]
    out = []
    for h, m in sorted(model["h"].items()):
        target = today + timedelta(days=h - 1)
        if target not in days:
            continue
        first = days[max(h, 1)] if (backfill and h == 1) else target
        rows = F.season_rows(region, year, cubes[year], days, cells, cubes, others, wx, last_day=target, h=h, aux=aux, first_day=first)
        rows["p"] = m["clf"].predict_proba(rows[feats])[:, 1]
        rows["mu"] = m["reg"].predict(rows[feats])
        rows["h"] = h
        out.append(rows)
    rows = pd.concat(out, ignore_index=True).merge(cells[["iy", "ix", "lat", "lon", "district"]].reset_index(drop=True), on=["iy", "ix"])
    return rows, cubes[year], days


def score_day(g, k=50):
    """How did the forecast for one day do once the satellites came in?"""
    tot = float(g.y.sum())
    top = g.sort_values("p", ascending=False).head(k)
    pers = (g.lag1 * 100 + g.nb1_lag1 * 10 + g.sum7 + 0.01 * g.clim)
    ptop = g.assign(_s=pers).sort_values("_s", ascending=False).head(k)
    by_d = g.groupby("district").agg(pred=("mu", "sum"), actual=("y", "sum")).sort_values("actual", ascending=False)
    lead = by_d.head(1)
    return {"date": str(g.date.iloc[0]), "flagged": k, "hits": int((top.y > 0).sum()), "precision": float((top.y > 0).mean()) if tot else None,
            "catch": float(top.y.sum() / tot) if tot else None, "persist_catch": float(ptop.y.sum() / tot) if tot else None,
            "predicted_total": round(float(g.mu.sum()), 1), "actual_total": int(tot),
            "top_district": lead.index[0] if len(lead) and tot else None,
            "top_district_pred": round(float(lead.pred.iloc[0]), 1) if len(lead) and tot else None,
            "top_district_actual": int(lead.actual.iloc[0]) if len(lead) and tot else None}


def green_villages(region, year, today, days, cube, fires, static, cells, cname):
    """Places that burned by this date last season and show no fires so far this season. Celebrate them."""
    di = days.index(today)
    if di < 7:
        return None
    shp = F.grid_shape(region)
    ldays, lcube = F.fire_cube(fires, region, year - 1, shp, static=static)
    if not lcube.sum():
        return None
    now, last = cube[:di].sum(0), lcube[:di].sum(0)
    iy, ix = cells.iy.to_numpy(), cells.ix.to_numpy()
    c = cells.assign(now=now[iy, ix], last=last[iy, ix])
    dist = c.groupby("district").agg(last=("last", "sum"), now=("now", "sum"))
    dist = dist[dist["last"] >= 5].sort_values("last", ascending=False).head(14)
    g = c[(c["last"] >= 3) & (c["now"] == 0)].sort_values("last", ascending=False)
    vill, seen = [], set()
    for r in g.itertuples():
        name = cname.get((r.iy, r.ix))
        if not name or (name, r.district) in seen:
            continue
        seen.add((name, r.district))
        vill.append([name, r.district, int(r.last), round(float(r.lat), 3), round(float(r.lon), 3)])
        if len(vill) >= 30:
            break
    return {"last_season": year - 1, "through": str(today - timedelta(days=1)), "season_day": di,
            "now_total": int(c.now.sum()), "last_total": int(c["last"].sum()), "green_cells": int(len(g)),
            "districts": [[d, int(r["last"]), int(r["now"])] for d, r in dist.iterrows()], "villages": vill}


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def telegram(text, region):
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get(f"TELEGRAM_CHAT_{region.upper()}") or os.environ.get("TELEGRAM_CHAT")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=30)


def run_region(region, today, teams, stops, backfill, live):
    reg = REGIONS[region]
    mp = MODELS / f"{region}.pkl"
    if not mp.exists():
        print(f"{region}: no model yet, run train.py first"); return None
    model = pickle.loads(mp.read_bytes())
    cells = grid.build_cells(region)
    if model.get("cells"):
        cells = F.keep_active(cells, set(map(tuple, model["cells"])))
    key = os.environ.get("FIRMS_MAP_KEY")
    if live and key:
        try:
            print(f"{region}: {fetch_live(region, key)} new satellite points")
            fetch_weather(region, cells)
        except Exception as e:  # keep going with what we have
            print(f"{region}: live fetch failed ({e}), using stored data")
    fires, wx = load_all(region)
    hp = DATA / f"harvest_{region}.csv.gz"
    harvest = None
    if hp.exists():
        harvest = pd.read_csv(hp); harvest["date"] = pd.to_datetime(harvest["date"]).dt.date
    places = load_places(region)
    cname = name_cells(cells, places)
    od = OUT / region; od.mkdir(parents=True, exist_ok=True)
    rows, cube, days = predict_season(region, today, fires, wx, cells, model, harvest, backfill)
    report = json.loads((MODELS / f"{region}_report.json").read_text()) if (MODELS / f"{region}_report.json").exists() else None
    extras = {"replay": read_json(MODELS / f"{region}_replay.json"), "people": read_json(DATA / f"people_{region}.json"),
              "bbox": reg["bbox"], "cell": CELL, "lang": reg["lang"],
              "season": [f"{m:02d}-{d:02d}" for m, d in reg["season"]]}
    if rows is None:
        latest = {"region": region, "label": reg["label"], "date": str(today), "in_season": False, "report": report} | extras
        (od / "latest.json").write_text(json.dumps(latest)); return latest

    rows["place"] = [cname.get((a, b), "") for a, b in zip(rows.iy, rows.ix)]
    pred = rows[(rows.date == today) & (rows.h == 1)].set_index(["iy", "ix"], drop=False)
    ahead = {h: rows[(rows.h == h) & (rows.date == today + timedelta(days=int(h) - 1))].set_index(["iy", "ix"], drop=False)
             for h in sorted(rows.h.unique())}
    allp = pred[["iy", "ix", "lat", "lon", "district", "place", "p", "mu"]].copy()
    for h, g in ahead.items():
        if h > 1:
            allp[f"p_{g.date.iloc[0]}"] = g.p.reindex(allp.index).to_numpy()
    allp.round(4).to_csv(od / f"pred_{today}.csv", index=False)

    # scoreboard: every finished day of this season that we have predictions for
    sb_path = od / "scoreboard.csv"
    sb = pd.read_csv(sb_path) if sb_path.exists() else pd.DataFrame()
    done_days = [d for d in days if d < today and (backfill or d == today - timedelta(days=1))]
    new = [score_day(g) for d, g in rows[(rows.h == 1) & rows.date.isin(done_days)].groupby("date")]
    if new:
        sb = pd.concat([sb, pd.DataFrame(new)], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        sb.to_csv(sb_path, index=False)
    yday = sb[sb.date == str(today - timedelta(days=1))].astype(object).where(lambda x: pd.notna(x), None).to_dict("records") if "date" in sb else []
    yscore = yday[0] if yday else None

    plan, summary = dispatch.plan(pred, teams, stops)
    ahead_tot = [(["today", "tomorrow", "day after"][h - 1], float(g.mu.sum())) for h, g in ahead.items() if h > 1]
    text = brief.make(reg["label"], reg["lang"], today, pred, plan, summary, yscore, ahead_tot)
    (od / f"brief_{today}.txt").write_text(text)

    w, s, e, n = reg["bbox"]
    yfires = []
    di = days.index(today)
    if di:
        ys, xs = np.nonzero(cube[di - 1])
        inside = set(zip(cells.iy, cells.ix))
        yfires = [[round(s + (a + .5) * CELL, 3), round(w + (b + .5) * CELL, 3), int(cube[di - 1, a, b])] for a, b in zip(ys, xs) if (a, b) in inside]
    static = set(map(tuple, model.get("static", [])))
    green = green_villages(region, season_of(region, today)[0], today, days, cube, fires, static, cells, cname)
    smoke_fc = smoke.forecast(region, today, pred, live=live)
    dn = sorted(pred.district.unique()); di_of = {d: i for i, d in enumerate(dn)}
    labels = {1: "today", 2: "tomorrow", 3: "day after"}

    def day_block(g, h):
        cand = g.sort_values("p", ascending=False).head(400)
        return {"h": int(h), "label": labels[h], "date": str(g.date.iloc[0]), "total_mu": round(float(g.mu.sum()), 1),
                "risk": [[int(r.iy), int(r.ix), round(float(r.p), 3), round(float(r.mu), 2), di_of[r.district]] for r in g.itertuples() if r.p >= 0.005],
                "candidates": [[int(r.iy), int(r.ix), round(float(r.lat), 4), round(float(r.lon), 4), r.district, round(float(r.p), 3), round(float(r.mu), 2), r.place] for r in cand.itertuples()],
                "districts": [[d, round(float(v), 1)] for d, v in g.groupby("district").mu.sum().sort_values(ascending=False).head(8).items()]}
    dayblocks = [day_block(g, h) for h, g in ahead.items()]
    cand = pred.sort_values("p", ascending=False).head(400)
    latest = {
        "region": region, "label": reg["label"], "lang": reg["lang"], "date": str(today), "in_season": True,
        "bbox": reg["bbox"], "cell": CELL, "season_day": di + 1, "season_len": len(days),
        "dnames": dn,
        "risk": [[int(r.iy), int(r.ix), round(float(r.p), 3), round(float(r.mu), 2), di_of[r.district]] for r in pred.itertuples() if r.p >= 0.005],
        "candidates": dayblocks[0]["candidates"], "days": dayblocks,
        "places": [[r.name, r.pin, r.district, float(r.lat), float(r.lon), int(r.iy), int(r.ix)] for r in places.itertuples()] if places is not None else [],
        "site_url": os.environ.get("SATARK_SITE_URL", ""),
        "total_mu": round(float(pred.mu.sum()), 1),
        "districts": [[d, round(float(v), 1)] for d, v in pred.groupby("district").mu.sum().sort_values(ascending=False).head(8).items()],
        "yesterday_fires": yfires, "yesterday": yscore,
        "scoreboard": sb.astype(object).where(pd.notna(sb), None).to_dict("records") if len(sb) else [],
        "plan": [[int(r.team), int(r.order), r.district, round(float(r.lat), 4), round(float(r.lon), 4), round(float(r.p), 3), r.place] for r in plan.itertuples()],
        "summary": summary, "brief": text, "report": report, "teams_default": teams, "stops_default": stops,
        "model": {"seasons": model["seasons"], "trained": model["trained"], "signals": model.get("groups", [])},
        "generated": datetime.now(IST).strftime("%d %b %Y, %H:%M IST"),
        "green": green, "smoke": smoke_fc,
    } | extras
    (od / "latest.json").write_text(json.dumps(latest))
    print(f"{region}: {today} expected {summary.get('expected_fires_total')} detections; brief at {od / f'brief_{today}.txt'}")
    if live:
        try:
            telegram(text, region)
        except Exception as ex:
            print("telegram failed:", ex)
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", action="append")
    ap.add_argument("--date")
    ap.add_argument("--teams", type=int, default=5)
    ap.add_argument("--stops", type=int, default=4)
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--offline", action="store_true", help="don't download anything")
    a = ap.parse_args()
    today = date.fromisoformat(a.date) if a.date else datetime.now(IST).date()
    out = {}
    for r in a.region or [x for x in ORDER if x in REGIONS]:
        res = run_region(r, today, a.teams, a.stops, a.backfill, not a.offline)
        if res:
            out[r] = res
    import dashboard
    dashboard.build(out)


if __name__ == "__main__":
    main()
