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
import grid, features as F, dispatch, brief, sources, smoke, train

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


def predict_season(region, today, fires, wx, cells, model, harvest=None, backfill=False, since=None):
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
        first = target
        if h == 1:   # also earlier days still to check against the satellites (with backfill, the whole season)
            ti = days.index(target)
            first = days[1] if backfill else days[max(1, min(ti - 3, days.index(since) if since in days else ti))] if ti else target
        rows = F.season_rows(region, year, cubes[year], days, cells, cubes, others, wx, last_day=target, h=h, aux=aux, first_day=first)
        rows["p"], rows["mu"] = train.predict(m, rows[feats])
        # what places are ranked by (chosen per region on the check season): chance, or expected fires
        rows["rank"] = rows.mu if model.get("rank_by") == "expected" else rows.p
        rows["h"] = h
        out.append(rows)
    rows = pd.concat(out, ignore_index=True).merge(cells[["iy", "ix", "lat", "lon", "district"]].reset_index(drop=True), on=["iy", "ix"])
    return rows, cubes[year], days


def score_day(g, k=50):
    """How did the forecast for one day do once the satellites came in?"""
    tot = float(g.y.sum())
    top = g.sort_values("rank", ascending=False).head(k)
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


def now_ist():
    return datetime.now(IST)


def as_published(od, d, g, rank_by=None):
    """Check the forecast exactly as it was published that morning (pred_<date>.csv), not one worked out again later."""
    f = od / f"pred_{d}.csv"
    if not f.exists():
        return g
    pub = pd.read_csv(f)
    if not {"iy", "ix", "p", "mu"} <= set(pub.columns):
        return g
    pub = pub.set_index(["iy", "ix"])
    key = pd.MultiIndex.from_arrays([g.iy.to_numpy(), g.ix.to_numpy()])
    g = g.copy()
    g["p"] = pub.p.reindex(key).fillna(0).to_numpy()
    g["mu"] = pub.mu.reindex(key).fillna(0).to_numpy()
    g["rank"] = pub["rank"].reindex(key).fillna(0).to_numpy() if "rank" in pub else (g.mu if rank_by == "expected" else g.p)
    return g


def record_issued(od, ahead):
    """Keep this morning's forecast totals as published. One made before its day is never replaced by one
    worked out later (a --date or --backfill rerun); a rerun on the same morning replaces that morning's."""
    path = od / "issued.csv"
    old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["for", "h", "expected", "made"])
    made = now_ist().strftime("%Y-%m-%d %H:%M")
    new = pd.DataFrame([{"for": str(g.date.iloc[0]), "h": str(int(h)), "expected": f"{float(g.mu.sum()):.1f}", "made": made}
                        for h, g in ahead.items()])
    # mornings from before this record existed: each pred_<date>.csv was written by that morning's run
    have = set(zip(old["for"], old["h"].astype(str))) | set(new["for"])
    seed = []
    for f in od.glob("pred_*.csv"):
        d = f.stem[5:]
        if (d, "1") not in have and d not in have:
            try:
                seed.append({"for": d, "h": "1", "expected": f"{pd.read_csv(f, usecols=['mu']).mu.sum():.1f}",
                             "made": f"{d} 00:00 (pred file)"})
            except (ValueError, KeyError):
                pass
    allr = pd.concat([old, pd.DataFrame(seed, columns=new.columns), new], ignore_index=True)
    allr["adv"] = allr.made.str[:10] <= allr["for"]
    allr = allr.sort_values(["adv", "made"]).drop_duplicates(["for", "h"], keep="last")
    allr = allr.drop(columns="adv").sort_values(["for", "h"])
    allr.to_csv(path, index=False)
    return allr


def fire_log(od, year, today, days, cube, cells, sb, issued):
    """Every day of this season: the fire detections the forecast expected, and what the satellites then saw.
    "Expected" is the total published that morning (or one and two days before). Days the system wasn't
    running that morning use the forecast worked out later, and are marked so."""
    iy, ix = cells.iy.to_numpy(), cells.ix.to_numpy()
    seen = cube[:, iy, ix].sum(1)
    iss = {(r["for"], int(r["h"])): r for r in issued.to_dict("records")}
    score = {r["date"]: r for r in sb.astype(object).where(pd.notna(sb), None).to_dict("records")} if len(sb) else {}

    def made_before(ds, h):
        r = iss.get((ds, h))
        return float(r["expected"]) if r and r["made"][:10] <= ds else None

    log = []
    for i, d in enumerate(days):
        if d > today + timedelta(days=2):
            break
        ds, s = str(d), score.get(str(d), {})
        e = made_before(ds, 1)
        later = s.get("predicted_total") if e is None and d < today else None
        log.append({"date": ds, "expected": e if e is not None else later, "made_that_morning": e is not None,
                    "expected_day_before": made_before(ds, 2), "expected_2_days_before": made_before(ds, 3),
                    "seen": int(seen[i]) if d < today else None,
                    "flagged": s.get("flagged"), "flagged_hit": s.get("hits"), "share_caught": s.get("catch")})
    log = [r for r in log if any(r[k] is not None for k in ("seen", "expected", "expected_day_before", "expected_2_days_before"))]
    yn = lambda v: "" if v is None else v
    pd.DataFrame([{"date": r["date"], "fire_detections_seen": yn(r["seen"]),
                   "expected_that_morning": yn(r["expected"]),
                   "made_that_morning": "" if r["expected"] is None else ("yes" if r["made_that_morning"] else "no, worked out later"),
                   "expected_a_day_ahead": yn(r["expected_day_before"]), "expected_two_days_ahead": yn(r["expected_2_days_before"]),
                   "flagged_places": yn(r["flagged"]), "flagged_places_that_burned": yn(r["flagged_hit"]),
                   "share_of_fires_in_flagged_places": "" if r["share_caught"] is None else f"{r['share_caught']:.3f}"}
                  for r in log]).to_csv(od / f"fire_log_{year}.csv", index=False)
    return log


def read_log(od):
    """The most recent season's fire log, for the page between seasons."""
    files = sorted(od.glob("fire_log_*.csv"))
    if not files:
        return None
    t = pd.read_csv(files[-1], dtype=str).fillna("")
    num = lambda v, f=float: f(v) if v != "" else None
    return [{"date": r["date"], "seen": num(r["fire_detections_seen"], lambda v: int(float(v))),
             "expected": num(r["expected_that_morning"]), "made_that_morning": r["made_that_morning"] == "yes",
             "expected_day_before": num(r["expected_a_day_ahead"]), "expected_2_days_before": num(r["expected_two_days_ahead"]),
             "flagged": num(r["flagged_places"], lambda v: int(float(v))), "flagged_hit": num(r["flagged_places_that_burned"], lambda v: int(float(v))),
             "share_caught": num(r["share_of_fires_in_flagged_places"])} for r in t.to_dict("records")]


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
    # days of this season the scoreboard is missing (say the run failed that morning), plus the last 3
    sb_path = od / "scoreboard.csv"
    sb = pd.read_csv(sb_path) if sb_path.exists() else pd.DataFrame()
    _, sdays = season_of(region, today)
    scored = set(sb.date) if "date" in sb else set()
    todo = [d for d in (sdays or [])[1:] if d < today and str(d) not in scored]
    rows, cube, days = predict_season(region, today, fires, wx, cells, model, harvest, backfill, since=todo[0] if todo else None)
    report = json.loads((MODELS / f"{region}_report.json").read_text()) if (MODELS / f"{region}_report.json").exists() else None
    extras = {"replay": read_json(MODELS / f"{region}_replay.json"), "people": read_json(DATA / f"people_{region}.json"),
              "bbox": reg["bbox"], "cell": CELL, "lang": reg["lang"],
              "season": [f"{m:02d}-{d:02d}" for m, d in reg["season"]]}
    if rows is None:
        latest = {"region": region, "label": reg["label"], "date": str(today), "in_season": False, "report": report,
                  "log": read_log(od)} | extras
        (od / "latest.json").write_text(json.dumps(latest)); return latest

    rows["place"] = [cname.get((a, b), "") for a, b in zip(rows.iy, rows.ix)]
    pred = rows[(rows.date == today) & (rows.h == 1)].set_index(["iy", "ix"], drop=False)
    ahead = {h: rows[(rows.h == h) & (rows.date == today + timedelta(days=int(h) - 1))].set_index(["iy", "ix"], drop=False)
             for h in sorted(rows.h.unique())}
    allp = pred[["iy", "ix", "lat", "lon", "district", "place", "p", "mu", "rank"]].copy()
    for h, g in ahead.items():
        if h > 1:
            allp[f"p_{g.date.iloc[0]}"] = g.p.reindex(allp.index).to_numpy()
    pf = od / f"pred_{today}.csv"
    if not (pf.exists() and now_ist().date() > today):   # a later rerun never replaces what was published that morning
        allp.round(4).to_csv(pf, index=False)

    # scoreboard: every finished day of this season that we have predictions for. Each morning re-checks the
    # last 3 days, so late satellite data counts; the forecast side is always the one published that morning.
    done_days = [d for d in days if d < today and (backfill or d >= today - timedelta(days=3) or d in todo)]
    new = [score_day(as_published(od, d, g, model.get("rank_by")))
           for d, g in rows[(rows.h == 1) & rows.date.isin(done_days)].groupby("date")]
    if new:
        sb = pd.concat([sb, pd.DataFrame(new)], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        sb.to_csv(sb_path, index=False)
    yday = sb[sb.date == str(today - timedelta(days=1))].astype(object).where(lambda x: pd.notna(x), None).to_dict("records") if "date" in sb else []
    yscore = yday[0] if yday else None
    log = fire_log(od, season_of(region, today)[0], today, days, cube, cells, sb, record_issued(od, ahead))

    plan, summary = dispatch.plan(pred, teams, stops, by="rank")
    ahead_tot = [(["today", "tomorrow", "day after"][h - 1], float(g.mu.sum())) for h, g in ahead.items() if h > 1]
    weak = bool(report) and report.get("beats_both_guesses") is False
    text = brief.make(reg["label"], reg["lang"], today, pred, plan, summary, yscore, ahead_tot, weak=weak)
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
        cand = g.sort_values("rank", ascending=False).head(400)   # the dashboard plans teams in this order
        return {"h": int(h), "label": labels[h], "date": str(g.date.iloc[0]), "total_mu": round(float(g.mu.sum()), 1),
                "risk": [[int(r.iy), int(r.ix), round(float(r.p), 3), round(float(r.mu), 2), di_of[r.district]] for r in g.itertuples() if r.p >= 0.005],
                "candidates": [[int(r.iy), int(r.ix), round(float(r.lat), 4), round(float(r.lon), 4), r.district, round(float(r.p), 3), round(float(r.mu), 2), r.place] for r in cand.itertuples()],
                "districts": [[d, round(float(v), 1)] for d, v in g.groupby("district").mu.sum().sort_values(ascending=False).head(8).items()]}
    dayblocks = [day_block(g, h) for h, g in ahead.items()]
    cand = pred.sort_values("rank", ascending=False).head(400)
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
        "generated": now_ist().strftime("%d %b %Y, %H:%M IST"),
        "green": green, "smoke": smoke_fc, "log": log,
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
    today = date.fromisoformat(a.date) if a.date else now_ist().date()
    out = {}
    for r in a.region or [x for x in ORDER if x in REGIONS]:
        res = run_region(r, today, a.teams, a.stops, a.backfill, not a.offline)
        if res:
            out[r] = res
    import dashboard
    dashboard.build(out)


if __name__ == "__main__":
    main()
