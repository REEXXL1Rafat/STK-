"""Train the next-day fire model and test it honestly on a season it never saw.

    python train.py punjab_haryana            # uses the extra signals chosen last time
    python train.py punjab_haryana --select   # re-test every extra signal first (after new data arrives)
    python train.py purba_medinipur

Writes models/<region>.pkl and models/<region>_report.json. The report compares
the model against two simple guesses a person could make without any AI:
  persistence  "tomorrow's fires will be where today's were"
  climatology  "fires will be where they usually are this week"
If the model can't beat those, it isn't worth sending anyone anywhere.
"""
import json, pickle, sys
from datetime import date
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from config import DATA, MODELS, REGIONS, FIRST_YEAR
import grid, features as F

KS = (25, 50, 100)


def load(region):
    parts = [F.load_fires(p) for p in (DATA / f"fires_{region}.csv.gz", DATA / f"live_{region}.csv.gz") if p.exists()]
    fires = pd.concat(parts, ignore_index=True).drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    wparts = [pd.read_csv(p) for p in (DATA / f"weather_{region}.csv.gz", DATA / f"wxlive_{region}.csv.gz") if p.exists()]
    wx = pd.concat(wparts, ignore_index=True).drop_duplicates(["wlat", "wlon", "date"], keep="last") if wparts else None
    return fires, wx


def load_harvest(region):
    p = DATA / f"harvest_{region}.csv.gz"
    if not p.exists():
        return None
    h = pd.read_csv(p)
    h["date"] = pd.to_datetime(h["date"]).dt.date
    return h


def available_groups(fires, wx, harvest):
    sats = set(fires.satellite.astype(str).unique())
    g = ["festival"]
    if sats & {"N21", "2"}: g.append("noaa21")
    if "frp" in fires: g.append("intensity")
    if sats & {"Terra", "Aqua", "T", "A"}: g.append("modis")
    if wx is not None and "wind_direction_10m_dominant" in wx: g.append("wind_dir")
    if harvest is not None: g.append("harvest")
    if harvest is not None and "scar" in harvest and harvest.scar.notna().any(): g.append("scar")
    return g


def complete_seasons(region, today):
    ys = []
    for y in range(FIRST_YEAR, today.year + 1):
        if F.season_dates(region, y)[-1] < today:
            ys.append(y)
    return ys


HORIZONS = (1, 2, 3)   # today, tomorrow, the day after


def build(region, years, pool, fires, wx, cells, cubes, dayss, h=1, aux=None):
    aux = aux or {}
    return pd.concat([F.season_rows(region, y, cubes[y], dayss[y], cells, cubes, pool, wx, h=h, aux=aux.get(y)) for y in years], ignore_index=True)


def choose_groups(region, years, fires, wx, cells, cubes, dayss, aux, groups):
    """Try each extra signal on the second-to-last season (never the test season). Keep it only if it helps."""
    tr_y, va_y = years[:-2], years[-2]
    tr = build(region, tr_y, tr_y, fires, wx, cells, cubes, dayss, 1, aux)
    va = build(region, [va_y], tr_y, fires, wx, cells, cubes, dayss, 1, aux)
    def score(feats):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80,
                                             l2_regularization=1.0, random_state=0).fit(tr[feats], (tr.y > 0).astype(int))
        p = clf.predict_proba(va[feats])[:, 1]
        return topk_scores(va, p)["catch@50"], average_precision_score(va.y > 0, p)
    base_c, base_ap = score(F.BASE)
    log = {"validated_on": va_y, "base": {"catch@50": base_c, "avg_precision": base_ap}, "tried": {}}
    keep = []
    for g in groups:
        if any(f not in tr or tr[f].notna().sum() == 0 for f in F.EXTRA[g]):   # e.g. a satellite that only started recently
            log["tried"][g] = {"kept": False, "note": "no data in the training seasons yet"}
            print(f"  {g:10} no data in the training seasons yet, skipped"); continue
        c, ap = score(F.BASE + F.EXTRA[g])
        ok = c >= base_c + 0.005 and ap >= base_ap - 0.002
        log["tried"][g] = {"catch@50": c, "avg_precision": ap, "kept": ok}
        print(f"  {g:10} catch@50 {c:.3f} (base {base_c:.3f})  AP {ap:.3f} (base {base_ap:.3f})  {'KEEP' if ok else 'drop'}")
        if ok:
            keep.append(g)
    log["kept"] = keep
    return keep, log


def baselines(df):
    return {
        "persistence": df.lag1 * 100 + df.nb1_lag1 * 10 + df.sum7 + 0.01 * df.clim,
        "climatology": df.clim + 0.001 * df.clim_total + 0.0001 * df.last_year.fillna(0),
    }


def topk_scores(df, score, ks=KS):
    """precision@K: share of the K flagged cells that burned. catch@K: share of all fires inside them."""
    out = {f"precision@{k}": [] for k in ks} | {f"catch@{k}": [] for k in ks}
    tot_fire = {k: 0.0 for k in ks}; tot_all = 0.0
    for d, g in df.assign(_s=score).groupby("date"):
        if g.y.sum() == 0:
            continue
        g = g.sort_values("_s", ascending=False)
        tot_all += g.y.sum()
        for k in ks:
            top = g.head(k)
            out[f"precision@{k}"].append(float((top.y > 0).mean()))
            tot_fire[k] += top.y.sum()
    res = {f"precision@{k}": float(np.mean(out[f"precision@{k}"])) for k in ks}
    res |= {f"catch@{k}": float(tot_fire[k] / tot_all) if tot_all else 0.0 for k in ks}
    return res


def daily_table(df, p, mu, base, k=50):
    rows = []
    for d, g in df.assign(_p=p, _mu=mu, _b=base).groupby("date"):
        tot = float(g.y.sum())
        rows.append({"date": str(d), "actual": tot, "predicted": round(float(g._mu.sum()), 1),
                     "model_catch": float(g.sort_values("_p", ascending=False).head(k).y.sum() / tot) if tot else None,
                     "persist_catch": float(g.sort_values("_b", ascending=False).head(k).y.sum() / tot) if tot else None})
    return rows


def fit(X, y):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(X, (y > 0).astype(int))
    reg = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                        min_samples_leaf=80, l2_regularization=1.0, random_state=0)
    reg.fit(X, y)
    return clf, reg


def replay(te, p, k=50, max_cells=140, min_p=0.08):
    """The test season, day by day: where fires were seen, and where the model said to look (it never saw this season)."""
    out = {"k": k, "days": []}
    for d, g in te.assign(pp=p).groupby("date"):
        g = g.sort_values("pp", ascending=False)
        top = g.head(k)
        fires = g[g.y > 0]
        risk = g[g.pp >= min_p].head(max_cells)
        tot = float(g.y.sum())
        out["days"].append({
            "date": str(d), "seen": int(tot), "caught": int(top.y.sum()),
            "fires": [int(v) for r in fires.itertuples() for v in (r.iy, r.ix, r.y)],
            "risk": [int(v) for r in risk.itertuples() for v in (r.iy, r.ix, round(r.pp * 100))],
        })
    return out


def evaluate(te, p):
    yb = (te.y > 0).astype(int)
    return {"auc": float(roc_auc_score(yb, p)), "avg_precision": float(average_precision_score(yb, p)), **topk_scores(te, p)}


def main(region, today=None, select=False, replay_only=False):
    today = today or date.today()
    fires, wx = load(region)
    harvest = load_harvest(region)
    cells = grid.build_cells(region)
    years = complete_seasons(region, today)
    if len(years) < 3:
        raise SystemExit(f"Need at least 3 complete seasons, found {years}")
    static = F.static_cells(fires, region)
    print(f"Ignoring {len(static)} cells that look like factories, flares or power plants")
    active = F.active_cells(fires, region, years, static) if REGIONS[region].get("active_only") else None
    if active is not None:
        n0 = len(cells); cells = F.keep_active(cells, active)
        print(f"Modelling the {len(cells):,} of {n0:,} cells that have burned before (and their neighbours)")
    cubes, dayss = F.all_cubes(fires, region, years, static)
    aux = {y: F.aux_cubes(fires, region, y, static, harvest) for y in years}
    gpath = MODELS / f"{region}_groups.json"
    if select:
        print("Testing extra signals on the validation season:")
        groups, glog = choose_groups(region, years, fires, wx, cells, cubes, dayss, aux, available_groups(fires, wx, harvest))
        MODELS.mkdir(parents=True, exist_ok=True); gpath.write_text(json.dumps(glog, indent=1))
    else:
        groups = json.loads(gpath.read_text())["kept"] if gpath.exists() else []
    feats = F.BASE + [f for g in groups for f in F.EXTRA[g]]
    test_year, train_years = years[-1], years[:-1]
    print(f"{REGIONS[region]['label']}: train on {train_years}, test on {test_year}; extra signals: {groups or 'none'}")

    report = {"region": region, "label": REGIONS[region]["label"], "train_seasons": train_years, "test_season": test_year,
              "cells": int(len(cells)), "static_cells_ignored": len(static), "extra_signals": groups, "horizons": {}}
    if gpath.exists():
        report["signal_tests"] = json.loads(gpath.read_text())
    for h in ((1,) if replay_only else HORIZONS):
        tr = build(region, train_years, train_years, fires, wx, cells, cubes, dayss, h, aux)
        te = build(region, [test_year], train_years, fires, wx, cells, cubes, dayss, h, aux)
        clf, reg = fit(tr[feats], tr.y)
        p, mu = clf.predict_proba(te[feats])[:, 1], reg.predict(te[feats])
        base = baselines(te)
        hr = {"model": evaluate(te, p)} | {n: evaluate(te, s) for n, s in base.items()}
        report["horizons"][str(h)] = hr
        print(f"  h={h} ({['today', 'tomorrow', 'day after'][h - 1]}): prec@50 {hr['model']['precision@50']:.2f} vs yesterday-guess "
              f"{hr['persistence']['precision@50']:.2f}; catch@50 {hr['model']['catch@50']:.2f} vs {hr['persistence']['catch@50']:.2f}")
        if h == 1:
            report |= {"test_cell_days": int(len(te)), "test_fires": int(te.y.sum()), "share_cell_days_with_fire": float((te.y > 0).mean())}
            report |= {k: hr[k] for k in ("model", "persistence", "climatology")}
            te2 = te.assign(_mu=mu).merge(cells[["iy", "ix", "district"]].reset_index(drop=True), on=["iy", "ix"])
            dd = te2.groupby(["date", "district"]).agg(actual=("y", "sum"), pred=("_mu", "sum"), yday=("dist_lag1", "first")).reset_index()
            busy = dd[dd.actual + dd.yday > 0]
            report["district_totals"] = {"model_mae": float((busy.pred - busy.actual).abs().mean()),
                                         "yesterday_mae": float((busy.yday - busy.actual).abs().mean()),
                                         "model_corr": float(np.corrcoef(busy.pred, busy.actual)[0, 1])}
            bins = np.array([0, .05, .1, .2, .3, .5, .7, 1.01])
            cal = pd.DataFrame({"p": p, "y": (te.y > 0).astype(int)}).assign(b=lambda x: pd.cut(x.p, bins)).groupby("b", observed=True).agg(said=("p", "mean"), saw=("y", "mean"), n=("y", "size"))
            report["calibration"] = [{"said": round(float(r.said), 3), "saw": round(float(r.saw), 3), "n": int(r.n)} for r in cal.itertuples()]
            report["backtest_daily"] = daily_table(te, p, mu, base["persistence"])
            MODELS.mkdir(parents=True, exist_ok=True)
            (MODELS / f"{region}_replay.json").write_text(json.dumps(replay(te, p), separators=(",", ":")))
            if replay_only:
                print(f"Wrote the {test_year} replay"); return report
    report["generated"] = str(today)

    # final models: learn from every complete season, including the test one
    models = {}
    for h in HORIZONS:
        full = build(region, years, years, fires, wx, cells, cubes, dayss, h, aux)
        models[h] = dict(zip(("clf", "reg"), fit(full[feats], full.y)))
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / f"{region}.pkl", "wb") as fh:
        pickle.dump({"h": models, "features": feats, "groups": groups, "seasons": years, "trained": str(today),
                     "static": sorted(static), "cells": sorted(active) if active is not None else None}, fh)
    (MODELS / f"{region}_report.json").write_text(json.dumps(report, indent=1))
    m, ps = report["model"], report["persistence"]
    print(f"Saved. Today's model on {test_year}: catch@50 {m['catch@50']:.2f} vs yesterday-guess {ps['catch@50']:.2f}")
    return report


def in_season(region, day):
    return any(day in F.season_dates(region, y) for y in (day.year - 1, day.year))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:   # every region; with --missing, only those without a model yet
        from config import ORDER
        for r in ORDER:
            if "--missing" in sys.argv and (MODELS / f"{r}.pkl").exists():
                continue
            man = json.loads((DATA / "fetched.json").read_text()) if (DATA / "fetched.json").exists() else None
            if not (DATA / f"fires_{r}.csv.gz").exists() or (man is not None and not man.get("weather", {}).get(r)):
                print(f"{r}: data still downloading, skipping"); continue
            if "--if-in-season" in sys.argv and (MODELS / f"{r}.pkl").exists() and not in_season(r, date.today()):
                print(f"{r}: not in season, keeping the current model"); continue
            try:
                main(r, select="--select" in sys.argv)
            except SystemExit as e:
                print(f"{r}: {e}")
        sys.exit(0)
    region = args[0] if args else "punjab_haryana"
    # the daily robot retrains weekly only while fields are burning, so the repo doesn't fill up with old models
    if "--if-in-season" in sys.argv and (MODELS / f"{region}.pkl").exists() and not in_season(region, date.today()):
        print(f"{region}: not in season, keeping the current model")
        sys.exit(0)
    main(region, select="--select" in sys.argv, replay_only="--replay-only" in sys.argv)
