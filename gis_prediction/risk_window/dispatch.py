"""Turn a risk map into a plan: which places each team visits today, in what order."""
import math
import numpy as np
import pandas as pd


def km(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def pick(pred, n, spacing=2, by="p"):
    """Top-n cells by risk, never two within `spacing` cells of each other (no double visits)."""
    chosen, taken = [], set()
    for r in pred.sort_values(by, ascending=False).itertuples():
        if any((r.iy + a, r.ix + b) in taken for a in range(-spacing + 1, spacing) for b in range(-spacing + 1, spacing)):
            continue
        chosen.append(r.Index); taken.add((r.iy, r.ix))
        if len(chosen) >= n:
            break
    return pred.loc[chosen]


def _kmeans(xy, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((xy[:, None] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = xy[lab == j].mean(0)
    return lab


def _balanced(xy, lab, k, cap):
    """Same areas as k-means, but every team gets at most `cap` stops."""
    cen = np.array([xy[lab == j].mean(0) if (lab == j).any() else xy[j % len(xy)] for j in range(k)])
    pairs = sorted(((float(((xy[i] - cen[j]) ** 2).sum()), i, j) for i in range(len(xy)) for j in range(k)))
    out, load = -np.ones(len(xy), int), np.zeros(k, int)
    for _, i, j in pairs:
        if out[i] < 0 and load[j] < cap:
            out[i] = j; load[j] += 1
    return out


MIN_P = 0.05   # never send a team to a place with less than a 5% chance of fire


def plan(pred, teams=5, stops=4, spacing=2, min_p=MIN_P):
    """Returns stops with a team number and visit order, plus a short summary."""
    pts = pick(pred[pred.p >= min_p], teams * stops, spacing)
    total = float(pred.mu.sum())
    if len(pts) == 0:
        return pts.assign(team=[], order=[]), {"teams": 0, "stops": 0, "expected_fires_total": round(total, 1),
                                               "expected_fires_covered": 0.0, "share_covered": 0.0}
    k = min(teams, len(pts))
    xy = np.c_[pts.lon.to_numpy() * math.cos(math.radians(pts.lat.mean())), pts.lat.to_numpy()]
    lab = _balanced(xy, _kmeans(xy, k), k, stops)
    pts = pts.assign(team=lab)
    # number teams west to east so team 1 is always the same side of the map
    order = pts.groupby("team").lon.mean().sort_values().index
    pts["team"] = pts.team.map({t: i + 1 for i, t in enumerate(order)})
    out = []
    for t, g in pts.groupby("team"):
        g = g.copy(); route = []
        cur = g.loc[g.p.idxmax()]                    # start at the riskiest stop
        left = set(g.index)
        while left:
            nxt = min(left, key=lambda i: km(cur.lat, cur.lon, g.at[i, "lat"], g.at[i, "lon"]))
            route.append(nxt); left.remove(nxt); cur = g.loc[nxt]
        g = g.loc[route].assign(order=range(1, len(route) + 1))
        out.append(g)
    res = pd.concat(out)
    summary = {"teams": int(k), "stops": int(len(res)), "expected_fires_total": round(float(total), 1),
               "expected_fires_covered": round(float(res.mu.sum()), 1),
               "share_covered": round(float(res.mu.sum() / total), 3) if total > 0 else 0.0}
    return res, summary


def maps_link(lat, lon):
    return f"https://maps.google.com/?q={lat:.4f},{lon:.4f}"
