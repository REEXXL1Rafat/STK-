"""The 5 km grid: cell ids, centres, and which district each cell sits in."""
import json, math
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from config import ROOT, CELL, REGIONS, wx_step


def cell_of(lat, lon, bbox):
    w, s = bbox[0], bbox[1]
    iy = np.floor((np.asarray(lat, float) - s) / CELL).astype(int)
    ix = np.floor((np.asarray(lon, float) - w) / CELL).astype(int)
    return iy, ix


def load_districts(states):
    fc = json.loads((ROOT / "geo" / "districts.geojson").read_text())
    return [(shape(f["geometry"]), f["properties"]["district"], f["properties"]["state"])
            for f in fc["features"] if f["properties"]["state"] in states]


def build_cells(region_name):
    reg = REGIONS[region_name]
    w, s, e, n = reg["bbox"]
    dists = load_districts(reg["states"])
    tree = STRtree([d[0] for d in dists])
    rows = []
    for iy in range(int(math.ceil((n - s) / CELL))):
        for ix in range(int(math.ceil((e - w) / CELL))):
            lat, lon = s + (iy + 0.5) * CELL, w + (ix + 0.5) * CELL
            p = Point(lon, lat)
            hit = [i for i in tree.query(p) if dists[i][0].contains(p)]
            if not hit:
                continue
            g, d, st = dists[hit[0]]
            rows.append((iy, ix, round(lat, 4), round(lon, 4), d, st))
    cells = pd.DataFrame(rows, columns=["iy", "ix", "lat", "lon", "district", "state"])
    # nearest weather point
    st = wx_step(region_name)
    cells["wlat"] = (np.floor((cells.lat - s) / st) * st + s + st / 2).round(3)
    cells["wlon"] = (np.floor((cells.lon - w) / st) * st + w + st / 2).round(3)
    cells.index = pd.MultiIndex.from_arrays([cells.iy, cells.ix], names=["iy", "ix"])
    return cells


def outlines(region_name):
    """District outlines as GeoJSON-like dicts for the dashboard map."""
    reg = REGIONS[region_name]
    w, s, e, n = reg["bbox"]
    fc = json.loads((ROOT / "geo" / "districts.geojson").read_text())
    out = []
    for f in fc["features"]:
        if f["properties"]["state"] not in reg["states"]:
            continue
        g = shape(f["geometry"])
        if not g.intersects(shape({"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]})):
            continue
        out.append(f)
    return out


def wx_points_for(cells):
    """The weather points the model actually needs (one per wx_step square that has a cell)."""
    return sorted(set(zip(cells.wlat.round(3), cells.wlon.round(3))))
