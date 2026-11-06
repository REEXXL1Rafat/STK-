"""Builds out/index.html and out/data/<region>.json.

The pilot regions are written into the page itself, so it opens even without internet. The big
states are separate files the page loads when you pick them (that part needs the web link).
"""
import json, os
from shapely.geometry import shape
from config import OUT, REGIONS, SHOW
import grid

import base64
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "dashboard_template.html"
ASSETS = HERE / "assets"  # SATARK brand: logo, riso field photo, DIVA drawing


def region_geo(region):
    feats = grid.outlines(region)
    outs, labels = [], []
    for f in feats:
        g = shape(f["geometry"])
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        rings = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys]
        outs.append({"d": f["properties"]["district"], "r": rings})
        pt = g.representative_point()
        labels.append([f["properties"]["district"], round(pt.y, 3), round(pt.x, 3)])
    return {"outlines": outs, "labels": labels}


def build(latest_by_region):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    inline, meta = {}, []
    for r in [x for x in SHOW if x in latest_by_region]:
        d = latest_by_region[r] | {"geo": region_geo(r)}
        js = json.dumps(d, separators=(",", ":"))
        (OUT / "data" / f"{r}.json").write_text(js)
        inline[r] = d if not REGIONS[r].get("active_only") else None
        meta.append([r, d["label"]])
    cfg = {"regions": meta, "voice": os.environ.get("SATARK_VOICE_URL", "").rstrip("/")}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(inline, separators=(",", ":")))
    html = html.replace("/*__CFG__*/null", json.dumps(cfg, separators=(",", ":")))
    html = html.replace("<!--__LOGO__-->", logo_symbol())
    for key, name in (("__BANNER__", "banner.webp"), ("__DIVA__", "harvester.webp")):
        f = ASSETS / name
        uri = "data:image/webp;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""
        html = html.replace(key, uri)
    (OUT / "index.html").write_text(html)
    print("dashboard:", OUT / "index.html", f"({len(html) // 1024} KB, {len(meta)} regions)")


def logo_symbol():
    """The SATARK wordmark as a reusable <symbol>, drawn with <use href="#satark-logo">."""
    f = ASSETS / "logo.svg"
    if not f.exists():
        return ""
    svg = f.read_text()
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0]
    inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return (f'<svg width="0" height="0" style="position:absolute" aria-hidden="true">'
            f'<symbol id="satark-logo" viewBox="{vb}" fill="currentColor">{inner}</symbol></svg>')
