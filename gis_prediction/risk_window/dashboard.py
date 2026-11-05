"""Builds out/index.html: one self-contained page (no internet needed once built)."""
import json
from shapely.geometry import shape
from config import OUT, REGIONS
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
    data = {}
    for r, latest in latest_by_region.items():
        data[r] = latest | {"geo": region_geo(r)}
    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("<!--__LOGO__-->", logo_symbol())
    for key, name in (("__BANNER__", "banner.webp"), ("__DIVA__", "harvester.webp")):
        f = ASSETS / name
        uri = "data:image/webp;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""
        html = html.replace(key, uri)
    (OUT / "index.html").write_text(html)
    print("dashboard:", OUT / "index.html")


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
