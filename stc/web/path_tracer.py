"""Builds the animated Path Tracer view (world map + rotating globe) as a self-contained HTML string.

Orbits are propagated in the browser with the same SGP4 code (stc/web/engine.js), so objects move
smoothly without re-running Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def build_path_tracer_html(prims, objs, sel_index: int, sim_jd: float, stations: list[dict],
                           conj_satnums: set[str], mask_deg: float = 5.0, threshold_km: float = 5.0,
                           window_h: float = 72.0, calc: dict | None = None) -> str:
    """Clicking an object in the view screens it against every other object for `window_h` hours and
    flashes the ones that come within `threshold_km` (same settings as the Collision Stats page).

    `calc` (optional) is the Calculator overlay from calculator.overlay.build_overlay(): objects A/B,
    manoeuvred trajectories, the closest approach and the calculator's ground station."""
    def compact(o):
        e = o.elements()   # [satnum, epoch JD, B*, inc, RAAN, ecc, argp, M, rev/day], rounded for size
        return [e["satnum"], round(e["jdepoch"], 8), float(f"{e['bstar']:.6g}"), round(e["inc"], 5), round(e["raan"], 5),
                round(e["ecc"], 8), round(e["argp"], 5), round(e["ma"], 5), round(e["n"], 9)]
    items = []
    for i, p in enumerate(prims):
        items.append({"n": p.name, "e": compact(p), "t": "PAYLOAD", "c": p.country or "IN", "sel": i == sel_index})
    for o in objs:
        items.append({"n": o.name, "e": compact(o), "t": o.obj_type, "c": o.country, "cj": o.satnum in conj_satnums,
                      **({"syn": 1} if getattr(o, "synthetic", False) else {})})
    cfg = {
        "simJd": sim_jd,
        "mask": mask_deg,
        "thr": threshold_km,
        "windowH": window_h,
        "calc": calc,
        "objects": items,
        "stations": [{"name": s["name"], "short": s.get("short", s["name"]), "lat": s["lat"], "lon": s["lon"],
                      "type": s.get("type", "")} for s in stations],
    }
    data = json.dumps(cfg, separators=(",", ":")).replace("</", "<\\/")
    html = (_DIR / "path_tracer.html").read_text(encoding="utf-8")
    engine = (_DIR / "engine.js").read_text(encoding="utf-8")
    return (html.replace("__THR__", f"{threshold_km:g}").replace("__ENGINE__", engine).replace("__DATA__", data))
