"""Turns the Calculator's results into an overlay for the map and globe.

Everything is computed with the calculator's own modules (python-sgp4 + the delta-propagation
manoeuvre model), so the map shows exactly what Phases 2-5 calculate:
  * Objects A and B (their SGP4 elements, propagated live in the browser)
  * the closest approach between A and B over the simulation window (Phase 3)
  * the manoeuvred trajectory of A from the Phase 3 burn
  * one manoeuvred trajectory per Phase 5 scenario
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SCENARIO_COLOURS = ["#F97316", "#0891B2", "#65A30D"]


def satrec_elements(sat) -> dict:
    """Mean elements of a python-sgp4 Satrec in the form the browser engine expects."""
    return {"satnum": str(sat.satnum), "jdepoch": float(sat.jdsatepoch + sat.jdsatepochF), "bstar": float(sat.bstar),
            "inc": math.degrees(sat.inclo), "raan": math.degrees(sat.nodeo), "ecc": float(sat.ecco),
            "argp": math.degrees(sat.argpo), "ma": math.degrees(sat.mo), "n": float(sat.no_kozai) * 1440 / (2 * math.pi)}


def overlay_key(cfg, manoeuvre, p3_win, scenarios) -> str:
    """Stable hash of everything the overlay depends on (used for caching)."""
    objs = {k: {"name": v["name"], "e": satrec_elements(v["satrec"])} for k, v in cfg["objects"].items()}
    blob = json.dumps([objs, cfg["ground_station"], cfg["simulation"], manoeuvre, p3_win, scenarios], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def _path(sat, jd_burn, dv_rtn, jd_end, step_s=30.0):
    from modules.avoidance import burn_trajectory
    n = int((jd_end - jd_burn) * 86400 / step_s) + 1
    grid = jd_burn + np.arange(max(n, 2)) * step_s / 86400.0
    r, _ = burn_trajectory(sat, jd_burn, dv_rtn, grid)
    ok = np.all(np.isfinite(r), axis=1)
    return [[float(j), float(x), float(y), float(z)] for j, (x, y, z) in zip(grid[ok], r[ok])]


def build_overlay(cfg, manoeuvre: dict | None, p3_win_h: float | None, scenarios: dict | None) -> dict:
    from modules.avoidance import dv_vector
    from modules.collision import close_approach
    from modules.orbital_mechanics import epoch_jd, jd_grid, propagate

    A, B = cfg["objects"]["A"], cfg["objects"]["B"]
    sat_a, sat_b = A["satrec"], B["satrec"]
    start = epoch_jd(sat_a)
    dur_h = float(cfg["simulation"]["duration_hours"])
    screen = float(cfg["simulation"]["screening_distance_km"])
    out = {
        "label": f"{A['name']} vs {B['name']} · {cfg.get('mode', '')} mode",
        "startJd": start,
        "objects": [{"id": k, "n": O["name"], "color": col,
                     "e": {**satrec_elements(O["satrec"]), "satnum": str(O.get("catalogue_number") or O["satrec"].satnum)}}
                    for k, O, col in (("A", A, "#E11D48"), ("B", B, "#7C3AED"))],
        "station": None,   # the calculator's station (Sriharikota by default) is one of the network stations already on the map
        "burns": [], "tca": None, "notes": [],
    }
    # Closest approach over the whole simulation window (as in Phase 3, section 1)
    try:
        ca = close_approach(sat_a, sat_b, jd_grid(sat_a, 0.0, dur_h * 60, 30.0), screen)
        r_a, _ = propagate(sat_a, [ca["tca_jd"]])
        out["tca"] = {"jd": ca["tca_jd"], "miss": ca["miss_km"], "within": bool(ca["within_screening"]),
                      "screen": screen, "r": [float(v) for v in r_a[0]]}
    except (ValueError, RuntimeError) as exc:
        out["notes"].append(f"Closest approach not available: {exc}")
    # Phase 3 manoeuvre
    if manoeuvre and float(manoeuvre.get("dv_ms", 0)) > 0:
        win = float(p3_win_h or min(6.0, dur_h))
        jd_burn = start + float(manoeuvre["burn_min"]) / 1440.0
        if jd_burn < start + win / 24:
            try:
                out["burns"].append({"name": f"Phase 3: {manoeuvre['dv_ms']:.2f} m/s {manoeuvre['direction'].split(' (')[0].lower()}",
                                     "color": "#E11D48", "jdBurn": jd_burn,
                                     "pts": _path(sat_a, jd_burn, dv_vector(manoeuvre["direction"], float(manoeuvre["dv_ms"])), start + win / 24)})
            except (ValueError, RuntimeError) as exc:
                out["notes"].append(f"Phase 3 path not available: {exc}")
    # Phase 5 scenarios
    if scenarios:
        win = float(scenarios.get("win_h", min(6.0, dur_h)))
        for i, s in enumerate(scenarios.get("list", [])):
            if float(s.get("dv_ms", 0)) <= 0:
                continue
            jd_burn = start + float(s["burn_min"]) / 1440.0
            if jd_burn >= start + win / 24:
                continue
            try:
                out["burns"].append({"name": f"Phase 5 · {s['name']}: {s['dv_ms']:.2f} m/s", "color": SCENARIO_COLOURS[i % 3],
                                     "jdBurn": jd_burn, "pts": _path(sat_a, jd_burn, dv_vector(s["direction"], float(s["dv_ms"])), start + win / 24)})
            except (ValueError, RuntimeError) as exc:
                out["notes"].append(f"{s['name']} path not available: {exc}")
    return out
