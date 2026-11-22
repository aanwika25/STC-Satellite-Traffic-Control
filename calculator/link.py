"""Keeps the dashboard and the Calculator on the same two objects and the same manoeuvre.

One shared pair lives in st.session_state["pair"] = {"a": NORAD of the satellite, "b": NORAD of the
other object, "jd": time of the selected close approach}. It is changed by
  * the sidebar satellite list            (A; B becomes that satellite's top close approach)
  * the event list on Collision Stats     (B and the event time)
  * Object A / Object B in Calculator Phase 1
and every page reads it. When the pair changes, the Calculator is filled in automatically (objects, a
window that contains the encounter, the Phase 3 burn 1.5 orbits before it), so the Phase 3 manoeuvre
is the one used by Alternate Path Mapping, New Path Stats and the map and globe.
"""
from __future__ import annotations

import math

import streamlit as st

P3_KEYS = ("p3_start", "p3_burn", "p3_dir", "p3_dv", "p3_target", "p3_win", "cat_dur", "p4_mwin", "p5_win", "p5_burn_0", "p5_burn_1", "p5_burn_2",
           "p3_find_result")


def get_pair() -> dict:
    return st.session_state.setdefault("pair", {"a": None, "b": None, "jd": None})


def set_pair(a, b, jd=None):
    get_pair().update(a=a, b=b if b != a else None, jd=jd)


def focus_for(a_obj, tca_jd: float) -> dict:
    """Calculator window that contains the encounter, and a burn 1.5 orbits before it."""
    tca_h = max(0.0, (tca_jd - a_obj.jdepoch) * 24.0)
    burn_min = max(0.0, round(tca_h * 60 - 1.5 * a_obj.period_min))
    return {"tca_h": tca_h, "duration_h": float(min(720, max(1, math.ceil(tca_h + 2)))),
            "win_h": float(min(720, max(1, math.ceil(tca_h + 1)))), "burn_min": float(burn_min)}


def build_cfg(a_obj, b_obj, focus: dict, screen_km: float) -> dict:
    """Calculator configuration for two dashboard objects, keeping the user's station/radio settings."""
    from calculator.catalogue_source import satrec_from_catalogue
    from modules.catalogue import load_settings
    base = st.session_state.get("cfg") or {}
    s = load_settings()
    sim = {**s["simulation"], **(base.get("simulation") or {"screening_distance_km": screen_km}),
           "duration_hours": focus["duration_h"]}
    return {"objects": {k: {"name": o.name, "catalogue_number": o.satnum, "satrec": satrec_from_catalogue(o),
                            "line1": o.line1, "line2": o.line2} for k, o in (("A", a_obj), ("B", b_obj))},
            "ground_station": dict(base.get("ground_station") or s["ground_station"]),
            "radio": dict(base.get("radio") or s["radio"]), "simulation": sim, "mode": "Automated", "linked": True}


def apply_pair(a_obj, b_obj, tca_jd: float, screen_km: float) -> str | None:
    """Push the pair into the Calculator. Returns an error message or None."""
    from modules.ui_state import get_manoeuvre
    focus = focus_for(a_obj, tca_jd)
    st.session_state["calc_focus"] = focus
    st.session_state["calc_p3_win"] = focus["win_h"]
    for k in P3_KEYS:                       # Phase 3 widgets pick up the new values
        st.session_state.pop(k, None)
    cfg = st.session_state.get("cfg")
    if cfg is None or cfg.get("linked"):    # never overwrite a Manual / file configuration
        try:
            cfg = build_cfg(a_obj, b_obj, focus, screen_km)
            st.session_state["cfg"] = cfg
        except Exception as err:            # e.g. python-sgp4 missing
            st.session_state["manoeuvre"] = {**get_manoeuvre(), "burn_min": focus["burn_min"]}
            return f"Calculator not updated: {err}"
    # Phase 3 manoeuvre: burn 1.5 orbits before the encounter, target = the calculator's screening distance,
    # delta-v = the calculator's own smallest-delta-v search (no placeholder value)
    man = {**get_manoeuvre(), "burn_min": focus["burn_min"],
           "target_km": float(cfg["simulation"]["screening_distance_km"])}
    man["dv_ms"], st.session_state["auto_dv_note"] = auto_delta_v(cfg, man, focus)
    st.session_state["manoeuvre"] = man
    return None


def auto_delta_v(cfg, man, focus):
    """Smallest delta-v (Phase 3 search) that lifts the miss after the burn to the target. Returns (m/s, note)."""
    from modules.avoidance import required_delta_v
    from modules.orbital_mechanics import epoch_jd, jd_grid
    sat_a, sat_b = cfg["objects"]["A"]["satrec"], cfg["objects"]["B"]["satrec"]
    start = max(0.0, man["burn_min"] - 10.0)
    grid = jd_grid(sat_a, start, max(30.0, focus["win_h"] * 60.0 - start), 30.0)
    try:
        with st.spinner("Calculator: finding the smallest delta-v…"):
            out = required_delta_v(sat_a, sat_b, grid, epoch_jd(sat_a) + man["burn_min"] / 1440.0,
                                   man["direction"], man["target_km"])
    except Exception as err:   # never block the page
        return 0.0, f"Delta-v search failed ({err}); set it in Phase 3."
    if out["found"]:
        return out["dv_ms"], (f"Calculator search: {out['dv_ms']:.3f} m/s {man['direction'].split(' (')[0].lower()} "
                              f"lifts the miss to {out['miss_km']:.2f} km (target {man['target_km']:g} km).")
    return 0.0, (f"Calculator search: no {man['direction'].split(' (')[0].lower()} burn up to 50 m/s reaches "
                 f"{man['target_km']:g} km at this burn time; delta-v left at 0. Change direction or burn time in Phase 3.")


def calc_is_linked() -> bool:
    cfg = st.session_state.get("cfg")
    return bool(cfg and cfg.get("linked"))
