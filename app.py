"""STC: Satellite Traffic Control - Streamlit dashboard (Team TRISOL, ASTRA 2026).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from stc.common.constants import RE, SECONDS_PER_DAY, MINUTES_PER_DAY
from stc.common.frames import (eci_to_ecef, ecef_to_eci, ecef_to_geodetic, geodetic_to_ecef, look_angles,
                                      coverage_half_angle, circle_points)
from stc.common.time_utils import fmt_utc, fmt_rel, fmt_duration, now_jd
from stc.comms.visibility import STATIONS, compare_passes, elevation_series
from stc.ingestion.spacetrack import load_spacetrack, SPACETRACK_DIR
from stc.ingestion.tle_loader import load_sample, load_from_text, load_from_celestrak, load_objects_text, DEBRIS_GROUPS
from stc.manoeuvre.planner import build_manoeuvre, miss_near, secondary_check, candidate_dv_values
from stc.risk.probability_of_collision import pc_isotropic, risk_level
from stc.screening.conjunction import screen_catalogue, separation_series, screen_pair, make_event
from calculator.link import get_pair, set_pair, apply_pair, calc_is_linked
from modules.avoidance import DIRECTIONS, dv_vector
from modules.ui_state import get_manoeuvre, set_manoeuvre
from stc.web.path_tracer import build_path_tracer_html

_ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="STC: Satellite Traffic Control",
                   page_icon=str(_ASSETS / "stc_logo_64.png") if (_ASSETS / "stc_logo_64.png").exists() else "🛰️",
                   layout="wide")


def _loading_logo_css():
    """Use the round STC logo (pulsing) in place of Streamlit's 'running' icon at the top right while a
    page recomputes. Loading messages (st.spinner) keep Streamlit's own spinner."""
    f = _ASSETS / "stc_logo.png"
    if not f.exists():
        return
    import base64
    url = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
    st.markdown(f"""<style>
@keyframes stc-pulse {{ 0%, 100% {{ transform: scale(0.88); opacity: .7; }} 50% {{ transform: scale(1.08); opacity: 1; }} }}
/* top-right "running" indicator: draw the logo on the running-man element itself, so it keeps the
   exact position and size; the original animation is pushed out of view with object-position */
[data-testid="stStatusWidget"] img {{ object-position: -9999px 0 !important;
  background: url("{url}") center / contain no-repeat !important; border-radius: 50%;
  animation: stc-pulse 1.2s ease-in-out infinite; }}
[data-testid="stStatusWidget"] svg {{ background: url("{url}") center / contain no-repeat !important; border-radius: 50%;
  animation: stc-pulse 1.2s ease-in-out infinite; }}
[data-testid="stStatusWidget"] svg * {{ fill: transparent !important; stroke: transparent !important; }}
</style>""", unsafe_allow_html=True)


_loading_logo_css()

# Charcoal and gold palette
BG, SURFACE, INK = "#FAFAF9", "#E7E5E4", "#1C1917"
GOLD, CHARCOAL, RED, STONE, GREY = "#CA8A04", "#292524", "#B91C1C", "#78716C", "#A8A29E"
RISK_ICON = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🔵 LOW"}
PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font=dict(color=INK))


def _stretch(fn, *args, **kw):
    """Full-width rendering across Streamlit versions (width='stretch' on new, use_container_width on old)."""
    try:
        return fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


def show_chart(fig):
    return _stretch(st.plotly_chart, fig)


def show_df(df, **kw):
    return _stretch(st.dataframe, df, **kw)


# ============================================================ data
@st.cache_resource(show_spinner=False)
def cached_sample():
    return load_sample()


def set_data(prims, objs, start_jd, label, key):
    st.session_state.data = dict(prims=prims, objs=objs, start=start_jd, label=label, key=key)
    for k in ("ev_idx", "pass_idx", "pair", "pair_sig"):
        st.session_state.pop(k, None)


@st.cache_resource(show_spinner="Loading Space-Track catalogue…")
def cached_spacetrack():
    return load_spacetrack()


def use_spacetrack():
    p, o, newest, info = cached_spacetrack()
    syn = f" incl. {info['synthetic']:,} synthetic demo debris" if info.get("synthetic") else ""
    set_data(p, o, now_jd(), f"Space-Track · {info['objects']:,} objects{syn}", "spacetrack")


if "data" not in st.session_state:
    if any(SPACETRACK_DIR.glob("*.xml")):
        use_spacetrack()
    else:
        p, o, s = cached_sample()
        set_data(p, o, s, "Sample dataset (synthetic)", "sample")

D = st.session_state.data
PRIMS, OBJS, START = D["prims"], D["objs"], D["start"]

# ============================================================ the shared pair (A = satellite, B = object it meets)
# One pair for every page and the Calculator; see calculator/link.py.
PAIR = get_pair()
_all = {o.satnum: o for o in PRIMS + OBJS}
if PAIR["a"] not in _all or _all[PAIR["a"]].obj_type in ("DEBRIS", "ROCKET BODY"):
    set_pair(PRIMS[0].satnum if PRIMS else None, None)
if PAIR["a"] is not None and PAIR["a"] not in {p.satnum for p in PRIMS}:
    # A satellite chosen in the Calculator that is not in the sidebar list: make it a tracked satellite
    _o = _all[PAIR["a"]]
    D.update(prims=PRIMS + [_o], objs=[o for o in OBJS if o is not _o], key=f"{D['key']}+A{_o.satnum}")
    PRIMS, OBJS = D["prims"], D["objs"]
ALL = {o.satnum: o for o in PRIMS + OBJS}
st.session_state["sat_idx"] = next((i for i, p in enumerate(PRIMS) if p.satnum == PAIR["a"]), 0)


def _on_sat_change():
    set_pair(st.session_state.data["prims"][st.session_state.sat_idx].satnum, None)


def _calc_value(section, key, fallback):
    """A value from the Calculator configuration (or the calculator's settings file before it exists)."""
    cfg = st.session_state.get("cfg")
    if cfg and key in cfg.get(section, {}):
        return float(cfg[section][key])
    try:
        from modules.catalogue import load_settings
        return float(load_settings()[section][key])
    except Exception:
        return fallback


def _set_calc_value(section, key, widget_key, calc_widget):
    cfg = st.session_state.get("cfg")
    if cfg:
        cfg[section][key] = float(st.session_state[widget_key])
    st.session_state.pop(calc_widget, None)      # Calculator Phase 1 shows the new value


def _on_event_change():
    b, jd = st.session_state["_event_keys"][st.session_state.ev_idx]
    set_pair(get_pair()["a"], b, jd)

# ============================================================ pages + persistent inputs
PAGES = ["🏠 Overview", "1 · Path Tracer", "2 · Collision Stats", "3 · Alternate Path Mapping", "4 · New Path Stats",
         "5 · Calculator"]
CALC_SECTIONS = {"Phase 1 · Input & configuration": "phase1", "Phase 2 · Orbital propagation": "phase2",
                 "Phase 3 · Encounter screening": "phase3", "Phase 4 · Doppler & ground station": "phase4",
                 "Phase 5 · Scenario comparison": "phase5"}
PAGE_INFO = {
    PAGES[1]: "Where is the satellite? How fast is it moving? Where will it be in the next few minutes? Live map of every tracked object.",
    PAGES[2]: "How close will the satellite come to another satellite or debris object, and when will they be closest?",
    PAGES[3]: "What happens if the satellite changes its velocity? How much should it change in a hypothetical manoeuvre?",
    PAGES[4]: "How does the hypothetical manoeuvre change ground-station visibility?",
    PAGES[5]: "Two-object calculator: inputs, propagation, encounter screening, Doppler and scenario comparison. "
              "Its objects and trajectories appear on the map and globe.",
}
DEFAULTS = dict(page=PAGES[0], ev_idx=0, mass=700.0, isp=220.0,
                stations=[s["name"] for s in STATIONS], pass_idx=0,
                calc_section=list(CALC_SECTIONS)[0])
for _k, _v in DEFAULTS.items():
    # Re-assigning keeps a widget's value alive while its page is not displayed.
    st.session_state[_k] = st.session_state[_k] if _k in st.session_state else _v


def go_to(page):
    st.session_state.page = page


def nav_footer(idx):
    st.divider()
    left, _, right = st.columns([1, 2, 1])
    if idx > 0:
        left.button(f"← {PAGES[idx - 1]}", on_click=go_to, args=(PAGES[idx - 1],), key=f"back{idx}")
    if idx < len(PAGES) - 1:
        right.button(f"{PAGES[idx + 1]} →", on_click=go_to, args=(PAGES[idx + 1],), type="primary", key=f"next{idx}")


# ============================================================ sidebar
with st.sidebar:
    _logo = _ASSETS / "stc_logo.png"
    if _logo.exists():
        import base64
        _b64 = base64.b64encode(_logo.read_bytes()).decode()
        st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin:0 0 .35rem">
<img src="data:image/png;base64,{_b64}" alt="STC logo" style="width:52px;height:52px;border-radius:50%">
<span style="font-size:1.75rem;font-weight:700;line-height:1">STC</span></div>""", unsafe_allow_html=True)
        st.markdown("**Satellite Traffic Control**")
    else:
        st.markdown("## STC\n**Satellite Traffic Control**")
    st.caption("Conjunction screening, collision avoidance & link-visibility analysis for LEO satellites · Team TRISOL")
    st.radio("Go to", PAGES, key="page")
    if st.session_state.page == PAGES[5]:
        st.radio("Calculator section", list(CALC_SECTIONS), key="calc_section")
    st.divider()

    with st.expander("Data source", expanded=False):
        src = st.radio("Load", ["Space-Track files", "Sample dataset (synthetic)", "Paste TLE / OMM", "CelesTrak"],
                       horizontal=False, label_visibility="collapsed")
        if src == "Space-Track files":
            files = sorted(SPACETRACK_DIR.glob("*.xml"))
            if not files:
                st.caption("Put Space-Track OMM XML files in `data/spacetrack/` to use them here.")
            else:
                _, _, newest, info = cached_spacetrack()
                st.caption(f"{', '.join(f.name for f in files)} · {info['objects']:,} objects · newest epoch "
                           f"{fmt_utc(newest, False)} UTC. Country, type and size come from Space-Track. "
                           "The clock starts at the current time.")
                if st.button("Use Space-Track data"):
                    use_spacetrack()
                    st.rerun()
        elif src == "Sample dataset (synthetic)":
            st.caption("3 sample Indian-style satellites, 288 synthetic objects with built-in close approaches.")
            if st.button("Use sample dataset"):
                p, o, s = cached_sample()
                set_data(p, o, s, "Sample dataset (synthetic)", "sample")
                st.rerun()
        elif src == "Paste TLE / OMM":
            st.caption("TLE, or OMM as JSON, CSV or XML (CelesTrak FORMAT=json/csv/xml). The format is detected automatically.")
            pt = st.text_area("Satellite(s) to protect", height=110, placeholder="CARTOSAT-3\n1 44804U ...\n2 44804 ...")
            ot = st.text_area("Objects to screen against", height=140)
            if st.button("Load pasted data"):
                try:
                    p, o = load_from_text(pt, ot)
                    set_data(p, o, now_jd(), f"Pasted data ({len(p)} sat, {len(o)} obj)", hashlib.md5((pt + ot).encode()).hexdigest())
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
        else:
            ids = st.text_input("NORAD IDs to protect", placeholder="e.g. 44804, 44233")
            groups = st.multiselect("Debris groups", list(DEBRIS_GROUPS), default=list(DEBRIS_GROUPS)[:2], format_func=DEBRIS_GROUPS.get)
            st.caption("Downloads OMM JSON and keeps a copy for 2 hours, as CelesTrak asks. "
                       "Repeat fetches within 2 hours use the saved copy.")
            if st.button("Fetch from CelesTrak"):
                try:
                    with st.spinner("Fetching…"):
                        p, o, info = load_from_celestrak(ids.split(","), groups)
                    st.session_state.fetch_info = info
                    set_data(p, o, now_jd(), f"CelesTrak ({len(p)} sat, {len(o)} obj)", f"ct-{ids}-{groups}-{now_jd()}")
                    st.rerun()
                except Exception as err:  # network, bad ID, blocked
                    st.error(f"Could not load from CelesTrak: {err}")
            for inf in st.session_state.get("fetch_info", []):
                note = {"download": "downloaded now", "cache": f"saved copy, {inf['age_min']:.0f} min old",
                        "stale cache": f"CelesTrak unreachable ({inf.get('error', '')}); using copy {inf['age_min']:.0f} min old"}[inf["source"]]
                st.caption(f"`{inf['query']}`: {note}")
        st.divider()
        st.markdown("**Add objects from a file**")
        st.caption("TLE, OMM (JSON/CSV/XML) or SpaceObjectsCatalog XML with ECI coordinates. "
                   "The objects are added to the current data and appear on every page.")
        up = st.file_uploader("Orbit file", type=["xml", "json", "csv", "txt", "tle"], label_visibility="collapsed")
        if up is not None and st.button("Add these objects"):
            try:
                new, notes = load_objects_text(up.getvalue().decode("utf-8", "replace"))
            except Exception as err:  # malformed file
                new, notes = [], [f"Could not read {up.name}: {err}"]
            have = {o.satnum for o in OBJS} | {p.satnum for p in PRIMS}
            new = [o for o in new if o.satnum not in have]
            st.session_state.file_notes = [f"{up.name}: added {len(new)} object(s)."] + notes
            if new:
                set_data(PRIMS, OBJS + new, START, D["label"] + f" + {up.name}",
                         D["key"] + "+" + hashlib.md5(up.getvalue()).hexdigest()[:8])
                st.rerun()
        for note in st.session_state.get("file_notes", []):
            st.caption(("⚠ " if "does not match" in note or "skipped" in note else "• ") + note)
    st.caption(f"Data: **{D['label']}**")

    sat_idx = st.selectbox("Satellite (Object A)", range(len(PRIMS)), format_func=lambda i: PRIMS[i].name, key="sat_idx",
                           on_change=_on_sat_change)
    SAT = PRIMS[sat_idx]

    st.markdown("**Screening**")
    window_h = st.select_slider("Window (hours)", [24, 48, 72], value=72)
    st.session_state["thr"] = min(100.0, max(0.5, _calc_value("simulation", "screening_distance_km", 10.0)))
    thr = st.number_input("Threshold (km) · = Calculator screening distance", 0.5, 100.0, step=0.5, key="thr",
                          on_change=_set_calc_value, args=("simulation", "screening_distance_km", "thr", "cat_screen"))
    sigma = st.number_input("Position uncertainty 1σ (m)", 20, 5000, 300, 10, help="Combined, assumed isotropic")
    hbr = st.number_input("Hard-body radius (m)", 1, 100, 10, 1)

    st.markdown("**Clock**")
    t_min = st.slider("Minutes from start", 0, window_h * 60, 0, 1)
    SIM = START + t_min / MINUTES_PER_DAY
    st.caption(f"`{fmt_utc(SIM)} UTC`  ·  {fmt_rel(SIM, START)}")


# ============================================================ cached computations
@st.cache_resource(show_spinner="Screening catalogue…")
def run_screening(data_key, sat_idx, window_h, thr, sigma, hbr):
    prims = st.session_state.data["prims"]
    sat = prims[sat_idx]
    objs = st.session_state.data["objs"] + [p for p in prims if p is not sat]
    start = st.session_state.data["start"]
    return screen_catalogue(sat, objs, start, start + window_h / 24, thr, 120.0, sigma, hbr)


@st.cache_resource(show_spinner="Propagating manoeuvre…")
def run_manoeuvre(data_key, sat_idx, ev_key, ev_jd, lead, dv):
    sat = st.session_state.data["prims"][sat_idx]
    m = build_manoeuvre(sat, ev_jd, lead, list(dv), horizon_jd=ev_jd - lead * sat.period_min / MINUTES_PER_DAY + 1)
    return m


@st.cache_resource(show_spinner="Checking catalogue for new close approaches…")
def run_secondary(data_key, sat_idx, window_h, thr, sigma, hbr, b_id, ev_jd, lead, dv):
    events, _ = run_screening(data_key, sat_idx, window_h, thr, sigma, hbr)
    man = run_manoeuvre(data_key, sat_idx, f"{b_id}-{ev_jd:.6f}", ev_jd, lead, dv)
    prims = st.session_state.data["prims"]
    found = secondary_check(man, st.session_state.data["objs"] + [p for p in prims if p is not prims[sat_idx]], events, thr)
    return [f for f in found if not (f["obj"].satnum == b_id and abs(f["jd"] - ev_jd) < 5 / MINUTES_PER_DAY)]


@st.cache_resource(show_spinner="Finding the closest approach of the chosen pair…")
def run_pair_event(data_key, a_id, b_id, jd0, jd1, sigma, hbr):
    """Closest approach of A and B in the window, even if it is outside the screening threshold."""
    allo = {o.satnum: o for o in st.session_state.data["prims"] + st.session_state.data["objs"]}
    sat, obj = allo[a_id], allo[b_id]
    mins = screen_pair(sat.at, obj.at, jd0, jd1, 60.0, 1e9)
    if not mins:
        return None
    jd, miss = min(mins, key=lambda m: m[1])
    ev = make_event(sat.at, obj, jd, miss).score(sigma, hbr)
    ev.custom = True
    return ev


@st.cache_resource(show_spinner="Computing ground-station passes…")
def run_passes(data_key, sat_idx, ev_key, ev_jd, lead, dv, station_names, mask):
    sat = st.session_state.data["prims"][sat_idx]
    man = run_manoeuvre(data_key, sat_idx, ev_key, ev_jd, lead, dv)
    sts = [s for s in STATIONS if s["name"] in station_names]
    return compare_passes(sat.at, man.positions, sts, man.jd_burn, man.jd_burn + 1, mask)


EVENTS, NCAND = run_screening(D["key"], sat_idx, window_h, thr, sigma, hbr)
st.session_state["_event_obj_ids"] = [e.obj.satnum for e in EVENTS]

# ---- Object B and the selected close approach, from the shared pair
if PAIR["b"] not in ALL or PAIR["b"] == SAT.satnum:
    set_pair(SAT.satnum, EVENTS[0].obj.satnum if EVENTS else None, EVENTS[0].jd if EVENTS else None)
EV_OPTS = list(EVENTS)
_match = [i for i, e in enumerate(EVENTS) if e.obj.satnum == PAIR["b"]]
if _match:
    EV_I = min(_match, key=lambda i: abs(EVENTS[i].jd - PAIR["jd"])) if PAIR["jd"] else _match[0]
elif PAIR["b"] is not None:
    _pe = run_pair_event(D["key"], SAT.satnum, PAIR["b"], START, START + window_h / 24, sigma, hbr)
    if _pe is not None:
        EV_OPTS.append(_pe)
    EV_I = len(EV_OPTS) - 1 if _pe is not None else None
else:
    EV_I = None
EV = EV_OPTS[EV_I] if EV_I is not None else None
if EV is not None:
    PAIR["jd"] = EV.jd
    st.session_state.ev_idx = EV_I
st.session_state["_event_keys"] = [(e.obj.satnum, e.jd) for e in EV_OPTS]

# ---- push a new pair into the Calculator (objects, window, Phase 3 burn 1.5 orbits before the encounter)
_sig = (D["key"], SAT.satnum, EV.obj.satnum if EV else None, round(EV.jd, 5) if EV else None)
if EV is not None and st.session_state.get("pair_sig") != _sig:
    st.session_state["pair_sig"] = _sig
    _msg = apply_pair(SAT, EV.obj, EV.jd, thr)
    if _msg:
        st.sidebar.warning(_msg)

MASK = _calc_value("radio", "minimum_elevation_deg", 10.0)   # elevation mask = Calculator minimum elevation

# ---- the manoeuvre: Calculator Phase 3 settings, shared with Alternate Path Mapping and New Path Stats
MP = get_manoeuvre()
JD_BURN = SAT.jdepoch + float(MP["burn_min"]) / MINUTES_PER_DAY
LEAD = (EV.jd - JD_BURN) * MINUTES_PER_DAY / SAT.period_min if EV else 0.0
DV = tuple(float(x) * 1000.0 for x in dv_vector(MP["direction"], float(MP["dv_ms"])))   # R, T, N in m/s
MAN_ARGS = (D["key"], sat_idx, f"{EV.obj.satnum}-{EV.jd:.6f}", EV.jd, round(LEAD, 6), DV) if EV and LEAD > 0 else None
MAN = run_manoeuvre(*MAN_ARGS) if MAN_ARGS else None
PAGE = st.session_state.page


# ============================================================ header
if PAGE != PAGES[5]:
    st.title("STC: Satellite Traffic Control" if PAGE == PAGES[0] else PAGE.split("· ")[1])
    if PAGE in PAGE_INFO:
        st.caption(PAGE_INFO[PAGE])
    upcoming = sorted([e for e in EVENTS if e.jd >= SIM], key=lambda e: e.jd)
    c1, c2, c3, c4 = st.columns([2.2, 2.6, 1.2, 1.2])
    c1.markdown(f"**Tracking:** {SAT.name}  \nNORAD {SAT.satnum}{' · ' + SAT.desc if SAT.desc else ''}")
    if upcoming:
        nx = upcoming[0]
        c2.markdown(f"**Next close approach:** {RISK_ICON[nx.risk]} **{nx.obj.name}** in {fmt_duration((nx.jd - SIM) * SECONDS_PER_DAY)}  \nmiss {nx.miss * 1000:.0f} m · {fmt_utc(nx.jd)} UTC")
    else:
        c2.markdown("**Next close approach:** none inside threshold")
    c3.metric("Events in window", len(EVENTS))
    c4.metric("High risk", sum(e.risk == "HIGH" for e in EVENTS))


# ============================================================ calculator overlay (shared by Path Tracer and Calculator)
@st.cache_resource(show_spinner="Drawing calculator trajectories…")
def _calc_overlay(key, _cfg, manoeuvre, p3_win, scenarios):
    from calculator.overlay import build_overlay
    return build_overlay(_cfg, manoeuvre, p3_win, scenarios)


def calc_overlay():
    """Overlay of the Calculator's objects and trajectories, or None if Phase 1 is not complete."""
    cfg = st.session_state.get("cfg")
    if not cfg:
        return None
    try:
        from calculator.overlay import overlay_key
        man = st.session_state.get("manoeuvre")
        win = st.session_state.get("calc_p3_win")
        scen = st.session_state.get("calc_scenarios")
        return _calc_overlay(overlay_key(cfg, man, win, scen), cfg, man, win, scen)
    except Exception as err:  # never let the overlay break the page
        st.caption(f"Calculator overlay unavailable: {err}")
        return None


def geo(jd, r):
    lat, lon, alt = ecef_to_geodetic(eci_to_ecef(r, jd))
    return lat, lon, alt


def break_antimeridian(lon, lat):
    lon, lat = list(lon), list(lat)
    out_lo, out_la = [], []
    for i in range(len(lon)):
        if i and abs(lon[i] - lon[i - 1]) > 180:
            out_lo.append(None); out_la.append(None)
        out_lo.append(lon[i]); out_la.append(lat[i])
    return out_lo, out_la


def line_chart(xs, ys, names, colors, x_title, y_title, hline=None, vline=None, dashes=None, height=320):
    fig = go.Figure()
    for i, (x, y, n, c) in enumerate(zip(xs, ys, names, colors)):
        fig.add_trace(go.Scatter(x=x, y=y, name=n, mode="lines", line=dict(color=c, width=2.5, dash=(dashes or [None] * len(xs))[i])))
    if hline is not None:
        fig.add_hline(y=hline[0], line=dict(color=STONE, dash="dash"), annotation_text=hline[1], annotation_font_color=STONE)
    if vline is not None:
        fig.add_vline(x=vline[0], line=dict(color=RED, dash="dot"), annotation_text=vline[1], annotation_font_color=RED)
    fig.update_layout(**PLOT_BG, height=height, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.08),
                      xaxis=dict(title=x_title, gridcolor=SURFACE, zerolinecolor=SURFACE), yaxis=dict(title=y_title, gridcolor=SURFACE, zerolinecolor=SURFACE))
    return fig


# ============================================================ Overview
if PAGE == PAGES[0]:
    st.markdown("Screen a satellite for close approaches, test a hypothetical avoidance manoeuvre and see how it "
                "changes ground-station contact. Pick a module below or use the menu in the sidebar.")
    k = st.columns(4)
    k[0].metric("Satellite", SAT.name)
    k[1].metric(f"Close approaches ({window_h} h)", len(EVENTS))
    k[2].metric("High risk", sum(e.risk == "HIGH" for e in EVENTS))
    k[3].metric("Closest approach", f"{min(e.miss for e in EVENTS) * 1000:.0f} m" if EVENTS else "—")
    st.write("")
    cols = st.columns(2)
    for i, p in enumerate(PAGES[1:]):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {p}")
                st.caption(PAGE_INFO[p])
                st.button("Open →", key=f"open{i}", on_click=go_to, args=(p,), type="primary")
    nav_footer(0)

# ============================================================ 1. Path tracer
def _dms(v: float, pos: str, neg: str) -> str:
    h = pos if v >= 0 else neg
    a = abs(v)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60) * 3600
    return f"{d}° {m:02d}′ {s:04.1f}″ {h}"


def _orbit_type(inc_deg: float, alt_km: float) -> str:
    if 95.5 <= inc_deg <= 100.5 and alt_km < 1500:
        return "Sun-synchronous (near-polar)"
    if abs(inc_deg - 90) < 10:
        return "Polar"
    if inc_deg > 90:
        return "Retrograde"
    if inc_deg < 5:
        return "Equatorial"
    return "Inclined, prograde"


def stat_dropdown(col, label: str, value: str, rows: list[tuple[str, str]]):
    with col.expander(f"**{label}** · {value}"):
        for k_, v_ in rows:
            st.markdown(f"<span style='color:#78716C'>{k_}</span><br><b>{v_}</b>", unsafe_allow_html=True)


if PAGE == PAGES[1]:
    r, v = SAT.at1(SIM)
    lat, lon, alt = (float(x) for x in geo(SIM, r))
    spd = float(np.linalg.norm(v))
    inc = float(np.degrees(SAT.inclo))
    st.caption(f"Values at the dashboard clock: **{fmt_utc(SIM)} UTC** (change it in the sidebar). Click a stat for details.")
    row1, row2 = st.columns(4), st.columns(4)
    stat_dropdown(row1[0], "Latitude", f"{lat:.3f}°", [("Degrees, minutes, seconds", _dms(lat, "N", "S")),
                                                        ("Hemisphere", "Northern" if lat >= 0 else "Southern")])
    stat_dropdown(row1[1], "Longitude", f"{lon:.3f}°", [("Degrees, minutes, seconds", _dms(lon, "E", "W")),
                                                         ("Side", "East of Greenwich" if lon >= 0 else "West of Greenwich")])
    stat_dropdown(row1[2], "Altitude", f"{alt:.1f} km", [("Above", "WGS-84 ellipsoid"),
                                                          ("Orbit band", f"{SAT.perigee_km:.0f} – {SAT.apogee_km:.0f} km ({SAT.regime})")])
    stat_dropdown(row1[3], "Speed", f"{spd:.3f} km/s", [("In km/h", f"{spd * 3600:,.0f} km/h"),
                                                         ("Frame", "Inertial (TEME), relative to Earth's centre")])
    stat_dropdown(row2[0], "Orbital period", f"{SAT.period_min:.2f} min", [("Revolutions per day", f"{1440 / SAT.period_min:.3f}"),
                                                                            ("Semi-major axis", f"{SAT.a_km:,.1f} km")])
    stat_dropdown(row2[1], "Inclination", f"{inc:.2f}°", [("Orbit type", _orbit_type(inc, alt)),
                                                           ("Highest latitude reached", f"{min(inc, 180 - inc):.1f}°")])
    stat_dropdown(row2[2], "Perigee / apogee", f"{SAT.perigee_km:.0f} / {SAT.apogee_km:.0f} km",
                  [("Eccentricity", f"{SAT.ecco:.5f}"), ("Difference", f"{SAT.apogee_km - SAT.perigee_km:.1f} km")])
    stat_dropdown(row2[3], "TLE age", f"{SIM - SAT.jdepoch:+.2f} days", [("TLE epoch", f"{fmt_utc(SAT.jdepoch)} UTC"),
                                                                          ("NORAD ID", SAT.satnum)])

    st.markdown("#### Live map and globe")
    st.caption("Starts at the dashboard clock and runs on its own. Indian satellites green, international satellites "
               "sky blue, debris and rocket bodies black. Hover an object for its name; click it to trace its path - "
               f"objects that come within {thr:g} km of it in the next {window_h} h blink gold. Use LEO / MEO / All under the map.")
    overlay = calc_overlay()
    if overlay and calc_is_linked():
        st.caption(f"Calculator link: Object A is {SAT.name} (the tracked satellite) and Object B is "
                   f"{EV.obj.name if EV else '—'}, shown in purple and blinking. The dashed red line is the alternate path "
                   "after the Phase 3 burn; ✕ marks the closest approach.")
    elif overlay:
        st.caption("Calculator objects A and B (Manual / file input) and their trajectories are shown too. They use their own "
                   "epoch: press **Go to calculator time** above the map to see them at the analysed time.")
    html = build_path_tracer_html(PRIMS, OBJS, sat_idx, SIM, STATIONS, {e.obj.satnum for e in EVENTS}, MASK,
                                  threshold_km=thr, window_h=window_h, calc=overlay)
    components.html(html, height=820, scrolling=True)

    st.markdown("#### Position forecast · next 10 minutes (SGP4)")
    jd = SIM + np.arange(0, 11) / MINUTES_PER_DAY
    rr, vv = SAT.at(jd)
    la, lo, al = geo(jd, rr)
    ecef = eci_to_ecef(rr, jd)
    vis = []
    for k in range(len(jd)):
        seen = [f"{s['short']} ({e:.0f}°)" for s in STATIONS
                for e in [float(look_angles(s["lat"], s["lon"], s.get("alt", 0.0), ecef[k])[0])] if e >= MASK]
        vis.append(", ".join(seen) or "—")
    show_df(pd.DataFrame({"Time (UTC)": [fmt_utc(x) for x in jd], "From now": [f"+{i} min" for i in range(11)],
                          "Lat (°)": la.round(3), "Lon (°)": lo.round(3), "Alt (km)": al.round(1),
                          "Speed (km/s)": np.linalg.norm(vv, axis=1).round(3), "Visible from": vis}), hide_index=True)
    nav_footer(1)


# ============================================================ 2. Collision Stats
def separation_chart(sat, e, wide: bool):
    """Distance between the satellite and the event's object around TCA.

    wide=True: a window sized so the curve reaches several hundred km (overall shape of the encounter).
    wide=False: zoomed so the curve rises to about 6x the miss distance, making the minimum visible.
    Both are recomputed from the selected objects' orbits whenever the selected event changes."""
    if wide:
        half = float(np.clip(600.0 / max(e.vrel, 0.01), 20.0, 1800.0))
    else:
        half = float(np.clip(6.0 * max(e.miss, 0.05) / max(e.vrel, 0.01), 0.05, 600.0))
    s, d = separation_series(sat.at, e.obj.at, e.jd, half, 401)
    unit, k = ("m", 1000.0) if (not wide and e.miss < 2.0) else ("km", 1.0)
    fig = line_chart([s], [d * k], ["Separation"], [GOLD], "seconds from time of closest approach",
                     f"distance ({unit})", vline=(0, "TCA"), height=260 if wide else 240)
    fig.add_trace(go.Scatter(x=[0], y=[e.miss * k], mode="markers", marker=dict(size=9, color=RED), name="Closest approach",
                             showlegend=False))
    fig.add_annotation(x=0, y=e.miss * k, text=f"miss {e.miss * 1000:,.0f} m at {fmt_utc(e.jd)} UTC", showarrow=True,
                       arrowhead=2, ax=70, ay=-35 if wide else -45, font=dict(color=INK, size=12), bgcolor="#FFFFFF")
    title = (f"{sat.name} × {e.obj.name}: separation over ±{half:.0f} s" if wide else
             f"Zoom: ±{half:.2f} s around closest approach")
    fig.update_layout(title=dict(text=title, font=dict(size=13)), margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    if not wide:
        fig.update_yaxes(range=[0, float(np.nanmax(d * k)) * 1.05])
    return fig



if PAGE == PAGES[2]:
    k = st.columns(5)
    k[0].metric("Objects in catalogue", len(OBJS))
    k[1].metric("After altitude filter", NCAND)
    k[2].metric(f"Events < {thr:g} km", len(EVENTS))
    k[3].metric("High-risk events", sum(e.risk == "HIGH" for e in EVENTS))
    k[4].metric("Closest approach", f"{min(e.miss for e in EVENTS) * 1000:.0f} m" if EVENTS else "—")

    if not EVENTS:
        st.info(f"No close approaches within {thr:g} km in the next {window_h} h.")
    if EVENTS:
        df = pd.DataFrame([{"Risk": RISK_ICON[e.risk], "Object": e.obj.name,
                            "Type": e.obj.obj_type + (" (synthetic demo)" if getattr(e.obj, "synthetic", False) else ""), "TCA (UTC)": fmt_utc(e.jd),
                            "From start": fmt_rel(e.jd, START), "Miss (m)": round(e.miss * 1000), "Rel. speed (km/s)": round(e.vrel, 2),
                            "Pc": f"{e.pc:.1e}"} for e in EVENTS])
        show_df(df, hide_index=True)
    if EV_OPTS:
        ev_idx = st.selectbox("Event to analyse · sets Object B everywhere (Calculator, Alternate Path Mapping, New Path Stats, map)",
                              range(len(EV_OPTS)), key="ev_idx", on_change=_on_event_change,
                              format_func=lambda i: f"{EV_OPTS[i].risk} · {EV_OPTS[i].obj.name} · {fmt_utc(EV_OPTS[i].jd)} · "
                                                    f"{EV_OPTS[i].miss * 1000:,.0f} m"
                                                    + (" · chosen in Calculator (outside threshold)" if getattr(EV_OPTS[i], "custom", False) else ""))
        e = EV_OPTS[ev_idx]
        if getattr(e, "custom", False):
            st.info(f"{e.obj.name} was chosen as Object B in the Calculator. Its closest approach to {SAT.name} in the "
                    f"next {window_h} h is {e.miss:,.2f} km, outside the {thr:g} km threshold, so it is not in the table above.")
        a, b = st.columns([1, 1.3])
        with a:
            rt = e.rtn_miss * 1000
            st.markdown(f"#### {RISK_ICON[e.risk]} {SAT.name} × {e.obj.name}")
            st.table(pd.DataFrame({"": ["Time of closest approach", "Miss distance", "Miss components R / T / N",
                                        "Relative speed", "Crossing angle", "Collision probability (assumed σ)", "Object"],
                                   "Value": [f"{fmt_utc(e.jd)} UTC ({fmt_rel(e.jd, START)})", f"{e.miss * 1000:.0f} m",
                                             f"{rt[0]:.0f} / {rt[1]:.0f} / {rt[2]:.0f} m", f"{e.vrel:.2f} km/s",
                                             f"{e.angle_deg:.0f}°", f"{e.pc:.1e}",
                                             f"{e.obj.obj_type} · RCS {e.obj.rcs or '—'} · NORAD {e.obj.satnum}"
                                             + (" · SYNTHETIC DEMO OBJECT" if getattr(e.obj, "synthetic", False) else "")]}).set_index(""))
        with b:
            show_chart(separation_chart(SAT, e, wide=True))
            show_chart(separation_chart(SAT, e, wide=False))
            st.caption(
                f"Distance between **{SAT.name}** and **{e.obj.name}**, computed from both objects' orbits (SGP4) "
                f"around the time of closest approach. They close at **{e.vrel:.2f} km/s** (crossing angle "
                f"{e.angle_deg:.0f}°), so the gap changes by about {e.vrel:.1f} km every second - the steeper the V, "
                f"the faster and briefer the encounter. The bottom of the V is the miss distance, "
                f"**{e.miss * 1000:,.0f} m**: too small to see on the wide view, so the lower chart zooms in.")
    nav_footer(2)


# ============================================================ 3. Alternate Path Mapping
P3_WIDGETS = ("p3_burn", "p3_dir", "p3_dv", "p3_target")


def _on_ap_change():
    """Alternate Path Mapping controls -> the shared Phase 3 manoeuvre."""
    tca_min, period = st.session_state["_ap_ctx"]
    set_manoeuvre({"burn_min": max(0.0, round(tca_min - float(st.session_state.ap_lead) * period, 3)),
                   "direction": st.session_state.ap_dir, "dv_ms": float(st.session_state.ap_dv),
                   "target_km": float(st.session_state.ap_target)})
    for k in P3_WIDGETS + ("p3_find_result",):
        st.session_state.pop(k, None)
    st.session_state.suggest_msg = ""


def _suggest(*_):
    """Same search as Calculator Phase 3's 'Find the smallest delta-v' button, applied everywhere."""
    from calculator.link import auto_delta_v
    cfg, man, focus = st.session_state.get("cfg"), get_manoeuvre(), st.session_state.get("calc_focus")
    if not cfg or not focus:
        st.session_state.suggest_msg = "The Calculator has no configuration yet."
        return
    dv, note = auto_delta_v(cfg, man, focus)
    set_manoeuvre({**man, "dv_ms": dv} if dv > 0 else man)
    for k in P3_WIDGETS + ("p3_find_result",):
        st.session_state.pop(k, None)
    st.session_state.suggest_msg = note


def _reset():
    set_manoeuvre({**get_manoeuvre(), "dv_ms": 0.0})
    for k in P3_WIDGETS:
        st.session_state.pop(k, None)
    st.session_state.suggest_msg = ""


def pair_map(start_jd, height=780):
    """Map and globe of the shared pair: A, B, the closest approach and A's alternate (manoeuvred) path."""
    overlay = calc_overlay()
    if not overlay:
        st.caption("Map unavailable: the Calculator has no configuration yet.")
        return
    if not calc_is_linked():
        st.warning("The Calculator uses Manual or file input, so the map shows the Calculator's own objects, not this pair. "
                   "Switch Calculator Phase 1 back to 'Dashboard catalogue' to link them.")
    for note in overlay.get("notes", []):
        st.caption("⚠ " + note)
    cfg_ = st.session_state.cfg
    html = build_path_tracer_html([], [], -1, start_jd, STATIONS, set(), 5.0,
                                  threshold_km=float(cfg_["simulation"]["screening_distance_km"]),
                                  window_h=float(cfg_["simulation"]["duration_hours"]), calc=overlay)
    components.html(html, height=height, scrolling=True)


if PAGE == PAGES[3]:
    if EV is None:
        st.info("No close approach selected. Pick a satellite with events, or choose Object B on Collision Stats or in the Calculator.")
    else:
        e = EV
        tca_min = (e.jd - SAT.jdepoch) * MINUTES_PER_DAY
        max_lead = max(0.25, tca_min / SAT.period_min)
        st.session_state["_ap_ctx"] = (tca_min, SAT.period_min)
        st.session_state.ap_lead = round(float(min(max(LEAD, 0.0), max_lead)), 3)
        st.session_state.ap_dir = MP["direction"]
        st.session_state.ap_dv = float(min(max(float(MP["dv_ms"]), 0.0), 50.0))
        st.session_state.ap_target = float(min(max(float(MP["target_km"]), 0.1), 100.0))
        ctl, res = st.columns([1, 2.2])
        with ctl:
            st.markdown(f"**Event:** {SAT.name} × {e.obj.name}  \n{fmt_utc(e.jd)} UTC · miss {e.miss * 1000:,.0f} m · {RISK_ICON[e.risk]}")
            st.caption("🔗 This is the **Calculator Phase 3 manoeuvre**: changes here update Phase 3, and changes in "
                       "Phase 3 update this page, New Path Stats and the map. Choose another event on Collision Stats.")
            st.number_input("Burn time (orbits before closest approach)", 0.0, float(max_lead), step=0.25, format="%.2f",
                            key="ap_lead", on_change=_on_ap_change)
            st.caption(f"Burn at {fmt_utc(JD_BURN)} UTC = Phase 3 burn time {float(MP['burn_min']):,.0f} min after "
                       f"{SAT.name}'s element epoch")
            st.selectbox("Direction (RTN)", list(DIRECTIONS), key="ap_dir", on_change=_on_ap_change)
            st.number_input("Δv magnitude (m/s)", 0.0, 50.0, step=0.005, format="%.3f", key="ap_dv", on_change=_on_ap_change)
            st.number_input("Target miss (km)", 0.1, 100.0, step=0.1, key="ap_target", on_change=_on_ap_change)
            b1, b2 = st.columns(2)
            b1.button("Suggest minimum Δv", type="primary", on_click=_suggest, disabled=MAN is None,
                      help="Runs the Calculator Phase 3 search for the chosen direction, burn time and target")
            b2.button("Set Δv to 0", on_click=_reset)
            if st.session_state.get("suggest_msg"):
                st.caption(st.session_state.suggest_msg)
            elif st.session_state.get("auto_dv_note"):
                st.caption("Set automatically for this encounter · " + st.session_state.auto_dv_note)
            mass = st.number_input("Spacecraft mass (kg)", 1.0, 10000.0, step=10.0, key="mass")
            isp = st.number_input("Thruster Isp (s)", 50.0, 3000.0, step=5.0, key="isp")
            st.caption("Tip: along-track burns an odd number of half-orbits (0.5, 1.5, 2.5…) before the encounter give the "
                       "most separation per m/s.")
        if MAN is None:
            res.warning("The burn is at or after the closest approach, so it cannot change it. Choose a burn time before it.")
        else:
            man = MAN
            jd_after, miss_after = miss_near(man.positions, e.obj, e.jd)
            pc_after = pc_isotropic(miss_after, sigma, hbr)
            with res:
                if man.jd_burn < START:
                    st.warning("The burn is before the dashboard clock start (it is in the past); use a smaller burn lead.")
                st.caption(f"Burn at **{fmt_utc(man.jd_burn)} UTC**, {fmt_duration((e.jd - man.jd_burn) * SECONDS_PER_DAY)} before TCA")
                k = st.columns(4)
                k[0].metric("Miss distance", f"{miss_after * 1000:.0f} m", f"{(miss_after - e.miss) * 1000:+.0f} m" if man.dv_mag > 0 else None)
                k[1].metric("Risk", risk_level(miss_after, pc_after), f"Pc {e.pc:.1e} → {pc_after:.1e}", delta_color="off")
                k[2].metric("Total Δv", f"{man.dv_mag:.3f} m/s", f"≈ {man.propellant_kg(mass, isp) * 1000:.1f} g propellant", delta_color="off")
                k[3].metric("Semi-major axis change", f"{man.sma_change_m():+.0f} m", f"period {man.period_change_s():+.2f} s", delta_color="off")
                zoom = max(MP['target_km'] * 2.5, miss_after * 1.6, e.miss * 2, 3.0)
                half = max(1.0, 1.2 * zoom / e.vrel) + abs(jd_after - e.jd) * SECONDS_PER_DAY
                s0, d0 = separation_series(SAT.at, e.obj.at, e.jd, half, 301)
                s1, d1 = separation_series(man.positions, e.obj.at, e.jd, half, 301)
                fig = line_chart([s0, s1], [d0, d1], ["Original orbit", "After manoeuvre"], [GOLD, CHARCOAL], "seconds from original TCA",
                                 "separation (km)", hline=(MP["target_km"], "target"), vline=(0, "original TCA"))
                fig.update_yaxes(range=[0, zoom])
                show_chart(fig)

                st.markdown("**Secondary conjunction check** · new or worse close approaches in the 24 h after the burn")
                if man.dv_mag == 0:
                    st.caption("No manoeuvre applied.")
                else:
                    found = run_secondary(D["key"], sat_idx, window_h, thr, sigma, hbr, e.obj.satnum, e.jd, round(LEAD, 6), DV)
                    if found:
                        st.error(f"{len(found)} new or worse close approach(es) within {thr:g} km")
                        show_df(pd.DataFrame([{"Object": f["obj"].name, "TCA (UTC)": fmt_utc(f["jd"]), "Miss (m)": round(f["miss"] * 1000),
                                                    "Before": f"{f['was'] * 1000:.0f} m" if f["was"] else "new"} for f in found]), hide_index=True)
                    else:
                        st.success(f"Clear: no new close approaches within {thr:g} km in the 24 h after the burn.")

        st.markdown("#### Original path and alternate path")
        st.caption(f"Red: {SAT.name} (Object A) on its original orbit · dashed red: the alternate path after the "
                   f"burn · purple: {e.obj.name} (Object B) · ✕ marks the original closest approach. The map starts 5 minutes "
                   "before the burn; press play, or **Go to calculator time**.")
        pair_map(JD_BURN - 5 / MINUTES_PER_DAY)
    nav_footer(3)


# ============================================================ 4. New Path Stats
if PAGE == PAGES[4]:
    man = MAN
    if EV is None or man is None:
        st.info("No manoeuvre to analyse: choose a close approach on Collision Stats and a burn before it on "
                "Alternate Path Mapping (or Calculator Phase 3).")
    else:
        st.caption(f"🔗 Manoeuvre from **Calculator Phase 3**: {float(MP['dv_ms']):.3f} m/s "
                   f"{MP['direction'].split(' (')[0].lower()} at {fmt_utc(man.jd_burn)} UTC, avoiding {EV.obj.name}. "
                   "Change it on Alternate Path Mapping or in Phase 3.")
        a, b = st.columns([1, 2.4])
        with a:
            names = st.multiselect("Ground stations", [s["name"] for s in STATIONS], key="stations")
            st.session_state["mask"] = min(90.0, max(0.0, MASK))
            mask = st.number_input("Elevation mask (°) · = Calculator minimum elevation", 0.0, 90.0, step=1.0, key="mask",
                                   on_change=_set_calc_value, args=("radio", "minimum_elevation_deg", "mask", "cat_elev"))
            st.caption(f"Window: burn ({fmt_utc(man.jd_burn, False)}) to +24 h  \nManoeuvre: {man.dv_mag:.3f} m/s")
            if man.dv_mag == 0:
                st.warning("Δv is 0: set one on Alternate Path Mapping or in Calculator Phase 3 to see a difference.")
        rows, sm = run_passes(*MAN_ARGS, tuple(names), mask)
        with b:
            k = st.columns(5)
            k[0].metric("Passes (24 h)", f"{sm['n_before']} → {sm['n_after']}")
            k[1].metric("Contact time", f"{sm['contact_after_s'] / 60:.1f} min", f"{sm['contact_after_s'] - sm['contact_before_s']:+.1f} s")
            k[2].metric("Max AOS/LOS shift", f"{sm['max_shift_s']:.1f} s")
            k[3].metric("Passes lost", sm["lost"])
            k[4].metric("New passes", sm["gained"])
        if rows:
            df = pd.DataFrame([{
                "Station": x["station"]["name"],
                "AOS before": fmt_utc(x["before"].aos) if x["before"] else "NEW",
                "AOS after": fmt_utc(x["after"].aos) if x["after"] else "LOST",
                "AOS shift (s)": round((x["after"].aos - x["before"].aos) * SECONDS_PER_DAY, 1) if x["before"] and x["after"] else None,
                "LOS shift (s)": round((x["after"].los - x["before"].los) * SECONDS_PER_DAY, 1) if x["before"] and x["after"] else None,
                "Duration before": fmt_duration(x["before"].duration_s) if x["before"] else "—",
                "Duration after": fmt_duration(x["after"].duration_s) if x["after"] else "—",
                "Max el. before (°)": round(x["before"].max_el, 1) if x["before"] else None,
                "Max el. after (°)": round(x["after"].max_el, 1) if x["after"] else None} for x in rows])
            show_df(df, hide_index=True)
            if st.session_state.pass_idx >= len(rows):
                st.session_state.pass_idx = 0
            pi = st.selectbox("Pass to plot", range(len(rows)), key="pass_idx",
                              format_func=lambda i: f"{rows[i]['station']['name']} · {fmt_utc((rows[i]['before'] or rows[i]['after']).aos)}")
            x = rows[pi]
            ref = x["before"] or x["after"]
            jd = np.arange(ref.aos - 3 / MINUTES_PER_DAY, ref.los + 3 / MINUTES_PER_DAY, 10 / SECONDS_PER_DAY)
            tm = (jd - ref.aos) * MINUTES_PER_DAY
            e0 = elevation_series(SAT.at, x["station"], jd)
            e1 = elevation_series(man.positions, x["station"], jd)
            show_chart(line_chart([tm, tm], [e0, e1], ["Original orbit", "After manoeuvre"], [GOLD, CHARCOAL], "minutes from AOS",
                                       "elevation (°)", hline=(mask, "mask"), dashes=[None, "dash"]))
        else:
            st.info(f"No passes above {mask:g}° for the selected stations in this window.")
        st.markdown("#### New path on the map")
        st.caption("Dashed red: the satellite's path after the burn; solid red: the original path. Ground-station "
                   "coverage circles show where passes are gained or lost.")
        pair_map(man.jd_burn - 5 / MINUTES_PER_DAY)
    nav_footer(4)

# ============================================================ 5. Calculator
if PAGE == PAGES[5]:
    import importlib
    section = st.session_state.calc_section
    try:
        mod = importlib.import_module(f"calculator.{CALC_SECTIONS[section]}")
    except ImportError as err:
        st.error(f"The calculator needs extra packages ({err.name}). Stop the app and run:  "
                 "`python -m pip install -r requirements.txt`")
        st.stop()
    mod.render()
    overlay = calc_overlay()
    if overlay:
        st.divider()
        st.markdown("#### Map and globe · calculator objects and trajectories")
        st.caption("Objects A and B, the Phase 3 manoeuvre, Phase 5 scenarios and the closest approach, shown at the "
                   "calculator's time. Object B blinks gold when the closest approach is inside the screening distance. "
                   "The full catalogue is on the Path Tracer page.")
        for note in overlay.get("notes", []):
            st.caption("⚠ " + note)
        cfg_ = st.session_state.cfg
        html = build_path_tracer_html([], [], -1, overlay["startJd"], STATIONS, set(), 5.0,
                                      threshold_km=float(cfg_["simulation"]["screening_distance_km"]),
                                      window_h=float(cfg_["simulation"]["duration_hours"]), calc=overlay)
        components.html(html, height=820, scrolling=True)
    nav_footer(5)

st.divider()
st.caption("Method: SGP4 (Vallado 2006, near-Earth) from TLEs · manoeuvre = impulsive Δv, effect propagated with two-body + J2 RK4 "
           "and added to the SGP4 baseline · Pc from an isotropic 2-D Gaussian with the assumed σ (screening only). "
           + ("Sample dataset objects are synthetic with built-in close approaches." if D["key"] == "sample" else ""))
