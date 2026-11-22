"""
Shared session state for the multi-page app.

Phase 1 (Home page) stores the validated configuration under 'cfg'. Later phase
pages read it with require_config(). Manoeuvre settings are stored under
'manoeuvre' so that Phases 3, 4 and 5 use the same hypothetical burn.
"""
import streamlit as st

from modules.avoidance import DIRECTIONS

DEFAULT_MANOEUVRE = {
    "burn_min": 60.0,
    "direction": list(DIRECTIONS.keys())[0],
    "dv_ms": 0.0,          # set by the calculator's smallest-delta-v search when a pair is linked
    "target_km": None,     # None = the calculator's screening distance
}


def require_config():
    """Return the loaded configuration, or stop the page with a pointer to Phase 1."""
    cfg = st.session_state.get("cfg")
    if cfg is None:
        st.warning("No configuration is loaded yet. Choose Phase 1 in the Calculator section menu and complete the inputs first.")
        st.stop()
    return cfg


def sidebar_summary(cfg):
    """Short summary of the loaded configuration, shown on every phase page."""
    with st.sidebar:
        st.markdown("**Loaded configuration**")
        st.write(f"Mode: {cfg.get('mode', '')}")
        st.write(f"A: {cfg['objects']['A']['name']}")
        st.write(f"B: {cfg['objects']['B']['name']}")
        gs = cfg["ground_station"]
        st.write(f"Station: {gs['latitude_deg']:.2f}, {gs['longitude_deg']:.2f}, {gs['altitude_km']:.3f} km")
        st.write(f"Carrier: {cfg['radio']['carrier_frequency_mhz']} MHz")


def get_manoeuvre():
    m = {**DEFAULT_MANOEUVRE, **st.session_state.get("manoeuvre", {})}
    if m["target_km"] is None:
        cfg = st.session_state.get("cfg")
        m["target_km"] = float(cfg["simulation"]["screening_distance_km"]) if cfg else 10.0
    return m


def set_manoeuvre(values):
    st.session_state["manoeuvre"] = dict(values)
