"""Phase 2: Orbital propagation.

Adapted from the STC Phase 1 calculator (pages/). The page body is wrapped in render() so the
dashboard can show it as a section of the Calculator page."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.orbital_mechanics import epoch_jd, jd_grid, orbit_elements, propagate, state_table
from modules.ui_state import require_config, sidebar_summary


def render():
    cfg = require_config()
    sidebar_summary(cfg)

    st.title("Calculator · Phase 2: Orbital Propagation")
    st.caption("SGP4 propagation of each object. Positions are in km and velocities in km/s, "
               "in the TEME frame that SGP4 returns. Times are minutes after each object's own epoch.")

    objects = cfg["objects"]
    choice = st.radio("Object", ["A", "B"], horizontal=True, key="p2_obj",
                      format_func=lambda k: f"Object {k}: {objects[k]['name']}")
    sat = objects[choice]["satrec"]

    st.subheader("Orbit parameters")
    els = orbit_elements(sat)
    m = st.columns(5)
    m[0].metric("Orbital period (min)", f"{els['period_min']:.2f}")
    m[1].metric("Inclination (deg)", f"{els['inclination_deg']:.2f}")
    m[2].metric("Eccentricity", f"{els['eccentricity']:.5f}")
    m[3].metric("Perigee altitude (km)", f"{els['perigee_altitude_km']:.1f}")
    m[4].metric("Apogee altitude (km)", f"{els['apogee_altitude_km']:.1f}")
    st.caption(f"Semi-major axis: {els['semi_major_axis_km']:.1f} km (from the mean motion).")

    st.subheader("State vector over time")
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Start (min after epoch)", value=0.0, step=1.0, key="p2_start")
    duration = c2.number_input("Duration (min)", min_value=1.0, max_value=1440.0, value=95.0, step=1.0, key="p2_dur")
    step_s = c3.number_input("Step (seconds)", min_value=10.0, max_value=600.0, value=60.0, step=10.0, key="p2_step")

    jd = jd_grid(sat, start, duration, step_s)
    r, v = propagate(sat, jd)
    epoch = epoch_jd(sat)
    table = pd.DataFrame({
        "Minutes after epoch": (jd - epoch) * 1440.0,
        "X (km)": r[:, 0], "Y (km)": r[:, 1], "Z (km)": r[:, 2],
        "Vx (km/s)": v[:, 0], "Vy (km/s)": v[:, 1], "Vz (km/s)": v[:, 2],
        "Speed (km/s)": np.linalg.norm(v, axis=1),
        "Altitude (km)": np.linalg.norm(r, axis=1) - 6378.137,
    })
    if table[["X (km)", "Vx (km/s)"]].isna().any().any():
        st.warning("SGP4 reported an error for some times in this window. Those rows are blank.")

    tab_pos, tab_vel, tab_alt, tab_table = st.tabs(["Positions", "Velocities", "Altitude and speed", "Table and ground track"])
    with tab_pos:
        st.plotly_chart(px.line(table, x="Minutes after epoch", y=["X (km)", "Y (km)", "Z (km)"],
                                title="Position components (TEME, km)"))
    with tab_vel:
        st.plotly_chart(px.line(table, x="Minutes after epoch", y=["Vx (km/s)", "Vy (km/s)", "Vz (km/s)"],
                                title="Velocity components (TEME, km/s)"))
    with tab_alt:
        st.plotly_chart(px.line(table, x="Minutes after epoch", y=["Altitude (km)", "Speed (km/s)"],
                                title="Altitude and speed"))
    with tab_table:
        st.dataframe(table, hide_index=True)
        st.dataframe(state_table(sat, jd), hide_index=True)
