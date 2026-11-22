"""Phase 4: Doppler shift, elevation/azimuth, visibility and communication windows.

Adapted from the STC Phase 1 calculator (pages/). The page body is wrapped in render() so the
dashboard can show it as a section of the Calculator page."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.avoidance import dv_vector
from modules.communication import link_analysis
from modules.doppler import doppler_profile, precompensated_tx_mhz
from modules.geometry import look_angles
from modules.orbital_mechanics import epoch_jd, jd_grid, propagate
from modules.ui_state import get_manoeuvre, require_config, sidebar_summary


def render():
    cfg = require_config()
    sidebar_summary(cfg)

    objects = cfg["objects"]
    station = cfg["ground_station"]
    carrier = cfg["radio"]["carrier_frequency_mhz"]
    min_el = cfg["radio"]["minimum_elevation_deg"]
    duration_h = cfg["simulation"]["duration_hours"]
    step_s = max(cfg["simulation"]["time_step_seconds"], 10.0)
    sat_a = objects["A"]["satrec"]
    epoch_a = epoch_jd(sat_a)

    st.title("Calculator · Phase 4: Doppler & Ground Station")
    st.caption(f"Ground station: {station['latitude_deg']:.3f}, {station['longitude_deg']:.3f}, "
               f"{station['altitude_km']:.3f} km. Carrier {carrier} MHz, minimum elevation {min_el} deg.")

    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Start (min after epoch)", value=0.0, step=1.0, key="p4_start")
    window = c2.number_input("Window (min)", 10.0, duration_h * 60, min(180.0, duration_h * 60), 10.0, key="p4_win")
    which = c3.radio("Object", ["A", "B"], horizontal=True, key="p4_obj")
    sat = objects[which]["satrec"]
    jd = jd_grid(sat, start, window, step_s)

    st.subheader("1. Elevation, azimuth and visibility")
    r, v = propagate(sat, jd)
    el, az, rng, rr = look_angles(station, jd, r, v)
    visible = el >= min_el
    passes = int(np.sum(np.diff(np.concatenate([[0], visible.astype(int)])) == 1))
    m = st.columns(4)
    m[0].metric("Visible time (min)", f"{visible.sum() * step_s / 60:.1f}")
    m[1].metric("Passes in window", passes)
    m[2].metric("Max elevation (deg)", f"{np.nanmax(el):.1f}")
    m[3].metric("Range at closest visible point (km)",
                f"{np.nanmin(rng[visible]):.0f}" if visible.any() else "n/a")
    geo = pd.DataFrame({"Minutes after epoch": (jd - epoch_a) * 1440.0, "Elevation (deg)": el,
                        "Azimuth (deg)": az, "Range (km)": rng, "Range rate (km/s)": rr, "Visible": visible})
    st.plotly_chart(px.line(geo, x="Minutes after epoch", y=["Elevation (deg)"],
                            title=f"Elevation of Object {which} (minimum {min_el} deg)"))
    st.plotly_chart(px.line(geo, x="Minutes after epoch", y=["Azimuth (deg)"], title=f"Azimuth of Object {which}"))

    st.subheader("2. Doppler shift and received frequency")
    df, summ = doppler_profile(sat, station, carrier, min_el, jd)
    c = st.columns(3)
    c[0].metric("Peak increase when approaching (kHz)",
                f"{summ['peak_approach_hz'] / 1e3:+.3f}" if np.isfinite(summ["peak_approach_hz"]) else "n/a")
    c[1].metric("Peak decrease when receding (kHz)",
                f"{summ['peak_recession_hz'] / 1e3:+.3f}" if np.isfinite(summ["peak_recession_hz"]) else "n/a")
    if visible.any():
        best = int(np.nanargmax(np.where(visible, el, -np.inf)))   # highest point of the pass
        tx_mhz = precompensated_tx_mhz(carrier, rr[best])
        c[2].metric("Transmit pre-correction at highest point (MHz)", f"{tx_mhz:.6f}",
                    delta=f"{(tx_mhz - carrier) * 1e3:+.3f} kHz")
    else:
        c[2].metric("Transmit pre-correction at highest point (MHz)", "n/a")
    df["Minutes after epoch"] = (df["JD"] - epoch_a) * 1440.0
    st.plotly_chart(px.line(df, x="Minutes after epoch", y="Doppler shift (Hz)",
                            title=f"Doppler shift at {carrier} MHz (positive = approaching)"))
    st.caption("Samples below the minimum elevation are shown for reference only.")

    st.subheader("3. Communication windows with the hypothetical manoeuvre")
    mv = get_manoeuvre()
    st.caption(f"Uses the burn set in Phase 3: {mv['dv_ms']:.2f} m/s, {mv['direction']}, at {mv['burn_min']:.0f} min. "
               "The ground station pre-corrects its transmitter using the no-manoeuvre prediction.")
    win_h = st.number_input("Analysis window (hours)", 0.5, float(duration_h), float(min(float((st.session_state.get("calc_focus") or {}).get("duration_h", 6.0)), duration_h)), 0.5, key="p4_mwin")
    if mv["burn_min"] >= win_h * 60:
        st.error("The burn time (set in Phase 3) must be earlier than this analysis window.")
    else:
        jd_burn = epoch_a + mv["burn_min"] / 1440.0
        jd5 = jd_grid(sat_a, 0.0, win_h * 60, step_s)
        tl, comm = link_analysis(sat_a, station, carrier, min_el, jd5, step_s, jd_burn,
                                 dv_vector(mv["direction"], mv["dv_ms"]))
        st.dataframe(comm, hide_index=True)
        tl["Minutes after epoch"] = (tl["JD"] - epoch_a) * 1440.0
        st.plotly_chart(px.line(tl, x="Minutes after epoch", y=["Doppler baseline (Hz)", "Doppler with manoeuvre (Hz)"],
                                title="Doppler shift, baseline vs manoeuvre (Object A)"))
        st.plotly_chart(px.line(tl, x="Minutes after epoch", y="Residual after pre-correction (Hz)",
                                title="Frequency error the receiver must still correct (Hz)"))
