"""Phase 5: Scenario comparison.

Adapted from the STC Phase 1 calculator (pages/). The page body is wrapped in render() so the
dashboard can show it as a section of the Calculator page."""
import pandas as pd
import streamlit as st
from modules.avoidance import DIRECTIONS, dv_vector, miss_distance
from modules.communication import link_analysis
from modules.orbital_mechanics import epoch_jd, jd_grid
from modules.ui_state import require_config, sidebar_summary


def render():
    cfg = require_config()
    sidebar_summary(cfg)

    sat_a = cfg["objects"]["A"]["satrec"]
    sat_b = cfg["objects"]["B"]["satrec"]
    station = cfg["ground_station"]
    carrier = cfg["radio"]["carrier_frequency_mhz"]
    min_el = cfg["radio"]["minimum_elevation_deg"]
    duration_h = cfg["simulation"]["duration_hours"]
    step_s = max(cfg["simulation"]["time_step_seconds"], 10.0)
    epoch_a = epoch_jd(sat_a)

    st.title("Calculator · Phase 5: Scenario Comparison")
    st.caption("Compare up to three hypothetical manoeuvres on Object A. Each scenario gets its own name, burn time, "
               "direction and size. Results show collision separation and communication effects together.")

    win = st.number_input("Analysis window (hours)", 0.5, float(duration_h), float(min(float((st.session_state.get("calc_focus") or {}).get("duration_h", 6.0)), duration_h)), 0.5, key="p5_win")
    defaults = [
        ("Scenario 1", 60.0, "Prograde (along-track +)", 0.5),
        ("Scenario 2", 60.0, "Prograde (along-track +)", 1.0),
        ("Scenario 3", 60.0, "Radial out (+)", 0.5),
    ]
    names = list(DIRECTIONS.keys())
    cols = st.columns(3)
    scenarios = []
    for i, (col, (name, burn, direction, dv)) in enumerate(zip(cols, defaults)):
        with col:
            st.markdown(f"**{name}**")
            label = col.text_input("Name", value=name, key=f"p5_name_{i}")
            b = col.number_input("Burn time (min after epoch)", 0.0, value=float((st.session_state.get("calc_focus") or {}).get("burn_min", burn)),
                                step=10.0, key=f"p5_burn_{i}")
            d = col.selectbox("Direction (RTN)", names, index=names.index(direction), key=f"p5_dir_{i}")
            v = col.number_input("Delta-v (m/s)", 0.0, 500.0, dv, 0.1, key=f"p5_dv_{i}")
            scenarios.append({"name": label, "burn_min": b, "direction": d, "dv_ms": v})

    use = st.multiselect("Scenarios to compare", [s["name"] for s in scenarios],
                         default=[s["name"] for s in scenarios], key="p5_use")
    chosen = [s for s in scenarios if s["name"] in use]
    st.session_state["calc_scenarios"] = {"win_h": win, "list": chosen}   # drawn on the map and globe

    if not chosen:
        st.info("Select at least one scenario.")
    elif any(s["burn_min"] >= win * 60 for s in chosen):
        st.error("Every burn time must be earlier than the analysis window.")
    else:
        grid = jd_grid(sat_a, 0.0, win * 60, 30.0)
        jd_grid_comm = jd_grid(sat_a, 0.0, win * 60, step_s)
        rows = []
        for s in chosen:
            jd_burn = epoch_a + s["burn_min"] / 1440.0
            dv_rtn = dv_vector(s["direction"], s["dv_ms"])
            miss = miss_distance(sat_a, sat_b, grid, jd_burn, dv_rtn)
            _, comm = link_analysis(sat_a, station, carrier, min_el, jd_grid_comm, step_s, jd_burn, dv_rtn)
            vals = dict(zip(comm["Metric"], comm["With manoeuvre"]))
            rows.append({
                "Scenario": s["name"],
                "Burn (min)": s["burn_min"],
                "Direction": s["direction"],
                "Delta-v (m/s)": s["dv_ms"],
                "Closest approach after burn (km)": round(miss["miss_km"], 3),
                "Visible time (min)": round(vals["Visible time (min)"], 1),
                "Number of passes": int(vals["Number of passes"]),
                "Peak |Doppler| (kHz)": round(vals["Peak |Doppler| without correction (kHz)"], 3),
                "Residual after pre-correction (Hz)": round(vals["Peak residual after pre-correction (Hz)"], 3),
            })
        st.subheader("Comparison")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Closest approach is measured after each burn. Residual is the frequency error the receiver must still "
                   "correct, after the ground station pre-corrects using the no-manoeuvre prediction.")

    st.divider()
    st.subheader("Not built yet")
    st.write("The following Phase 5 items from the plan are not implemented in this version:")
    st.markdown("- Hohmann transfer calculations\n- Plane change calculations\n- Traffic analysis across many objects\n- Report export")
