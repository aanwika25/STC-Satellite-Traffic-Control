"""Phase 3: Encounter and collision screening, with hypothetical manoeuvres.

Adapted from the STC Phase 1 calculator (pages/). The page body is wrapped in render() so the
dashboard can show it as a section of the Calculator page."""
import plotly.express as px
import streamlit as st
from modules.avoidance import DIRECTIONS, dv_vector, miss_distance, required_delta_v
from modules.collision import close_approach
from modules.orbital_mechanics import epoch_jd, jd_grid
from modules.ui_state import get_manoeuvre, require_config, set_manoeuvre, sidebar_summary


def _find_and_apply(sat_a, sat_b, grid, jd_burn, direction, target_km):
    """Button callback: search the smallest burn and make it the shared Phase 3 delta-v, so Alternate Path
    Mapping, New Path Stats, Phases 4-5 and the map all use it. Runs before the page is drawn, so the
    Delta-v box shows the new value."""
    with st.spinner("Searching burn sizes..."):
        out = required_delta_v(sat_a, sat_b, grid, jd_burn, direction, target_km)
    if out["found"]:
        set_manoeuvre({**get_manoeuvre(), "dv_ms": out["dv_ms"]})
        st.session_state.pop("p3_dv", None)
        msg = (f"Applied: {out['dv_ms']:.3f} m/s '{direction}' raises the closest approach after the burn to "
               f"{out['miss_km']:.3f} km. Alternate Path Mapping, New Path Stats and the map now use this burn.")
    else:
        msg = (f"No burn up to 50 m/s in the '{direction}' direction reaches {target_km:.2f} km in this window, so the "
               "delta-v was not changed. Try another direction or an earlier burn time.")
    st.session_state["p3_find_result"] = (out["found"], msg, out["table"])


def render():
    cfg = require_config()
    sidebar_summary(cfg)

    sat_a = cfg["objects"]["A"]["satrec"]
    sat_b = cfg["objects"]["B"]["satrec"]
    duration_h = cfg["simulation"]["duration_hours"]
    step_s = max(cfg["simulation"]["time_step_seconds"], 10.0)
    screen_km = cfg["simulation"]["screening_distance_km"]
    epoch_a = epoch_jd(sat_a)

    st.title("Calculator · Phase 3: Encounter & Collision Screening")
    st.caption(f"Object A: {cfg['objects']['A']['name']} | Object B: {cfg['objects']['B']['name']}. "
               "Times are minutes after Object A's epoch.")

    st.subheader("1. Closest approach")
    start = st.number_input("Start (min after epoch)", value=0.0, step=1.0, key="p3_start")
    res = close_approach(sat_a, sat_b, jd_grid(sat_a, start, duration_h * 60, step_s), screen_km)
    c = st.columns(4)
    c[0].metric("Closest approach (km)", f"{res['miss_km']:.3f}")
    c[1].metric("Closest approach time (UTC)", res["tca_utc"])
    c[2].metric("Relative speed (km/s)", f"{res['relative_speed_km_s']:.3f}")
    c[3].metric("Within screening distance?", "Yes" if res["within_screening"] else "No")
    prof = res["profile"].copy()
    prof["Minutes after epoch"] = (prof["JD"] - epoch_a) * 1440.0
    st.plotly_chart(px.line(prof, x="Minutes after epoch", y="Separation (km)",
                            title="Separation between the two objects (km)"))

    st.subheader("2. Hypothetical manoeuvre on Object A")
    st.caption("These settings are shared with Phases 4 and 5. RTN frame: R = radial (away from Earth), "
               "T = along-track (direction of motion), N = normal (orbit-normal).")
    cur = get_manoeuvre()
    m1, m2, m3, m4 = st.columns(4)
    burn_min = m1.number_input("Burn time (min after epoch)", 0.0, value=float(cur["burn_min"]), step=10.0, key="p3_burn")
    direction = m2.selectbox("Direction (RTN)", list(DIRECTIONS.keys()),
                             index=list(DIRECTIONS.keys()).index(cur["direction"]), key="p3_dir")
    dv_ms = m3.number_input("Delta-v magnitude (m/s)", 0.0, 500.0, float(cur["dv_ms"]), 0.1, key="p3_dv")
    target_km = m4.number_input("Target miss distance (km)", 0.1, value=float(cur["target_km"]), step=1.0, key="p3_target")
    set_manoeuvre({"burn_min": burn_min, "direction": direction, "dv_ms": dv_ms, "target_km": target_km})

    st.subheader("3. Effect of the manoeuvre")
    focus_win = (st.session_state.get("calc_focus") or {}).get("win_h", 6.0)
    win = st.number_input("Analysis window (hours)", 0.5, float(duration_h), float(min(focus_win, duration_h)), 0.5, key="p3_win")
    st.session_state["calc_p3_win"] = win   # used to draw the manoeuvred path on the map and globe
    if burn_min >= win * 60:
        st.error("The burn time must be earlier than the end of the analysis window.")
    else:
        jd_burn = epoch_a + burn_min / 1440.0
        dv_rtn = dv_vector(direction, dv_ms)
        grid = jd_grid(sat_a, 0.0, win * 60, 30.0)
        before = miss_distance(sat_a, sat_b, grid, jd_burn)
        after = miss_distance(sat_a, sat_b, grid, jd_burn, dv_rtn)
        c = st.columns(3)
        c[0].metric("Closest approach after burn, no manoeuvre (km)", f"{before['miss_km']:.3f}")
        c[1].metric("Closest approach after burn, with manoeuvre (km)", f"{after['miss_km']:.3f}",
                    delta=f"{after['miss_km'] - before['miss_km']:+.3f} km")
        c[2].metric("Manoeuvre size (m/s)", f"{dv_ms:.2f}")
        st.caption(f"The closest approach before the burn is {after['pre_burn_miss_km']:.3f} km. A burn at this time "
                   "cannot change it, so the target applies to encounters after the burn.")

        st.button("Find the smallest delta-v that reaches the target and use it", key="p3_find", on_click=_find_and_apply,
                  args=(sat_a, sat_b, grid, jd_burn, direction, target_km))
        found = st.session_state.get("p3_find_result")
        if found:
            ok, msg, table = found
            (st.success if ok else st.warning)(msg)
            st.dataframe(table, hide_index=True)
