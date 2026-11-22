"""Phase 1: Input and configuration (two objects, ground station, radio, simulation).

Adapted from the STC Phase 1 calculator (pages/). The page body is wrapped in render() so the
dashboard can show it as a section of the Calculator page."""
import streamlit as st
from modules.input_handler import validate_simulation, validate_station
from modules.ui_inputs import automated_inputs, manual_inputs


def render():
    st.title("Calculator · Phase 1: Input & Configuration")
    st.write(
        "Choose a mode, define two orbital objects and the ground-station, radio and simulation values. "
        "When the configuration validates, Phases 2 to 5 work in the Calculator section menu in the sidebar, "
        "and objects A and B appear on the map and globe."
    )

    mode = st.radio("Mode", ["Automated (from files)", "Manual (hypothetical scenarios)"],
                    horizontal=True, key="mode")

    cfg = None
    try:
        cfg = automated_inputs() if mode.startswith("Automated") else manual_inputs()
    except ValueError as exc:
        st.error(str(exc))

    st.divider()
    if cfg is None:
        st.session_state["cfg"] = None
        st.info("Complete the inputs above to continue to Phases 2 to 5.")
    else:
        try:
            validate_station(cfg["ground_station"]["latitude_deg"], cfg["ground_station"]["longitude_deg"],
                             cfg["ground_station"]["altitude_km"])
            validate_simulation(cfg["simulation"]["duration_hours"], cfg["simulation"]["time_step_seconds"],
                                cfg["radio"]["carrier_frequency_mhz"], cfg["radio"]["minimum_elevation_deg"],
                                cfg["simulation"]["screening_distance_km"])
            st.session_state["cfg"] = cfg
            st.success("Phase 1 configuration validated. Open Phases 2 to 5 from the Calculator section menu in the sidebar.")
            a, b = cfg["objects"]["A"], cfg["objects"]["B"]
            st.write(f"**Object A:** {a['name']} (catalogue {a['catalogue_number']})")
            st.write(f"**Object B:** {b['name']} (catalogue {b['catalogue_number']})")
        except ValueError as exc:
            st.session_state["cfg"] = None
            st.error(str(exc))
