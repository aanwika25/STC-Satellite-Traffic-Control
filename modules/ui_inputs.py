"""
Shared input widgets for Phase 1 (Automated and Manual modes).
Used by the Home page (app.py). Each widget returns a configuration dictionary.
"""
import streamlit as st

from modules.catalogue import DATA_DIR, list_catalogue_files, load_settings, read_catalogue
from modules.input_handler import validate_tle
from modules.manual_orbit import default_epoch, satrec_from_elements
from modules.tle_loader import load_tle, parse_tle_text


# ---------------------------------------------------------------------------
# Shared input widgets
# ---------------------------------------------------------------------------
def station_and_radio_inputs(defaults, key):
    """Ground station, radio and simulation values, pre-filled from defaults."""
    gs, rd, sim = defaults["ground_station"], defaults["radio"], defaults["simulation"]
    st.subheader("Ground station, radio and simulation")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude (deg)", -90.0, 90.0, float(gs["latitude_deg"]), 0.01, key=f"{key}_lat")
    lon = c2.number_input("Longitude (deg)", -180.0, 180.0, float(gs["longitude_deg"]), 0.01, key=f"{key}_lon")
    alt = c3.number_input("Ground-station altitude (km)", -0.5, 10.0, float(gs["altitude_km"]), 0.01, format="%.3f", key=f"{key}_alt")
    c4, c5, c6, c7 = st.columns(4)
    freq = c4.number_input("Carrier frequency (MHz)", 0.001, value=float(rd["carrier_frequency_mhz"]), step=0.1, key=f"{key}_freq")
    elev = c5.number_input("Minimum elevation (deg)", 0.0, 90.0, float(rd["minimum_elevation_deg"]), 1.0, key=f"{key}_elev")
    dur = c6.number_input("Simulation duration (hours)", 0.1, 720.0, float(sim["duration_hours"]), 1.0, key=f"{key}_dur")
    step = c7.number_input("Time step (seconds)", 0.1, 3600.0, float(sim["time_step_seconds"]), 1.0, key=f"{key}_step")
    screen = st.number_input("Encounter screening distance (km)", 0.1,
                             value=float(sim["screening_distance_km"]), step=1.0, key=f"{key}_screen")
    return {
        "ground_station": {"latitude_deg": lat, "longitude_deg": lon, "altitude_km": alt},
        "radio": {"carrier_frequency_mhz": freq, "minimum_elevation_deg": elev},
        "simulation": {"duration_hours": dur, "time_step_seconds": step, "screening_distance_km": screen},
    }


def tle_object(name, satrec, line1="", line2=""):
    return {"name": name.strip() or f"Object {satrec.satnum}", "catalogue_number": satrec.satnum,
            "satrec": satrec, "line1": line1, "line2": line2}


# ---------------------------------------------------------------------------
# Automated mode: records come from files in data/ or an upload
# ---------------------------------------------------------------------------
def automated_inputs():
    settings = load_settings()
    st.header("1. Automated: satellite data from the catalogue or files")

    source = st.radio("Source", ["Dashboard catalogue (loaded in the sidebar)", "Catalogue file in data/ folder", "Upload a file"],
                      horizontal=True, key="auto_source")
    records, label = [], ""
    if source.startswith("Dashboard catalogue"):
        from calculator.catalogue_source import catalogue_objects
        return catalogue_objects(station_and_radio_inputs, settings)
    if source == "Catalogue file in data/ folder":
        files = list_catalogue_files()
        if not files:
            st.info(f"No .txt or .tle files found in {DATA_DIR}. Add a catalogue there, or upload one instead.")
            return None
        label = st.selectbox("Catalogue file", files, key="auto_file")
        records = read_catalogue(label)
    else:
        up = st.file_uploader("Upload a TLE file", type=["txt", "tle"], key="auto_upload")
        if up is None:
            return None
        records = parse_tle_text(up.getvalue().decode("utf-8-sig"))

    if len(records) < 2:
        st.error("The file must contain at least two complete TLE records.")
        return None

    names = [r[0] for r in records]
    c1, c2 = st.columns(2)
    idx_a = c1.selectbox("Object A", range(len(records)), format_func=lambda i: names[i], key="auto_a")
    idx_b = c2.selectbox("Object B", range(len(records)), index=1, format_func=lambda i: names[i], key="auto_b")
    if idx_a == idx_b:
        st.warning("Choose two different records.")
        return None

    try:
        objs = {}
        for key, idx in (("A", idx_a), ("B", idx_b)):
            name, l1, l2 = records[idx]
            validate_tle(l1, l2)
            objs[key] = load_tle(name, l1, l2)
    except ValueError as exc:
        st.error(str(exc))
        return None

    settings_in = station_and_radio_inputs(settings, key="auto")
    return {"objects": objs, **settings_in, "mode": "Automated"}


# ---------------------------------------------------------------------------
# Manual mode: name objects, paste TLEs or type orbital elements, hypothetical values
# ---------------------------------------------------------------------------
def manual_object(label, key):
    st.subheader(label)
    name = st.text_input("Name", key=f"{key}_name", placeholder="e.g. Debris-7 or Hypothetical sat")
    method = st.radio("How to define it", ["Paste TLE lines", "Enter orbital elements"],
                      horizontal=True, key=f"{key}_method")

    if method == "Paste TLE lines":
        l1 = st.text_input("TLE line 1", key=f"{key}_l1", placeholder="Starts with 1")
        l2 = st.text_input("TLE line 2", key=f"{key}_l2", placeholder="Starts with 2")
        l1, l2 = l1.strip(), l2.strip()
        if not l1 and not l2:
            return None
        try:
            validate_tle(l1, l2)
            return tle_object(name, load_tle(name, l1, l2)["satrec"], l1, l2) | {
                "catalogue_number": int(l1[2:7])}
        except ValueError as exc:
            st.error(f"{label}: {exc}")
            return None

    c1, c2, c3 = st.columns(3)
    alt = c1.number_input("Mean altitude (km)", 150.0, 40000.0, 400.0, 10.0, key=f"{key}_alt")
    ecc = c2.number_input("Eccentricity", 0.0, 0.89, 0.0005, 0.0005, format="%.4f", key=f"{key}_ecc")
    inc = c3.number_input("Inclination (deg)", 0.0, 180.0, 51.6, 0.1, key=f"{key}_inc")
    c4, c5, c6 = st.columns(3)
    raan = c4.number_input("RAAN (deg)", 0.0, 360.0, 0.0, 0.1, key=f"{key}_raan")
    argp = c5.number_input("Argument of perigee (deg)", 0.0, 360.0, 0.0, 0.1, key=f"{key}_argp")
    ma = c6.number_input("Mean anomaly (deg)", 0.0, 360.0, 0.0, 0.1, key=f"{key}_ma")
    try:
        sat = satrec_from_elements(alt, ecc, inc, raan, argp, ma, default_epoch())
    except ValueError as exc:
        st.error(f"{label}: {exc}")
        return None
    return tle_object(name, sat) | {"catalogue_number": "hypothetical"}


def manual_inputs():
    st.header("1. Manual: define objects and values")
    st.caption("Use this to build hypothetical situations. Objects can be pasted from TLEs or entered from orbital elements.")
    col1, col2 = st.columns(2)
    with col1:
        obj_a = manual_object("Object A", "ma")
    with col2:
        obj_b = manual_object("Object B", "mb")
    if obj_a is None or obj_b is None:
        st.info("Define both objects to continue.")
        return None
    settings_in = station_and_radio_inputs(load_settings(), key="man")
    return {"objects": {"A": obj_a, "B": obj_b}, **settings_in, "mode": "Manual"}
