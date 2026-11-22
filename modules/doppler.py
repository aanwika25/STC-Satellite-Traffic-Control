"""
Calculator 4: Doppler shift
How much will the satellite's movement change the frequency received by a ground station?

Sign convention: range rate > 0 means the satellite is moving away, which lowers
the received frequency. Non-relativistic formulas are used, which is accurate to
far better than the precision needed for LEO links.
"""
import numpy as np
import pandas as pd

from modules.geometry import C_KM_S, look_angles
from modules.orbital_mechanics import propagate


def doppler_shift_hz(carrier_mhz, range_rate_km_s):
    """Received frequency minus carrier. Positive when the satellite approaches."""
    return -carrier_mhz * 1e6 * np.asarray(range_rate_km_s, dtype=float) / C_KM_S


def precompensated_tx_mhz(target_mhz, range_rate_km_s):
    """Transmit frequency that makes the receiver see target_mhz for this range rate."""
    return target_mhz / (1.0 - np.asarray(range_rate_km_s, dtype=float) / C_KM_S)


def doppler_profile(sat, station, carrier_mhz, min_el_deg, jd):
    r, v = propagate(sat, jd)
    el, az, rng, rr = look_angles(station, jd, r, v)
    shift = doppler_shift_hz(carrier_mhz, rr)
    visible = el >= min_el_deg
    df = pd.DataFrame({
        "JD": jd,
        "Elevation (deg)": el,
        "Azimuth (deg)": az,
        "Range (km)": rng,
        "Range rate (km/s)": rr,
        "Doppler shift (Hz)": shift,
        "Visible": visible,
    })
    if np.any(visible):
        summary = {
            "peak_approach_hz": float(np.nanmax(shift[visible])),
            "peak_recession_hz": float(np.nanmin(shift[visible])),
        }
    else:
        summary = {"peak_approach_hz": float("nan"), "peak_recession_hz": float("nan")}
    return df, summary
