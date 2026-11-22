"""
Calculator 5: Communication impact
How does the hypothetical manoeuvre change communication visibility, Doppler shift
and receiver frequency correction?

Frequency correction model
--------------------------
The ground station pre-compensates its transmitter using the baseline (no-burn)
prediction. If the satellite then follows the manoeuvred path, the receiver sees
a residual error. That residual is what the receiver must still correct.
"""
import numpy as np
import pandas as pd

from modules.avoidance import burn_trajectory
from modules.doppler import doppler_shift_hz, precompensated_tx_mhz
from modules.geometry import C_KM_S, look_angles
from modules.orbital_mechanics import propagate


def _count_passes(visible):
    edges = np.diff(np.concatenate([[0], visible.astype(int)]))
    return int(np.sum(edges == 1))


def _peak_abs(values, mask):
    return float(np.nanmax(np.abs(values[mask]))) if np.any(mask) else float("nan")


def link_analysis(sat, station, carrier_mhz, min_el_deg, jd, step_s,
                  jd_burn=None, dv_rtn=None):
    jd = np.asarray(jd, dtype=float)

    # Baseline: no manoeuvre
    r0, v0 = propagate(sat, jd)
    el0, _, _, rr0 = look_angles(station, jd, r0, v0)

    # With the hypothetical manoeuvre
    if dv_rtn is None:
        r1, v1 = r0, v0
    else:
        r1, v1 = burn_trajectory(sat, jd_burn, dv_rtn, jd)
    el1, _, _, rr1 = look_angles(station, jd, r1, v1)

    vis0 = el0 >= min_el_deg
    vis1 = el1 >= min_el_deg

    shift0 = doppler_shift_hz(carrier_mhz, rr0)
    shift1 = doppler_shift_hz(carrier_mhz, rr1)

    # Transmitter pre-compensated from the baseline prediction, then flown on the new path
    f_tx = precompensated_tx_mhz(carrier_mhz, rr0)
    received = f_tx * 1e6 * (1.0 - rr1 / C_KM_S)
    residual = received - carrier_mhz * 1e6

    timeline = pd.DataFrame({
        "JD": jd,
        "Elevation baseline (deg)": el0,
        "Elevation with manoeuvre (deg)": el1,
        "Doppler baseline (Hz)": shift0,
        "Doppler with manoeuvre (Hz)": shift1,
        "Residual after pre-correction (Hz)": residual,
        "Visible baseline": vis0,
        "Visible with manoeuvre": vis1,
    })

    summary = pd.DataFrame([
        {"Metric": "Visible time (min)",
         "Baseline (no manoeuvre)": vis0.sum() * step_s / 60.0,
         "With manoeuvre": vis1.sum() * step_s / 60.0},
        {"Metric": "Number of passes",
         "Baseline (no manoeuvre)": _count_passes(vis0),
         "With manoeuvre": _count_passes(vis1)},
        {"Metric": "Peak |Doppler| without correction (kHz)",
         "Baseline (no manoeuvre)": _peak_abs(shift0 / 1e3, vis0),
         "With manoeuvre": _peak_abs(shift1 / 1e3, vis1)},
        {"Metric": "Peak residual after pre-correction (Hz)",
         "Baseline (no manoeuvre)": 0.0,
         "With manoeuvre": _peak_abs(residual, vis1)},
    ])
    return timeline, summary
