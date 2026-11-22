"""
Calculator 1: Orbital mechanics
Where is the satellite? How fast is it moving? Where will it be in the next few minutes?

All times are Julian dates (UT). Use epoch_jd() and jd_grid() to build them.
"""
import numpy as np
import pandas as pd

from modules.geometry import MU_KM3_S2, RE_KM, subpoint, utc_string


def epoch_jd(satrec):
    """TLE epoch as a single Julian date."""
    return satrec.jdsatepoch + satrec.jdsatepochF


def jd_grid(ref_satrec, start_min, duration_min, step_s):
    """Julian-date grid starting start_min after the reference TLE epoch."""
    start = epoch_jd(ref_satrec) + start_min / 1440.0
    n = int(np.floor(duration_min * 60.0 / step_s + 1e-9)) + 1
    return start + np.arange(n) * step_s / 86400.0


def propagate(satrec, jd):
    """
    SGP4 propagation to absolute Julian dates.
    Returns position (km) and velocity (km/s) in TEME as (N, 3) arrays.
    Rows where SGP4 reports an error are NaN.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    r = np.full((len(jd), 3), np.nan)
    v = np.full((len(jd), 3), np.nan)
    for i, t in enumerate(jd):
        err, ri, vi = satrec.sgp4(satrec.jdsatepoch, t - satrec.jdsatepoch)
        if err == 0:
            r[i], v[i] = ri, vi
    return r, v


def orbit_elements(satrec):
    """Summary orbit description derived from the TLE mean elements."""
    n_rad_s = satrec.no_kozai / 60.0
    a = (MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)
    e = satrec.ecco
    return {
        "period_min": 2.0 * np.pi / satrec.no_kozai,
        "inclination_deg": float(np.degrees(satrec.inclo)),
        "eccentricity": float(e),
        "semi_major_axis_km": float(a),
        "perigee_altitude_km": float(a * (1.0 - e) - RE_KM),
        "apogee_altitude_km": float(a * (1.0 + e) - RE_KM),
    }


def state_table(satrec, jd):
    """Position, altitude and speed at each time, as a table."""
    r, v = propagate(satrec, jd)
    lat, lon = subpoint(jd, r)
    return pd.DataFrame({
        "UTC": [utc_string(t) for t in jd],
        "Latitude (deg)": lat,
        "Longitude (deg)": lon,
        "Altitude (km)": np.linalg.norm(r, axis=1) - RE_KM,
        "Speed (km/s)": np.linalg.norm(v, axis=1),
    })
