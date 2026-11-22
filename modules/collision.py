"""
Calculator 2: Collision and close-approach
How close will two satellites (or a satellite and debris) come? When will they be closest?
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from modules.geometry import utc_string
from modules.orbital_mechanics import propagate


def _separation(jd, sat_a, sat_b):
    ra, _ = propagate(sat_a, [jd])
    rb, _ = propagate(sat_b, [jd])
    return float(np.linalg.norm(ra[0] - rb[0]))


def close_approach(sat_a, sat_b, jd, screening_km):
    """
    Find the time of closest approach (TCA) and miss distance.

    A coarse search over the grid `jd` finds the closest sample, then a bounded
    scalar optimiser refines the time between its neighbours.
    """
    jd = np.asarray(jd, dtype=float)
    ra, va = propagate(sat_a, jd)
    rb, vb = propagate(sat_b, jd)
    dist = np.linalg.norm(ra - rb, axis=1)
    if not np.any(np.isfinite(dist)):
        raise ValueError("SGP4 could not propagate either object over this window.")

    # The closest grid sample is not always on the closest pass: at ~10 km/s relative speed a coarse grid
    # can miss the true minimum by tens of km. Refine the most promising local minima and keep the best.
    d = np.where(np.isfinite(dist), dist, np.inf)
    i = int(np.argmin(d))
    tca, miss = jd[i], dist[i]
    is_min = np.r_[True, d[1:] <= d[:-1]] & np.r_[d[:-1] <= d[1:], True]
    cand = np.flatnonzero(is_min & np.isfinite(d))
    for k in cand[np.argsort(d[cand])][:12]:
        lo, hi = jd[max(k - 1, 0)], jd[min(k + 1, len(jd) - 1)]
        if hi <= lo:
            continue
        # search in seconds from `lo`: scipy's relative tolerance on a Julian date (~2.46e6) is ~50 minutes
        res = minimize_scalar(lambda t: _separation(lo + t / 86400.0, sat_a, sat_b), bounds=(0.0, (hi - lo) * 86400.0),
                              method="bounded", options={"xatol": 1e-4})
        if np.isfinite(res.fun) and res.fun < miss:
            tca, miss = lo + res.x / 86400.0, res.fun

    _, va_t = propagate(sat_a, [tca])
    _, vb_t = propagate(sat_b, [tca])
    return {
        "tca_jd": float(tca),
        "tca_utc": utc_string(tca),
        "miss_km": float(miss),
        "relative_speed_km_s": float(np.linalg.norm(va_t[0] - vb_t[0])),
        "within_screening": bool(miss <= screening_km),
        "profile": pd.DataFrame({"JD": jd, "Separation (km)": dist}),
    }
