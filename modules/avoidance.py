"""
Calculator 3: Collision avoidance
What happens if the satellite changes its velocity? How much should it change?

Method ("delta propagation")
----------------------------
The baseline trajectory is the SGP4 prediction, which includes drag and other
effects that a simple integrator would miss. A burn is modelled by integrating
the same initial state twice, once with the burn and once without, using
two-body gravity plus J2. The difference between the two integrations is then
added to the SGP4 trajectory. This keeps SGP4 accuracy while isolating the
effect of the burn.

The burn is given in the RTN frame:
  R = radial (away from Earth), T = along-track (direction of motion),
  N = normal (orbit-normal).
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from modules.geometry import J2, MU_KM3_S2, RE_KM
from modules.orbital_mechanics import propagate

DIRECTIONS = {
    "Prograde (along-track +)": ("T", 1.0),
    "Retrograde (along-track -)": ("T", -1.0),
    "Radial out (+)": ("R", 1.0),
    "Radial in (-)": ("R", -1.0),
    "Normal (+)": ("N", 1.0),
    "Normal (-)": ("N", -1.0),
}
_AXIS = {"R": 0, "T": 1, "N": 2}


def dv_vector(direction, dv_ms):
    """RTN delta-v vector in km/s for a direction key and magnitude in m/s."""
    axis, sign = DIRECTIONS[direction]
    vec = np.zeros(3)
    vec[_AXIS[axis]] = sign * dv_ms / 1000.0
    return vec


def _accel(r):
    """Two-body gravity plus J2 acceleration (km/s^2)."""
    x, y, z = r
    rn = np.linalg.norm(r)
    fac = -MU_KM3_S2 / rn ** 3
    j = 1.5 * J2 * (RE_KM / rn) ** 2
    z2 = (z / rn) ** 2
    return np.array([
        fac * x * (1.0 + j * (1.0 - 5.0 * z2)),
        fac * y * (1.0 + j * (1.0 - 5.0 * z2)),
        fac * z * (1.0 + j * (3.0 - 5.0 * z2)),
    ])


def _rhs(_t, s):
    return np.concatenate([s[3:], _accel(s[:3])])


def rtn_to_eci(r, v, dv_rtn):
    """Convert an RTN delta-v into the inertial frame."""
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return dv_rtn[0] * radial + dv_rtn[1] * along + dv_rtn[2] * normal


def _integrate(r0, v0, t_end_s):
    sol = solve_ivp(_rhs, (0.0, t_end_s), np.concatenate([r0, v0]),
                    method="DOP853", rtol=1e-9, atol=1e-9, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Orbit integration failed: {sol.message}")
    return sol


def burn_trajectory(sat, jd_burn, dv_rtn, jd_grid):
    """
    Position and velocity of Object A on jd_grid, with a burn at jd_burn.
    Times before the burn are the unchanged SGP4 prediction.
    """
    jd_grid = np.asarray(jd_grid, dtype=float)
    r_sgp, v_sgp = propagate(sat, jd_grid)
    t_samp = (jd_grid - jd_burn) * 86400.0
    after = t_samp >= -1e-6
    if not np.any(t_samp > 1.0):
        raise ValueError("The manoeuvre must happen before the end of the analysis window.")

    r0, v0 = propagate(sat, [jd_burn])
    r0, v0 = r0[0], v0[0]
    if not np.all(np.isfinite(r0)):
        raise ValueError("SGP4 could not propagate Object A at the burn time.")

    dv = rtn_to_eci(r0, v0, np.asarray(dv_rtn, dtype=float))
    ts = np.clip(t_samp[after], 0.0, None)
    t_end = ts.max()
    sol_burn = _integrate(r0, v0 + dv, t_end)
    sol_none = _integrate(r0, v0, t_end)
    yb = sol_burn.sol(ts)
    yn = sol_none.sol(ts)

    r_out, v_out = r_sgp.copy(), v_sgp.copy()
    r_out[after] += (yb[:3] - yn[:3]).T
    v_out[after] += (yb[3:] - yn[3:]).T
    return r_out, v_out


def miss_distance(sat_a, sat_b, jd_grid, jd_burn=None, dv_rtn=None):
    """
    Closest approach between A and B.

    If jd_burn is given, 'miss_km' is the closest approach AFTER the burn, because
    a burn cannot change an encounter that has already happened. The closest
    approach before the burn is returned separately as 'pre_burn_miss_km'.
    """
    jd_grid = np.asarray(jd_grid, dtype=float)
    if dv_rtn is None:
        ra, _ = propagate(sat_a, jd_grid)
    else:
        ra, _ = burn_trajectory(sat_a, jd_burn, dv_rtn, jd_grid)
    rb, _ = propagate(sat_b, jd_grid)
    d = np.linalg.norm(ra - rb, axis=1)

    if jd_burn is None:
        after = np.ones(len(jd_grid), dtype=bool)
    else:
        after = jd_grid > jd_burn + 1e-9   # strictly after the burn: the burn cannot affect its own instant
    if not np.any(after) or not np.any(np.isfinite(d[after])):
        raise ValueError("No valid positions after the burn time.")

    idx_after = np.where(after)[0]
    i = int(idx_after[np.nanargmin(d[after])])
    pre = d[~after]
    pre_min = float(np.nanmin(pre)) if pre.size and np.any(np.isfinite(pre)) else float("nan")
    tca, miss = _refine_after(sat_a, sat_b, jd_grid, d, after, jd_burn, dv_rtn)
    return {
        "miss_km": miss,
        "tca_jd": tca,
        "pre_burn_miss_km": pre_min,
        "profile_km": d,
    }


def _refine_after(sat_a, sat_b, jd_grid, d, after, jd_burn, dv_rtn, n_cand=3):
    """Refine the closest approach after the burn below the grid spacing.

    At ~10 km/s relative speed a 30 s grid can miss the true minimum by many km, so the three
    most promising local minima are resampled twice (about 0.25 s, then about 2 ms)."""
    dd = np.where(after & np.isfinite(d), d, np.inf)
    is_min = np.r_[True, dd[1:] <= dd[:-1]] & np.r_[dd[:-1] <= dd[1:], True] & np.isfinite(dd)
    cand = np.flatnonzero(is_min)
    cand = cand[np.argsort(dd[cand])][:n_cand]
    best_i = int(np.argmin(dd))
    best = (float(jd_grid[best_i]), float(dd[best_i]))
    spans = [(jd_grid[max(k - 1, 0)], jd_grid[min(k + 1, len(jd_grid) - 1)]) for k in cand]
    for _ in range(2):
        fine = []
        for lo, hi in spans:
            if jd_burn is not None:
                lo = max(lo, jd_burn + 1e-8)
            if hi > lo:
                fine.append(np.linspace(lo, hi, 241))
        if not fine:
            break
        g = np.concatenate(fine)
        ra = propagate(sat_a, g)[0] if dv_rtn is None else burn_trajectory(sat_a, jd_burn, dv_rtn, g)[0]
        dist = np.linalg.norm(ra - propagate(sat_b, g)[0], axis=1)
        spans = []
        for seg in range(len(fine)):
            ds = dist[seg * 241:(seg + 1) * 241]
            if not np.any(np.isfinite(ds)):
                continue
            k = int(np.nanargmin(ds))
            gs = fine[seg]
            if ds[k] < best[1]:
                best = (float(gs[k]), float(ds[k]))
            spans.append((gs[max(k - 1, 0)], gs[min(k + 1, 240)]))
    return best


SCAN_DV_MS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def required_delta_v(sat_a, sat_b, jd_grid, jd_burn, direction, target_km,
                     max_dv_ms=50.0, tol_ms=0.005):
    """
    Smallest burn magnitude (m/s) in one RTN direction that lifts the miss
    distance to target_km.

    Magnitudes are scanned on a roughly logarithmic ladder (0.01 ... 50 m/s); once one
    reaches the target, bisection between it and the previous step narrows the answer
    to within tol_ms.
    """
    axis, sign = DIRECTIONS[direction]
    unit = np.zeros(3)
    unit[_AXIS[axis]] = sign
    rows = []

    def miss(mag):
        res = miss_distance(sat_a, sat_b, jd_grid, jd_burn, unit * mag / 1000.0)
        rows.append({"Delta-v (m/s)": round(float(mag), 4), "Miss distance (km)": round(res["miss_km"], 4)})
        return res["miss_km"]

    lo = 0.0
    for mag in [m for m in SCAN_DV_MS if m <= max_dv_ms]:
        m_hi = miss(mag)
        if m_hi >= target_km:
            hi = mag
            while hi - lo > tol_ms:
                mid = 0.5 * (lo + hi)
                m_mid = miss(mid)
                if m_mid >= target_km:
                    hi, m_hi = mid, m_mid
                else:
                    lo = mid
            table = pd.DataFrame(rows).sort_values("Delta-v (m/s)").reset_index(drop=True)
            return {"found": True, "dv_ms": float(np.ceil(hi * 1000) / 1000), "miss_km": m_hi, "table": table}
        lo = mag
    return {"found": False, "table": pd.DataFrame(rows)}
