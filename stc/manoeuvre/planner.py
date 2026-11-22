"""Hypothetical collision-avoidance manoeuvre.

The manoeuvre is an impulsive delta-v in the satellite's RTN frame applied
`lead_orbits` orbital periods before TCA. Because SGP4 cannot propagate a state
vector, the manoeuvre's *effect* is computed numerically:

    r_manoeuvred(t) = r_SGP4(t) + [ r_num(t; v0 + dv) - r_num(t; v0) ]

Both numerical runs start from the same SGP4 state at burn time, so model
differences cancel and the SGP4 baseline is preserved exactly when dv = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..common.constants import MU, G0, SECONDS_PER_DAY, MINUTES_PER_DAY
from ..common.frames import rtn_basis, rv_to_coe
from ..propagation.numerical import propagate, Trajectory
from ..screening.conjunction import golden_min, altitude_overlap


@dataclass
class Manoeuvre:
    sat: object
    jd_burn: float
    jd_end: float
    dv_rtn: np.ndarray       # m/s
    dv_eci: np.ndarray       # km/s
    r0: np.ndarray
    v0: np.ndarray
    traj: Trajectory         # K=2: [baseline, manoeuvred]

    @property
    def dv_mag(self) -> float:
        return float(np.linalg.norm(self.dv_rtn))

    def positions(self, jd):
        """Manoeuvred (r, v) arrays (N,3) at Julian dates jd."""
        jd = np.atleast_1d(np.asarray(jd, float))
        r, v = self.sat.at(jd)
        after = jd > self.jd_burn
        if np.any(after):
            p, vel = self.traj.sample(jd[after])
            r = r.copy(); v = v.copy()
            r[after] += p[:, 1] - p[:, 0]
            v[after] += vel[:, 1] - vel[:, 0]
        return r, v

    def propellant_kg(self, mass_kg: float, isp_s: float) -> float:
        return mass_kg * (1 - math.exp(-self.dv_mag / (isp_s * G0)))

    def sma_change_m(self) -> float:
        a0 = rv_to_coe(self.r0, self.v0, MU)["a"]
        a1 = rv_to_coe(self.r0, self.v0 + self.dv_eci, MU)["a"]
        return (a1 - a0) * 1000.0

    def period_change_s(self) -> float:
        a0 = rv_to_coe(self.r0, self.v0, MU)["a"]
        a1 = rv_to_coe(self.r0, self.v0 + self.dv_eci, MU)["a"]
        per = lambda a: 2 * math.pi * math.sqrt(a ** 3 / MU)
        return per(a1) - per(a0)


def build_manoeuvre(sat, jd_tca: float, lead_orbits: float, dv_rtn_ms, horizon_jd: float | None = None,
                    dt_s: float = 10.0) -> Manoeuvre:
    dv_rtn = np.asarray(dv_rtn_ms, float)
    jd_burn = jd_tca - lead_orbits * sat.period_min / MINUTES_PER_DAY
    r0, v0 = sat.at1(jd_burn)
    R, T, N = rtn_basis(r0, v0)
    dv_eci = (dv_rtn[0] * R + dv_rtn[1] * T + dv_rtn[2] * N) / 1000.0
    jd_end = max(horizon_jd or 0.0, jd_tca + 15 / MINUTES_PER_DAY)
    traj = propagate(np.stack([r0, r0]), np.stack([v0, v0 + dv_eci]), jd_burn, jd_end, dt_s, dt_s)
    return Manoeuvre(sat, jd_burn, jd_end, dv_rtn, dv_eci, r0, v0, traj)


def miss_near(pos_fn, obj, jd_center: float, half_window_s: float = 480.0) -> tuple[float, float]:
    """Closest approach to obj within +-half_window_s of jd_center. Returns (jd, miss_km)."""
    s = np.arange(-half_window_s, half_window_s + 1e-9, 2.0)
    jd = jd_center + s / SECONDS_PER_DAY
    ra, _ = pos_fn(jd)
    rb, _ = obj.at(jd)
    d = np.linalg.norm(rb - ra, axis=-1)
    k = int(np.nanargmin(d))
    f = lambda x: float(np.linalg.norm(obj.at(np.array([x]))[0][0] - pos_fn(np.array([x]))[0][0]))
    return golden_min(f, jd[k] - 2 / SECONDS_PER_DAY, jd[k] + 2 / SECONDS_PER_DAY, 0.0005 / SECONDS_PER_DAY)


def candidate_dv_values(max_ms: float = 0.5):
    vals, v = [], 0.005
    while v <= max_ms + 1e-9:
        vals.append(round(v, 3))
        v += 0.005 if v < 0.05 else 0.01 if v < 0.2 else 0.025
    return vals


def suggest_minimum_dv(sat, event, lead_orbits: float, target_km: float, max_ms: float = 0.5):
    """Smallest single-axis dv (along-track first, then radial) that reaches target miss distance.
    Returns dict(dv_rtn=[R,T,N] m/s, miss=km) or None."""
    for axis in (1, 0):  # T, then R
        for v in candidate_dv_values(max_ms):
            for sgn in (1, -1):
                dv = [0.0, 0.0, 0.0]
                dv[axis] = sgn * v
                m = build_manoeuvre(sat, event.jd, lead_orbits, dv)
                _, miss = miss_near(m.positions, event.obj, event.jd)
                if miss >= target_km:
                    return {"dv_rtn": dv, "miss": miss}
    return None


def secondary_check(man: Manoeuvre, objects, baseline_events, threshold_km: float, hours: float = 24.0):
    """Close approaches within threshold in the `hours` after the burn that are new or worse than before."""
    from ..screening.conjunction import screen_objects
    jd0, jd1 = man.jd_burn, min(man.jd_end, man.jd_burn + hours / 24)
    cands = [o for o in objects if not o.error and o.satnum != man.sat.satnum and altitude_overlap(man.sat, o, threshold_km + 30)]
    found = []
    for o, jd, miss in screen_objects(man.positions, cands, jd0, jd1, threshold_km):
        was = next((e for e in baseline_events if e.obj is o and abs(e.jd - jd) < 5 / MINUTES_PER_DAY), None)
        if was is not None and miss >= was.miss - 0.05:
            continue
        found.append({"obj": o, "jd": jd, "miss": miss, "was": was.miss if was else None})
    return found
