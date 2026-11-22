"""Close-approach (conjunction) screening.

Method
1. Apogee/perigee filter removes objects whose altitude band cannot overlap the satellite's.
2. Both objects are propagated on a coarse time grid (default 60 s, vectorised).
3. Local minima of distance are found where the range-rate changes sign (- to +).
4. Each minimum is refined: dense 0.5 s sampling, then golden-section search to 0.5 ms (a few metres at 15 km/s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..common.constants import SECONDS_PER_DAY
from ..common.frames import to_rtn
from ..risk.probability_of_collision import pc_isotropic, risk_level

PosFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]  # jd array -> (r, v) arrays (N,3)


@dataclass
class Conjunction:
    obj: object
    jd: float              # time of closest approach (TCA)
    miss: float            # km
    vrel: float            # km/s
    rel_pos: np.ndarray    # object minus satellite, km (TEME)
    r_sat: np.ndarray
    v_sat: np.ndarray
    v_obj: np.ndarray
    pc: float = 0.0
    risk: str = "LOW"
    rtn_miss: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angle_deg: float = 0.0

    def score(self, sigma_m: float, hbr_m: float):
        self.pc = pc_isotropic(self.miss, sigma_m, hbr_m)
        self.risk = risk_level(self.miss, self.pc)
        return self


def golden_min(f: Callable[[float], float], a: float, b: float, tol: float) -> tuple[float, float]:
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def _dist(pa: PosFn, pb: PosFn, jd):
    ra, _ = pa(jd)
    rb, _ = pb(jd)
    return np.linalg.norm(rb - ra, axis=-1)


def refine_minimum(pa: PosFn, pb: PosFn, jd_lo: float, jd_hi: float) -> tuple[float, float]:
    """Return (jd_tca, miss_km) for the minimum of separation inside [jd_lo, jd_hi]."""
    n = max(3, int((jd_hi - jd_lo) * SECONDS_PER_DAY / 0.5) + 1)
    grid = np.linspace(jd_lo, jd_hi, n)
    d = _dist(pa, pb, grid)
    if np.all(np.isnan(d)):
        return jd_lo, float("nan")
    k = int(np.nanargmin(d))
    lo, hi = grid[max(0, k - 1)], grid[min(n - 1, k + 1)]
    f = lambda jd: float(_dist(pa, pb, np.array([jd]))[0])
    return golden_min(f, lo, hi, 0.0005 / SECONDS_PER_DAY)


def screen_pair(pa: PosFn, pb: PosFn, jd0: float, jd1: float, step_s: float, threshold_km: float,
                grid: np.ndarray | None = None, sat_states=None) -> list[tuple[float, float]]:
    """All local minima with miss < threshold. Returns list of (jd_tca, miss_km)."""
    if grid is None:
        grid = np.arange(jd0, jd1 + 1e-9, step_s / SECONDS_PER_DAY)
    ra, va = sat_states if sat_states is not None else pa(grid)
    rb, vb = pb(grid)
    dr, dv = rb - ra, vb - va
    d = np.linalg.norm(dr, axis=-1)
    rr = np.einsum("ij,ij->i", dr, dv) / d
    # a minimum lies between k-1 and k when range-rate goes from negative to non-negative.
    # 7.5 km/s * step bounds how far the true minimum can be below the sampled distances.
    gate = 0.5 * 15.0 * step_s + threshold_km
    idx = np.where((rr[:-1] < 0) & (rr[1:] >= 0) & (np.minimum(d[:-1], d[1:]) < gate))[0]
    out = []
    for k in idx:
        jd_tca, miss = refine_minimum(pa, pb, grid[k], grid[k + 1])
        if miss < threshold_km:
            out.append((jd_tca, miss))
    return out


def altitude_overlap(sat, obj, pad_km: float) -> bool:
    return not (obj.perigee_km > sat.apogee_km + pad_km or obj.apogee_km < sat.perigee_km - pad_km)


def make_event(sat_pos: PosFn, obj, jd: float, miss: float) -> Conjunction:
    ra, va = sat_pos(np.array([jd]))
    rb, vb = obj.at(np.array([jd]))
    ra, va, rb, vb = ra[0], va[0], rb[0], vb[0]
    ev = Conjunction(obj=obj, jd=jd, miss=miss, vrel=float(np.linalg.norm(vb - va)),
                     rel_pos=rb - ra, r_sat=ra, v_sat=va, v_obj=vb)
    ev.rtn_miss = to_rtn(ev.rel_pos, ra, va)
    cosang = np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)
    ev.angle_deg = float(np.degrees(np.arccos(cosang)))
    return ev


def screen_objects(sat_pos: PosFn, objects: list, jd0: float, jd1: float, threshold_km: float,
                   step_s: float = 120.0, chunk: int = 300, progress: Callable[[float], None] | None = None):
    """Close approaches of one trajectory against many objects. Returns [(obj, jd_tca, miss_km)].

    Near-Earth objects are propagated together with SGP4Batch on a coarse grid. Each local minimum
    (range-rate sign change) gets a straight-line closest-approach estimate; only those that could be
    inside the threshold are refined precisely (dense sampling + golden section on the exact models).
    """
    from ..propagation.sgp4_batch import SGP4Batch
    grid = np.arange(jd0, jd1 + 1e-9, step_s / SECONDS_PER_DAY)
    ra, va = sat_pos(grid)
    gate = threshold_km + 20.0 * (step_s / 120.0) ** 2       # covers the straight-line approximation error
    near = [o for o in objects if not getattr(o, "deep", False)]
    deep = [o for o in objects if getattr(o, "deep", False)]
    out, done, total = [], 0, max(1, len(objects))
    for i in range(0, len(near), chunk):
        part = near[i:i + chunk]
        R, V = SGP4Batch(part).at(grid)
        dr, dv = R - ra[None], V - va[None]
        d = np.linalg.norm(dr, axis=-1)
        rr = np.einsum("ntk,ntk->nt", dr, dv) / d
        with np.errstate(invalid="ignore"):
            n_idx, k_idx = np.where((rr[:, :-1] < 0) & (rr[:, 1:] >= 0))
        if len(n_idx):
            a, b = dr[n_idx, k_idx], dv[n_idx, k_idx]
            tau = np.clip(-np.einsum("ij,ij->i", a, b) / np.einsum("ij,ij->i", b, b), 0, step_s)
            dlin = np.linalg.norm(a + tau[:, None] * b, axis=1)
            for n, k in zip(n_idx[dlin < gate], k_idx[dlin < gate]):
                o = part[n]
                jd, miss = refine_minimum(sat_pos, o.at, grid[k], grid[k + 1])
                if miss < threshold_km:
                    out.append((o, jd, miss))
        done += len(part)
        if progress:
            progress(done / total)
    for o in deep:   # rare for LEO screening (altitude filter): exact per-object path
        for jd, miss in screen_pair(sat_pos, o.at, jd0, jd1, 60.0, threshold_km):
            out.append((o, jd, miss))
        done += 1
    if progress:
        progress(1.0)
    return out


def screen_catalogue(sat, objects: Iterable, jd0: float, jd1: float, threshold_km: float = 5.0,
                     step_s: float = 120.0, sigma_m: float = 300.0, hbr_m: float = 10.0,
                     sat_pos: PosFn | None = None, progress: Callable[[float], None] | None = None):
    """Screen one satellite against a catalogue. Returns (events sorted by risk, n_candidates)."""
    sat_pos = sat_pos or sat.at
    cands = [o for o in objects if not o.error and o.satnum != sat.satnum
             and altitude_overlap(sat, o, threshold_km + 10)]
    events = [make_event(sat_pos, o, jd, miss).score(sigma_m, hbr_m)
              for o, jd, miss in screen_objects(sat_pos, cands, jd0, jd1, threshold_km, step_s, progress=progress)]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    events.sort(key=lambda e: (order[e.risk], e.miss))
    return events, len(cands)


def separation_series(pa: PosFn, pb: PosFn, jd_center: float, half_width_s: float, n: int = 241):
    """Seconds-from-centre and separation (km) arrays, for plotting."""
    s = np.linspace(-half_width_s, half_width_s, n)
    return s, _dist(pa, pb, jd_center + s / SECONDS_PER_DAY)
