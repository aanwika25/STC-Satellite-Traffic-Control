"""Ground-station visibility: passes above an elevation mask, before vs after a manoeuvre."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.constants import SECONDS_PER_DAY
from ..common.frames import eci_to_ecef, look_angles

from ..ingestion.ground_stations import load_ground_stations

# Indian ground stations, loaded from data/ground_stations.xml (edit that file to change them)
STATIONS = load_ground_stations()


@dataclass
class Pass:
    station: str
    aos: float      # JD of acquisition of signal
    los: float      # JD of loss of signal
    max_el: float   # deg
    t_max: float

    @property
    def duration_s(self) -> float:
        return (self.los - self.aos) * SECONDS_PER_DAY


def elevation_series(pos_fn, st: dict, jd):
    r, _ = pos_fn(jd)
    el, az, rng = look_angles(st["lat"], st["lon"], st.get("alt", 0.0), eci_to_ecef(r, jd))
    return el


def find_passes(pos_fn, st: dict, jd0: float, jd1: float, step_s: float = 10.0, mask_deg: float = 5.0) -> list[Pass]:
    jd = np.arange(jd0, jd1 + 1e-9, step_s / SECONDS_PER_DAY)
    el = elevation_series(pos_fn, st, jd)
    above = el > mask_deg
    out = []
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(jd) - 1]

    def cross(k):  # linear interpolation of the mask crossing between samples k-1 and k
        if k == 0 or k >= len(jd):
            return jd[min(k, len(jd) - 1)]
        e0, e1 = el[k - 1], el[k]
        f = (mask_deg - e0) / (e1 - e0) if e1 != e0 else 0.0
        return jd[k - 1] + f * (jd[k] - jd[k - 1])

    for s, e in zip(starts, ends):
        seg = el[s:e] if e > s else el[s:s + 1]
        k = s + int(np.argmax(seg))
        out.append(Pass(st["name"], float(cross(s)), float(cross(e)), float(el[k]), float(jd[k])))
    return out


def compare_passes(base_fn, man_fn, stations: list[dict], jd0: float, jd1: float, mask_deg: float = 5.0):
    """Match passes before/after by nearest AOS (within 15 min). Returns rows + summary dict."""
    rows = []
    summary = dict(n_before=0, n_after=0, contact_before_s=0.0, contact_after_s=0.0, max_shift_s=0.0, lost=0, gained=0)
    for st in stations:
        pb = find_passes(base_fn, st, jd0, jd1, 10.0, mask_deg)
        pm = find_passes(man_fn, st, jd0, jd1, 10.0, mask_deg)
        summary["n_before"] += len(pb)
        summary["n_after"] += len(pm)
        summary["contact_before_s"] += sum(p.duration_s for p in pb)
        summary["contact_after_s"] += sum(p.duration_s for p in pm)
        used = set()
        for b in pb:
            best, bd = None, 15 / 1440
            for j, a in enumerate(pm):
                if j not in used and abs(a.aos - b.aos) < bd:
                    best, bd = j, abs(a.aos - b.aos)
            a = pm[best] if best is not None else None
            if a is None:
                summary["lost"] += 1
            else:
                used.add(best)
                summary["max_shift_s"] = max(summary["max_shift_s"], abs(a.aos - b.aos) * SECONDS_PER_DAY,
                                             abs(a.los - b.los) * SECONDS_PER_DAY)
            rows.append({"station": st, "before": b, "after": a})
        for j, a in enumerate(pm):
            if j not in used:
                summary["gained"] += 1
                rows.append({"station": st, "before": None, "after": a})
    rows.sort(key=lambda x: (x["before"] or x["after"]).aos)
    return rows, summary
