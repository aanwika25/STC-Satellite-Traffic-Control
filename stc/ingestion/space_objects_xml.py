"""Reader for <SpaceObjectsCatalog> XML files (ECI state vectors plus Keplerian parameters).

Each <SpaceObject> may carry
  * PositionCoordinates / VelocityCoordinates (ECI, km and km/s) at <Timestamp>, and/or
  * KeplerianElements: Inclination_deg, Eccentricity, RightAscension_deg, MeanAnomaly_deg,
    OperatingAltitude_km (mean altitude), optionally ArgumentOfPerigee_deg.

The two descriptions are cross-checked. The state vector is used when it agrees with the stated
orbit (it is the more precise description); otherwise the Keplerian parameters are used and a note
explains why. Every note is returned so the dashboard can show it.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

from ..common.constants import MU_WGS72, RE_WGS72, RE
from ..common.frames import rv_to_coe
from ..common.time_utils import jd_from_datetime
from ..propagation.sgp4_prop import Satellite


def _f(el, path):
    node = el.find(path)
    if node is None or node.text is None or not node.text.strip():
        return None
    try:
        return float(node.text)
    except ValueError:
        return None


def _rev_per_day(a_km: float) -> float:
    return math.sqrt(MU_WGS72 / a_km ** 3) * 86400 / (2 * math.pi)


def fit_state_vector(name: str, satnum: str, jd: float, r, v, **meta) -> Satellite:
    """Mean elements whose SGP4 (or Kepler) propagation reproduces r, v at jd (fixed-point iteration)."""
    r_t, v_t = np.asarray(r, float), np.asarray(v, float)
    r_g, v_g = r_t.copy(), v_t.copy()
    sat = None
    for _ in range(40):
        c = rv_to_coe(r_g, v_g, MU_WGS72)
        sat = Satellite.from_elements(name, satnum, jd, 0.0, math.degrees(c["i"]), math.degrees(c["raan"]), max(c["e"], 1e-7),
                                      math.degrees(c["argp"]), math.degrees(c["M"]), _rev_per_day(c["a"]), **meta)
        rs, vs = sat.at1(jd)
        dr, dv = r_t - rs, v_t - vs
        if np.linalg.norm(dr) < 1e-3 and np.linalg.norm(dv) < 1e-6:
            break
        r_g, v_g = r_g + dr, v_g + dv
    return sat


def _check(r, v, kep):
    """Compare the state vector with the stated Keplerian parameters. Returns (ok, details, derived)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    c = rv_to_coe(r, v, MU_WGS72)
    alt_now = np.linalg.norm(r) - RE
    perigee = c["a"] * (1 - c["e"]) - RE
    derived = {"alt_now": alt_now, "perigee": perigee, "apogee": c["a"] * (1 + c["e"]) - RE,
               "e": c["e"], "inc": math.degrees(c["i"])}
    problems = []
    if perigee < 100:
        problems.append(f"its orbit would dip to {perigee:.0f} km altitude (below 100 km, i.e. re-entry)")
    if kep.get("alt") is not None:
        tol = max(100.0, 0.03 * kep["alt"])
        if not (derived["perigee"] - tol <= kep["alt"] <= derived["apogee"] + tol):
            problems.append(f"altitude {alt_now:.0f} km / range {derived['perigee']:.0f}-{derived['apogee']:.0f} km vs stated {kep['alt']:.0f} km")
    if kep.get("inc") is not None and abs(derived["inc"] - kep["inc"]) > 1.0:
        problems.append(f"inclination {derived['inc']:.1f}° vs stated {kep['inc']:.2f}°")
    if kep.get("e") is not None and abs(derived["e"] - kep["e"]) > 0.01:
        problems.append(f"eccentricity {derived['e']:.3f} vs stated {kep['e']}")
    return not problems, problems, derived


def parse_space_objects_xml(text: str, guess_country=None):
    """Returns (satellites, notes). notes is a list of plain-language strings, one or more per object."""
    root = ET.fromstring(text.strip())
    sats, notes = [], []
    for so in root.iter("SpaceObject"):
        name = (so.findtext("ObjectName") or f"NORAD {so.get('noradId', '?')}").strip()
        satnum = so.get("noradId", "") or name
        ts = (so.findtext("Timestamp") or "").strip().replace("Z", "")
        try:
            jd = jd_from_datetime(datetime.fromisoformat(ts).replace(tzinfo=timezone.utc))
        except ValueError:
            notes.append(f"{name}: skipped - missing or unreadable Timestamp.")
            continue
        pos = [_f(so, f"PositionCoordinates/{k}") for k in ("X_km", "Y_km", "Z_km")]
        vel = [_f(so, f"VelocityCoordinates/{k}") for k in ("Vx_kms", "Vy_kms", "Vz_kms")]
        kep = {"inc": _f(so, "KeplerianElements/Inclination_deg"), "e": _f(so, "KeplerianElements/Eccentricity"),
               "raan": _f(so, "KeplerianElements/RightAscension_deg"), "ma": _f(so, "KeplerianElements/MeanAnomaly_deg"),
               "alt": _f(so, "KeplerianElements/OperatingAltitude_km"), "argp": _f(so, "KeplerianElements/ArgumentOfPerigee_deg")}
        status = (so.get("status") or "").strip()
        meta = {"obj_type": "DEBRIS" if "DEB" in name.upper() else "PAYLOAD"}
        if guess_country:
            meta["country"] = guess_country(name, meta["obj_type"])
        has_sv = all(x is not None for x in pos + vel)
        kep_ok = all(kep[k] is not None for k in ("inc", "e", "raan", "ma", "alt"))
        sat = None
        if has_sv:
            ok, problems, d = _check(pos, vel, kep)
            if ok:
                sat = fit_state_vector(name, satnum, jd, pos, vel, **meta)
                notes.append(f"{name}: used the ECI position and velocity (consistent with the stated orbit).")
            else:
                notes.append(f"{name}: the position/velocity does not match the stated orbit - " + "; ".join(problems) + ".")
        if sat is None and kep_ok:
            a = RE_WGS72 + kep["alt"]
            argp = kep["argp"] if kep["argp"] is not None else 0.0
            sat = Satellite.from_elements(name, satnum, jd, 0.0, kep["inc"], kep["raan"], max(kep["e"], 1e-7), argp, kep["ma"],
                                          _rev_per_day(a), **meta)
            notes.append(f"{name}: used the Keplerian parameters ({kep['alt']:.0f} km, {kep['inc']}°)"
                         + ("; argument of perigee not given, assumed 0°." if kep["argp"] is None else "."))
        if sat is None:
            notes.append(f"{name}: skipped - neither a consistent state vector nor complete Keplerian parameters.")
            continue
        sat.desc = f"{so.get('orbitCategory', sat.regime)} · {status}" if status else sat.regime
        sats.append(sat)
    return sats, notes
