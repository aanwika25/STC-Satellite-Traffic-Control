"""Generate a SYNTHETIC debris catalogue in Space-Track OMM XML format, for demonstrations.

The population is modelled on the real tracked debris environment: the main fragmentation clouds at
their real inclinations and altitude bands, rocket bodies, geostationary-transfer stages and a few MEO
stages. It is NOT real data: catalogue numbers start at 880001 (outside the real range), the header says
SYNTHETIC, and every record carries <USER_DEFINED parameter="DATA_SOURCE">SYNTHETIC DEMO</USER_DEFINED>.

It also places a few fragments on orbits computed (with the same SGP4) to pass close to Indian satellites
in data/spacetrack/ at chosen future times, so a demo always has debris encounters to show.

Usage:  python scripts/make_demo_debris.py [--seed 7] [--out data/spacetrack/demo_debris_synthetic.xml]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stc.common.constants import MU, RE, MU_WGS72          # noqa: E402
from stc.common.frames import rv_to_coe                     # noqa: E402
from stc.common.time_utils import datetime_from_jd, jday   # noqa: E402
from stc.propagation.sgp4_prop import Satellite             # noqa: E402

# name, object id prefix, country, launch date, site, inclination (mean, sd), mean-altitude band (km),
# eccentricity scale, count, type, RCS weights (small, medium, large)
CLOUDS = [
    ("FENGYUN 1C DEB", "1999-025", "PRC", "1999-05-10", "TSC", (98.65, 0.35), (720, 1150), 0.020, 1000, "DEBRIS", (0.80, 0.17, 0.03)),
    ("COSMOS 2251 DEB", "1993-036", "CIS", "1993-06-16", "PKMTR", (74.03, 0.20), (650, 1050), 0.012, 520, "DEBRIS", (0.78, 0.18, 0.04)),
    ("IRIDIUM 33 DEB", "1997-051", "US", "1997-09-14", "TTMTR", (86.38, 0.15), (650, 900), 0.008, 210, "DEBRIS", (0.82, 0.15, 0.03)),
    ("NOAA 16 DEB", "2000-055", "US", "2000-09-21", "AFWTR", (98.95, 0.15), (820, 900), 0.004, 160, "DEBRIS", (0.85, 0.13, 0.02)),
    ("NOAA 17 DEB", "2002-032", "US", "2002-06-24", "AFWTR", (98.55, 0.12), (780, 830), 0.004, 60, "DEBRIS", (0.85, 0.13, 0.02)),
    ("DMSP 5D-2 F13 DEB", "1995-015", "US", "1995-03-24", "AFWTR", (98.80, 0.15), (800, 870), 0.005, 90, "DEBRIS", (0.85, 0.13, 0.02)),
    ("CZ-6A DEB", "2024-149", "PRC", "2024-08-06", "TSC", (89.00, 0.25), (760, 900), 0.006, 330, "DEBRIS", (0.75, 0.20, 0.05)),
    ("COSMOS 1408 DEB", "1982-092", "CIS", "1982-09-16", "PKMTR", (82.55, 0.15), (380, 470), 0.003, 6, "DEBRIS", (0.9, 0.1, 0.0)),
    ("CZ-2C DEB", "2023-110", "PRC", "2023-07-23", "JSC", (35.0, 4.0), (420, 620), 0.004, 70, "DEBRIS", (0.8, 0.2, 0.0)),
    ("FALCON 9 DEB", "2023-001", "US", "2023-01-03", "AFETR", (52.0, 8.0), (400, 600), 0.003, 50, "DEBRIS", (0.7, 0.3, 0.0)),
    ("PSLV DEB", "2017-008", "IND", "2017-02-15", "SRI", (97.5, 0.3), (430, 650), 0.005, 18, "DEBRIS", (0.8, 0.2, 0.0)),
    ("SL-8 R/B", "1985-000", "CIS", "1985-01-01", "PKMTR", (74.0, 0.1), (730, 1000), 0.003, 110, "ROCKET BODY", (0.0, 0.0, 1.0)),
    ("SL-8 R/B", "1987-000", "CIS", "1987-01-01", "PKMTR", (82.9, 0.1), (900, 1000), 0.003, 70, "ROCKET BODY", (0.0, 0.0, 1.0)),
    ("SL-16 R/B", "1990-000", "CIS", "1990-01-01", "TTMTR", (71.0, 0.1), (830, 850), 0.002, 20, "ROCKET BODY", (0.0, 0.0, 1.0)),
    ("CZ-4C R/B", "2016-000", "PRC", "2016-01-01", "TSC", (98.3, 0.5), (500, 900), 0.010, 60, "ROCKET BODY", (0.0, 0.2, 0.8)),
    ("PSLV R/B", "2018-000", "IND", "2018-01-01", "SRI", (97.6, 0.4), (450, 700), 0.006, 24, "ROCKET BODY", (0.0, 0.1, 0.9)),
    ("DELTA 2 R/B", "1998-000", "US", "1998-01-01", "AFWTR", (98.0, 1.5), (600, 1300), 0.040, 30, "ROCKET BODY", (0.0, 0.2, 0.8)),
]
GTO = [  # elongated transfer-orbit stages: perigee in LEO, apogee near GEO
    ("ARIANE 5 R/B", "2015-000", "FR", "AFETR", (6.0, 2.0), 70),
    ("FALCON 9 R/B", "2019-000", "US", "AFETR", (27.0, 1.5), 40),
    ("CZ-3B R/B", "2018-000", "PRC", "XSC", (27.5, 1.0), 50),
    ("GSLV R/B", "2018-000", "IND", "SRI", (20.5, 1.0), 8),
]
MEO = [  # stages near the navigation-satellite shells
    ("SL-12 R/B(2)", "2010-000", "CIS", "TTMTR", 64.8, (18800, 19500), 30),
    ("DELTA 2 R/B", "2005-000", "US", "AFETR", 55.0, (19500, 20400), 12),
    ("CZ-3B R/B", "2019-001", "PRC", "XSC", 55.0, (20800, 21800), 14),
]
DEMO_TARGETS = ["CARTOSAT 3", "RISAT 2B", "EOS-4", "EOS-6", "NISAR", "CARTOSAT 2F", "DRISHTI", "RISAT-2BR1", "EOS-8"]


def rev_per_day(a_km):
    return math.sqrt(MU / a_km ** 3) * 86400 / (2 * math.pi)


def bstar_for(perigee_km, rng, kind):
    base = -3.3 if perigee_km < 600 else -3.9 if perigee_km < 900 else -4.6
    if kind == "ROCKET BODY":
        base -= 0.8
    return 10 ** rng.uniform(base - 0.5, base + 0.4)


class Writer:
    def __init__(self, epoch_jd):
        self.parts, self.n, self.epoch_jd = [], 880001, epoch_jd

    def add(self, name, obj_id, country, launch, site, kind, rcs, inc, raan, ecc, argp, ma, n_rev, bstar, epoch_jd):
        a = (MU / (n_rev * 2 * math.pi / 86400) ** 2) ** (1 / 3)
        per = 1440 / n_rev
        years = max(0.1, (epoch_jd - jday(int(launch[:4]), int(launch[5:7]), int(launch[8:10]))) / 365.25)
        ep = datetime_from_jd(epoch_jd)
        nnum = self.n
        self.n += 1
        ud = {"SEMIMAJOR_AXIS": f"{a:.3f}", "PERIOD": f"{per:.3f}", "APOAPSIS": f"{a * (1 + ecc) - RE:.3f}",
              "PERIAPSIS": f"{a * (1 - ecc) - RE:.3f}", "OBJECT_TYPE": kind, "RCS_SIZE": rcs, "COUNTRY_CODE": country,
              "LAUNCH_DATE": launch, "SITE": site, "DECAY_DATE": "", "FILE": "0", "GP_ID": str(900000000 + nnum),
              "DATA_SOURCE": "SYNTHETIC DEMO"}
        self.parts.append(
            "<segment><metadata>"
            f"<OBJECT_NAME>{escape(name)}</OBJECT_NAME><OBJECT_ID>{obj_id}</OBJECT_ID><CENTER_NAME>EARTH</CENTER_NAME>"
            "<REF_FRAME>TEME</REF_FRAME><TIME_SYSTEM>UTC</TIME_SYSTEM><MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY></metadata>"
            "<data><meanElements>"
            f"<EPOCH>{ep.strftime('%Y-%m-%dT%H:%M:%S.%f')}</EPOCH><MEAN_MOTION>{n_rev:.8f}</MEAN_MOTION>"
            f"<ECCENTRICITY>{ecc:.8f}</ECCENTRICITY><INCLINATION>{inc:.4f}</INCLINATION><RA_OF_ASC_NODE>{raan % 360:.4f}</RA_OF_ASC_NODE>"
            f"<ARG_OF_PERICENTER>{argp % 360:.4f}</ARG_OF_PERICENTER><MEAN_ANOMALY>{ma % 360:.4f}</MEAN_ANOMALY></meanElements>"
            "<tleParameters><EPHEMERIS_TYPE>0</EPHEMERIS_TYPE><CLASSIFICATION_TYPE>U</CLASSIFICATION_TYPE>"
            f"<NORAD_CAT_ID>{nnum}</NORAD_CAT_ID><ELEMENT_SET_NO>999</ELEMENT_SET_NO><REV_AT_EPOCH>{int(years * 365.25 * n_rev) % 100000}</REV_AT_EPOCH>"
            f"<BSTAR>{bstar:.14f}</BSTAR><MEAN_MOTION_DOT>{bstar * 0.05:.8f}</MEAN_MOTION_DOT><MEAN_MOTION_DDOT>0.0000000000000</MEAN_MOTION_DDOT>"
            "</tleParameters><userDefinedParameters>"
            + "".join(f'<USER_DEFINED parameter="{k}">{v}</USER_DEFINED>' for k, v in ud.items())
            + "</userDefinedParameters></data></segment>")
        return nnum

    def xml(self, created):
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<ndm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-3.0.0-master-3.0.xsd">'
                '<omm id="CCSDS_OMM_VERS" version="3.0"><header>'
                "<COMMENT>SYNTHETIC DEMO DEBRIS CATALOGUE - NOT REAL DATA. Generated by STC scripts/make_demo_debris.py, "
                "modelled on the real debris environment. Catalogue numbers 880001+.</COMMENT>"
                f"<CREATION_DATE>{created}</CREATION_DATE><ORIGINATOR>STC DEMO (SYNTHETIC)</ORIGINATOR></header><body>"
                + "\n".join(self.parts) + "\n</body></omm></ndm>\n")


def rcs_pick(rng, w):
    return rng.choices(["SMALL", "MEDIUM", "LARGE"], weights=w)[0]


def fit_to_state(r_t, v_t, jd_c, jd_epoch, bstar):
    """Mean elements at jd_epoch whose SGP4 propagation reaches (r_t, v_t) at jd_c (fixed-point iteration)."""
    r_g, v_g = r_t.copy(), v_t.copy()
    dt_min = (jd_c - jd_epoch) * 1440
    sat = None
    for _ in range(80):
        c = rv_to_coe(r_g, v_g, MU_WGS72)
        if not (0 <= c["e"] < 0.3) or not (RE + 200 < c["a"] < RE + 3000):
            return None
        rev = math.sqrt(MU_WGS72 / c["a"] ** 3) * 86400 / (2 * math.pi)
        tmp = Satellite.from_elements("t", 0, jd_epoch, bstar, math.degrees(c["i"]), 0, max(c["e"], 1e-6), 0, 0, rev)
        if tmp.error or tmp.deep:
            return None
        els = (math.degrees(c["i"]), math.degrees(c["raan"] - tmp.nodedot * dt_min), max(c["e"], 1e-6),
               math.degrees(c["argp"] - tmp.argpdot * dt_min), math.degrees(c["M"] - tmp.mdot * dt_min), rev)
        sat = Satellite.from_elements("d", 0, jd_epoch, bstar, *els)
        rs, vs = sat.at1(jd_c)
        if not np.all(np.isfinite(rs)):
            return None
        er, ev = r_t - rs, v_t - vs
        if np.linalg.norm(er) < 0.002 and np.linalg.norm(ev) < 1e-5:
            return els
        r_g, v_g = r_g + er, v_g + ev
    return els if np.linalg.norm(er) < 0.01 else None


PLAN_CLOUDS = [("FENGYUN 1C DEB", 98.65, 4), ("COSMOS 2251 DEB", 74.03, 3), ("IRIDIUM 33 DEB", 86.38, 2),
               ("CZ-6A DEB", 89.0, 2), ("NOAA 16 DEB", 98.95, 1)]


def _plan(rng, r, v, miss, jd_c, epoch_hi):
    """One attempt at a fragment orbit from a real cloud (its inclination) passing `miss` km from the
    satellite state (r, v) at jd_c. Returns (elements, B*, epoch, cloud name) or None."""
    name, inc, _ = rng.choices(PLAN_CLOUDS, weights=[c[2] for c in PLAN_CLOUDS])[0]
    rn = np.linalg.norm(r)
    up = r / rn
    east = np.cross([0.0, 0.0, 1.0], up); east /= np.linalg.norm(east)
    north = np.cross(up, east)
    coslat = math.sqrt(max(1e-9, 1 - up[2] ** 2))
    sA = math.cos(math.radians(inc)) / coslat            # heading that gives this inclination here
    if abs(sA) > 1:
        return None
    A = math.asin(sA) if rng.random() < 0.5 else math.pi - math.asin(sA)
    a = rn + rng.uniform(60, 420)                         # fragment orbit a bit higher on average: eccentric
    speed = math.sqrt(MU_WGS72 * (2 / rn - 1 / a))
    fpa = math.radians(rng.uniform(-2.0, 2.0))            # small flight-path angle
    v_d = speed * (math.cos(fpa) * (math.cos(A) * north + math.sin(A) * east) + math.sin(fpa) * up)
    if np.linalg.norm(v_d - v) < 0.5:                      # avoid near co-orbital geometry
        return None
    u = (v_d - v) / np.linalg.norm(v_d - v)
    a1 = up - u * (up @ u); a1 /= np.linalg.norm(a1)
    a2 = np.cross(u, a1); phi = rng.uniform(0, 2 * math.pi)
    r_d = r + miss * (a1 * math.cos(phi) + a2 * math.sin(phi))
    ep = epoch_hi - rng.uniform(0.05, 1.2)
    bstar = 10 ** rng.uniform(-5.0, -4.3)
    els = fit_to_state(r_d, v_d, jd_c, ep, bstar)
    return (els, bstar, ep, name) if els else None


CLOUD_META = [("FENGYUN 1C DEB", "PRC", "1999-05-10", "TSC", "1999-025"), ("NOAA 16 DEB", "US", "2000-09-21", "AFWTR", "2000-055"),
              ("CZ-6A DEB", "PRC", "2024-08-06", "TSC", "2024-149"), ("IRIDIUM 33 DEB", "US", "1997-09-14", "TTMTR", "1997-051"),
              ("COSMOS 2251 DEB", "CIS", "1993-06-16", "PKMTR", "1993-036")]


def name_for_inclination(inc):
    for name, inc0, cc, launch, site, oid in [("FENGYUN 1C DEB", 98.65, "PRC", "1999-05-10", "TSC", "1999-025"),
                                              ("NOAA 16 DEB", 98.95, "US", "2000-09-21", "AFWTR", "2000-055"),
                                              ("CZ-6A DEB", 89.0, "PRC", "2024-08-06", "TSC", "2024-149"),
                                              ("IRIDIUM 33 DEB", 86.38, "US", "1997-09-14", "TTMTR", "1997-051"),
                                              ("COSMOS 2251 DEB", 74.03, "CIS", "1993-06-16", "PKMTR", "1993-036")]:
        if abs(inc - inc0) < 1.6:
            return name, cc, launch, site, oid
    return "CZ-4C DEB", "PRC", "2016-11-11", "TSC", "2016-066"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "spacetrack" / "demo_debris_synthetic.xml"))
    ap.add_argument("--start", default="2026-11-22T00:00", help="first planned encounter (UTC)")
    ap.add_argument("--days", type=float, default=21.0, help="spread planned encounters over this many days")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    epoch_hi = jday(2026, 11, 21, 18, 0, 0)
    W = Writer(epoch_hi)
    epoch = lambda: epoch_hi - rng.uniform(0.05, 1.6)
    piece = {}

    for name, oid, cc, launch, site, (i0, isd), (h0, h1), escale, count, kind, w in CLOUDS:
        for _ in range(count):
            hm = h0 + (h1 - h0) * rng.betavariate(2.2, 2.6)          # mean altitude, peaked inside the band
            a = RE + hm
            e = min(abs(rng.gauss(0, escale)), 0.12)
            if a * (1 - e) - RE < 330:                                 # perigee too low: it would have decayed
                e = max(0.0005, (a - RE - 330) / a * 0.9)
            piece[oid] = piece.get(oid, 0) + 1
            suffix = "" if kind == "ROCKET BODY" else ""
            obj_id = f"{oid}{'A' if kind == 'ROCKET BODY' else ''}" if oid.endswith("000") else f"{oid}{_letters(piece[oid])}"
            W.add(name + suffix, obj_id, cc, launch, site, kind, rcs_pick(rng, w), rng.gauss(i0, isd), rng.uniform(0, 360),
                  max(e, 0.0003), rng.uniform(0, 360), rng.uniform(0, 360), rev_per_day(a), bstar_for(a * (1 - e) - RE, rng, kind), epoch())
    for name, oid, cc, site, (i0, isd), count in GTO:
        for _ in range(count):
            hp, ha = rng.uniform(180, 650), rng.uniform(29000, 36200)
            a = RE + (hp + ha) / 2
            e = (ha - hp) / (2 * a)
            year = rng.randint(2008, 2025)
            W.add(name, f"{year}-{rng.randint(1, 120):03d}B", cc, f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", site,
                  "ROCKET BODY", "LARGE", rng.gauss(i0, isd), rng.uniform(0, 360), e, rng.uniform(0, 360), rng.uniform(0, 360),
                  rev_per_day(a), bstar_for(hp, rng, "ROCKET BODY") * 0.3, epoch())
    for name, oid, cc, site, inc, (h0, h1), count in MEO:
        for _ in range(count):
            a = RE + rng.uniform(h0, h1)
            year = rng.randint(1995, 2022)
            W.add(name, f"{year}-{rng.randint(1, 90):03d}B", cc, f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", site,
                  "ROCKET BODY", "LARGE", rng.gauss(inc, 0.5), rng.uniform(0, 360), rng.uniform(0.001, 0.03), rng.uniform(0, 360),
                  rng.uniform(0, 360), rev_per_day(a), 0.0, epoch())

    # Planned demo encounters with Indian satellites (same SGP4 as the dashboard, so they appear exactly)
    from stc.ingestion.spacetrack import load_spacetrack
    prims, _, _, _ = load_spacetrack([p for p in (ROOT / "data" / "spacetrack").glob("*.xml") if "synthetic" not in p.name])
    targets = [p for name in DEMO_TARGETS for p in prims if p.name == name]
    y, mo, d = (int(x) for x in args.start[:10].split("-"))
    hh, mm = (int(x) for x in args.start[11:16].split(":"))
    jd0 = jday(y, mo, d, hh, mm, 0)
    n_enc = 18
    planned = []
    for k in range(n_enc):
        tgt = targets[k % len(targets)]
        jd_c = jd0 + (k + rng.uniform(0.1, 0.9)) * args.days / n_enc
        miss = rng.choice([0.12, 0.25, 0.4, 0.6, 0.9, 1.4, 2.2, 3.5])
        r, v = tgt.at1(jd_c)
        for attempt in range(20):
            els = _plan(rng, r, v, miss, jd_c, epoch_hi)
            if els:
                break
        else:
            print("   could not plan an encounter for", tgt.name); continue
        els, bstar, ep, cloud = els
        name, cc, launch, site, oid = next(c for c in CLOUD_META if c[0] == cloud)
        piece[oid] = piece.get(oid, 0) + 1
        num = W.add(name, f"{oid}{_letters(piece[oid] + 400)}", cc, launch, site, "DEBRIS", "SMALL", *els, bstar, ep)
        planned.append((tgt.name, name, num, datetime_from_jd(jd_c).strftime("%Y-%m-%d %H:%M"), miss))
        continue
        R = r / np.linalg.norm(r)
        th = math.radians(rng.choice([25, 40, 60, 110, 150, 170]) * rng.choice([1, -1]))
        vrot = v * math.cos(th) + np.cross(R, v) * math.sin(th) + R * (R @ v) * (1 - math.cos(th))
        v_d = vrot * rng.uniform(1.004, 1.03) + R * rng.uniform(-0.05, 0.05)      # slightly eccentric fragment orbit
        vrel = v_d - v
        u = vrel / np.linalg.norm(vrel)
        a1 = R - u * (R @ u); a1 /= np.linalg.norm(a1)
        a2 = np.cross(u, a1); phi = rng.uniform(0, 2 * math.pi)
        r_d = r + miss * (a1 * math.cos(phi) + a2 * math.sin(phi))
        ep = epoch_hi - rng.uniform(0.05, 1.2)
        bstar = 10 ** rng.uniform(-4.4, -3.6)
        els = fit_to_state(r_d, v_d, jd_c, ep, bstar)
        name, cc, launch, site, oid = name_for_inclination(els[0])
        piece[oid] = piece.get(oid, 0) + 1
        num = W.add(name, f"{oid}{_letters(piece[oid] + 400)}", cc, launch, site, "DEBRIS", "SMALL", *els, bstar, ep)
        planned.append((tgt.name, name, num, datetime_from_jd(jd_c).strftime("%Y-%m-%d %H:%M"), miss))

    created = datetime_from_jd(epoch_hi + 0.01).strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(W.xml(created), encoding="utf-8")
    print(f"wrote {W.n - 880001} objects to {args.out}")
    print("planned demo encounters (target, debris, NORAD, UTC, planned miss km):")
    for p in planned:
        print("  ", p)


def _letters(n):
    """Piece letters like Space-Track: A..Z, AA..ZZ, AAA... (skipping I and O)."""
    abc = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    s = ""
    n += 1
    while n > 0:
        n, r = divmod(n - 1, len(abc))
        s = abc[r] + s
    return s


if __name__ == "__main__":
    main()
