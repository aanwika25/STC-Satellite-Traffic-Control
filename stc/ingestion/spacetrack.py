"""Space-Track OMM XML catalogues (as downloaded from space-track.org, FORMAT=xml).

Reads data/spacetrack/*.xml, keeps Space-Track's own metadata (OBJECT_TYPE, RCS_SIZE, COUNTRY_CODE,
LAUNCH_DATE), drops decayed objects, removes duplicates (newest epoch wins) and splits the result into
Indian LEO payloads (selectable "satellites to protect") and everything else.

Note: Space-Track's user agreement restricts passing its data on to third parties. Keep these files
out of public repositories (they are listed in .gitignore) and let each user download their own.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from ..common.time_utils import jd_from_datetime
from ..propagation.sgp4_prop import Satellite

SPACETRACK_DIR = Path(__file__).resolve().parents[2] / "data" / "spacetrack"
INDIA_CODES = {"IND", "IN", "ISRO"}
FEATURED = ["CARTOSAT 3", "RISAT 2B", "EOS-4", "EOS-6", "NISAR", "CARTOSAT 2F", "RISAT-2BR1", "DRISHTI", "EOS-8", "SARAL"]


def iter_omm_records(path):
    """Yield one flat dict per <segment>, including USER_DEFINED parameters (streams large files)."""
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag.split("}")[-1] != "segment":
            continue
        rec = {}
        for e in el.iter():
            tag = e.tag.split("}")[-1]
            if tag == "USER_DEFINED":
                rec[e.get("parameter", "")] = (e.text or "").strip()
            elif len(e) == 0 and e.text and e.text.strip():
                rec[tag] = e.text.strip()
        yield rec
        el.clear()


def apply_catalogue_metadata(sat: Satellite, rec: dict) -> Satellite:
    """Use Space-Track's OBJECT_TYPE / RCS_SIZE / COUNTRY_CODE when the record has them."""
    t = (rec.get("OBJECT_TYPE") or "").upper()
    if t:
        sat.obj_type = {"ROCKET BODY": "ROCKET BODY", "DEBRIS": "DEBRIS", "PAYLOAD": "PAYLOAD"}.get(t, t)
    if rec.get("RCS_SIZE"):
        sat.rcs = rec["RCS_SIZE"].upper()
    cc = (rec.get("COUNTRY_CODE") or "").upper()
    if cc:
        sat.country = "" if sat.obj_type in ("DEBRIS", "ROCKET BODY") else ("IN" if cc in INDIA_CODES else "INTL")
        sat.country_code = cc
    if rec.get("LAUNCH_DATE"):
        sat.launch_date = rec["LAUNCH_DATE"]
    sat.synthetic = (rec.get("DATA_SOURCE") or "").upper().startswith("SYNTHETIC")
    return sat


def load_spacetrack(paths=None):
    """Returns (indian_leo_payloads, other_objects, start_jd, info)."""
    paths = paths or sorted(SPACETRACK_DIR.glob("*.xml"))
    best: dict[str, tuple[float, Satellite]] = {}
    skipped = {"decayed": 0, "invalid": 0}
    for p in paths:
        for rec in iter_omm_records(p):
            if rec.get("DECAY_DATE"):
                skipped["decayed"] += 1
                continue
            try:
                sat = apply_catalogue_metadata(Satellite.from_omm(rec), rec)
            except (KeyError, TypeError, ValueError):
                skipped["invalid"] += 1
                continue
            if sat.error:
                skipped["invalid"] += 1
                continue
            prev = best.get(sat.satnum)
            if prev is None or sat.jdepoch > prev[0]:
                best[sat.satnum] = (sat.jdepoch, sat)
    sats = [s for _, s in best.values()]
    prims = [s for s in sats if s.country == "IN" and s.obj_type == "PAYLOAD" and s.regime == "LEO"]
    rank = {n: i for i, n in enumerate(FEATURED)}
    prims.sort(key=lambda s: (rank.get(s.name, len(FEATURED)), s.name))
    for s in prims:
        s.desc = f"Indian satellite, launched {getattr(s, 'launch_date', '?')}"
    ids = {s.satnum for s in prims}
    objs = [s for s in sats if s.satnum not in ids]
    newest = max(s.jdepoch for s in sats) if sats else None
    info = {"files": [p.name for p in paths], "objects": len(sats), "newest_epoch_jd": newest,
            "synthetic": sum(1 for s in sats if getattr(s, "synthetic", False)), **skipped}
    return prims, objs, newest, info
