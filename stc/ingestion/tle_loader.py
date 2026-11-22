"""Loading orbital data: bundled sample dataset, pasted TLE/OMM text, or CelesTrak.

CelesTrak usage follows https://celestrak.org/NORAD/documentation/gp-data-formats.php:
OMM JSON (works for catalogue numbers above 99999, unlike TLE), the .org domain, a 2-hour local
cache (GP data only changes every 2 hours), HTTP 200 checks, and at most 3 attempts per request.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..propagation.sgp4_prop import Satellite, parse_orbit_text, detect_format
from .space_objects_xml import parse_space_objects_xml

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
DEBRIS_GROUPS = {
    "cosmos-1408-debris": "COSMOS 1408 debris",
    "fengyun-1c-debris": "FENGYUN 1C debris",
    "iridium-33-debris": "IRIDIUM 33 debris",
    "cosmos-2251-debris": "COSMOS 2251 debris",
}
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_dataset.json"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_MAX_AGE_S = 2 * 3600          # CelesTrak updates GP data at most every 2 hours
MAX_ATTEMPTS = 3                    # CelesTrak asks for no more than 2-3 tries before giving up


# Name prefixes of Indian satellites (ISRO and Indian operators) used to colour live TLEs
INDIAN_NAMES = ("CARTOSAT", "RISAT", "OCEANSAT", "RESOURCESAT", "EOS-", "EOS ", "IRNSS", "NVS-", "GSAT", "INSAT", "KALPANA",
                "ASTROSAT", "MEGHA-TROPIQUES", "SARAL", "SCATSAT", "HYSIS", "MICROSAT", "EMISAT", "GISAT", "AZAADISAT",
                "ANAND", "PIXXEL", "SHAKUNTALA", "INS-", "IMS-", "CHANDRAYAAN", "ADITYA", "XPOSAT", "BHASKARA", "ROHINI",
                "IRS-", "TES", "KALAMSAT", "STUDSAT", "JUGNU", "SRMSAT", "NIUSAT", "PRATHAM", "EXSEED")


def guess_country(name: str, obj_type: str) -> str:
    if obj_type in ("DEBRIS", "ROCKET BODY"):
        return ""
    n = name.upper()
    return "IN" if any(n.startswith(k) or f" {k}" in n for k in INDIAN_NAMES) else "INTL"


def guess_type(name: str) -> str:
    n = name.upper()
    if " DEB" in n:
        return "DEBRIS"
    if "R/B" in n or "ROCKET" in n:
        return "ROCKET BODY"
    return "PAYLOAD"


def load_sample(path: Path = SAMPLE_PATH):
    """Returns (satellites_to_protect, objects, start_jd)."""
    d = json.loads(Path(path).read_text())
    prims = [Satellite(p["name"], p["l1"], p["l2"], desc=p.get("desc", ""), obj_type="PAYLOAD", country=p.get("country", "IN"))
             for p in d["primaries"]]
    objs = [Satellite(o["name"], o["l1"], o["l2"], obj_type=o["type"], rcs=o.get("rcs", ""), country=o.get("country", ""))
            for o in d["objects"]]
    return prims, [o for o in objs if not o.error], d["epochJd"]


def load_objects_text(text: str):
    """Read objects from any supported text: TLE, OMM JSON/CSV/XML or <SpaceObjectsCatalog> XML.
    Returns (satellites, notes)."""
    text = text.strip()
    if not text:
        return [], []
    if detect_format(text) == "xml" and "<SpaceObjectsCatalog" in text[:500]:
        return parse_space_objects_xml(text, guess_country)
    sats = parse_orbit_text(text)
    for o in sats:   # keep catalogue metadata (e.g. Space-Track COUNTRY_CODE); otherwise guess from the name
        if not o.obj_type:
            o.obj_type = guess_type(o.name)
        if not getattr(o, "country_code", ""):
            o.country = guess_country(o.name, o.obj_type)
    fmt = {"json": "OMM JSON", "csv": "OMM CSV", "xml": "OMM XML", "tle": "TLE"}[detect_format(text)]
    return sats, [f"Read {len(sats)} object(s) as {fmt}."]


def load_from_text(primary_text: str, objects_text: str):
    prims, _ = load_objects_text(primary_text)
    if not prims:
        raise ValueError("No valid orbit data found for the satellite(s) to protect. Paste TLE, OMM or SpaceObjectsCatalog XML.")
    ids = {p.satnum for p in prims}
    objs = [o for o in load_objects_text(objects_text)[0] if o.satnum not in ids]
    for p in prims:
        p.obj_type = "PAYLOAD"
    return prims, objs


class CelesTrakError(RuntimeError):
    pass


def _cache_path(query: str, fmt: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9_-]+", "_", query) + "." + fmt.lower())


def fetch_celestrak(query: str, fmt: str = "JSON", timeout: float = 20.0, force: bool = False):
    """Download GP data for one query (e.g. 'CATNR=44804' or 'GROUP=cosmos-1408-debris').

    Returns (text, info). info["source"] is "cache", "download" or "stale cache"."""
    path = _cache_path(query, fmt)
    if path.exists() and not force:
        age = time.time() - path.stat().st_mtime
        if age < CACHE_MAX_AGE_S:
            return path.read_text(encoding="utf-8"), {"query": query, "source": "cache", "age_min": age / 60}
    url = f"{CELESTRAK_GP}?{query}&FORMAT={fmt.upper()}"
    req = urllib.request.Request(url, headers={"User-Agent": "STC-SatelliteTrafficControl/0.1 (student project)"})
    last_err = "unknown error"
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    raise CelesTrakError(f"HTTP {r.status}")
                text = r.read().decode("utf-8", "replace")
            if fmt.upper() == "JSON" and not text.lstrip().startswith(("[", "{")):
                raise CelesTrakError(text.strip()[:120] or "empty response")   # e.g. "No GP data found"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return text, {"query": query, "source": "download", "age_min": 0.0}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (403, 404, 301):   # blocked / bad query: retrying makes it worse
                break
        except CelesTrakError as e:
            last_err = str(e)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(getattr(e, "reason", e))
        time.sleep(1.5 * (attempt + 1))
    if path.exists():
        age = (time.time() - path.stat().st_mtime) / 60
        return path.read_text(encoding="utf-8"), {"query": query, "source": "stale cache", "age_min": age, "error": last_err}
    raise CelesTrakError(f"{query}: {last_err}")


def load_from_celestrak(norad_ids: list[str], groups: list[str], force: bool = False):
    """Returns (satellites_to_protect, objects, info list)."""
    ids = [i.strip() for i in norad_ids if i.strip()]
    if not ids:
        raise ValueError("Enter at least one NORAD catalogue number.")
    for i in ids:
        if not i.isdigit():
            raise ValueError(f"'{i}' is not a NORAD catalogue number.")
    info, ptexts, otexts = [], [], []
    for i in ids:
        t, inf = fetch_celestrak(f"CATNR={i}", force=force); ptexts.append(t); info.append(inf)
    for g in groups:
        t, inf = fetch_celestrak(f"GROUP={g}", force=force); otexts.append(t); info.append(inf)
    prims, objs = [], []
    for t in ptexts:
        prims += parse_orbit_text(t)
    for t in otexts:
        objs += parse_orbit_text(t)
    if not prims:
        raise ValueError("CelesTrak returned no usable data for those NORAD IDs.")
    ids_set = {p.satnum for p in prims}
    objs = [o for o in objs if o.satnum not in ids_set]
    for o in objs:
        o.obj_type = guess_type(o.name)
        o.country = guess_country(o.name, o.obj_type)
    for p in prims:
        p.obj_type = "PAYLOAD"
        p.country = guess_country(p.name, "PAYLOAD")
    return prims, objs, info
