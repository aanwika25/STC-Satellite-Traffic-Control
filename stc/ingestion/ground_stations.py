"""Ground-station list loaded from data/ground_stations.xml (WGS-84)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

STATIONS_XML = Path(__file__).resolve().parents[2] / "data" / "ground_stations.xml"

_FALLBACK = [
    {"id": "ISTRAC-BENGALURU", "name": "ISTRAC Bengaluru", "short": "Bengaluru", "lat": 13.0358, "lon": 77.5970, "alt": 0.9,
     "type": "Telemetry Tracking and Command", "city": "Bengaluru", "notes": ""},
]


def _num(el, tag, default=0.0):
    node = el.find(tag)
    try:
        return float(node.text) if node is not None and node.text else default
    except ValueError:
        return default


def load_ground_stations(path: Path = STATIONS_XML) -> list[dict]:
    """Parse the ground-station XML. Elevation is converted from metres to km."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return list(_FALLBACK)
    out = []
    for gs in root.findall("groundStation"):
        c = gs.find("coordinates")
        if c is None:
            continue
        name = (gs.findtext("name") or gs.get("id") or "Station").strip()
        short = (gs.findtext("shortName") or gs.findtext("city") or name).strip()
        elev = c.find("elevation")
        elev_m = float(elev.text) if elev is not None and elev.text else 0.0
        if elev is not None and elev.get("unit", "m").lower() in ("km", "kilometres", "kilometers"):
            elev_m *= 1000
        out.append({
            "id": gs.get("id", name), "name": name, "short": short,
            "lat": _num(c, "latitude"), "lon": _num(c, "longitude"), "alt": elev_m / 1000.0,
            "type": (gs.findtext("type") or "").strip(), "city": (gs.findtext("city") or "").strip(),
            "operator": (gs.findtext("operator") or "").strip(), "status": (gs.findtext("status") or "").strip(),
            "notes": (gs.findtext("notes") or "").strip(),
        })
    return out or list(_FALLBACK)
