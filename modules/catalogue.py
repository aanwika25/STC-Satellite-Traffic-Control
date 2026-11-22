"""
Automated mode: read satellite data and default settings from the project's data/ folder.

Put TLE catalogues (.txt or .tle) in data/ and edit data/settings.json for the
default ground station, radio and simulation values.
"""
import json
from pathlib import Path

from modules.tle_loader import parse_tle_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "ground_station": {"latitude_deg": 13.7199, "longitude_deg": 80.2304, "altitude_km": 0.010},   # Sriharikota
    "radio": {"carrier_frequency_mhz": 145.8, "minimum_elevation_deg": 10.0},
    "simulation": {"duration_hours": 24.0, "time_step_seconds": 10.0, "screening_distance_km": 10.0},
}


def list_catalogue_files():
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in (".txt", ".tle"))


def read_catalogue(filename):
    """Return the TLE records (name, line1, line2) from a file in data/."""
    text = (DATA_DIR / filename).read_text(encoding="utf-8-sig")
    return parse_tle_text(text)


def load_settings():
    """Default settings from data/settings.json, falling back to built-in defaults."""
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for section, values in stored.items():
                if section in settings and isinstance(values, dict):
                    settings[section].update(values)
            # Older settings files stored the station height in metres
            gs = settings["ground_station"]
            if "altitude_m" in gs and "altitude_km" not in gs:
                gs["altitude_km"] = gs.pop("altitude_m") / 1000.0
            gs.pop("altitude_m", None)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not read data/settings.json: {exc}")
    return settings
