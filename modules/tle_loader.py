from sgp4.api import Satrec
from modules.input_handler import validate_tle


def load_tle(name: str, line1: str, line2: str) -> dict:
    validate_tle(line1, line2)
    satellite = Satrec.twoline2rv(line1, line2)
    return {
        "name": name.strip() or f"Object {satellite.satnum}",
        "catalogue_number": satellite.satnum,
        "satrec": satellite,
        "line1": line1,
        "line2": line2
    }


def parse_tle_text(text: str):
    """Parse a standard 3-line name/line1/line2 TLE text file."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = f"Object {len(records) + 1}"
            line1, line2 = lines[i], lines[i + 1]
            records.append((name, line1, line2))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            records.append((lines[i], lines[i + 1], lines[i + 2]))
            i += 3
        else:
            raise ValueError(f"Unexpected TLE format near line: {lines[i][:30]}")
    return records
