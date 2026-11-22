"""Add country tags, extra Indian/international LEO satellites and MEO objects to the sample dataset.

Run once:  python scripts/extend_sample_data.py
All added objects are synthetic (not real TLEs).
"""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stc.propagation.sgp4_prop import format_tle  # noqa: E402

MU, RE = 398600.4418, 6378.137
path = ROOT / "data" / "sample_dataset.json"
d = json.loads(path.read_text())
EPOCH = d["epochJd"]
rng = random.Random(42)


def rev_per_day(alt_km):
    a = RE + alt_km
    return math.sqrt(MU / a ** 3) * 86400 / (2 * math.pi)


for p in d["primaries"]:
    p["country"] = "IN"
for o in d["objects"]:
    o.setdefault("country", "INTL" if o["type"] == "INACTIVE SAT" else "")
d["objects"] = [o for o in d["objects"] if not o["name"].startswith(("IND SAT", "INTL SAT", "INTL NAV", "SYN R/B M"))]

num = 93001


def add(name, typ, country, alt, inc, raan, ecc=0.0005, argp=0.0, ma=0.0, bstar=1e-5, rcs="LARGE"):
    global num
    l1, l2 = format_tle(num, EPOCH, bstar, inc, raan, ecc, argp, ma, rev_per_day(alt), intl="26" + f"{num % 1000:03d}" + "C")
    d["objects"].append({"name": name, "type": typ, "rcs": rcs, "country": country, "l1": l1, "l2": l2})
    num += 1


# Indian LEO satellites (sample)
for i in range(10):
    shell = rng.choice([(97.4, 98.6, 480, 820), (97.4, 98.6, 480, 820), (19.5, 20.5, 620, 680), (36.9, 37.1, 520, 560)])
    add(f"IND SAT S{i + 1:02d}", "PAYLOAD", "IN", rng.uniform(shell[2], shell[3]), rng.uniform(shell[0], shell[1]),
        rng.uniform(0, 360), argp=rng.uniform(0, 360), ma=rng.uniform(0, 360))
# International LEO satellites (sample)
for i in range(40):
    if i < 15:
        alt, inc = rng.uniform(540, 560), 53.0
    elif i < 30:
        alt, inc = rng.uniform(500, 700), rng.uniform(97.3, 98.2)
    else:
        alt, inc = rng.uniform(1190, 1210), 87.9
    add(f"INTL SAT L{i + 1:02d}", "PAYLOAD", "INTL", alt, inc, rng.uniform(0, 360), argp=rng.uniform(0, 360), ma=rng.uniform(0, 360))
# MEO navigation constellations (sample, GPS-/Galileo-/GLONASS-like geometry)
for tag, alt, inc, planes, per_plane in [("G", 20180, 55.0, 6, 4), ("E", 23222, 56.0, 3, 4), ("R", 19130, 64.8, 3, 4)]:
    k = 1
    for p in range(planes):
        for s in range(per_plane):
            add(f"INTL NAV-{tag} {k:02d}", "PAYLOAD", "INTL", alt, inc, p * 360 / planes,
                ecc=0.002, argp=rng.uniform(0, 360), ma=s * 360 / per_plane + p * 15, bstar=0.0)
            k += 1
# MEO rocket bodies / debris (eccentric transfer-type orbits)
for i in range(8):
    add(f"SYN R/B M{i + 1:02d}", "ROCKET BODY", "", rng.uniform(17000, 21000), rng.uniform(50, 66), rng.uniform(0, 360),
        ecc=rng.uniform(0.02, 0.12), argp=rng.uniform(0, 360), ma=rng.uniform(0, 360), bstar=0.0)

path.write_text(json.dumps(d))
print("objects:", len(d["objects"]))
