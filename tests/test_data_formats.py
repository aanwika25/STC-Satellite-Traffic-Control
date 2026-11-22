"""OMM (JSON/CSV/XML) parsing and the CelesTrak cache/retry rules. Run: python -m pytest -q"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.common.time_utils import datetime_from_jd
from stc.ingestion import tle_loader
from stc.propagation.sgp4_prop import Satellite, detect_format, parse_orbit_text

L1 = "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985"
L2 = "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774"
REF = Satellite("DELTA 1 DEB", L1, L2)


def omm_record(satnum="6251", name="DELTA 1 DEB"):
    e = REF.elements()
    return {"OBJECT_NAME": name, "OBJECT_ID": "1962-025E",
            "EPOCH": datetime_from_jd(REF.jdepoch).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "MEAN_MOTION": f"{e['n']:.8f}", "ECCENTRICITY": f"{e['ecc']:.7f}", "INCLINATION": f"{e['inc']:.4f}",
            "RA_OF_ASC_NODE": f"{e['raan']:.4f}", "ARG_OF_PERICENTER": f"{e['argp']:.4f}", "MEAN_ANOMALY": f"{e['ma']:.4f}",
            "EPHEMERIS_TYPE": "0", "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": satnum, "ELEMENT_SET_NO": "999",
            "REV_AT_EPOCH": "677", "BSTAR": f"{e['bstar']:.8g}", "MEAN_MOTION_DOT": "0.00008885", "MEAN_MOTION_DDOT": "0"}


def same_orbit(s):
    r0, _ = REF.sgp4(np.array([0.0, 360.0]))
    r1, _ = s.sgp4(np.array([0.0, 360.0]))
    return np.max(np.linalg.norm(r0 - r1, axis=1)) < 0.01   # km (epoch ISO string rounds to microseconds)


def test_omm_json():
    text = json.dumps([omm_record()])
    assert detect_format(text) == "json"
    [s] = parse_orbit_text(text)
    assert s.name == "DELTA 1 DEB" and same_orbit(s)


def test_omm_csv():
    rec = omm_record()
    text = ",".join(rec) + "\n" + ",".join(rec.values())
    assert detect_format(text) == "csv"
    [s] = parse_orbit_text(text)
    assert same_orbit(s)


def test_omm_xml():
    rec = omm_record()
    body = "".join(f"<{k}>{v}</{k}>" for k, v in rec.items())
    text = f'<?xml version="1.0"?><ndm><omm><body><segment><data><meanElements>{body}</meanElements></data></segment></body></omm></ndm>'
    assert detect_format(text) == "xml"
    [s] = parse_orbit_text(text)
    assert same_orbit(s)


def test_six_digit_catalogue_number():
    [s] = parse_orbit_text(json.dumps([omm_record(satnum="270123", name="ANALYST OBJECT")]))
    assert s.satnum == "270123" and same_orbit(s)


def test_tle_still_supported():
    [s] = parse_orbit_text(f"DELTA 1 DEB\n{L1}\n{L2}")
    assert same_orbit(s)


class _Resp(io.BytesIO):
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_cache_and_retry_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(tle_loader, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tle_loader.time, "sleep", lambda s: None)
    calls = []
    payload = json.dumps([omm_record()]).encode()

    def ok(req, timeout=0):
        calls.append(req.full_url); return _Resp(payload)
    monkeypatch.setattr(tle_loader.urllib.request, "urlopen", ok)
    t1, i1 = tle_loader.fetch_celestrak("CATNR=6251")
    t2, i2 = tle_loader.fetch_celestrak("CATNR=6251")
    assert i1["source"] == "download" and i2["source"] == "cache" and len(calls) == 1
    assert calls[0] == "https://celestrak.org/NORAD/elements/gp.php?CATNR=6251&FORMAT=JSON"

    # 403 (blocked): no retries, fall back to the saved copy
    def blocked(req, timeout=0):
        calls.append(req.full_url); raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
    monkeypatch.setattr(tle_loader.urllib.request, "urlopen", blocked)
    n = len(calls)
    t3, i3 = tle_loader.fetch_celestrak("CATNR=6251", force=True)
    assert len(calls) == n + 1 and i3["source"] == "stale cache" and t3 == t1

    # network down: at most 3 attempts, then an error when there is no saved copy
    def down(req, timeout=0):
        calls.append(req.full_url); raise urllib.error.URLError("offline")
    monkeypatch.setattr(tle_loader.urllib.request, "urlopen", down)
    n = len(calls)
    try:
        tle_loader.fetch_celestrak("GROUP=cosmos-1408-debris")
        assert False, "expected an error"
    except tle_loader.CelesTrakError:
        pass
    assert len(calls) == n + 3


def test_no_gp_data_message_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tle_loader, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tle_loader.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"No GP data found"))
    try:
        tle_loader.fetch_celestrak("CATNR=99999999")
        assert False
    except tle_loader.CelesTrakError as e:
        assert "No GP data" in str(e)


# ---------------------------------------------------------------- SpaceObjectsCatalog XML (ECI coordinates)
from stc.ingestion.space_objects_xml import parse_space_objects_xml  # noqa: E402

def _catalog(r, v, alt, inc, e):
    return f"""<?xml version="1.0"?><SpaceObjectsCatalog><SpaceObject noradId="70001" orbitCategory="LEO" status="Active">
      <ObjectName>TEST SAT</ObjectName>
      <PositionCoordinates referenceFrame="ECI"><X_km>{r[0]}</X_km><Y_km>{r[1]}</Y_km><Z_km>{r[2]}</Z_km></PositionCoordinates>
      <VelocityCoordinates referenceFrame="ECI"><Vx_kms>{v[0]}</Vx_kms><Vy_kms>{v[1]}</Vy_kms><Vz_kms>{v[2]}</Vz_kms></VelocityCoordinates>
      <KeplerianElements><Inclination_deg>{inc}</Inclination_deg><Eccentricity>{e}</Eccentricity><RightAscension_deg>10</RightAscension_deg>
      <MeanAnomaly_deg>20</MeanAnomaly_deg><OperatingAltitude_km>{alt}</OperatingAltitude_km></KeplerianElements>
      <Timestamp>2026-11-22T12:00:00Z</Timestamp></SpaceObject></SpaceObjectsCatalog>"""


def test_consistent_state_vector_is_used():
    jd = 2461002.0  # 2026-11-22 12:00 UTC
    r, v = REF.at1(REF.jdepoch)
    sats, notes = parse_space_objects_xml(_catalog(r.round(6), v.round(9), 330, 58.06, 0.003).replace(
        "2026-11-22T12:00:00Z", datetime_from_jd(REF.jdepoch).strftime("%Y-%m-%dT%H:%M:%S.%f")))
    assert len(sats) == 1 and "used the ECI position" in notes[0]
    assert np.linalg.norm(sats[0].at1(sats[0].jdepoch)[0] - r) < 0.01


def test_inconsistent_state_vector_falls_back_to_keplerian():
    path = Path(__file__).resolve().parents[1] / "data" / "LEO_MEO_example.xml"
    sats, notes = parse_space_objects_xml(path.read_text())
    assert [s.regime for s in sats] == ["LEO", "MEO"]
    assert sum("does not match" in n for n in notes) == 2
    iss = sats[0]
    assert 410 < iss.perigee_km < 430 and abs(np.degrees(iss.inclo) - 51.64) < 0.01
