"""Validation tests. Run:  python -m pytest -q   (or: python tests/test_engine.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stc.propagation.sgp4_prop import Satellite, format_tle, parse_tle_text
from stc.propagation.numerical import propagate
from stc.ingestion.tle_loader import load_sample
from stc.screening.conjunction import screen_catalogue
from stc.manoeuvre.planner import build_manoeuvre, miss_near, suggest_minimum_dv
from stc.comms.visibility import STATIONS, compare_passes

L1 = "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985"
L2 = "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774"


def test_sgp4_reference_case_06251():
    """Vallado SGP4 verification output (tcppver.out) for NORAD 06251."""
    s = Satellite("06251", L1, L2)
    r, v = s.sgp4([0.0, 120.0])
    assert np.allclose(r[0], [3988.31022699, 5498.96657235, 0.90055879], atol=1e-6)
    assert np.allclose(v[0], [-3.290032738, 2.357652820, 6.496623475], atol=1e-8)
    assert np.allclose(r[1], [-3935.69800083, 409.10980837, 5471.33577327], atol=1e-5)


def test_tle_roundtrip():
    s = Satellite("06251", L1, L2)
    a, b = format_tle(90001, s.jdepoch, 1.2808e-4, 58.0579, 54.0425, 0.0030035, 139.1568, 221.1854, 15.56387291)
    s2 = parse_tle_text(f"X\n{a}\n{b}")[0]
    assert np.linalg.norm(s.sgp4(500.0)[0] - s2.sgp4(500.0)[0]) < 1e-6


def test_numerical_vs_sgp4_short_arc():
    s = Satellite("06251", L1, L2)
    r0, v0 = s.at1(s.jdepoch)
    tr = propagate(r0, v0, s.jdepoch, s.jdepoch + 3 / 24)
    p, _ = tr.sample([s.jdepoch + 1.5 / 24])
    assert np.linalg.norm(p[0, 0] - s.at1(s.jdepoch + 1.5 / 24)[0]) < 2.0  # km, drag + J3/J4 not modelled


def test_sample_screening_finds_engineered_events():
    prims, objs, start = load_sample()
    ev, ncand = screen_catalogue(prims[0], objs, start, start + 3, threshold_km=5)
    misses = sorted(round(e.miss, 2) for e in ev)
    assert ncand > 50
    assert any(abs(m - 0.08) < 0.01 for m in misses)       # engineered 80 m event at T+44.1 h
    assert any(abs(m - 0.35) < 0.02 for m in misses)       # engineered 350 m event at T+6.2 h
    assert ev[0].risk == "HIGH"


def test_manoeuvre_zero_dv_preserves_baseline_and_dv_increases_miss():
    prims, objs, start = load_sample()
    ev, _ = screen_catalogue(prims[0], objs, start, start + 3, threshold_km=5)
    e = min(ev, key=lambda x: x.miss)
    m0 = build_manoeuvre(prims[0], e.jd, 1.5, [0, 0, 0])
    _, miss0 = miss_near(m0.positions, e.obj, e.jd)
    assert abs(miss0 - e.miss) < 1e-3
    sug = suggest_minimum_dv(prims[0], e, 1.5, 2.0)
    assert sug is not None and sug["miss"] >= 2.0 and abs(sum(sug["dv_rtn"])) <= 0.2


def test_passes_shift_small_for_small_dv():
    prims, objs, start = load_sample()
    sat = prims[0]
    e_jd = start + 44.1 / 24
    m = build_manoeuvre(sat, e_jd, 1.5, [0, 0.1, 0], horizon_jd=e_jd + 1)
    rows, summ = compare_passes(sat.at, m.positions, STATIONS[:4], m.jd_burn, m.jd_burn + 1, 5.0)
    assert summ["n_before"] > 5 and summ["max_shift_s"] < 30


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
