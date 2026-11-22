# STC: Satellite Traffic Control

Team TRISOL: Amit Jadhav, Aanwika Mishra, Palak Tapadiya

Conjunction screening, collision avoidance and link-visibility analysis for LEO satellites.
ASTRA 2026 (Anterix × VIT Bhopal) - Domain 06 Sustainable Space, with Domain 04 Satellite Communication.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the validation tests:

```bash
python -m pytest -q
```

## Calculator page (Phases 1-5)

**5 · Calculator** in the Go to menu is the team's STC Phase 1 calculator (two-object workflow), merged
into the dashboard. When you open it, a **Calculator section** menu appears in the sidebar (and only then):
Phase 1 input and configuration, Phase 2 orbital propagation, Phase 3 encounter screening and manoeuvres,
Phase 4 Doppler and ground station, Phase 5 scenario comparison. Its code is in `modules/` (calculation
modules, unchanged) and `calculator/` (each phase page wrapped as a section, plus `overlay.py`).

Phase 1's **Dashboard catalogue** source picks Object A from every satellite and Object B from the debris
and rocket bodies (or any object) in the data loaded in the sidebar - searchable by name or NORAD number.
A shortcut button copies the close approach selected on the Collision Stats page into A and B and sizes
the simulation window, Phase 3 burn time and analysis window around the encounter.

Everything the calculator computes is drawn on the map and globe - at the bottom of every calculator
section and on the Path Tracer page: objects A and B with their tracks, the closest approach (✕), the
Phase 3 manoeuvred trajectory, one trajectory per Phase 5 scenario, and the calculator's ground station.
Object B blinks gold when the closest approach is inside the screening distance. The calculator uses its
objects' own epoch, so on the Path Tracer press **Go to calculator time**. The calculator needs `sgp4`,
`skyfield` and `scipy` (in `requirements.txt`).

## Using the dashboard

The sidebar has a page menu: **Overview** (summary + a button for each module) and the four module pages.
Every page also has Back / Next buttons at the bottom. Values you set (selected event, Δv, stations, mask)
are kept when you move between pages.

Run it with `python -m streamlit run app.py` from this folder (not with the editor's Run ▶ button).

## The four modules

| # | Module | Question it answers |
|---|---|---|
| 1 | Path Tracer | Where is the satellite, how fast is it moving, where will it be next? Live map + rotating globe of every tracked object (LEO / MEO / All filter). |
| 2 | Collision Stats | How close will it come to another satellite or debris object, and when? |
| 3 | Alternate Path Mapping | What happens if it changes velocity? How much Δv is needed? |
| 4 | New Path Stats | How does the manoeuvre change ground-station visibility? |

## Project layout

```
stc_dashboard/
├── app.py                       # Streamlit dashboard (UI only)
├── data/sample_dataset.json     # sample satellites + synthetic objects (TLEs)
├── stc/
│   ├── common/
│   │   ├── constants.py         # WGS-72 / WGS-84 constants
│   │   ├── time_utils.py        # Julian dates, GMST, formatting
│   │   └── frames.py            # TEME->ECEF, geodetic, look angles, RTN, elements
│   ├── ingestion/
│   │   └── tle_loader.py        # sample data, pasted TLEs, CelesTrak fetch
│   ├── propagation/
│   │   ├── sgp4_prop.py         # SGP4 near-Earth, vectorised with NumPy
│   │   └── numerical.py         # two-body + J2 RK4 integrator
│   ├── screening/
│   │   └── conjunction.py       # altitude filter, coarse grid, TCA refinement
│   ├── risk/
│   │   └── probability_of_collision.py
│   ├── manoeuvre/
│   │   └── planner.py           # Δv in RTN, minimum-Δv search, secondary check
│   └── comms/
│       └── visibility.py        # ground-station passes, before vs after
└── tests/test_engine.py         # SGP4 reference case, screening, manoeuvre, passes
```

The layout follows the same split as the larger `autoca` design (common / ingestion / propagation /
risk / manoeuvre), so CDM ingestion, covariance fusion and the ML models can be added later as new
packages without changing the dashboard.

## Method

- **Propagation:** SGP4 (Vallado et al. 2006, "improved" mode, WGS-72), implemented from scratch and
  vectorised over time. Verified against the published reference output for NORAD 06251
  (`tests/test_engine.py`). Deep-space objects (period ≥ 225 min) are skipped.
- **Screening:** perigee/apogee overlap filter → 60 s grid → range-rate sign change marks each local
  minimum → 0.5 s dense sampling + golden-section search gives TCA to ~10 ms.
- **Collision probability:** simplified isotropic 2-D Gaussian,
  `Pc ≈ HBR²/(2σ²)·exp(−d²/(2σ²))`, with an assumed combined σ. Screening-grade only.
- **Manoeuvre:** impulsive Δv in the RTN frame at a chosen lead time. SGP4 cannot propagate a modified
  state, so the effect is integrated numerically (two-body + J2, RK4, 10 s steps) for the baseline and
  manoeuvred states from the same starting point; the difference is added to the SGP4 baseline.
  With Δv = 0 the baseline is reproduced exactly.
- **New Path Stats (communication impact):** elevation above a mask for Indian ground stations (approximate
  coordinates), pass AOS/LOS found on a 10 s grid with interpolated crossings, passes matched before
  vs after by nearest AOS.

## Data

- **Space-Track catalogue** (default when `data/spacetrack/*.xml` exists): OMM XML downloaded from
  space-track.org - currently `leo.xml` (17,805 payloads) and `meo.xml` (426 payloads), 18,231 objects
  after removing duplicates. Object type, radar size and country come from Space-Track's own fields
  (`COUNTRY_CODE = IND` makes an object Indian/green). All 54 Indian LEO payloads are selectable as the
  satellite to protect (CARTOSAT 3 by default). The clock starts at the current time. To refresh, replace
  the files with new downloads. These files contain payloads only; add debris/rocket bodies by downloading
  them from Space-Track too. **Space-Track's user agreement restricts sharing its data**, so
  `data/spacetrack/` is in `.gitignore` - do not push it to a public repository.
- **Linked pages:** one pair of objects - A (the satellite) and B (the object it meets) - and one manoeuvre
  are shared by every page. Choosing the satellite in the sidebar, an event on Collision Stats, or Object A /
  Object B in Calculator Phase 1 changes it everywhere, and the Calculator is filled in automatically (window
  containing the encounter, Phase 3 burn 1.5 orbits before it). Alternate Path Mapping and New Path Stats use
  the Calculator Phase 3 manoeuvre (editing either one changes both) and show the original and alternate paths
  on the map and globe. See `calculator/link.py`.
- **Demo debris (synthetic):** `data/spacetrack/demo_debris_synthetic.xml` is a generated debris catalogue in
  the same Space-Track OMM XML format - 3,070 fragments and rocket bodies modelled on the real environment
  (Fengyun-1C, Cosmos 2251/Iridium 33, NOAA 16/17, DMSP F13 and CZ-6A clouds at their real inclinations,
  most debris at 750-1,000 km, SL-8/SL-16/CZ-4C/PSLV rocket bodies, GTO and MEO stages). It is NOT real
  data: catalogue numbers are 880001+, the header and every record say SYNTHETIC, and the dashboard labels
  these objects "synthetic demo". 18 fragments are placed to pass 0.1-3.5 km from Indian satellites between
  22 Nov and 12 Dec 2026 so demos show debris encounters. Regenerate (e.g. other dates) with
  `python scripts/make_demo_debris.py --start 2026-12-01T00:00 --days 21`; delete the file to use real data only.
- **Fast screening for big catalogues:** near-Earth objects are propagated in batches (`sgp4_batch.py`,
  identical to the single-object SGP4) on a 2-minute grid; a straight-line closest-approach estimate picks
  the passes worth refining. Verified against exhaustive 30-second screening (identical events).
- **Sample dataset** (synthetic, in the Data source menu): 3 sample Indian-style satellites (Cartosat-, RISAT- and Oceansat-class
  orbits) and 288 **synthetic** objects, epoch 22 Nov 2026. Eight close approaches are engineered in
  so every module has something to show. These are not real TLEs.
- **Ground stations:** read from `data/ground_stations.xml` (8 ISRO stations). Edit that file to add,
  remove or move stations; elevation is given in metres.
- **Colours on the Path Tracer:** Indian satellites green, international satellites sky blue, debris and
  rocket bodies black. For live TLEs, Indian satellites are recognised by name (CARTOSAT, RISAT, NVS…).
- **Live data:** paste TLE or OMM (JSON, CSV or XML - detected automatically), or use "CelesTrak" in the
  sidebar to fetch by NORAD ID and debris group (needs internet).
- **Your own coordinates:** "Add objects from a file" in the sidebar accepts TLE, OMM or a
  `<SpaceObjectsCatalog>` XML with ECI position/velocity and Keplerian parameters (see
  `data/LEO_MEO_example.xml`). The position/velocity is cross-checked against the stated orbit: if they
  agree it is used (fitted to SGP4 mean elements); if not, the Keplerian parameters are used and the
  sidebar explains the mismatch. The example file's vectors do not match their stated orbits (the ISS
  vector would put perigee below the surface), so its Keplerian parameters are used.
- **Path Tracer interaction:** hover an object for its name; click it to draw its path and check it
  against every other object - anything within the screening threshold blinks gold.
- **CelesTrak rules followed** ([GP data formats](https://celestrak.org/NORAD/documentation/gp-data-formats.php)):
  downloads use OMM JSON, which also covers catalogue numbers above 99999 that TLEs cannot hold; each query
  is saved in `data/cache/` and reused for 2 hours (GP data changes at most every 2 hours); only HTTP 200
  responses are accepted; at most 3 attempts, with no retry after a 403/404 block; if CelesTrak is
  unreachable the last saved copy is used and the sidebar says how old it is.

## MEO objects

Near-Earth SGP4 covers orbits shorter than 225 minutes. MEO objects (navigation satellites at ~20,000 km)
need SDP4, which is not implemented; they are propagated with a two-body Kepler model plus J2 drift of the
node and perigee. That is accurate enough for the map and globe, not for close-approach screening, and
screening only ever pairs objects whose altitude bands overlap with the LEO satellite.

## Limitations

TEME is treated as inertial and Earth rotation uses GMST only; no covariance from CDMs; drag and
J3/J4 are not in the manoeuvre perturbation model (they cancel to first order because only the
difference between two numerical runs is used). Suitable for a prototype and teaching, not for
operational decisions.
