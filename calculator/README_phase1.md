# STC — Satellite Traffic Control (Phase 1)

This is the first phase of the original STC calculator project. It keeps the simple two-object workflow and adds two ways to provide TLE data:
- Paste TLE lines manually.
- Upload a standard `.txt` or `.tle` file containing multiple TLE records.

It validates two orbital objects (including TLE checksums), ground-station settings, radio settings and simulation settings. It does not yet calculate orbital propagation, encounters or Doppler shift.

## Setup on Windows

Open this folder in VS Code. In the VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

## TLE file format

A standard TLE text file can contain repeated records:

```text
ISS (ZARYA)
1 ...
2 ...
OBJECT NAME
1 ...
2 ...
```

Each numbered TLE line (line 1 and line 2) must be copied in full, with no truncation. Do not use the shortened examples above as real data.

## Calculation modules

Section 4 of the app unlocks once a valid Phase 1 configuration is loaded.

| # | Module | File | Answers |
|---|--------|------|---------|
| 1 | Orbital mechanics | `modules/orbital_mechanics.py` | Position, speed, altitude and ground track over the next few minutes |
| 2 | Collision and close approach | `modules/collision.py` | Miss distance, time of closest approach, relative speed |
| 3 | Collision avoidance | `modules/avoidance.py` | Effect of a burn, and the smallest burn that reaches a target miss distance |
| 4 | Doppler shift | `modules/doppler.py` | Frequency change at the ground station over a pass |
| 5 | Communication impact | `modules/communication.py` | Visibility, Doppler and receiver correction with and without the manoeuvre |

Shared frame and look-angle code is in `modules/geometry.py`.

### Assumptions and limits

- SGP4 positions are in TEME, which is treated as inertial. Ground stations are rotated into that frame using GMST.
- Manoeuvres use the delta-propagation method: the burn's effect is computed with two-body plus J2 gravity and added to the SGP4 prediction. This keeps SGP4's drag and other effects in the baseline.
- Burns are instantaneous and given in the RTN frame.
- A burn cannot change an encounter that has already happened. Closest approach after the burn is therefore reported separately from the closest approach before it.
- Doppler uses non-relativistic formulas.
- The receiver correction assumes the ground station pre-compensates using the no-manoeuvre prediction.
- Use verified station coordinates for meaningful pass predictions.

## Pages

The app is split into one page per phase. Use the sidebar to move between them.

| Page | File | Contents |
|------|------|----------|
| Home (Phase 1) | `app.py` | Automated or Manual input, validation |
| Phase 2 | `pages/2_Phase_2_Orbital_Propagation.py` | Orbit parameters, X/Y/Z, Vx/Vy/Vz, altitude and speed |
| Phase 3 | `pages/3_Phase_3_Encounter_Screening.py` | Closest approach, screening check, hypothetical manoeuvre and delta-v search |
| Phase 4 | `pages/4_Phase_4_Doppler_Ground_Station.py` | Elevation, azimuth, visibility, Doppler, transmit pre-correction, communication windows |
| Phase 5 | `pages/5_Phase_5_Scenarios_and_Dashboard.py` | Side-by-side comparison of up to three named manoeuvres |

Phase 1 must be completed on the Home page before the other pages show results. The manoeuvre set in Phase 3 is reused by Phases 4 and 5.

Shared code lives in `modules/ui_inputs.py` (Phase 1 widgets) and `modules/ui_state.py` (session state).

Not yet built: Hohmann transfers, plane-change calculations, traffic analysis across many objects, and report export.
