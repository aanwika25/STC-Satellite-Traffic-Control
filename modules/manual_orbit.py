"""
Manual mode: build an orbit from plain orbital elements for hypothetical scenarios.

Altitude is treated as the mean orbital altitude above the equatorial radius, so the
mean motion is derived from a circular-orbit approximation. The resulting object is
a real SGP4 propagator, so every calculator works on it unchanged.
"""
import math
from datetime import datetime, timezone

from sgp4.api import Satrec, WGS72

from modules.geometry import MU_KM3_S2, RE_KM

JD_1949_DEC31 = 2433281.5


def epoch_days_1949(dt_utc):
    """Days since 1949-12-31 00:00 UT, the epoch convention used by sgp4init."""
    jd = dt_utc.timestamp() / 86400.0 + 2440587.5
    return jd - JD_1949_DEC31


def satrec_from_elements(altitude_km, eccentricity, inclination_deg, raan_deg,
                         arg_perigee_deg, mean_anomaly_deg, epoch_utc, bstar=0.0,
                         satnum=0):
    if not 0.0 <= eccentricity < 0.9:
        raise ValueError("Eccentricity must be between 0 and 0.9.")
    if altitude_km <= 0:
        raise ValueError("Altitude must be positive.")
    if not 0.0 <= inclination_deg <= 180.0:
        raise ValueError("Inclination must be between 0 and 180 degrees.")

    a = RE_KM + altitude_km
    no_kozai = math.sqrt(MU_KM3_S2 / a ** 3) * 60.0   # rad/min
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", satnum, epoch_days_1949(epoch_utc), bstar, 0.0, 0.0,
        eccentricity,
        math.radians(arg_perigee_deg),
        math.radians(inclination_deg),
        math.radians(mean_anomaly_deg),
        no_kozai,
        math.radians(raan_deg),
    )
    if sat.error != 0:
        raise ValueError(f"SGP4 could not build this orbit (error code {sat.error}).")
    return sat


def default_epoch():
    """Midnight UTC today, so the epoch is always a clean reference time."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)
