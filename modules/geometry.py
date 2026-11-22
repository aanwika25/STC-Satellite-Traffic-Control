"""Shared frame, time and look-angle helpers.

Frames
------
SGP4 returns states in the TEME frame. Here TEME is treated as Earth-centred
inertial (ECI), which is the usual approximation in simple tracking tools.
Ground stations are rotated into ECI with Greenwich Mean Sidereal Time (GMST).
"""
import numpy as np
from skyfield.api import load

RE_KM = 6378.137                 # WGS-84 equatorial radius (km)
MU_KM3_S2 = 398600.4418          # Earth gravitational parameter (km^3/s^2)
J2 = 1.08263e-3                  # Earth oblateness coefficient
WGS84_F = 1.0 / 298.257223563    # WGS-84 flattening
OMEGA_EARTH = 7.2921150e-5       # Earth rotation rate (rad/s)
C_KM_S = 299792.458              # Speed of light (km/s)

_ts = load.timescale()


def utc_string(jd):
    """Readable UTC time for a single Julian date."""
    return _ts.ut1_jd(float(jd)).utc_strftime("%Y-%m-%d %H:%M:%S")


def gmst_rad(jd):
    """GMST in radians for an array of Julian dates."""
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    hours = np.asarray(_ts.ut1_jd(jd).gmst, dtype=float)
    return np.radians(hours * 15.0)


def rotate_z(vec, theta):
    """Rotate an (N, 3) array of vectors about the z-axis by theta (N,) radians."""
    c, s = np.cos(theta), np.sin(theta)
    x = c * vec[:, 0] - s * vec[:, 1]
    y = s * vec[:, 0] + c * vec[:, 1]
    return np.column_stack([x, y, vec[:, 2]])


def station_eci(station, jd):
    """Position (km) and velocity (km/s) of a fixed ground station in ECI."""
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    lat = np.radians(station["latitude_deg"])
    lon = np.radians(station["longitude_deg"])
    h_km = station["altitude_km"]
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = RE_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    ecef = np.array([
        (n + h_km) * np.cos(lat) * np.cos(lon),
        (n + h_km) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - e2) + h_km) * np.sin(lat),
    ])
    r = rotate_z(np.tile(ecef, (len(jd), 1)), gmst_rad(jd))
    v = np.column_stack([-OMEGA_EARTH * r[:, 1], OMEGA_EARTH * r[:, 0], np.zeros(len(jd))])
    return r, v


def look_angles(station, jd, sat_r, sat_v):
    """
    Elevation (deg), azimuth (deg), range (km) and range rate (km/s)
    of a satellite as seen from a ground station.
    Range rate is positive when the satellite is moving away.
    """
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    theta = gmst_rad(jd)
    sta_r, sta_v = station_eci(station, jd)

    rho = sat_r - sta_r
    rho_dot = sat_v - sta_v
    rng = np.linalg.norm(rho, axis=1)

    # Express the line of sight in the station's East-North-Up frame
    x, y, z = rotate_z(rho, -theta).T
    lat = np.radians(station["latitude_deg"])
    lon = np.radians(station["longitude_deg"])
    east = -np.sin(lon) * x + np.cos(lon) * y
    north = (-np.sin(lat) * np.cos(lon) * x - np.sin(lat) * np.sin(lon) * y
             + np.cos(lat) * z)
    up = (np.cos(lat) * np.cos(lon) * x + np.cos(lat) * np.sin(lon) * y
          + np.sin(lat) * z)

    el = np.degrees(np.arcsin(np.clip(up / rng, -1.0, 1.0)))
    az = np.degrees(np.arctan2(east, north)) % 360.0
    rr = np.einsum("ij,ij->i", rho, rho_dot) / rng
    return el, az, rng, rr


def subpoint(jd, sat_r):
    """Geocentric latitude and longitude (deg) of the satellite's ground track."""
    theta = gmst_rad(jd)
    x, y, z = rotate_z(sat_r, -theta).T
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return lat, lon
