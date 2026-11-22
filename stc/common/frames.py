"""Reference-frame conversions: TEME/ECI <-> ECEF, geodetic, topocentric, RTN.

SGP4 outputs TEME. For this prototype TEME is treated as the inertial frame and
the ECEF rotation uses GMST only (polar motion and equation of equinoxes ignored,
error < ~30 m), which is ample for screening and visibility.
"""
from __future__ import annotations

import numpy as np

from .constants import RE, F_WGS84, DEG
from .time_utils import gmst

E2 = F_WGS84 * (2.0 - F_WGS84)


def eci_to_ecef(r, jd):
    """r: (...,3) km; jd: scalar or (...) array."""
    r = np.asarray(r, dtype=float)
    g = gmst(jd)
    c, s = np.cos(g), np.sin(g)
    x = c * r[..., 0] + s * r[..., 1]
    y = -s * r[..., 0] + c * r[..., 1]
    return np.stack([x, y, r[..., 2]], axis=-1)


def ecef_to_eci(p, jd):
    p = np.asarray(p, dtype=float)
    g = gmst(jd)
    c, s = np.cos(g), np.sin(g)
    return np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1)


def ecef_to_geodetic(p):
    """Returns (lat_deg, lon_deg, alt_km) arrays. WGS-84, iterative."""
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rxy = np.hypot(x, y)
    lat = np.arctan2(z, rxy)
    h = np.zeros_like(lat)
    for _ in range(6):
        sl = np.sin(lat)
        n = RE / np.sqrt(1 - E2 * sl * sl)
        h = rxy / np.cos(lat) - n
        lat = np.arctan2(z, rxy * (1 - E2 * n / (n + h)))
    return lat / DEG, lon / DEG, h


def geodetic_to_ecef(lat_deg, lon_deg, h_km=0.0):
    lat, lon = np.asarray(lat_deg) * DEG, np.asarray(lon_deg) * DEG
    sl = np.sin(lat)
    n = RE / np.sqrt(1 - E2 * sl * sl)
    return np.stack([(n + h_km) * np.cos(lat) * np.cos(lon),
                     (n + h_km) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2) + h_km) * sl], axis=-1)


def look_angles(lat_deg, lon_deg, h_km, sat_ecef):
    """Elevation, azimuth (deg) and range (km) from a ground station to satellite ECEF position(s)."""
    st = geodetic_to_ecef(lat_deg, lon_deg, h_km)
    rho = np.asarray(sat_ecef) - st
    lat, lon = lat_deg * DEG, lon_deg * DEG
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    south = sl * co * rho[..., 0] + sl * so * rho[..., 1] - cl * rho[..., 2]
    east = -so * rho[..., 0] + co * rho[..., 1]
    zen = cl * co * rho[..., 0] + cl * so * rho[..., 1] + sl * rho[..., 2]
    rng = np.linalg.norm(rho, axis=-1)
    el = np.degrees(np.arcsin(zen / rng))
    az = np.mod(np.degrees(np.arctan2(east, -south)), 360.0)
    return el, az, rng


def rtn_basis(r, v):
    """Radial, transverse (along-track), normal unit vectors for a single state."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    N = h / np.linalg.norm(h)
    T = np.cross(N, R)
    return R, T, N


def to_rtn(vec, r, v):
    R, T, N = rtn_basis(r, v)
    vec = np.asarray(vec, float)
    return np.array([vec @ R, vec @ T, vec @ N])


def from_rtn(comp, r, v):
    R, T, N = rtn_basis(r, v)
    return comp[0] * R + comp[1] * T + comp[2] * N


def rv_to_coe(r, v, mu):
    """Osculating classical elements: a(km), e, i, raan, argp, M (rad)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rm, vm = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hm = np.linalg.norm(h)
    n = np.array([-h[1], h[0], 0.0])
    nm = np.linalg.norm(n)
    ev = ((vm * vm - mu / rm) * r - (r @ v) * v) / mu
    e = np.linalg.norm(ev)
    a = 1.0 / (2.0 / rm - vm * vm / mu)
    i = np.arccos(h[2] / hm)
    raan = np.arccos(np.clip(n[0] / nm, -1, 1))
    if n[1] < 0:
        raan = 2 * np.pi - raan
    argp = np.arccos(np.clip(n @ ev / (nm * e), -1, 1))
    if ev[2] < 0:
        argp = 2 * np.pi - argp
    nu = np.arccos(np.clip(ev @ r / (e * rm), -1, 1))
    if r @ v < 0:
        nu = 2 * np.pi - nu
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    M = np.mod(E - e * np.sin(E), 2 * np.pi)
    return dict(a=a, e=e, i=i, raan=raan, argp=argp, M=M)


def coverage_half_angle(alt_km: float, mask_deg: float) -> float:
    """Earth-central half-angle (rad) of a station's visibility circle for a satellite at alt_km."""
    e = mask_deg * DEG
    return float(np.arccos(RE / (RE + alt_km) * np.cos(e)) - e)


def circle_points(lat_deg, lon_deg, half_angle_rad, n=73):
    """Lat/lon points of a small circle on the sphere (for coverage footprints)."""
    la, lo = lat_deg * DEG, lon_deg * DEG
    b = np.linspace(0, 2 * np.pi, n)
    a = half_angle_rad
    la2 = np.arcsin(np.sin(la) * np.cos(a) + np.cos(la) * np.sin(a) * np.cos(b))
    lo2 = lo + np.arctan2(np.sin(b) * np.sin(a) * np.cos(la), np.cos(a) - np.sin(la) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540) % 360 - 180
