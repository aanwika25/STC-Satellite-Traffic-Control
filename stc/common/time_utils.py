"""Julian dates, sidereal time and formatting helpers."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import DEG, TWO_PI, SECONDS_PER_DAY

JD_UNIX_EPOCH = 2440587.5


def jday(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: float = 0.0) -> float:
    """Julian date for a UTC calendar date (valid 1900-2100)."""
    return (367 * y - math.floor(7 * (y + math.floor((mo + 9) / 12)) * 0.25)
            + math.floor(275 * mo / 9) + d + 1721013.5 + ((s / 60 + mi) / 60 + h) / 24)


def jd_from_datetime(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / SECONDS_PER_DAY + JD_UNIX_EPOCH


def datetime_from_jd(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - JD_UNIX_EPOCH) * SECONDS_PER_DAY)


def now_jd(round_to_minute: bool = True) -> float:
    jd = jd_from_datetime(datetime.now(timezone.utc))
    return math.floor(jd * 1440) / 1440 if round_to_minute else jd


def gmst(jd):
    """Greenwich mean sidereal time (rad), IAU-82, matching SGP4's gstime. Accepts arrays."""
    t = (np.asarray(jd, dtype=float) - 2451545.0) / 36525.0
    g = (-6.2e-6 * t ** 3 + 0.093104 * t ** 2
         + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841)
    return np.mod(g * DEG / 240.0, TWO_PI)


def fmt_utc(jd: float, seconds: bool = True) -> str:
    d = datetime_from_jd(jd)
    return d.strftime("%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M")


def fmt_rel(jd: float, start_jd: float) -> str:
    s = (jd - start_jd) * SECONDS_PER_DAY
    sign = "-" if s < 0 else "+"
    s = abs(s)
    return f"T{sign}{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def fmt_duration(sec: float) -> str:
    s = abs(round(sec))
    h, m, ss = s // 3600, (s % 3600) // 60, s % 60
    out = f"{h}h {m:02d}m" if h else f"{m}m {ss:02d}s"
    return ("-" if sec < 0 else "") + out
