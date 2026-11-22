"""Numerical propagator: two-body + J2, fixed-step RK4.

Used for manoeuvre what-ifs. Several trajectories are integrated together
(states shape (K,3)) so the baseline and the manoeuvred orbit share one loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.constants import MU, RE, J2, SECONDS_PER_DAY


def accel(r: np.ndarray) -> np.ndarray:
    """Gravity acceleration (km/s^2) with J2 for positions r of shape (...,3)."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rm = np.sqrt(r2)
    r3 = r2 * rm
    k = 1.5 * J2 * MU * RE * RE / (r2 * r2 * rm)
    zz = 5 * z * z / r2
    return np.stack([-MU * x / r3 + k * x * (zz - 1),
                     -MU * y / r3 + k * y * (zz - 1),
                     -MU * z / r3 + k * z * (zz - 3)], axis=-1)


def rk4_step(r, v, h):
    a1 = accel(r)
    r2, v2 = r + v * h / 2, v + a1 * h / 2
    a2 = accel(r2)
    r3, v3 = r + v2 * h / 2, v + a2 * h / 2
    a3 = accel(r3)
    r4, v4 = r + v3 * h, v + a3 * h
    a4 = accel(r4)
    return (r + (v + 2 * v2 + 2 * v3 + v4) * h / 6,
            v + (a1 + 2 * a2 + 2 * a3 + a4) * h / 6)


@dataclass
class Trajectory:
    """Sampled trajectories: pos/vel arrays of shape (n_samples, K, 3), sample spacing dt seconds."""
    jd0: float
    dt: float
    pos: np.ndarray
    vel: np.ndarray

    def sample(self, jd):
        """Cubic Hermite interpolation. jd: (N,) -> positions (N,K,3), velocities (N,K,3) (linear)."""
        jd = np.atleast_1d(np.asarray(jd, float))
        x = (jd - self.jd0) * SECONDS_PER_DAY / self.dt
        k = np.clip(np.floor(x).astype(int), 0, len(self.pos) - 2)
        f = (x - k)[:, None, None]
        p0, p1, v0, v1, h = self.pos[k], self.pos[k + 1], self.vel[k], self.vel[k + 1], self.dt
        f2, f3 = f * f, f * f * f
        pos = ((2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * h * v0
               + (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * h * v1)
        vel = v0 * (1 - f) + v1 * f
        return pos, vel


def propagate(r0, v0, jd0: float, jd1: float, dt_out: float = 10.0, h_step: float = 10.0) -> Trajectory:
    """Integrate one or more initial states (r0, v0 of shape (3,) or (K,3)) from jd0 to jd1."""
    r = np.atleast_2d(np.asarray(r0, float)).copy()
    v = np.atleast_2d(np.asarray(v0, float)).copy()
    n = int(np.floor((jd1 - jd0) * SECONDS_PER_DAY / dt_out)) + 1
    sub = max(1, round(dt_out / h_step))
    h = dt_out / sub
    pos = np.empty((n, r.shape[0], 3))
    vel = np.empty_like(pos)
    for i in range(n):
        pos[i], vel[i] = r, v
        for _ in range(sub):
            r, v = rk4_step(r, v, h)
    return Trajectory(jd0, dt_out, pos, vel)
