"""SGP4 for many near-Earth objects at once: arrays of shape (n_objects, n_times).

Same equations as Satellite.sgp4 (Vallado 2006), with every per-object coefficient stacked into a
column so NumPy evaluates hundreds of objects in one pass. Used for catalogue-wide screening, where
calling SGP4 object by object is dominated by Python overhead. Deep-space objects are not handled here.
"""
from __future__ import annotations

import numpy as np

from ..common.constants import TWO_PI, RE_WGS72 as RE, XKE, J2_WGS72 as J2

X2O3 = 2.0 / 3.0
_FIELDS = ["mo", "mdot", "argpo", "argpdot", "nodeo", "nodedot", "nodecf", "cc1", "bstar", "cc4", "t2cof",
           "omgcof", "xmcof", "eta", "delmo", "d2", "d3", "d4", "cc5", "sinmao", "t3cof", "t4cof", "t5cof",
           "no", "ecco", "inclo", "aycof", "xlcof", "con41", "x1mth2", "x7thm1", "jdepoch"]


class SGP4Batch:
    def __init__(self, sats):
        self.sats = [s for s in sats if not s.error and not getattr(s, "deep", False)]
        n = len(self.sats)
        for f in _FIELDS:
            setattr(self, f, np.array([float(getattr(s, f, 0.0)) for s in self.sats]).reshape(n, 1))
        self.full = np.array([0.0 if s.isimp == 1 else 1.0 for s in self.sats]).reshape(n, 1)

    def __len__(self):
        return len(self.sats)

    def at(self, jd):
        """Positions and velocities at Julian dates jd (T,) -> r, v of shape (N, T, 3); NaN where invalid."""
        t = (np.asarray(jd, float)[None, :] - self.jdepoch) * 1440.0
        f = self.full
        xmdf = self.mo + self.mdot * t
        argpdf = self.argpo + self.argpdot * t
        nodedf = self.nodeo + self.nodedot * t
        t2 = t * t
        nodem = nodedf + self.nodecf * t2
        delomg = self.omgcof * t
        delm = self.xmcof * ((1 + self.eta * np.cos(xmdf)) ** 3 - self.delmo)
        temp = (delomg + delm) * f
        mm = xmdf + temp
        argpm = argpdf - temp
        t3, t4 = t2 * t, t2 * t2
        tempa = 1 - self.cc1 * t - (self.d2 * t2 + self.d3 * t3 + self.d4 * t4) * f
        tempe = self.bstar * self.cc4 * t + self.bstar * self.cc5 * (np.sin(mm) - self.sinmao) * f
        templ = self.t2cof * t2 + (self.t3cof * t3 + t4 * (self.t4cof + t * self.t5cof)) * f
        am = (XKE / self.no) ** X2O3 * tempa * tempa
        nm = XKE / np.abs(am) ** 1.5
        em = self.ecco - tempe
        bad = (em >= 1) | (em < -0.001) | (am < 0.95)
        em = np.maximum(em, 1e-6)
        mm = mm + self.no * templ
        xlm = np.mod(mm + argpm + nodem, TWO_PI)
        nodem = np.mod(nodem, TWO_PI)
        argpm = np.mod(argpm, TWO_PI)
        mm = np.mod(xlm - argpm - nodem, TWO_PI)
        sinip, cosip = np.sin(self.inclo), np.cos(self.inclo)
        axnl = em * np.cos(argpm)
        temp = 1 / (am * (1 - em * em))
        aynl = em * np.sin(argpm) + temp * self.aycof
        xl = mm + argpm + nodem + temp * self.xlcof * axnl
        u = np.mod(xl - nodem, TWO_PI)
        eo1 = u.copy()
        for _ in range(10):
            s_, c_ = np.sin(eo1), np.cos(eo1)
            eo1 = eo1 + np.clip((u - aynl * c_ + axnl * s_ - eo1) / (1 - c_ * axnl - s_ * aynl), -0.95, 0.95)
        sineo1, coseo1 = np.sin(eo1), np.cos(eo1)
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        el2 = axnl * axnl + aynl * aynl
        pl = am * (1 - el2)
        bad |= pl < 0
        pl = np.where(pl < 0, np.nan, pl)
        rl = am * (1 - ecose)
        rdotl = np.sqrt(am) * esine / rl
        rvdotl = np.sqrt(pl) / rl
        betal = np.sqrt(1 - el2)
        temp = esine / (1 + betal)
        sinu = am / rl * (sineo1 - aynl - axnl * temp)
        cosu = am / rl * (coseo1 - axnl + aynl * temp)
        su = np.arctan2(sinu, cosu)
        sin2u = (cosu + cosu) * sinu
        cos2u = 1 - 2 * sinu * sinu
        temp = 1 / pl
        temp1 = 0.5 * J2 * temp
        temp2 = temp1 * temp
        mrt = rl * (1 - 1.5 * temp2 * betal * self.con41) + 0.5 * temp1 * self.x1mth2 * cos2u
        su = su - 0.25 * temp2 * self.x7thm1 * sin2u
        xnode = nodem + 1.5 * temp2 * cosip * sin2u
        xinc = self.inclo + 1.5 * temp2 * cosip * sinip * cos2u
        mvt = rdotl - nm * temp1 * self.x1mth2 * sin2u / XKE
        rvdot = rvdotl + nm * temp1 * (self.x1mth2 * cos2u + 1.5 * self.con41) / XKE
        sinsu, cossu = np.sin(su), np.cos(su)
        snod, cnod = np.sin(xnode), np.cos(xnode)
        sini, cosi = np.sin(xinc), np.cos(xinc)
        xmx, xmy = -snod * cosi, cnod * cosi
        ux, uy, uz = xmx * sinsu + cnod * cossu, xmy * sinsu + snod * cossu, sini * sinsu
        vx, vy, vz = xmx * cossu - cnod * sinsu, xmy * cossu - snod * sinsu, sini * cossu
        bad |= mrt < 1
        vk = RE * XKE / 60.0
        r = np.stack([mrt * ux, mrt * uy, mrt * uz], axis=-1) * RE
        v = np.stack([mvt * ux + rvdot * vx, mvt * uy + rvdot * vy, mvt * uz + rvdot * vz], axis=-1) * vk
        r[bad] = np.nan
        v[bad] = np.nan
        return r, v
