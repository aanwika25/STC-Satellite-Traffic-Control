"""SGP4 near-Earth propagator (Vallado et al. 2006, 'improved' mode), vectorised over time with NumPy.

Validated against the published SGP4 verification case (NORAD 06251) - see tests/.
Deep-space objects (period >= 225 min, e.g. MEO navigation satellites) are not covered by near-Earth
SGP4; for those a simplified Kepler + J2-secular model is used (good for display, not for screening).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..common.constants import (DEG, TWO_PI, RE_WGS72 as RE, XKE, J2_WGS72 as J2,
                                J3_WGS72 as J3, J4_WGS72 as J4, MINUTES_PER_DAY)
from ..common.time_utils import jday

J3OJ2 = J3 / J2
X2O3 = 2.0 / 3.0


# ---------------------------------------------------------------- TLE parsing / formatting
def _parse_exp(s: str) -> float:
    """' 12345-3' -> 0.12345e-3"""
    s = s.strip()
    if not s:
        return 0.0
    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    if len(s) >= 2 and s[-2] in "+-":
        return sign * float("0." + s[:-2]) * 10 ** int(s[-2:])
    try:
        return sign * float(s)
    except ValueError:
        return 0.0


def tle_checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


def _fmt_exp(x: float) -> str:
    if x == 0:
        return " 00000-0"
    sign = "-" if x < 0 else " "
    x = abs(x)
    e = math.floor(math.log10(x)) + 1
    m = round(x / 10 ** e * 1e5)
    if m >= 100000:
        m, e = round(m / 10), e + 1
    return f"{sign}{m:05d}{'-' if e < 0 else '+'}{abs(e)}"


def format_tle(satnum: int, jdepoch: float, bstar: float, inc_deg: float, raan_deg: float, ecc: float,
               argp_deg: float, ma_deg: float, revs_per_day: float, intl: str = "26001A") -> tuple[str, str]:
    """Build a TLE line pair (with checksums) from mean elements."""
    from ..common.time_utils import datetime_from_jd
    y = datetime_from_jd(jdepoch).year
    doy = jdepoch - jday(y, 1, 1) + 1
    l1 = (f"1 {satnum:05d}U {intl:<8} {y % 100:02d}{doy:012.8f}  .00000000  00000-0 {_fmt_exp(bstar)} 0  999")[:68]
    l2 = (f"2 {satnum:05d} {inc_deg:8.4f} {raan_deg % 360:8.4f} {round(ecc * 1e7):07d} "
          f"{argp_deg % 360:8.4f} {ma_deg % 360:8.4f} {revs_per_day:11.8f}    1")[:68]
    return l1 + str(tle_checksum(l1)), l2 + str(tle_checksum(l2))


def orbit_regime(mean_alt_km: float) -> str:
    if mean_alt_km < 2000:
        return "LEO"
    if mean_alt_km < 35000:
        return "MEO"
    return "GEO"


@dataclass
class Satellite:
    """An orbiting object with initialised SGP4 coefficients.

    Build it from a TLE (Satellite(name, line1, line2)) or from mean elements / OMM records
    (Satellite.from_elements / from_omm), which also covers catalogue numbers above 99999.
    """
    name: str
    line1: str = ""
    line2: str = ""
    obj_type: str = ""
    rcs: str = ""
    desc: str = ""
    country: str = ""
    satnum: str = field(init=False)
    jdepoch: float = field(init=False)
    error: int = field(init=False, default=0)

    def __post_init__(self):
        if not self.line1 and not self.line2:
            return  # elements are filled in by from_elements()
        l1, l2 = self.line1.ljust(69), self.line2.ljust(69)
        if l1[0] != "1" or l2[0] != "2":
            raise ValueError(f"Bad TLE lines for {self.name!r}")
        self.satnum = l1[2:7].strip()
        yr2 = int(l1[18:20])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        self.jdepoch = jday(year, 1, 1) + float(l1[20:32]) - 1
        self.bstar = _parse_exp(l1[53:61])
        self.inclo = float(l2[8:16]) * DEG
        self.nodeo = float(l2[17:25]) * DEG
        self.ecco = float("0." + l2[26:33].strip())
        self.argpo = float(l2[34:42]) * DEG
        self.mo = float(l2[43:51]) * DEG
        self.no_kozai = float(l2[52:63]) * TWO_PI / MINUTES_PER_DAY
        self._init()

    @classmethod
    def from_elements(cls, name: str, satnum, jdepoch: float, bstar: float, inc_deg: float, raan_deg: float, ecc: float,
                      argp_deg: float, ma_deg: float, rev_per_day: float, **meta) -> "Satellite":
        """Create from SGP4 mean elements (angles in degrees, mean motion in revolutions per day)."""
        s = cls(name, "", "", **meta)
        s.satnum = str(satnum).strip()
        s.jdepoch = float(jdepoch)
        s.bstar = float(bstar)
        s.inclo, s.nodeo = inc_deg * DEG, raan_deg * DEG
        s.ecco = float(ecc)
        s.argpo, s.mo = argp_deg * DEG, ma_deg * DEG
        s.no_kozai = rev_per_day * TWO_PI / MINUTES_PER_DAY
        if not (0 <= s.ecco < 1) or rev_per_day <= 0:
            raise ValueError(f"Invalid elements for {name!r}")
        s._init()
        return s

    @classmethod
    def from_omm(cls, rec: dict, **meta) -> "Satellite":
        """Create from a CCSDS OMM record using CelesTrak's keywords (JSON, CSV or XML)."""
        g = lambda k, d=None: rec.get(k, rec.get(k.lower(), d))
        epoch = str(g("EPOCH")).strip().replace("Z", "")
        from datetime import datetime, timezone
        from ..common.time_utils import jd_from_datetime
        fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in epoch else "%Y-%m-%dT%H:%M:%S"
        jd = jd_from_datetime(datetime.strptime(epoch, fmt).replace(tzinfo=timezone.utc))
        name = (g("OBJECT_NAME") or f"NORAD {g('NORAD_CAT_ID')}").strip()
        return cls.from_elements(name, g("NORAD_CAT_ID"), jd, float(g("BSTAR", 0) or 0), float(g("INCLINATION")),
                                 float(g("RA_OF_ASC_NODE")), float(g("ECCENTRICITY")), float(g("ARG_OF_PERICENTER")),
                                 float(g("MEAN_ANOMALY")), float(g("MEAN_MOTION")), **meta)

    def elements(self) -> dict:
        """Mean elements for the browser engine (degrees, rev/day)."""
        return {"satnum": self.satnum, "jdepoch": self.jdepoch, "bstar": self.bstar,
                "inc": math.degrees(self.inclo), "raan": math.degrees(self.nodeo), "ecc": self.ecco,
                "argp": math.degrees(self.argpo), "ma": math.degrees(self.mo), "n": self.no_kozai * MINUTES_PER_DAY / TWO_PI}

    # ------------------------------------------------------------ sgp4init
    def _init(self):
        ecco, inclo = self.ecco, self.inclo
        eccsq = ecco * ecco
        omeosq = 1 - eccsq
        rteosq = math.sqrt(omeosq)
        cosio = math.cos(inclo)
        cosio2 = cosio * cosio
        ak = (XKE / self.no_kozai) ** X2O3
        d1 = 0.75 * J2 * (3 * cosio2 - 1) / (rteosq * omeosq)
        dl = d1 / (ak * ak)
        adel = ak * (1 - dl * dl - dl * (1 / 3 + 134 * dl * dl / 81))
        dl = d1 / (adel * adel)
        self.no = self.no_kozai / (1 + dl)
        ao = (XKE / self.no) ** X2O3
        sinio = math.sin(inclo)
        po = ao * omeosq
        con42 = 1 - 5 * cosio2
        self.con41 = -con42 - cosio2 - cosio2
        posq = po * po
        rp = ao * (1 - ecco)
        self.period_min = TWO_PI / self.no
        self.a_km = ao * RE
        self.perigee_km = ao * (1 - ecco) * RE - RE
        self.apogee_km = ao * (1 + ecco) * RE - RE
        self.regime = orbit_regime(self.a_km - RE)
        if self.period_min >= 225:
            # Deep space: SDP4 is not implemented. Use Kepler + J2 secular drift (display-grade).
            self.deep = True
            p = ao * omeosq
            k = 1.5 * J2 / (p * p)
            self.k_nodedot = -k * self.no * cosio
            self.k_argpdot = 0.5 * k * self.no * (5 * cosio2 - 1)
            self.k_ao = ao
            return
        self.deep = False
        self.isimp = 1 if rp < 220 / RE + 1 else 0
        sfour = 78 / RE + 1
        qzms24 = ((120 - 78) / RE) ** 4
        perige = (rp - 1) * RE
        if perige < 156:
            sfour = 20 if perige < 98 else perige - 78
            qzms24 = ((120 - sfour) / RE) ** 4
            sfour = sfour / RE + 1
        pinvsq = 1 / posq
        tsi = 1 / (ao - sfour)
        self.eta = ao * ecco * tsi
        etasq = self.eta ** 2
        eeta = ecco * self.eta
        psisq = abs(1 - etasq)
        coef = qzms24 * tsi ** 4
        coef1 = coef / psisq ** 3.5
        cc2 = coef1 * self.no * (ao * (1 + 1.5 * etasq + eeta * (4 + etasq))
                                 + 0.375 * J2 * tsi / psisq * self.con41 * (8 + 3 * etasq * (8 + etasq)))
        self.cc1 = self.bstar * cc2
        cc3 = -2 * coef * tsi * J3OJ2 * self.no * sinio / ecco if ecco > 1e-4 else 0.0
        self.x1mth2 = 1 - cosio2
        self.cc4 = 2 * self.no * coef1 * ao * omeosq * (
            self.eta * (2 + 0.5 * etasq) + ecco * (0.5 + 2 * etasq)
            - J2 * tsi / (ao * psisq) * (-3 * self.con41 * (1 - 2 * eeta + etasq * (1.5 - 0.5 * eeta))
                                         + 0.75 * self.x1mth2 * (2 * etasq - eeta * (1 + etasq)) * math.cos(2 * self.argpo)))
        self.cc5 = 2 * coef1 * ao * omeosq * (1 + 2.75 * (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * J2 * pinvsq * self.no
        temp2 = 0.5 * temp1 * J2 * pinvsq
        temp3 = -0.46875 * J4 * pinvsq * pinvsq * self.no
        self.mdot = self.no + 0.5 * temp1 * rteosq * self.con41 + 0.0625 * temp2 * rteosq * (13 - 78 * cosio2 + 137 * cosio4)
        self.argpdot = (-0.5 * temp1 * con42 + 0.0625 * temp2 * (7 - 114 * cosio2 + 395 * cosio4)
                        + temp3 * (3 - 36 * cosio2 + 49 * cosio4))
        xhdot1 = -temp1 * cosio
        self.nodedot = xhdot1 + (0.5 * temp2 * (4 - 19 * cosio2) + 2 * temp3 * (3 - 7 * cosio2)) * cosio
        self.omgcof = self.bstar * cc3 * math.cos(self.argpo)
        self.xmcof = -X2O3 * coef * self.bstar / eeta if ecco > 1e-4 else 0.0
        self.nodecf = 3.5 * omeosq * xhdot1 * self.cc1
        self.t2cof = 1.5 * self.cc1
        den = 1 + cosio if abs(cosio + 1) > 1.5e-12 else 1.5e-12
        self.xlcof = -0.25 * J3OJ2 * sinio * (3 + 5 * cosio) / den
        self.aycof = -0.5 * J3OJ2 * sinio
        self.delmo = (1 + self.eta * math.cos(self.mo)) ** 3
        self.sinmao = math.sin(self.mo)
        self.x7thm1 = 7 * cosio2 - 1
        if self.isimp != 1:
            cc1sq = self.cc1 ** 2
            self.d2 = 4 * ao * tsi * cc1sq
            temp = self.d2 * tsi * self.cc1 / 3
            self.d3 = (17 * ao + sfour) * temp
            self.d4 = 0.5 * temp * ao * tsi * (221 * ao + 31 * sfour) * self.cc1
            self.t3cof = self.d2 + 2 * cc1sq
            self.t4cof = 0.25 * (3 * self.d3 + self.cc1 * (12 * self.d2 + 10 * cc1sq))
            self.t5cof = 0.2 * (3 * self.d4 + 12 * self.cc1 * self.d3 + 6 * self.d2 ** 2 + 15 * cc1sq * (2 * self.d2 + cc1sq))

    # ------------------------------------------------------------ sgp4
    def _kepler(self, t):
        """Two-body Kepler orbit with J2 secular node/perigee drift (used for deep-space objects)."""
        e, a = self.ecco, self.k_ao
        M = np.mod(self.mo + self.no * t, TWO_PI)
        node = self.nodeo + self.k_nodedot * t
        argp = self.argpo + self.k_argpdot * t
        E = M.copy()
        for _ in range(12):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        cE, sE = np.cos(E), np.sin(E)
        rr = a * (1 - e * cE)
        xp, yp = a * (cE - e), a * math.sqrt(1 - e * e) * sE
        nrate = self.no * a * a / rr                    # dE/dt * a (er/min)
        vxp, vyp = -sE * nrate, math.sqrt(1 - e * e) * cE * nrate
        ci, si = math.cos(self.inclo), math.sin(self.inclo)
        cO, sO, cw, sw = np.cos(node), np.sin(node), np.cos(argp), np.sin(argp)
        P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
        Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)
        r = (xp[:, None] * P + yp[:, None] * Q) * RE
        v = (vxp[:, None] * P + vyp[:, None] * Q) * RE / 60.0
        return r, v

    def sgp4(self, tsince_min):
        """Position (km) and velocity (km/s) in TEME. tsince_min: scalar or array (minutes from TLE epoch).
        Returns arrays of shape (N,3); rows are NaN where the propagation fails (e.g. decayed)."""
        t = np.atleast_1d(np.asarray(tsince_min, dtype=float))
        n = t.size
        if self.error:
            nan = np.full((n, 3), np.nan)
            return nan, nan.copy()
        if self.deep:
            return self._kepler(t)
        xmdf = self.mo + self.mdot * t
        argpdf = self.argpo + self.argpdot * t
        nodedf = self.nodeo + self.nodedot * t
        argpm, mm = argpdf.copy(), xmdf.copy()
        t2 = t * t
        nodem = nodedf + self.nodecf * t2
        tempa = 1 - self.cc1 * t
        tempe = self.bstar * self.cc4 * t
        templ = self.t2cof * t2
        if self.isimp != 1:
            delomg = self.omgcof * t
            delm = self.xmcof * ((1 + self.eta * np.cos(xmdf)) ** 3 - self.delmo)
            temp = delomg + delm
            mm = xmdf + temp
            argpm = argpdf - temp
            t3, t4 = t2 * t, t2 * t2
            tempa = tempa - self.d2 * t2 - self.d3 * t3 - self.d4 * t4
            tempe = tempe + self.bstar * self.cc5 * (np.sin(mm) - self.sinmao)
            templ = templ + self.t3cof * t3 + t4 * (self.t4cof + t * self.t5cof)
        am = (XKE / self.no) ** X2O3 * tempa * tempa
        nm = XKE / np.abs(am) ** 1.5
        em = self.ecco - tempe
        bad = (em >= 1) | (em < -0.001) | (am < 0.95)
        em = np.maximum(em, 1e-6)
        mm = mm + self.no * templ
        xlm = mm + argpm + nodem
        nodem = np.mod(nodem, TWO_PI)
        argpm = np.mod(argpm, TWO_PI)
        xlm = np.mod(xlm, TWO_PI)
        mm = np.mod(xlm - argpm - nodem, TWO_PI)
        sinip, cosip = math.sin(self.inclo), math.cos(self.inclo)
        axnl = em * np.cos(argpm)
        temp = 1 / (am * (1 - em * em))
        aynl = em * np.sin(argpm) + temp * self.aycof
        xl = mm + argpm + nodem + temp * self.xlcof * axnl
        u = np.mod(xl - nodem, TWO_PI)
        eo1 = u.copy()
        for _ in range(10):  # Kepler's equation (Newton-Raphson, clipped)
            sineo1, coseo1 = np.sin(eo1), np.cos(eo1)
            tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / (1 - coseo1 * axnl - sineo1 * aynl)
            eo1 = eo1 + np.clip(tem5, -0.95, 0.95)
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
        ux = xmx * sinsu + cnod * cossu
        uy = xmy * sinsu + snod * cossu
        uz = sini * sinsu
        vx = xmx * cossu - cnod * sinsu
        vy = xmy * cossu - snod * sinsu
        vz = sini * cossu
        bad |= mrt < 1
        vk = RE * XKE / 60.0
        r = np.stack([mrt * ux, mrt * uy, mrt * uz], axis=-1) * RE
        v = np.stack([mvt * ux + rvdot * vx, mvt * uy + rvdot * vy, mvt * uz + rvdot * vz], axis=-1) * vk
        r[bad] = np.nan
        v[bad] = np.nan
        return r, v

    def at(self, jd):
        """State at Julian date(s). Returns (r, v) with shape (N,3)."""
        return self.sgp4((np.asarray(jd, dtype=float) - self.jdepoch) * MINUTES_PER_DAY)

    def at1(self, jd: float):
        """Single state as two length-3 arrays."""
        r, v = self.at(jd)
        return r[0], v[0]


# ---------------------------------------------------------------- loading helpers
def detect_format(text: str) -> str:
    t = text.lstrip()
    if t.startswith("[") or t.startswith("{"):
        return "json"
    if t.startswith("<"):
        return "xml"
    first = t.splitlines()[0] if t else ""
    if "OBJECT_NAME" in first.upper() or "NORAD_CAT_ID" in first.upper():
        return "csv"
    return "tle"


def parse_omm_json(text: str, **meta) -> list[Satellite]:
    import json
    data = json.loads(text)
    recs = data if isinstance(data, list) else [data]
    return _from_records(recs, **meta)


def parse_omm_csv(text: str, **meta) -> list[Satellite]:
    import csv, io
    return _from_records(list(csv.DictReader(io.StringIO(text.strip()))), **meta)


def parse_omm_xml(text: str, **meta) -> list[Satellite]:
    """CCSDS OMM XML (as served by CelesTrak FORMAT=xml): one <omm> per object."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text.strip())
    strip = lambda tag: tag.split("}")[-1]
    recs = []
    for omm in root.iter():
        if strip(omm.tag) != "omm":
            continue
        rec = {}
        for el in omm.iter():
            if el.text and el.text.strip() and len(el) == 0:
                key = el.get("parameter") if strip(el.tag) == "USER_DEFINED" else strip(el.tag)
                rec[key] = el.text.strip()
        recs.append(rec)
    return _from_records(recs, **meta)


def _from_records(recs, **meta) -> list[Satellite]:
    out = []
    for r in recs:
        try:
            s = Satellite.from_omm(r, **meta)
            from ..ingestion.spacetrack import apply_catalogue_metadata   # Space-Track OBJECT_TYPE / COUNTRY_CODE
            apply_catalogue_metadata(s, r)
            if not s.error:
                out.append(s)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_orbit_text(text: str, **meta) -> list[Satellite]:
    """Parse TLE (2/3-line), OMM JSON, OMM CSV or OMM XML text - the format is detected automatically."""
    fmt = detect_format(text)
    if fmt == "json":
        return parse_omm_json(text, **meta)
    if fmt == "csv":
        return parse_omm_csv(text, **meta)
    if fmt == "xml":
        return parse_omm_xml(text, **meta)
    return parse_tle_text(text, **meta)


def parse_tle_text(text: str, **meta) -> list[Satellite]:
    """Parse 2-line or 3-line TLE text into Satellite objects (deep-space ones skipped)."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out, i = [], 0
    while i < len(lines):
        a = lines[i]
        b = lines[i + 1] if i + 1 < len(lines) else ""
        c = lines[i + 2] if i + 2 < len(lines) else ""
        try:
            if a.startswith("1 ") and b.startswith("2 "):
                out.append(Satellite(f"NORAD {a[2:7].strip()}", a, b, **meta)); i += 2; continue
            if b.startswith("1 ") and c.startswith("2 "):
                name = a[2:] if a.startswith("0 ") else a
                out.append(Satellite(name.strip(), b, c, **meta)); i += 3; continue
        except (ValueError, IndexError):
            pass
        i += 1
    return [s for s in out if not s.error]
