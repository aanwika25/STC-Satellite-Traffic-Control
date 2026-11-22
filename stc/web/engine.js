// OrbitGuard orbit engine: SGP4 (near-Earth), TLE I/O, frames, J2 numerical propagator, passes.
// Works in the browser (global OG) and in Node (module.exports).
(function (root) {
  "use strict";
  const PI = Math.PI, TWOPI = 2 * Math.PI, DEG = PI / 180;
  // WGS-72 constants used by SGP4
  const MU = 398600.8, RE = 6378.135;
  const XKE = 60.0 / Math.sqrt((RE * RE * RE) / MU);
  const J2 = 0.001082616, J3 = -0.00000253881, J4 = -0.00000165597, J3OJ2 = J3 / J2;
  const X2O3 = 2 / 3;
  // WGS-84 for geodesy and the numerical propagator
  const MU84 = 398600.4418, RE84 = 6378.137, F84 = 1 / 298.257223563, J2_84 = 1.08262668e-3;
  const OMEGA_E = 7.292115e-5; // rad/s

  const mod2pi = (x) => { x %= TWOPI; return x < 0 ? x + TWOPI : x; };

  // ---------- time ----------
  function jday(y, mo, d, h, mi, s) {
    return 367 * y - Math.floor((7 * (y + Math.floor((mo + 9) / 12))) * 0.25) + Math.floor((275 * mo) / 9) + d + 1721013.5 + ((s / 60 + mi) / 60 + h) / 24;
  }
  const jdFromDate = (dt) => dt.getTime() / 86400000 + 2440587.5;
  const dateFromJd = (jd) => new Date((jd - 2440587.5) * 86400000);
  function gmst(jd) {
    const t = (jd - 2451545.0) / 36525.0;
    let g = -6.2e-6 * t * t * t + 0.093104 * t * t + (876600.0 * 3600 + 8640184.812866) * t + 67310.54841;
    return mod2pi((g * DEG) / 240.0);
  }

  // ---------- TLE ----------
  const checksum = (line) => { let c = 0; for (let i = 0; i < 68; i++) { const ch = line[i]; if (ch >= "0" && ch <= "9") c += +ch; else if (ch === "-") c += 1; } return c % 10; };
  function parseExp(s) { // " 12345-3" -> 0.12345e-3
    s = s.trim(); if (!s) return 0;
    let sign = 1; if (s[0] === "-") { sign = -1; s = s.slice(1); } else if (s[0] === "+") s = s.slice(1);
    const m = s.match(/^(\d+)([-+]\d)$/); if (!m) return parseFloat(s) || 0;
    return sign * parseFloat("0." + m[1]) * Math.pow(10, parseInt(m[2], 10));
  }
  function parseTLE(name, l1, l2) {
    l1 = l1.padEnd(69); l2 = l2.padEnd(69);
    if (l1[0] !== "1" || l2[0] !== "2") throw new Error("Bad TLE lines for " + name);
    const yr2 = parseInt(l1.slice(18, 20), 10), days = parseFloat(l1.slice(20, 32));
    const year = yr2 < 57 ? 2000 + yr2 : 1900 + yr2;
    const rec = {
      name: (name || "").trim(), satnum: l1.slice(2, 7).trim(), line1: l1.trimEnd(), line2: l2.trimEnd(),
      jdepoch: jday(year, 1, 1, 0, 0, 0) + days - 1,
      bstar: parseExp(l1.slice(53, 61)),
      inclo: parseFloat(l2.slice(8, 16)) * DEG,
      nodeo: parseFloat(l2.slice(17, 25)) * DEG,
      ecco: parseFloat("0." + l2.slice(26, 33).trim()),
      argpo: parseFloat(l2.slice(34, 42)) * DEG,
      mo: parseFloat(l2.slice(43, 51)) * DEG,
      no_kozai: (parseFloat(l2.slice(52, 63)) * TWOPI) / 1440.0,
    };
    sgp4init(rec);
    return rec;
  }
  // Build a record from mean elements (degrees, rev/day) - used for OMM data and 6-digit catalogue numbers
  function fromElements(name, e) {
    if (Array.isArray(e)) e = { satnum: e[0], jdepoch: e[1], bstar: e[2], inc: e[3], raan: e[4], ecc: e[5], argp: e[6], ma: e[7], n: e[8] };
    const rec = { name, satnum: String(e.satnum), jdepoch: e.jdepoch, bstar: e.bstar || 0, inclo: e.inc * DEG, nodeo: e.raan * DEG,
      ecco: e.ecc, argpo: e.argp * DEG, mo: e.ma * DEG, no_kozai: (e.n * TWOPI) / 1440.0 };
    sgp4init(rec); return rec;
  }
  function parseTLEText(text) {
    const lines = text.split(/\r?\n/).map((l) => l.replace(/\s+$/, "")).filter((l) => l.length);
    const out = []; let i = 0;
    while (i < lines.length) {
      if (lines[i][0] === "1" && lines[i + 1] && lines[i + 1][0] === "2") { out.push(parseTLE("NORAD " + lines[i].slice(2, 7).trim(), lines[i], lines[i + 1])); i += 2; }
      else if (lines[i + 1] && lines[i + 1][0] === "1" && lines[i + 2] && lines[i + 2][0] === "2") { out.push(parseTLE(lines[i].replace(/^0 /, ""), lines[i + 1], lines[i + 2])); i += 3; }
      else i++;
    }
    return out;
  }
  function fmtExp(x) { // 0.00012345 -> " 12345-3"
    if (x === 0) return " 00000-0";
    const sign = x < 0 ? "-" : " "; x = Math.abs(x);
    let e = Math.floor(Math.log10(x)) + 1, m = Math.round(x / Math.pow(10, e) * 1e5);
    if (m >= 100000) { m = Math.round(m / 10); e += 1; }
    return sign + String(m).padStart(5, "0") + (e < 0 ? "-" : "+") + Math.abs(e);
  }
  function formatTLE(o) { // o: {satnum, intl, jdepoch, bstar, incDeg, raanDeg, ecc, argpDeg, maDeg, revsPerDay}
    const d = dateFromJd(o.jdepoch), y = d.getUTCFullYear();
    const doy = o.jdepoch - jday(y, 1, 1, 0, 0, 0) + 1;
    const f = (v, w, p) => v.toFixed(p).padStart(w, " ");
    let l1 = "1 " + String(o.satnum).padStart(5, "0") + "U " + (o.intl || "26001A").padEnd(8) + " " + String(y % 100).padStart(2, "0") +
      doy.toFixed(8).padStart(12, "0") + "  .00000000  00000-0 " + fmtExp(o.bstar) + " 0  999";
    let l2 = "2 " + String(o.satnum).padStart(5, "0") + " " + f(o.incDeg, 8, 4) + " " + f(mod2pi(o.raanDeg * DEG) / DEG, 8, 4) + " " +
      Math.round(o.ecc * 1e7).toString().padStart(7, "0") + " " + f(mod2pi(o.argpDeg * DEG) / DEG, 8, 4) + " " + f(mod2pi(o.maDeg * DEG) / DEG, 8, 4) + " " +
      f(o.revsPerDay, 11, 8) + "    1";
    l1 = l1.slice(0, 68); l2 = l2.slice(0, 68);
    return [l1 + checksum(l1), l2 + checksum(l2)];
  }

  // ---------- SGP4 (near-Earth, Vallado 2006 'i' mode) ----------
  function sgp4init(s) {
    s.error = 0;
    const ecco = s.ecco, inclo = s.inclo;
    const eccsq = ecco * ecco, omeosq = 1 - eccsq, rteosq = Math.sqrt(omeosq);
    const cosio = Math.cos(inclo), cosio2 = cosio * cosio;
    const ak = Math.pow(XKE / s.no_kozai, X2O3);
    const d1 = (0.75 * J2 * (3 * cosio2 - 1)) / (rteosq * omeosq);
    let del = d1 / (ak * ak);
    const adel = ak * (1 - del * del - del * (1 / 3 + (134 * del * del) / 81));
    del = d1 / (adel * adel);
    s.no = s.no_kozai / (1 + del);
    const ao = Math.pow(XKE / s.no, X2O3);
    const sinio = Math.sin(inclo), po = ao * omeosq, con42 = 1 - 5 * cosio2;
    s.con41 = -con42 - cosio2 - cosio2;
    const posq = po * po, rp = ao * (1 - ecco);
    s.periodMin = TWOPI / s.no; s.aKm = ao * RE;
    s.perigeeKm = ao * (1 - ecco) * RE - RE; s.apogeeKm = ao * (1 + ecco) * RE - RE;
    if (s.periodMin >= 225) { // deep space: Kepler + J2 secular drift (display-grade, mirrors the Python fallback)
      s.deep = true; const p = ao * omeosq, k = 1.5 * J2 / (p * p);
      s.kNodedot = -k * s.no * cosio; s.kArgpdot = 0.5 * k * s.no * (5 * cosio2 - 1); s.kAo = ao; return s;
    }
    s.deep = false;
    s.isimp = rp < 220 / RE + 1 ? 1 : 0;
    let sfour = 78 / RE + 1, qzms24 = Math.pow((120 - 78) / RE, 4);
    const perige = (rp - 1) * RE;
    if (perige < 156) {
      sfour = perige < 98 ? 20 : perige - 78;
      qzms24 = Math.pow((120 - sfour) / RE, 4);
      sfour = sfour / RE + 1;
    }
    const pinvsq = 1 / posq, tsi = 1 / (ao - sfour);
    s.eta = ao * ecco * tsi;
    const etasq = s.eta * s.eta, eeta = ecco * s.eta, psisq = Math.abs(1 - etasq);
    const coef = qzms24 * Math.pow(tsi, 4), coef1 = coef / Math.pow(psisq, 3.5);
    const cc2 = coef1 * s.no * (ao * (1 + 1.5 * etasq + eeta * (4 + etasq)) + ((0.375 * J2 * tsi) / psisq) * s.con41 * (8 + 3 * etasq * (8 + etasq)));
    s.cc1 = s.bstar * cc2;
    let cc3 = 0; if (ecco > 1e-4) cc3 = (-2 * coef * tsi * J3OJ2 * s.no * sinio) / ecco;
    s.x1mth2 = 1 - cosio2;
    s.cc4 = 2 * s.no * coef1 * ao * omeosq * (s.eta * (2 + 0.5 * etasq) + ecco * (0.5 + 2 * etasq) -
      ((J2 * tsi) / (ao * psisq)) * (-3 * s.con41 * (1 - 2 * eeta + etasq * (1.5 - 0.5 * eeta)) + 0.75 * s.x1mth2 * (2 * etasq - eeta * (1 + etasq)) * Math.cos(2 * s.argpo)));
    s.cc5 = 2 * coef1 * ao * omeosq * (1 + 2.75 * (etasq + eeta) + eeta * etasq);
    const cosio4 = cosio2 * cosio2;
    const temp1 = 1.5 * J2 * pinvsq * s.no, temp2 = 0.5 * temp1 * J2 * pinvsq, temp3 = -0.46875 * J4 * pinvsq * pinvsq * s.no;
    s.mdot = s.no + 0.5 * temp1 * rteosq * s.con41 + 0.0625 * temp2 * rteosq * (13 - 78 * cosio2 + 137 * cosio4);
    s.argpdot = -0.5 * temp1 * con42 + 0.0625 * temp2 * (7 - 114 * cosio2 + 395 * cosio4) + temp3 * (3 - 36 * cosio2 + 49 * cosio4);
    const xhdot1 = -temp1 * cosio;
    s.nodedot = xhdot1 + (0.5 * temp2 * (4 - 19 * cosio2) + 2 * temp3 * (3 - 7 * cosio2)) * cosio;
    s.omgcof = s.bstar * cc3 * Math.cos(s.argpo);
    s.xmcof = ecco > 1e-4 ? (-X2O3 * coef * s.bstar) / eeta : 0;
    s.nodecf = 3.5 * omeosq * xhdot1 * s.cc1;
    s.t2cof = 1.5 * s.cc1;
    s.xlcof = Math.abs(cosio + 1) > 1.5e-12 ? (-0.25 * J3OJ2 * sinio * (3 + 5 * cosio)) / (1 + cosio) : (-0.25 * J3OJ2 * sinio * (3 + 5 * cosio)) / 1.5e-12;
    s.aycof = -0.5 * J3OJ2 * sinio;
    s.delmo = Math.pow(1 + s.eta * Math.cos(s.mo), 3);
    s.sinmao = Math.sin(s.mo);
    s.x7thm1 = 7 * cosio2 - 1;
    if (s.isimp !== 1) {
      const cc1sq = s.cc1 * s.cc1;
      s.d2 = 4 * ao * tsi * cc1sq;
      const temp = (s.d2 * tsi * s.cc1) / 3;
      s.d3 = (17 * ao + sfour) * temp;
      s.d4 = 0.5 * temp * ao * tsi * (221 * ao + 31 * sfour) * s.cc1;
      s.t3cof = s.d2 + 2 * cc1sq;
      s.t4cof = 0.25 * (3 * s.d3 + s.cc1 * (12 * s.d2 + 10 * cc1sq));
      s.t5cof = 0.2 * (3 * s.d4 + 12 * s.cc1 * s.d3 + 6 * s.d2 * s.d2 + 15 * cc1sq * (2 * s.d2 + cc1sq));
    }
    s.periodMin = TWOPI / s.no;
    s.aKm = ao * RE;
    s.perigeeKm = ao * (1 - ecco) * RE - RE;
    s.apogeeKm = ao * (1 + ecco) * RE - RE;
    return s;
  }

  // tsince in minutes from epoch. Returns {r:[km], v:[km/s]} in TEME, or null.
  function kepler(s, t) {
    const e = s.ecco, a = s.kAo, M = mod2pi(s.mo + s.no * t), node = s.nodeo + s.kNodedot * t, w = s.argpo + s.kArgpdot * t;
    let E = M; for (let i = 0; i < 12; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    const cE = Math.cos(E), sE = Math.sin(E), rr = a * (1 - e * cE), b = Math.sqrt(1 - e * e);
    const xp = a * (cE - e), yp = a * b * sE, nr = (s.no * a * a) / rr, vxp = -sE * nr, vyp = b * cE * nr;
    const ci = Math.cos(s.inclo), si = Math.sin(s.inclo), cO = Math.cos(node), sO = Math.sin(node), cw = Math.cos(w), sw = Math.sin(w);
    const P = [cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], Q = [-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si];
    return { r: [0, 1, 2].map((i) => (xp * P[i] + yp * Q[i]) * RE), v: [0, 1, 2].map((i) => ((vxp * P[i] + vyp * Q[i]) * RE) / 60) };
  }

  function sgp4(s, t) {
    if (s.error) return null;
    if (s.deep) return kepler(s, t);
    const xmdf = s.mo + s.mdot * t, argpdf = s.argpo + s.argpdot * t, nodedf = s.nodeo + s.nodedot * t;
    let argpm = argpdf, mm = xmdf;
    const t2 = t * t;
    let nodem = nodedf + s.nodecf * t2, tempa = 1 - s.cc1 * t, tempe = s.bstar * s.cc4 * t, templ = s.t2cof * t2;
    if (s.isimp !== 1) {
      const delomg = s.omgcof * t;
      const delm = s.xmcof * (Math.pow(1 + s.eta * Math.cos(xmdf), 3) - s.delmo);
      const temp = delomg + delm;
      mm = xmdf + temp; argpm = argpdf - temp;
      const t3 = t2 * t, t4 = t3 * t;
      tempa = tempa - s.d2 * t2 - s.d3 * t3 - s.d4 * t4;
      tempe = tempe + s.bstar * s.cc5 * (Math.sin(mm) - s.sinmao);
      templ = templ + s.t3cof * t3 + t4 * (s.t4cof + t * s.t5cof);
    }
    let nm = s.no, em = s.ecco;
    const inclm = s.inclo;
    const am = Math.pow(XKE / nm, X2O3) * tempa * tempa;
    nm = XKE / Math.pow(am, 1.5);
    em = em - tempe;
    if (em >= 1 || em < -0.001 || am < 0.95) return null;
    if (em < 1e-6) em = 1e-6;
    mm = mm + s.no * templ;
    let xlm = mm + argpm + nodem;
    nodem = mod2pi(nodem); argpm = mod2pi(argpm); xlm = mod2pi(xlm);
    mm = mod2pi(xlm - argpm - nodem);
    const sinip = Math.sin(inclm), cosip = Math.cos(inclm);
    const axnl = em * Math.cos(argpm);
    let temp = 1 / (am * (1 - em * em));
    const aynl = em * Math.sin(argpm) + temp * s.aycof;
    const xl = mm + argpm + nodem + temp * s.xlcof * axnl;
    const u = mod2pi(xl - nodem);
    let eo1 = u, tem5 = 9999.9, ktr = 1, sineo1 = 0, coseo1 = 0;
    while (Math.abs(tem5) >= 1e-12 && ktr <= 10) {
      sineo1 = Math.sin(eo1); coseo1 = Math.cos(eo1);
      tem5 = 1 - coseo1 * axnl - sineo1 * aynl;
      tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
      if (Math.abs(tem5) >= 0.95) tem5 = tem5 > 0 ? 0.95 : -0.95;
      eo1 += tem5; ktr++;
    }
    const ecose = axnl * coseo1 + aynl * sineo1, esine = axnl * sineo1 - aynl * coseo1;
    const el2 = axnl * axnl + aynl * aynl, pl = am * (1 - el2);
    if (pl < 0) return null;
    const rl = am * (1 - ecose), rdotl = (Math.sqrt(am) * esine) / rl, rvdotl = Math.sqrt(pl) / rl;
    const betal = Math.sqrt(1 - el2);
    temp = esine / (1 + betal);
    const sinu = (am / rl) * (sineo1 - aynl - axnl * temp), cosu = (am / rl) * (coseo1 - axnl + aynl * temp);
    let su = Math.atan2(sinu, cosu);
    const sin2u = (cosu + cosu) * sinu, cos2u = 1 - 2 * sinu * sinu;
    temp = 1 / pl;
    const temp1 = 0.5 * J2 * temp, temp2 = temp1 * temp;
    const mrt = rl * (1 - 1.5 * temp2 * betal * s.con41) + 0.5 * temp1 * s.x1mth2 * cos2u;
    su = su - 0.25 * temp2 * s.x7thm1 * sin2u;
    const xnode = nodem + 1.5 * temp2 * cosip * sin2u;
    const xinc = inclm + 1.5 * temp2 * cosip * sinip * cos2u;
    const mvt = rdotl - (nm * temp1 * s.x1mth2 * sin2u) / XKE;
    const rvdot = rvdotl + (nm * temp1 * (s.x1mth2 * cos2u + 1.5 * s.con41)) / XKE;
    const sinsu = Math.sin(su), cossu = Math.cos(su), snod = Math.sin(xnode), cnod = Math.cos(xnode), sini = Math.sin(xinc), cosi = Math.cos(xinc);
    const xmx = -snod * cosi, xmy = cnod * cosi;
    const ux = xmx * sinsu + cnod * cossu, uy = xmy * sinsu + snod * cossu, uz = sini * sinsu;
    const vx = xmx * cossu - cnod * sinsu, vy = xmy * cossu - snod * sinsu, vz = sini * cossu;
    if (mrt < 1) return null;
    const vk = (RE * XKE) / 60;
    return {
      r: [mrt * ux * RE, mrt * uy * RE, mrt * uz * RE],
      v: [(mvt * ux + rvdot * vx) * vk, (mvt * uy + rvdot * vy) * vk, (mvt * uz + rvdot * vz) * vk],
    };
  }
  const propJd = (s, jd) => sgp4(s, (jd - s.jdepoch) * 1440);

  // ---------- vectors ----------
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.sqrt(dot(a, a));
  const unit = (a) => mul(a, 1 / norm(a));

  // ---------- frames ----------
  function eciToEcef(r, jd) { const g = gmst(jd), c = Math.cos(g), s = Math.sin(g); return [c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]]; }
  function ecefToGeodetic(p) {
    const e2 = F84 * (2 - F84), lon = Math.atan2(p[1], p[0]), rxy = Math.hypot(p[0], p[1]);
    let lat = Math.atan2(p[2], rxy), h = 0;
    for (let i = 0; i < 6; i++) { const sl = Math.sin(lat), N = RE84 / Math.sqrt(1 - e2 * sl * sl); h = rxy / Math.cos(lat) - N; lat = Math.atan2(p[2], rxy * (1 - (e2 * N) / (N + h))); }
    return { lat, lon, alt: h };
  }
  function geodeticToEcef(latDeg, lonDeg, hKm) {
    const e2 = F84 * (2 - F84), lat = latDeg * DEG, lon = lonDeg * DEG, sl = Math.sin(lat), N = RE84 / Math.sqrt(1 - e2 * sl * sl);
    return [(N + hKm) * Math.cos(lat) * Math.cos(lon), (N + hKm) * Math.cos(lat) * Math.sin(lon), (N * (1 - e2) + hKm) * sl];
  }
  function lookAngles(stEcef, stLatDeg, stLonDeg, satEcef) {
    const lat = stLatDeg * DEG, lon = stLonDeg * DEG, rho = sub(satEcef, stEcef), range = norm(rho);
    const sl = Math.sin(lat), cl = Math.cos(lat), so = Math.sin(lon), co = Math.cos(lon);
    const S = sl * co * rho[0] + sl * so * rho[1] - cl * rho[2], E = -so * rho[0] + co * rho[1], Z = cl * co * rho[0] + cl * so * rho[1] + sl * rho[2];
    return { el: Math.asin(Z / range) / DEG, az: mod2pi(Math.atan2(E, -S)) / DEG, range };
  }
  // RTN unit vectors (radial, along-track/transverse, normal)
  function rtn(r, v) { const R = unit(r), N = unit(cross(r, v)), T = cross(N, R); return { R, T, N }; }

  // ---------- osculating elements ----------
  function rv2coe(r, v, mu = MU) {
    const rm = norm(r), vm = norm(v), h = cross(r, v), hm = norm(h);
    const nvec = [-h[1], h[0], 0], nm = norm(nvec);
    const evec = mul(sub(mul(r, vm * vm - mu / rm), mul(v, dot(r, v))), 1 / mu), e = norm(evec);
    const a = 1 / (2 / rm - (vm * vm) / mu), i = Math.acos(h[2] / hm);
    let raan = Math.acos(Math.max(-1, Math.min(1, nvec[0] / nm))); if (nvec[1] < 0) raan = TWOPI - raan;
    let argp = Math.acos(Math.max(-1, Math.min(1, dot(nvec, evec) / (nm * e)))); if (evec[2] < 0) argp = TWOPI - argp;
    let nu = Math.acos(Math.max(-1, Math.min(1, dot(evec, r) / (e * rm)))); if (dot(r, v) < 0) nu = TWOPI - nu;
    const E = 2 * Math.atan(Math.sqrt((1 - e) / (1 + e)) * Math.tan(nu / 2));
    const M = mod2pi(E - e * Math.sin(E));
    return { a, e, i, raan, argp, M };
  }

  // ---------- numerical propagator (two-body + J2, RK4) for manoeuvre what-ifs ----------
  function accel(r) {
    const x = r[0], y = r[1], z = r[2], r2 = x * x + y * y + z * z, rm = Math.sqrt(r2), r3 = r2 * rm;
    const k = 1.5 * J2_84 * MU84 * RE84 * RE84 / (r2 * r2 * rm), zz = (5 * z * z) / r2;
    return [-MU84 * x / r3 + k * x * (zz - 1), -MU84 * y / r3 + k * y * (zz - 1), -MU84 * z / r3 + k * z * (zz - 3)];
  }
  function rk4Step(r, v, h) {
    const a1 = accel(r);
    const r2 = add(r, mul(v, h / 2)), v2 = add(v, mul(a1, h / 2)), a2 = accel(r2);
    const r3 = add(r, mul(v2, h / 2)), v3 = add(v, mul(a2, h / 2)), a3 = accel(r3);
    const r4 = add(r, mul(v3, h)), v4 = add(v, mul(a3, h)), a4 = accel(r4);
    return [
      add(r, mul(add(add(v, mul(v2, 2)), add(mul(v3, 2), v4)), h / 6)),
      add(v, mul(add(add(a1, mul(a2, 2)), add(mul(a3, 2), a4)), h / 6)),
    ];
  }
  // Propagate from (r,v) at jd0 to jd1, sampling every dtOut seconds. Returns {jd0, dt, pos:[[x,y,z],...], vel}
  function propagateNumeric(r, v, jd0, jd1, dtOut = 10, hStep = 10) {
    const n = Math.floor(((jd1 - jd0) * 86400) / dtOut) + 1, pos = [], vel = [];
    const sub_ = Math.max(1, Math.round(dtOut / hStep)), h = dtOut / sub_;
    let rr = r.slice(), vv = v.slice();
    for (let k = 0; k < n; k++) {
      pos.push(rr); vel.push(vv);
      for (let j = 0; j < sub_; j++) [rr, vv] = rk4Step(rr, vv, h);
    }
    return { jd0, dt: dtOut, pos, vel };
  }
  function sampleTraj(tr, jd) { // linear interpolation (fine at 10 s with Hermite-ish accuracy using velocity)
    const x = ((jd - tr.jd0) * 86400) / tr.dt, k = Math.max(0, Math.min(tr.pos.length - 2, Math.floor(x))), f = x - k;
    const p0 = tr.pos[k], p1 = tr.pos[k + 1], v0 = tr.vel[k], v1 = tr.vel[k + 1], h = tr.dt;
    // cubic Hermite
    const f2 = f * f, f3 = f2 * f, h00 = 2 * f3 - 3 * f2 + 1, h10 = f3 - 2 * f2 + f, h01 = -2 * f3 + 3 * f2, h11 = f3 - f2;
    return [0, 1, 2].map((i) => h00 * p0[i] + h10 * h * v0[i] + h01 * p1[i] + h11 * h * v1[i]);
  }

  // ---------- close approach ----------
  function goldenMin(f, a, b, tol) {
    const g = (Math.sqrt(5) - 1) / 2; let c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
    while (b - a > tol) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } }
    const x = (a + b) / 2; return { x, f: f(x) };
  }
  // posA(jd), posB(jd) -> {r,v}. Screens [jd0, jd1] with step (s). Returns events with miss < thresholdKm.
  function screenPair(pa, pb, jd0, jd1, stepSec, thresholdKm) {
    const step = stepSec / 86400, events = [];
    let prev = null;
    for (let jd = jd0; jd <= jd1 + 1e-9; jd += step) {
      const A = pa(jd), B = pb(jd); if (!A || !B) { prev = null; continue; }
      const dr = sub(B.r, A.r), dv = sub(B.v, A.v), d = norm(dr), rr = dot(dr, dv) / d;
      if (prev && prev.rr < 0 && rr >= 0 && Math.min(prev.d, d) < 0.5 * 15 * stepSec + thresholdKm) {
        const res = goldenMin((x) => { const a = pa(x), b = pb(x); return a && b ? norm(sub(b.r, a.r)) : 1e9; }, prev.jd, jd, 0.0005 / 86400);
        if (res.f < thresholdKm) {
          const a = pa(res.x), b = pb(res.x);
          events.push({ jd: res.x, miss: res.f, vrel: norm(sub(b.v, a.v)), relPos: sub(b.r, a.r), rA: a.r, vA: a.v });
        }
      }
      prev = { jd, d, rr };
    }
    return events;
  }

  // ---------- passes ----------
  function passes(posAt, st, jd0, jd1, stepSec, maskDeg) {
    const stE = geodeticToEcef(st.lat, st.lon, st.alt || 0), out = [], step = stepSec / 86400;
    const elAt = (jd) => { const r = posAt(jd); return r ? lookAngles(stE, st.lat, st.lon, eciToEcef(r, jd)).el : -90; };
    let prevEl = elAt(jd0), cur = prevEl > maskDeg ? { aos: jd0, maxEl: prevEl, tMax: jd0 } : null;
    const cross_ = (a, b) => { // refine crossing time by bisection
      let lo = a, hi = b; const sLo = elAt(lo) > maskDeg;
      for (let i = 0; i < 20; i++) { const m = (lo + hi) / 2; if ((elAt(m) > maskDeg) === sLo) lo = m; else hi = m; }
      return (lo + hi) / 2;
    };
    for (let jd = jd0 + step; jd <= jd1 + 1e-9; jd += step) {
      const el = elAt(jd);
      if (!cur && el > maskDeg) cur = { aos: cross_(jd - step, jd), maxEl: el, tMax: jd };
      if (cur && el > cur.maxEl) { cur.maxEl = el; cur.tMax = jd; }
      if (cur && el <= maskDeg && prevEl > maskDeg) { cur.los = cross_(jd - step, jd); cur.dur = (cur.los - cur.aos) * 86400; out.push(cur); cur = null; }
      prevEl = el;
    }
    if (cur) { cur.los = jd1; cur.dur = (cur.los - cur.aos) * 86400; cur.open = true; out.push(cur); }
    return out;
  }

  const OG = {
    DEG, RE, MU, RE84, MU84, OMEGA_E, jday, jdFromDate, dateFromJd, gmst, parseTLE, parseTLEText, fromElements, formatTLE, sgp4init, sgp4, propJd,
    sub, add, mul, dot, cross, norm, unit, eciToEcef, ecefToGeodetic, geodeticToEcef, lookAngles, rtn, rv2coe,
    propagateNumeric, sampleTraj, rk4Step, goldenMin, screenPair, passes, mod2pi,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = OG; else root.OG = OG;
})(typeof window !== "undefined" ? window : globalThis);
