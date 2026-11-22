"""Physical constants used across STC."""
import math

PI = math.pi
TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0

# WGS-72 (required by SGP4 for consistency with TLE generation)
MU_WGS72 = 398600.8            # km^3/s^2
RE_WGS72 = 6378.135            # km
XKE = 60.0 / math.sqrt(RE_WGS72 ** 3 / MU_WGS72)
J2_WGS72 = 0.001082616
J3_WGS72 = -0.00000253881
J4_WGS72 = -0.00000165597

# WGS-84 (geodesy and the numerical propagator)
MU = 398600.4418               # km^3/s^2
RE = 6378.137                  # km
F_WGS84 = 1.0 / 298.257223563
J2 = 1.08262668e-3
OMEGA_EARTH = 7.292115e-5      # rad/s
G0 = 9.80665                   # m/s^2

SECONDS_PER_DAY = 86400.0
MINUTES_PER_DAY = 1440.0
