"""Collision probability and risk classification.

Screening-grade model: the combined position uncertainty in the encounter plane is
assumed isotropic Gaussian with standard deviation sigma, and the hard-body radius
HBR is small compared with sigma. Then

    Pc ~= HBR^2 / (2 sigma^2) * exp(-d^2 / (2 sigma^2))

Real operations use the full covariance from CDMs (see the autoca roadmap).
"""
import math


def pc_isotropic(miss_km: float, sigma_m: float = 300.0, hbr_m: float = 10.0) -> float:
    s = sigma_m / 1000.0
    hbr = hbr_m / 1000.0
    return min(1.0, hbr * hbr / (2 * s * s) * math.exp(-(miss_km ** 2) / (2 * s * s)))


def risk_level(miss_km: float, pc: float) -> str:
    if miss_km < 1.0 or pc > 1e-4:
        return "HIGH"
    if miss_km < 2.5 or pc > 1e-6:
        return "MEDIUM"
    return "LOW"
