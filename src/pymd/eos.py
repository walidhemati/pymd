"""Johnson, Zollweg & Gubbins (1993) modified Benedict-Webb-Rubin equation of state for the
full (untruncated) Lennard-Jones fluid, plus the NIST SRSW reference data used in validation.

The 32 MBWR coefficients are transcribed from NIST's teqp (``johnson.hpp``) and agree digit
for digit with Clapeyron.jl. The pressure and residual-energy expressions follow the
standard MBWR form (Johnson et al. 1993, Mol. Phys. 78, 591).
"""

from __future__ import annotations

import numpy as np

GAMMA = 3.0
X = np.array([
    0.8623085097507421, 2.976218765822098, -8.402230115796038, 0.1054136629203555,
    -0.8564583828174598, 1.582759470107601, 0.7639421948305453, 1.753173414312048,
    2.798291772190376e03, -4.8394220260857657e-2, 0.9963265197721935, -3.698000291272493e01,
    2.084012299434647e01, 8.305402124717285e01, -9.574799715203068e02, -1.477746229234994e02,
    6.398607852471505e01, 1.603993673294834e01, 6.805916615864377e01, -2.791293578795945e03,
    -6.245128304568454, -8.116836104958410e03, 1.488735559561229e01, -1.059346754655084e04,
    -1.131607632802822e02, -8.867771540418822e03, -3.986982844450543e01, -4.689270299917261e03,
    2.593535277438717e02, -2.694523589434903e03, -7.218487631550215e02, 1.721802063863269e02,
])


def _ab(t: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([[0.0], X])  # 1-based
    st = np.sqrt(t)
    a = np.array([
        x[1] * t + x[2] * st + x[3] + x[4] / t + x[5] / t**2,
        x[6] * t + x[7] + x[8] / t + x[9] / t**2,
        x[10] * t + x[11] + x[12] / t,
        x[13],
        x[14] / t + x[15] / t**2,
        x[16] / t,
        x[17] / t + x[18] / t**2,
        x[19] / t**2,
    ])
    b = np.array([
        x[20] / t**2 + x[21] / t**3,
        x[22] / t**2 + x[23] / t**4,
        x[24] / t**2 + x[25] / t**3,
        x[26] / t**2 + x[27] / t**4,
        x[28] / t**2 + x[29] / t**3,
        x[30] / t**2 + x[31] / t**3 + x[32] / t**4,
    ])
    return a, b


def _cd(t: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([[0.0], X])
    st = np.sqrt(t)
    c = np.array([
        x[2] * st / 2 + x[3] + 2 * x[4] / t + 3 * x[5] / t**2,
        x[7] + 2 * x[8] / t + 3 * x[9] / t**2,
        x[11] + 2 * x[12] / t,
        x[13],
        2 * x[14] / t + 3 * x[15] / t**2,
        2 * x[16] / t,
        2 * x[17] / t + 3 * x[18] / t**2,
        3 * x[19] / t**2,
    ])
    d = np.array([
        3 * x[20] / t**2 + 4 * x[21] / t**3,
        3 * x[22] / t**2 + 5 * x[23] / t**4,
        3 * x[24] / t**2 + 4 * x[25] / t**3,
        3 * x[26] / t**2 + 5 * x[27] / t**4,
        3 * x[28] / t**2 + 4 * x[29] / t**3,
        3 * x[30] / t**2 + 4 * x[31] / t**3 + 5 * x[32] / t**4,
    ])
    return c, d


def _g(rho: float) -> np.ndarray:
    f = np.exp(-GAMMA * rho**2)
    g = np.zeros(6)
    g[0] = (1 - f) / (2 * GAMMA)
    for i in range(2, 7):
        g[i - 1] = -(f * rho ** (2 * (i - 1)) - 2 * (i - 1) * g[i - 2]) / (2 * GAMMA)
    return g


def jzg_pressure(rho: float, t: float) -> float:
    """Pressure P* of the full LJ fluid."""
    a, b = _ab(t)
    f = np.exp(-GAMMA * rho**2)
    i8 = np.arange(1, 9)
    i6 = np.arange(1, 7)
    return float(rho * t + np.sum(a * rho ** (i8 + 1)) + f * np.sum(b * rho ** (2 * i6 + 1)))


def jzg_energy(rho: float, t: float) -> float:
    """Residual (configurational) internal energy per particle U*/N."""
    c, d = _cd(t)
    i8 = np.arange(1, 9)
    return float(np.sum(c * rho**i8 / i8) + np.sum(d * _g(rho)))


#: NIST SRSW NVT Monte Carlo, LJ truncated at 3 sigma + standard long-range corrections,
#: N = 500 (https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm): (T, rho, U, dU, P, dP)
NIST_MC = [
    (0.85, 0.776, -5.5121, 4.55e-4, 6.7714e-3, 1.77e-3),
    (0.85, 0.780, -5.5386, 7.26e-4, 4.7924e-2, 3.18e-3),
    (0.85, 0.820, -5.7947, 6.03e-4, 5.5355e-1, 4.13e-3),
    (0.85, 0.860, -6.0305, 2.38e-3, 1.2660, 1.36e-2),
    (0.85, 0.900, -6.2391, 5.27e-3, 2.2314, 2.72e-2),
    (0.90, 0.776, -5.4689, 4.20e-4, 2.4056e-1, 2.74e-3),
    (0.90, 0.780, -5.4956, 7.86e-4, 2.7851e-1, 2.97e-3),
    (0.90, 0.820, -5.7456, 7.51e-4, 8.2386e-1, 2.85e-3),
    (0.90, 0.860, -5.9753, 5.53e-4, 1.5781, 3.29e-3),
    (0.90, 0.900, -6.1773, 1.57e-3, 2.5848, 9.54e-3),
]

#: LAMMPS bench/log.15Jul25.lj.fixed.g++.1 (in.lj, 32000 atoms): thermo output
LAMMPS_IN_LJ = {
    0: {"temp": 1.44, "epair": -6.7733681, "etotal": -4.6134356, "press": -5.0197073},
    100: {"temp": 0.7574531, "epair": -5.7585055, "etotal": -4.6223613, "press": 0.20726105},
    "ave_neighs_half": 37.588531,
    "neighbor_builds": 5,
    "matom_step_per_s_1core": 3.752,
}
