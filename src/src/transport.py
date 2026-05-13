"""
transport.py
============
Transport-coefficient calculations for the 1D Axial Dispersion Model.

Implements:
  - Stokes–Einstein molecular diffusivity  (Eq. 17 of thesis)
  - Graetz–Lévêque wall mass-transfer coefficient  (Eq. 16)
  - Taylor–Aris effective axial dispersion coefficient  (Eq. 20)
  - Dimensionless group map: viscosity → (τ_transfer, Pe)

All formulae follow Chapter 3 and Appendix A.2 of the thesis.
"""

import numpy as np
from .parameters import kB, T, r0, L, d, rm, u, f


# ── Molecular diffusivity ─────────────────────────────────────────────────────

def diffusivity(mu: float) -> float:
    """
    Stokes–Einstein molecular diffusivity of glucose [m²/s].

    D_m = k_B T / (6 π μ r_0)

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].

    Returns
    -------
    float
        D_m [m²/s].
    """
    return kB * T / (6.0 * np.pi * mu * r0)


# ── Wall mass-transfer coefficient ───────────────────────────────────────────

def mass_transfer_coeff(mu: float) -> float:
    """
    Graetz–Lévêque lumen-to-wall mass-transfer coefficient K [m/s].

    K = 1.62 · (u D_m² / (L d))^(1/3)

    Scales as K ∝ μ^{-2/3} (dominant viscosity pathway for absorption).

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].

    Returns
    -------
    float
        K [m/s].
    """
    Dm = diffusivity(mu)
    return 1.62 * ((u * Dm**2) / (L * d)) ** (1.0 / 3.0)


# ── Dimensionless absorption parameter ───────────────────────────────────────

def tau_transfer(mu: float) -> float:
    """
    Dimensionless wall-absorption parameter.

    τ_transfer = (2 f K / r_m) · (L / u)

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].

    Returns
    -------
    float
        τ_transfer (dimensionless).
    """
    K = mass_transfer_coeff(mu)
    return (2.0 * f * K / rm) * (L / u)


# ── Taylor–Aris axial dispersion coefficient ─────────────────────────────────

def axial_dispersion(mu: float, alpha: float = 1.0) -> float:
    """
    Taylor–Aris effective axial dispersion coefficient [m²/s].

    D_ax = D_m + u² d_h² / (192 D_m)

    where d_h = 2 r_m / α is a parametric hydraulic diameter.
    α = 1  → full cylindrical lumen (Taylor–Aris, Eq. 20).
    α ≫ 1  → hydrodynamic contribution suppressed (→ plug-flow limit).

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].
    alpha : float
        Hydraulic-diameter scale factor (parametric control; default 1).

    Returns
    -------
    float
        D_ax [m²/s].
    """
    Dm   = diffusivity(mu)
    d_h  = 2.0 * rm / max(alpha, 1.0)
    return Dm + (u**2 * d_h**2) / (192.0 * Dm)


# ── Péclet number ─────────────────────────────────────────────────────────────

def peclet(mu: float, alpha: float = 1.0) -> float:
    """
    Axial Péclet number  Pe = u L / D_ax.

    Pe → ∞ recovers the plug-flow (Moxon) limit.

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].
    alpha : float
        Hydraulic-diameter scale factor (default 1).

    Returns
    -------
    float
        Pe (dimensionless).
    """
    return u * L / axial_dispersion(mu, alpha)


# ── Combined map: μ → (τ_transfer, Pe) ───────────────────────────────────────

def transport_params(mu: float, alpha: float = 1.0) -> tuple:
    """
    Map viscosity to the two transport dimensionless groups.

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s].
    alpha : float
        Hydraulic-diameter scale factor (default 1).

    Returns
    -------
    tau_tr : float
        τ_transfer (dimensionless wall-absorption strength).
    Pe : float
        Axial Péclet number.
    """
    return tau_transfer(mu), peclet(mu, alpha)


# ── Dimensionless time-conversion helpers ─────────────────────────────────────

def tau_emptying_from_halflife(t_half_min: float) -> float:
    """
    Dimensionless gastric emptying parameter from half-life [min].

    τ_emptying = g · L/u,  where g = ln2 / (t_{1/2} · 60)
    """
    g = np.log(2.0) / (t_half_min * 60.0)
    return g * (L / u)


def tau_R_from_Vmax(vmax_mM_per_min: float, C0_mM: float) -> float:
    """
    Dimensionless reaction parameter.

    τ_R = (L/u) · (V_max / 60) / C_0
    """
    return (L / u) * (vmax_mM_per_min / C0_mM) / 60.0


def hours_to_tau(hours: float) -> float:
    """Convert physical time [h] to dimensionless τ."""
    return (u / L) * (hours * 3600.0)


def tau_to_hours(tau: np.ndarray) -> np.ndarray:
    """Convert dimensionless τ to physical time [h]."""
    return tau * (L / u) / 3600.0
