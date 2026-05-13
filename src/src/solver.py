"""
solver.py
=========
BDF time-integration wrapper for the 1D Axial Dispersion Model.

Uses scipy.integrate.solve_ivp with the implicit BDF method
(variable-order, variable-step), which handles the stiffness
introduced by the dispersion term (eigenvalue ~ Pe⁻¹/Δξ²).

Reference: Appendix A.3 and Section 5.4 of the thesis.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .parameters import (
    NX, RTOL, ATOL, N_TEVAL, T_SIM_HOURS,
    M0_g, MW_AGU, Km_mM, rm, L,
)
from .model     import build_rhs, mass_balance
from .transport import (
    transport_params,
    tau_emptying_from_halflife,
    tau_R_from_Vmax,
    hours_to_tau,
    tau_to_hours,
    diffusivity,
)


def simulate(
    mu: float,
    vmax_mM_per_min: float,
    t_half_min: float = 20.0,
    alpha: float = 1.0,
    Pe_override: float | None = None,
    nx: int = NX,
    rtol: float = RTOL,
    atol: float = ATOL,
    t_sim_hours: float = T_SIM_HOURS,
    n_teval: int = N_TEVAL,
) -> dict:
    """
    Run one simulation of the 1D Axial Dispersion Model.

    Parameters
    ----------
    mu : float
        Dynamic viscosity of chyme [Pa·s].
    vmax_mM_per_min : float
        Maximum enzymatic hydrolysis rate V_max [mM/min].
    t_half_min : float
        Gastric half-emptying time [min] (default 20 min).
    alpha : float
        Hydraulic-diameter scale factor for Taylor–Aris D_ax.
        α=1  → full dispersion (Eq. 20 of thesis).
        α≫1  → dispersion suppressed → plug-flow limit.
    Pe_override : float or None
        If given, overrides the computed Péclet number (used to
        construct the Pe=5000 reference curve in Figure 5).
    nx : int
        Number of spatial nodes.
    rtol, atol : float
        BDF solver tolerances.
    t_sim_hours : float
        Simulation duration [h].
    n_teval : int
        Number of output time points.

    Returns
    -------
    result : dict with keys:
        t_h            – physical time [h]
        tau            – dimensionless time τ
        absorbed_fraction – G'_abs(τ)  (fraction of initial starch mass)
        absorbed_g     – absorbed glucose [g]
        mb_error       – max |M(τ)-1| over simulation (mass-balance check)
        Pe_used        – Péclet number actually used
        tau_transfer   – dimensionless absorption parameter
        lumen_S        – Σ S'_i · Δξ  (lumen starch integral)
        lumen_G        – Σ G'_i · Δξ  (lumen glucose integral)
        S_stomach      – S'_s(τ)
        S_field        – S'(ξ, τ) array, shape (nx, n_times)
        G_field        – G'(ξ, τ) array
        xi             – dimensionless spatial grid
    """
    import numpy as np

    # ── Dimensional → dimensionless scalings ──────────────────────────────
    import math
    A      = math.pi * rm**2
    V_m3   = A * L
    C0_mM  = (M0_g / MW_AGU) / (V_m3 * 1e3)   # [mol/m³] → [mM]

    tau_tr, Pe_computed = transport_params(mu, alpha)
    Pe_used   = Pe_override if Pe_override is not None else Pe_computed
    tau_em    = tau_emptying_from_halflife(t_half_min)
    tau_R     = tau_R_from_Vmax(vmax_mM_per_min, C0_mM)
    KmII      = Km_mM / C0_mM
    tau_end   = hours_to_tau(t_sim_hours)

    # ── Build and integrate ───────────────────────────────────────────────
    rhs, y0, n = build_rhs(tau_tr, tau_em, tau_R, KmII, Pe_used, nx)

    t_eval = tau_end * np.linspace(0.0, 1.0, n_teval)
    sol = solve_ivp(
        rhs, (0.0, tau_end), y0,
        t_eval=t_eval, method="BDF",
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    # ── Unpack solution ───────────────────────────────────────────────────
    dxi    = 1.0 / (nx - 1)
    S_int  = sol.y[0:n, :]
    G_int  = sol.y[n:2 * n, :]
    Sst    = sol.y[2 * n,     :]
    Gabs   = sol.y[2 * n + 1, :]
    Sout   = sol.y[2 * n + 2, :]
    Gout   = sol.y[2 * n + 3, :]

    # Full spatial fields (add closed-inlet node = 0)
    S_field = np.zeros((nx, sol.t.size))
    G_field = np.zeros_like(S_field)
    S_field[1:, :] = S_int
    G_field[1:, :] = G_int

    lumen_S = dxi * np.sum(S_int, axis=0)
    lumen_G = dxi * np.sum(G_int, axis=0)

    # ── Mass-balance diagnostic ───────────────────────────────────────────
    M    = mass_balance(sol.y, n, dxi)
    mb_error = float(np.max(np.abs(M - 1.0)))

    return dict(
        t_h               = tau_to_hours(sol.t),
        tau               = sol.t,
        absorbed_fraction = Gabs,
        absorbed_g        = M0_g * Gabs,
        mb_error          = mb_error,
        Pe_used           = Pe_used,
        tau_transfer      = tau_tr,
        lumen_S           = lumen_S,
        lumen_G           = lumen_G,
        S_stomach         = Sst,
        S_out             = Sout,
        G_out             = Gout,
        S_field           = S_field,
        G_field           = G_field,
        xi                = np.linspace(0.0, 1.0, nx),
    )
