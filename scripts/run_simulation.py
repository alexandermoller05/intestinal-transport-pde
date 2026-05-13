#!/usr/bin/env python3
"""
run_simulation.py
=================
Entry-point script: reproduces all five figures from the thesis.

  Figure 2 – Mass conservation error
  Figure 3 – Ideal PFR absorption (Vmax sweep)
  Figure 4 – ADM absorption at Pe=3331 (plug-flow limit recovery)
  Figure 5 – Absorbed fraction at t=3h vs viscosity
  Figure 6 – Isolated dispersion contribution ΔF

Outputs are saved to  ../figures/  relative to this file.

Usage
-----
    cd intestinal-transport-pde
    python scripts/run_simulation.py

Runtime
-------
Figures 2–4 : seconds.
Figures 5–6 : ~1–3 minutes (viscosity sweep over 20 log-spaced points,
              3 configurations per point; parallelisable).
"""

import sys
import os
import math
import numpy as np

# ── Ensure src/ is importable regardless of working directory ─────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.solver      import simulate
from src.transport   import (
    transport_params, tau_emptying_from_halflife, tau_R_from_Vmax,
    hours_to_tau, tau_to_hours, peclet,
)
from src.model       import mass_balance
from src.parameters  import (
    NX, RTOL, ATOL, RTOL_SWEEP, ATOL_SWEEP,
    M0_g, MW_AGU, Km_mM, rm, L,
)
from src.plotting    import (
    plot_mass_conservation,
    plot_pfr_absorption,
    plot_adm_absorption,
    plot_absorption_vs_viscosity,
    plot_dispersion_contribution,
)

FIGURES_DIR = os.path.join(_REPO, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Shared parameters ─────────────────────────────────────────────────────────
VMAX_LIST       = [4.0, 9.0, 16.0]   # mM/min
T_HALF_MIN      = 20.0               # gastric half-emptying time [min]
MU_WATER        = 1e-3               # water-like chyme [Pa·s]
MU_SWEEP        = np.logspace(-3, 1, 20)   # 20 points: 1e-3 … 10 Pa·s
VMAX_SWEEP      = 9.0                # mM/min  (fixed for Figs 5–6)
TAU_EMPTY_SWEEP = 10.0               # dimensionless (τ_emptying)

# ── Concentration scale (needed for KmII and τ_R) ────────────────────────────
import math as _math
_A    = _math.pi * rm**2
_V    = _A * L
C0_mM = (M0_g / MW_AGU) / (_V * 1e3)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run sweep returning absorbed fraction at t=3h
# ─────────────────────────────────────────────────────────────────────────────

def _absorbed_at_3h(mu, vmax, t_half_min, alpha=1.0, Pe_override=None):
    """Scalar: absorbed fraction at exactly t=3h."""
    res = simulate(
        mu, vmax, t_half_min,
        alpha=alpha, Pe_override=Pe_override,
        nx=NX, rtol=RTOL_SWEEP, atol=ATOL_SWEEP,
    )
    return float(np.interp(3.0, res["t_h"], res["absorbed_fraction"]))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Mass conservation error
# ─────────────────────────────────────────────────────────────────────────────

print("── Figure 2: mass conservation error ──────────────────────────────")
mu_list_fig2 = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
results_fig2 = {}

for mu in mu_list_fig2:
    label = fr"$\mu={mu:.0e}$ Pa·s ($Pe={peclet(mu):.0f}$)"
    res   = simulate(mu, VMAX_LIST[1], T_HALF_MIN, nx=NX, rtol=RTOL, atol=ATOL)
    # Build per-time-step mass-balance error series
    from scipy.integrate import solve_ivp
    from src.model import build_rhs
    from src.transport import tau_transfer as _tt, hours_to_tau, tau_R_from_Vmax
    tau_tr  = _tt(mu)
    tau_em  = tau_emptying_from_halflife(T_HALF_MIN)
    tau_R   = tau_R_from_Vmax(VMAX_LIST[1], C0_mM)
    KmII    = Km_mM / C0_mM
    Pe_val  = peclet(mu)
    tau_end = hours_to_tau(3.0)
    rhs, y0, n = build_rhs(tau_tr, tau_em, tau_R, KmII, Pe_val, NX)
    dxi = 1.0 / (NX - 1)
    t_eval = tau_end * np.linspace(0.0, 1.0, 800)
    sol = solve_ivp(rhs, (0.0, tau_end), y0, t_eval=t_eval,
                    method="BDF", rtol=RTOL, atol=ATOL)
    M_series = mass_balance(sol.y, n, dxi)
    t_h_series = tau_to_hours(sol.t)
    results_fig2[label] = dict(
        t_h=t_h_series,
        mb_error_series=np.abs(M_series - 1.0),
    )
    print(f"  μ={mu:.0e}  Pe={Pe_val:.0f}  max|M-1|={np.max(np.abs(M_series-1)):.2e}")

fig2 = plot_mass_conservation(results_fig2,
                               save_path=os.path.join(FIGURES_DIR, "fig2_mass_conservation.png"))
print("  Saved fig2_mass_conservation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Ideal PFR absorption (Pe → ∞  via  Pe_override=1e9)
# ─────────────────────────────────────────────────────────────────────────────

print("── Figure 3: ideal PFR absorption ─────────────────────────────────")
results_pfr = {}
for vmax in VMAX_LIST:
    res = simulate(MU_WATER, vmax, T_HALF_MIN,
                   Pe_override=1e9, nx=NX, rtol=RTOL, atol=ATOL)
    results_pfr[vmax] = res
    print(f"  Vmax={vmax}  absorbed(3h)={res['absorbed_g'][-1]:.1f}g  mb={res['mb_error']:.2e}")

fig3 = plot_pfr_absorption(results_pfr,
                            save_path=os.path.join(FIGURES_DIR, "fig3_pfr_absorption.png"))
print("  Saved fig3_pfr_absorption.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – ADM at Pe=3331 (water-like, μ=1e-3 Pa·s, α=1)
# ─────────────────────────────────────────────────────────────────────────────

print("── Figure 4: ADM at Pe=3331 ────────────────────────────────────────")
Pe_fig4   = peclet(MU_WATER, alpha=1.0)
results_adm = {}
for vmax in VMAX_LIST:
    res = simulate(MU_WATER, vmax, T_HALF_MIN,
                   alpha=1.0, nx=NX, rtol=RTOL, atol=ATOL)
    results_adm[vmax] = res
    print(f"  Vmax={vmax}  absorbed(3h)={res['absorbed_g'][-1]:.1f}g  mb={res['mb_error']:.2e}")

fig4 = plot_adm_absorption(results_adm, Pe=Pe_fig4,
                            save_path=os.path.join(FIGURES_DIR, "fig4_adm_absorption.png"))
print(f"  Saved fig4_adm_absorption.png  (Pe={Pe_fig4:.0f})")


# ─────────────────────────────────────────────────────────────────────────────
# Figures 5 & 6 – Viscosity sweep (3 configurations)
# ─────────────────────────────────────────────────────────────────────────────

print("── Figures 5 & 6: viscosity sweep ─────────────────────────────────")
print(f"  {len(MU_SWEEP)} viscosity points × 3 configs …")

# Override gastric emptying to match thesis (τ_emptying = 10)
# Back-calculate t_half from τ_emptying = g·L/u, g = ln2/(t_half·60)
from src.parameters import u as _u
_t_half_sweep = (np.log(2) / (TAU_EMPTY_SWEEP * _u / L)) / 60.0   # min

F_full   = np.empty(len(MU_SWEEP))   # α=1
F_pfr    = np.empty(len(MU_SWEEP))   # α=600  (plug-flow reference)
F_highpe = np.empty(len(MU_SWEEP))   # Pe=5000, α=1

for j, mu in enumerate(MU_SWEEP):
    F_full[j]   = _absorbed_at_3h(mu, VMAX_SWEEP, _t_half_sweep, alpha=1.0)
    F_pfr[j]    = _absorbed_at_3h(mu, VMAX_SWEEP, _t_half_sweep, alpha=600.0)
    F_highpe[j] = _absorbed_at_3h(mu, VMAX_SWEEP, _t_half_sweep,
                                   alpha=1.0, Pe_override=5000.0)
    print(f"  μ={mu:.2e}  F_full={F_full[j]:.3f}  F_pfr={F_pfr[j]:.3f}  "
          f"F_highpe={F_highpe[j]:.3f}")

delta_F = F_full - F_pfr

fig5 = plot_absorption_vs_viscosity(MU_SWEEP, F_full, F_pfr, F_highpe,
                                     save_path=os.path.join(FIGURES_DIR,
                                                             "fig5_absorption_vs_viscosity.png"))
print("  Saved fig5_absorption_vs_viscosity.png")

fig6 = plot_dispersion_contribution(MU_SWEEP, delta_F,
                                     save_path=os.path.join(FIGURES_DIR,
                                                             "fig6_dispersion_contribution.png"))
print("  Saved fig6_dispersion_contribution.png")

print("\n✓ All figures generated in", FIGURES_DIR)
