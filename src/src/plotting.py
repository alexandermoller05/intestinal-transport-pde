"""
plotting.py
===========
Visualisation utilities reproducing all five figures from the thesis:

  Figure 2 – Mass conservation error  |M(τ)−1|  vs time
  Figure 3 – Ideal PFR absorption curves (Vmax sweep, Dax=0)
  Figure 4 – ADM absorption curves at Pe=3331 (water-like, α=1)
  Figure 5 – Absorbed fraction at t=3h vs viscosity (3 configurations)
  Figure 6 – Isolated dispersion contribution ΔF = F(α=1) − F(PFR)

Usage
-----
    from src.plotting import (
        plot_mass_conservation,
        plot_pfr_absorption,
        plot_adm_absorption,
        plot_absorption_vs_viscosity,
        plot_dispersion_contribution,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Style matching the thesis (serif, STIX math) ─────────────────────────────
_STYLE = {
    "font.family":      "serif",
    "mathtext.fontset": "stix",
    "font.size":        13,
    "axes.labelsize":   13,
    "axes.titlesize":   13,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.linewidth":   1.0,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Mass conservation error
# ─────────────────────────────────────────────────────────────────────────────

def plot_mass_conservation(results_by_mu: dict, save_path: str | None = None) -> plt.Figure:
    """
    Figure 2: |M(τ)−1| vs time for multiple viscosity values.

    Parameters
    ----------
    results_by_mu : dict
        Keys: viscosity label strings (e.g. "1e-03 Pa·s").
        Values: result dicts from solver.simulate(), each containing
                't_h' and 'mb_error_series' (array of |M-1| at each step).
    save_path : str, optional
        If given, save figure to this path.
    """
    from .model import mass_balance

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, res in results_by_mu.items():
            ax.semilogy(res["t_h"], res["mb_error_series"], lw=1.4, label=label)

        # Solver tolerance reference line
        ax.axhline(1e-8, ls="--", color="k", lw=0.9, label="Solver tolerance")

        ax.set_xlabel("Time [h]")
        ax.set_ylabel(r"$|M(\tau) - 1|$")
        ax.set_title("Mass conservation error — conservative solver")
        ax.set_xlim(0, 3)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Ideal PFR absorption (reproduced from Moxon et al.)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pfr_absorption(results_by_vmax: dict, save_path: str | None = None) -> plt.Figure:
    """
    Figure 3: Absorbed glucose [g] vs time for the ideal PFR (Pe → ∞).

    Parameters
    ----------
    results_by_vmax : dict
        Keys: Vmax values (float, mM/min).
        Values: result dicts with 't_h' and 'absorbed_g'.
    save_path : str, optional
    """
    linestyles = [":", "--", "-"]

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for (vmax, res), ls in zip(results_by_vmax.items(), linestyles):
            ax.plot(
                res["t_h"], res["absorbed_g"], lw=2, ls=ls,
                label=fr"$V_{{\max}}={vmax:g}$ mM/min",
            )

        ax.set_xlim(0, 3)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Absorbed glucose [g]")
        ax.set_title("Model 3 — Glucose Absorption (Ideal PFR)")
        ax.legend(frameon=False)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – ADM at Pe = 3331 (plug-flow limit recovery)
# ─────────────────────────────────────────────────────────────────────────────

def plot_adm_absorption(results_by_vmax: dict, Pe: float, save_path: str | None = None) -> plt.Figure:
    """
    Figure 4: ADM absorption curves at a given Pe (typically 3331).

    Parameters
    ----------
    results_by_vmax : dict
        Same structure as for plot_pfr_absorption.
    Pe : float
        Péclet number used (displayed in title).
    save_path : str, optional
    """
    linestyles = [":", "--", "-"]

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for (vmax, res), ls in zip(results_by_vmax.items(), linestyles):
            ax.plot(
                res["t_h"], res["absorbed_g"], lw=2, ls=ls,
                label=fr"$V_{{\max}}={vmax:g}$ mM/min",
            )

        ax.set_xlim(0, 3)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Absorbed glucose [g]")
        ax.set_title(f"Model 3: Glucose Absorption ($Pe = {Pe:.0f}$)")
        ax.legend(frameon=False)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Absorbed fraction at t = 3 h vs viscosity
# ─────────────────────────────────────────────────────────────────────────────

def plot_absorption_vs_viscosity(
    mu_vals: np.ndarray,
    F_full: np.ndarray,
    F_pfr: np.ndarray,
    F_highpe: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Figure 5: Dimensionless absorbed fraction at t=3h vs chyme viscosity.

    Three curves:
      α=1     (F_full)   – full Taylor–Aris dispersion
      α=600   (F_pfr)    – dispersion suppressed → plug-flow reference
      Pe=5000 (F_highpe) – high-Pe override (isolates wall pathway)

    Parameters
    ----------
    mu_vals : ndarray
        Viscosity values [Pa·s] (x-axis, log scale).
    F_full, F_pfr, F_highpe : ndarray
        Absorbed fractions at t=3h for each configuration.
    save_path : str, optional
    """
    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogx(mu_vals, F_full,   lw=2, color="tab:blue",
                    label=r"$\alpha=1$ (full Taylor–Aris)")
        ax.semilogx(mu_vals, F_pfr,    lw=2, color="tab:orange", ls="--",
                    label=r"$\alpha=600$ (plug-flow limit)")
        ax.semilogx(mu_vals, F_highpe, lw=2, color="tab:green",  ls="-.",
                    label=r"$Pe=5000$, $\alpha=1$ (high-$Pe$ override)")

        ax.set_xlabel(r"Viscosity $\mu$ [Pa·s]")
        ax.set_ylabel("Absorbed glucose (dimensionless) at 3 h")
        ax.set_title(
            r"Absorbed glucose after 3 h vs viscosity"
            "\n"
            r"($V_{\max}=9.0$ mM/min, $\tau_{\mathrm{emptying}}=10.0$, conservative)"
        )
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Isolated dispersion contribution  ΔF
# ─────────────────────────────────────────────────────────────────────────────

def plot_dispersion_contribution(
    mu_vals: np.ndarray,
    delta_F: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Figure 6: ΔF = F(α=1) − F(PFR) vs viscosity.

    Parameters
    ----------
    mu_vals : ndarray
        Viscosity values [Pa·s].
    delta_F : ndarray
        Absolute difference in absorbed fraction.
    save_path : str, optional
    """
    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.semilogx(mu_vals, delta_F, "o-", lw=1.8, ms=4,
                    color="tab:blue",
                    label=r"$F_{\alpha=1} - F_{\mathrm{PFR}}$")

        ax.set_xlabel(r"Viscosity $\mu$ [Pa·s]")
        ax.set_ylabel("Absolute difference in absorbed fraction")
        ax.set_title(
            r"Effect of axial dispersion: $F_{\alpha=1} - F_{\mathrm{PFR}}$"
        )
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
