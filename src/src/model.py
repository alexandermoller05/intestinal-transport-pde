"""
model.py
========
Conservative finite-volume assembly of the 1D Axial Dispersion Model (ADM).

Governing dimensionless system (Section 4.5 of thesis):

  dS'_s/dτ  = -τ_emptying · S'_s

  ∂S'/∂τ    = -∂S'/∂ξ + (1/Pe) ∂²S'/∂ξ²
               - τ_R · S'/(K_mII + S')
               + τ_emptying · S'_s · δ(ξ)

  ∂G'/∂τ    = -∂G'/∂ξ + (1/Pe) ∂²G'/∂ξ²
               + τ_R · S'/(K_mII + S')
               - τ_transfer · G'

Spatial discretisation: conservative flux-divergence form (Section 5.2)
  J_{i+1/2} = C_i  -  (1/Pe) · (C_{i+1} - C_i) / Δξ
  dC_i/dτ   = -(J_{i+1/2} - J_{i-1/2}) / Δξ  +  sources_i

Boundary conditions (Section 3.7 / Eq. 46):
  Inlet  (ξ=0): advective face flux = 0; dispersive flux = 0 (Neumann)
  Outlet (ξ=1): dispersive face flux = 0 (Neumann); advective flux = C_N

State vector layout (Eq. 54):
  y = [S'_2, ..., S'_N,  G'_2, ..., G'_N,  S'_s,  G'_abs,  S'_out,  G'_out]
"""

import numpy as np


def build_rhs(
    tau_transfer_val: float,
    tau_empty_val: float,
    tau_R: float,
    KmII: float,
    Pe: float,
    nx: int,
):
    """
    Return the RHS callable  f(τ, y)  for scipy's solve_ivp.

    Parameters
    ----------
    tau_transfer_val : float
        Dimensionless wall-absorption parameter τ_transfer.
    tau_empty_val : float
        Dimensionless gastric-emptying parameter τ_emptying.
    tau_R : float
        Dimensionless reaction parameter.
    KmII : float
        Normalised Michaelis constant K_mII = K_m / S_s0.
    Pe : float
        Axial Péclet number.
    nx : int
        Number of spatial nodes (includes both boundary nodes).

    Returns
    -------
    rhs : callable
        RHS function with signature rhs(tau, y) -> dy/dτ.
    y0 : ndarray
        Initial state vector (all mass in stomach, intestine empty).
    n : int
        Number of interior unknowns per species (= nx - 1).
    """
    n    = nx - 1
    xi   = np.linspace(0.0, 1.0, nx)
    dxi  = xi[1] - xi[0]

    TT = tau_transfer_val
    TE = tau_empty_val
    TR = tau_R
    Km = KmII

    # ── Initial state: all starch in stomach, intestine empty ─────────────
    y0           = np.zeros(2 * n + 4)
    y0[2 * n]    = 1.0   # S'_s(0) = 1

    # ── Pre-allocate work arrays ───────────────────────────────────────────
    S     = np.empty(nx)
    G     = np.empty(nx)
    J_S   = np.empty(nx)
    J_G   = np.empty(nx)

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        S_int = y[0:n]
        G_int = y[n:2 * n]
        Sst   = y[2 * n]

        # Full fields (node 0 = closed inlet → 0)
        S[0]  = 0.0;  S[1:] = S_int
        G[0]  = 0.0;  G[1:] = G_int

        # ── Face fluxes: upwind advection + central dispersion ─────────────
        # Interior faces i = 0 … nx-2
        inv_Pe_dxi = 1.0 / (Pe * dxi)
        for i in range(nx - 1):
            J_S[i] = S[i] - inv_Pe_dxi * (S[i + 1] - S[i])
            J_G[i] = G[i] - inv_Pe_dxi * (G[i + 1] - G[i])

        # Outlet face (ξ=1): Neumann → pure advection
        J_S[nx - 1] = S[-1]
        J_G[nx - 1] = G[-1]

        # ── Michaelis–Menten hydrolysis ────────────────────────────────────
        R = TR * (S_int / (Km + S_int + 1e-50))

        # ── Gastric δ-source at inlet node (i=0, node index 1) ────────────
        srcS      = np.zeros(n)
        srcS[0]   = (TE * Sst) / dxi

        # ── Cell updates ───────────────────────────────────────────────────
        dS_int = np.empty(n)
        dG_int = np.empty(n)
        for i in range(n):
            node   = i + 1
            fl_S   = 0.0 if node == 1 else J_S[node - 1]
            fl_G   = 0.0 if node == 1 else J_G[node - 1]
            dS_int[i] = -(J_S[node] - fl_S) / dxi - R[i] + srcS[i]
            dG_int[i] = -(J_G[node] - fl_G) / dxi + R[i] - TT * G_int[i]

        # ── Integral states ────────────────────────────────────────────────
        dG_abs = TT * dxi * np.sum(G_int)   # glucose absorbed through wall
        dS_out = J_S[nx - 1]                # starch exiting at outlet
        dG_out = J_G[nx - 1]                # glucose exiting at outlet
        dSst   = -TE * Sst                  # stomach emptying

        return np.concatenate([dS_int, dG_int, [dSst, dG_abs, dS_out, dG_out]])

    return rhs, y0, n


def mass_balance(y: np.ndarray, n: int, dxi: float) -> np.ndarray:
    """
    Compute the total dimensionless mass M(τ) at every saved time step.

    M(τ) = S'_s + Δξ·Σ(S'_i + G'_i) + G'_abs + S'_out + G'_out

    Conservation requires M(τ) = 1 ∀ τ ≥ 0.

    Parameters
    ----------
    y : ndarray, shape (2n+4, n_times)
        Full solution array from solve_ivp.
    n : int
        Number of interior unknowns per species.
    dxi : float
        Dimensionless grid spacing.

    Returns
    -------
    M : ndarray, shape (n_times,)
    """
    S_int  = y[0:n, :]
    G_int  = y[n:2 * n, :]
    Sst    = y[2 * n,     :]
    Gabs   = y[2 * n + 1, :]
    Sout   = y[2 * n + 2, :]
    Gout   = y[2 * n + 3, :]

    return Sst + dxi * (np.sum(S_int, axis=0) + np.sum(G_int, axis=0)) + Gabs + Sout + Gout
