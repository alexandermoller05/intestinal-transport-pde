"""
parameters.py
=============
Physical constants, anatomical parameters, and numerical settings
for the 1D Axial Dispersion Model of glucose absorption in the
human small intestine.

All values follow Table 2 of the thesis (Møller Rivera, 2025/2026),
which inherits from Moxon et al. (2016) unless otherwise noted.
"""

# ── Physical constants ────────────────────────────────────────────────────────
kB   = 1.380649e-23   # Boltzmann constant [J/K]
T    = 310.0          # Body temperature [K]  (~37 °C)

# ── Molecular weights ─────────────────────────────────────────────────────────
MW_AGU = 162.141      # Anhydroglucose unit in starch [g/mol]
MW_GLU = 180.156      # Glucose [g/mol]

# ── Anatomical / flow parameters (Moxon et al., Table 2) ─────────────────────
L   = 2.85            # Small-intestine length [m]
rm  = 0.018           # Mean intestinal radius [m]  (1.8 cm)
d   = 2.0 * rm        # Diameter [m]
u   = 1.7e-4          # Mean axial chyme velocity [m/s]
f   = 12.0            # Surface-area amplification factor (folds/villi/microvilli)
rho = 1000.0          # Chyme density [kg/m³]
r0  = 0.38e-9         # Hydrodynamic radius of glucose molecule [m]

# ── Initial conditions ────────────────────────────────────────────────────────
M0_g = 50.0           # Initial starch mass in stomach [g]

# ── Enzymatic kinetics ────────────────────────────────────────────────────────
Km_mM = 9.0           # Michaelis constant [mM]
# Vmax swept in scripts; reference values: 4, 9, 16 mM/min

# ── Numerical settings ────────────────────────────────────────────────────────
NX          = 400     # Spatial nodes (grid-convergence study: <1% change 400→800)
RTOL        = 1e-8    # BDF relative tolerance (production runs)
ATOL        = 1e-11   # BDF absolute tolerance (production runs)
RTOL_SWEEP  = 1e-6    # BDF relative tolerance (parameter sweeps)
ATOL_SWEEP  = 1e-9    # BDF absolute tolerance (parameter sweeps)
N_TEVAL     = 800     # Number of output time points

# ── Simulation duration ───────────────────────────────────────────────────────
T_SIM_HOURS = 3.0     # Digestion window [h]
