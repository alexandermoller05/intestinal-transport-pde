"""
intestinal-transport-pde  –  src package
=========================================
1D Axial Dispersion Model of glucose absorption in the human small intestine.
Møller Rivera, École Polytechnique, 2025/2026.
"""

from .parameters import *          # noqa: F401,F403
from .transport  import (          # noqa: F401
    diffusivity,
    mass_transfer_coeff,
    tau_transfer,
    axial_dispersion,
    peclet,
    transport_params,
    tau_emptying_from_halflife,
    tau_R_from_Vmax,
    hours_to_tau,
    tau_to_hours,
)
from .model  import build_rhs, mass_balance   # noqa: F401
from .solver import simulate                  # noqa: F401
