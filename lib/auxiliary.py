"""Auxilliary functions"""

import numpy as np

# -----------------------------------------------------------------------------
#       Physics
# -----------------------------------------------------------------------------

def fermi(E, mu, beta):
    """Fermi-Dirac distribution
    Use with scalar or 1D array E,
    i.e., watch out for broadcasting and zero elements."""
    return 1 / (np.exp(beta * (E - mu)) + 1)

# -----------------------------------------------------------------------------
#       Convenience/shorthand
# -----------------------------------------------------------------------------

def ar(n):
    """Shorthand for np.arange(n) to reduce code length."""
    return np.arange(n)