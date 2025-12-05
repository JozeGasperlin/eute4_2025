"""Computing observables from arrays"""

import numpy as np
from scipy.optimize import fsolve
from .arrays import density_matrix

# -----------------------------------------------------------------------------
#      Occupation, chemical potential
# -----------------------------------------------------------------------------

def filling(rho):
    """Compute the filling (occupation) from the density matrix rho."""
    return np.sum(np.trace(rho, axis1=-2, axis2=-1)).real / (rho.shape[0]**2)

def chemical_potential(mu, D_x, D_y, U_x, U_y, beta, fill):
    """Find chemical potential mu for given filling.
    mu ... initial guess for chemical potential
    D_x, D_y ... arrays of eigenvalues for p_x and p_y orbital spaces
    U_x, U_y ... arrays of eigenvectors for p_x and p_y orbital spaces
    beta ... inverse temperature
    fill ... target filling (occupation)
    """
    def _f(mu, D_x, D_y, U_x, U_y, beta, fill):
        rho_x, rho_y = density_matrix(D_x, D_y, U_x, U_y, mu, beta)
        return (filling(rho_x) + filling(rho_y)) / 2 - fill
    return fsolve(_f, mu, args=(D_x, D_y, U_x, U_y, beta, fill))[0]

# -----------------------------------------------------------------------------
#      Energy
# -----------------------------------------------------------------------------

def lattice_energy(K, a, alpha):
    """Compute the lattice energy.
    K ... stiffness constant
    a ... lattice constant
    alpha ... array of PLD displacement parameters
    """
    return np.sum((2 * K * a**2 * alpha**2)).real / len(alpha)

def electronic_energy(h, rho):
    """Compute the electronic energy.
    h ... Hamiltonian array
    rho ... single-particle density matrix
    """
    N_k = h.shape[0]
    return 2 * (np.sum(np.trace(h @ rho, axis1=-2, axis2=-1)) / N_k**2).real