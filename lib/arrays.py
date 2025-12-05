"""Construction functions for Hamiltonian matrix and density matrix and
related routines."""

import numpy as np
from .auxiliary import ar, fermi

# -----------------------------------------------------------------------------
#       k-space
# -----------------------------------------------------------------------------

def k_space(N_k, a):
    """
    Returns k_x, k_y meshgrid arrays of the 2D square grid FBZ.

    N_k ... number of k-points in each direction
    a   ... lattice constant
    """
    k = np.linspace(-np.pi/a, np.pi/a, N_k, endpoint=False)
    k_x, k_y = np.meshgrid(k, k)
    return k_x, k_y

def k_space_new(N_k, a):
    """
    Returns k_x, k_y meshgrid arrays of the RBZ
    The grid is parametrised to retain a constant area element.

    N_k ... number of k-points in each direction
    a   ... lattice constant
    """
    # Reciprocal lattice vectors
    b_1 = np.array([2 * np.pi / (3 * a), 2 * np.pi / (3 * a)])
    b_2 = np.array([0.0, 2 * np.pi / a])
    # Parametrisation
    u = np.linspace(-0.5, 0.5, N_k, endpoint=False)
    v = np.linspace(-0.5, 0.5, N_k, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')
    # Linear combination
    kgrid = U[..., None] * b_1 + V[..., None] * b_2
    k_x = kgrid[..., 0]
    k_y = kgrid[..., 1]
    return k_x, k_y

# -----------------------------------------------------------------------------
#       Hamiltonian matrix elements
# -----------------------------------------------------------------------------

# Tight binding bands
def epsilon_x(k_x, k_y, t_sigma, t_pi, a):
    """
    Diagonal matrix element for p_x orbital.
    
    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    a         ... lattice constant
    """
    return -2 * (t_sigma * np.cos(k_x * a) + t_pi * np.cos(k_y * a))
    
def epsilon_y(k_x, k_y, t_sigma, t_pi, a):
    """
    Diagonal matrix element for p_y orbital.
    
    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    a         ... lattice constant
    """
    return -2 * (t_sigma * np.cos(k_y * a) + t_pi * np.cos(k_x * a))
    
# Electron-lattice coupling terms
def Delta_term(k, t, g, a, alpha, theta):
    """
    Generic term for intra-layer off-diagonal matrix element due to 
    electron-lattice coupling.

    k      ... wavevector component
    t      ... hopping integral
    g      ... electron-lattice coupling constant
    a      ... lattice constant
    alpha  ... PLD amplitude
    theta  ... PLD phase
    """
    return -2j * a * alpha * t * g * np.cos(k * a + np.pi/3) * np.exp(1j * (theta - np.pi/2))

def Delta_x(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x, theta_y):
    """
    Off-diagonal matrix element for p_x orbital.

    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    g_sigma   ... electron-lattice coupling constant for end-on hopping
    g_pi      ... electron-lattice coupling constant for side-on hopping
    a         ... lattice constant
    alpha     ... PLD amplitude
    theta_x   ... PLD phase for x-component
    theta_y   ... PLD phase for y-component
    """
    return Delta_term(k_x, t_sigma, g_sigma, a, alpha, theta_x) + Delta_term(k_y, t_pi, g_pi, a, alpha, theta_y)
    
def Delta_y(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x, theta_y):
    """
    Off-diagonal matrix element for p_y orbital.

    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    g_sigma   ... electron-lattice coupling constant for end-on hopping
    g_pi      ... electron-lattice coupling constant for side-on hopping
    a         ... lattice constant
    alpha     ... PLD amplitude
    theta_x   ... PLD phase for x-component
    theta_y   ... PLD phase for y-component
    """
    return Delta_term(k_y, t_sigma, g_sigma, a, alpha, theta_y) + Delta_term(k_x, t_pi, g_pi, a, alpha, theta_x)
    
# Interlayer hybridisation
def inter_aligned(t_al):
    """
    Interlayer coupling term for aligned stacking.
    t_al ... interlayer hopping integral
    """
    return -t_al

def inter_staggered(k_x, k_y, t_st, a):
    """
    Interlayer coupling term for staggered stacking.
    
    k_x, k_y ... wavevector components
    t_st      ... interlayer hopping integral
    a         ... lattice constant
    """
    return - t_st * (1 + np.exp(1j * k_x * a)) * (1 + np.exp(1j * k_y * a)) * np.exp(-0.5j * (k_x + k_y) * a)

# -----------------------------------------------------------------------------
#       Hamiltonian blocks
# -----------------------------------------------------------------------------

# Single layer block
def hamiltonian_single(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x, theta_y):
    """
    Returns the single layer Hamiltonian blocks H_x and H_y.

    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    g_sigma   ... electron-lattice coupling constant for end-on hopping
    g_pi      ... electron-lattice coupling constant for side-on hopping
    a         ... lattice constant
    alpha     ... PLD amplitude
    theta_x   ... PLD phase for x-component
    theta_y   ... PLD phase for y-component
    """
    q = 2 * np.pi / (3 * a)
    H_x = np.zeros((k_x.shape + (3, 3)), dtype='complex')
    H_y = np.zeros_like(H_x)
    for i in range(3):
        # Diagonal elements
        H_x[..., i, i] = epsilon_x(k_x + q * i, k_y + q * i, t_sigma, t_pi, a)
        H_y[..., i, i] = epsilon_y(k_x + q * i, k_y + q * i, t_sigma, t_pi, a)
        # Off-diagonal elements
        H_x[..., (i+1)%3, i] = Delta_x(k_x + q * i, k_y + q * i, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x, theta_y)
        H_x[..., i, (i+1)%3] = H_x[..., (i+1)%3, i].conj()
        H_y[..., (i+1)%3, i] = Delta_y(k_x + q * i, k_y + q * i, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x, theta_y)
        H_y[..., i, (i+1)%3] = H_y[..., (i+1)%3, i].conj()
    return H_x, H_y

# Interlayer coupling blocks
def coupling_aligned(k_x, t_al):
    """
    Returns interlayer coupling blocks for aligned stacking.
    k_x ... wavevector component
    t_al ... interlayer hopping integral
    """
    T_al_x = np.zeros((k_x.shape + (3, 3)), dtype='complex')
    T_al_y = np.zeros_like(T_al_x)
    T_al_x[..., ar(3), ar(3)] = inter_aligned(t_al)
    T_al_y[..., ar(3), ar(3)] = inter_aligned(t_al)
    return T_al_x, T_al_y

def coupling_staggered(k_x, k_y, t_st, a):
    """
    Returns interlayer coupling blocks for staggered stacking.
    k_x, k_y ... wavevector components
    t_st      ... interlayer hopping integral
    a         ... lattice constant
    """
    q = 2 * np.pi / (3 * a)
    T_st_x = np.zeros((k_x.shape + (3, 3)), dtype='complex')
    T_st_y = np.zeros_like(T_st_x)
    for i in range(3):
        T_st_x[..., i, i] = inter_staggered(k_x + q * i, k_y + q * i, t_st, a)
        T_st_y[..., i, i] = T_st_x[..., i, i].copy()
    return T_st_x, T_st_y

# -----------------------------------------------------------------------------
#       Full electronic Hamiltonian
# -----------------------------------------------------------------------------

def electronic_hamiltonian(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, t_al, t_st, a, alpha, theta_x, theta_y, pattern=None, PBC=None):
    """
    Constructs the full electronic Hamiltonian matrices H_x and H_y
    for a multilayer system.
    k_x, k_y ... wavevector components
    t_sigma   ... end-on hopping integral
    t_pi      ... side-on hopping integral
    g_sigma   ... electron-lattice coupling constant for end-on hopping
    g_pi      ... electron-lattice coupling constant for side-on hopping
    t_al      ... interlayer hopping integral for aligned stacking
    t_st      ... interlayer hopping integral for staggered stacking
    a         ... lattice constant
    alpha     ... list of PLD amplitudes for each layer
    theta_x   ... list of PLD phases for x-component for each layer
    theta_y   ... list of PLD phases for y-component for each layer
    pattern   ... list defining stacking pattern between layers
                  0 = aligned, 1 = staggered
    PBC       ... if True, applies periodic boundary conditions in stacking direction
    """
    # Number of layers
    L = len(alpha)
    if len(alpha) != len(theta_x) or len(alpha) != len(theta_y):
        raise ValueError('Inconsistent number of layers between alpha, theta_x, and theta_y')
    
    H_x = np.zeros((k_x.shape + (3 * L, 3 * L)), dtype='complex')
    H_y = np.zeros_like(H_x)

    # Single layer blocks
    for i in range(L):
        H_x[..., 3*i:3*(i+1), 3*i:3*(i+1)], H_y[..., 3*i:3*(i+1), 3*i:3*(i+1)] = hamiltonian_single(
            k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, alpha[i], theta_x[i], theta_y[i]
        )

    # Interlayer coupling blocks
    T_al_x, T_al_y = coupling_aligned(k_x, t_al)
    T_st_x, T_st_y = coupling_staggered(k_x, k_y, t_st, a)

    for i in range(L) if (PBC and L > 2) else range(L-1):
        if pattern[i] == 0:
            if i != L-1:
                H_x[..., 3*i:3*(i+1), 3*(i+1):3*(i+2)] = T_al_x
                H_x[..., 3*(i+1):3*(i+2), 3*i:3*(i+1)] = T_al_x
                H_y[..., 3*i:3*(i+1), 3*(i+1):3*(i+2)] = T_al_y
                H_y[..., 3*(i+1):3*(i+2), 3*i:3*(i+1)] = T_al_y
            else:
                H_x[..., 3*i:3*(i+1), 0:3] = T_al_x
                H_x[..., 0:3, 3*i:3*(i+1)] = T_al_x
                H_y[..., 3*i:3*(i+1), 0:3] = T_al_y
                H_y[..., 0:3, 3*i:3*(i+1)] = T_al_y
        elif pattern[i] == 1:
            if i != L-1:
                H_x[..., 3*i:3*(i+1), 3*(i+1):3*(i+2)] = T_st_x
                H_x[..., 3*(i+1):3*(i+2), 3*i:3*(i+1)] = T_st_x.conj()
                H_y[..., 3*i:3*(i+1), 3*(i+1):3*(i+2)] = T_st_y
                H_y[..., 3*(i+1):3*(i+2), 3*i:3*(i+1)] = T_st_y.conj()
            else:
                H_x[..., 3*i:3*(i+1), 0:3] = T_st_x
                H_x[..., 0:3, 3*i:3*(i+1)] = T_st_x.conj()
                H_y[..., 3*i:3*(i+1), 0:3] = T_st_y
                H_y[..., 0:3, 3*i:3*(i+1)] = T_st_y.conj()
    return H_x, H_y

# -----------------------------------------------------------------------------
#       Diagonalisation and density matrix
# -----------------------------------------------------------------------------

# Diagonalisation
def diagonalise(h_x, h_y):
    """Diagonalises the Hamiltonian matrices H_x and H_y and returns eigenvalues and eigenvectors."""
    D_x, U_x = np.linalg.eigh(h_x)
    D_y, U_y = np.linalg.eigh(h_y)
    return D_x, U_x, D_y, U_y

# Single-particle density matrix
def density_matrix(D_x, D_y, U_x, U_y, mu, beta):
    """Computes the single-particle density matrices rho_x and rho_y from the eigenvalues and eigenvectors.
    Here we assume a 3L x 3L Hamiltonian for L layers and normalise the density matrices accordingly."""
    L = D_x.shape[-1] // 3
    rho_x = fermi(D_x, mu, beta)
    rho_y = fermi(D_y, mu, beta)
    if U_x.ndim == 4:
        rho_x = (U_x * rho_x[..., None, :]) @ U_x.conj().transpose(0, 1, 3, 2)
        rho_y = (U_y * rho_y[..., None, :]) @ U_y.conj().transpose(0, 1, 3, 2)
    elif U_x.ndim == 3:
        rho_x = (U_x * rho_x[..., None]) @ U_x.conj().transpose(0, 2, 1)
        rho_y = (U_y * rho_y[..., None]) @ U_y.conj().transpose(0, 2, 1)
    return rho_x / (3 * L), rho_y / (3 * L)


    
