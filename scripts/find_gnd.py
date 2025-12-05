"""Optimise L-layer system to it's 'nearest' stable state 
w.r.t. PLD parameters alpha, theta_x, theta_y"""

# -----------------------------------------------------------------------------
#    Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as  plt
from scipy.optimize import root

# Project library
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.arrays import  *
from lib.eom import *
from lib.observables import *
from lib.auxiliary import *

# -----------------------------------------------------------------------------
#    Input parameters
# -----------------------------------------------------------------------------

N_k = 100

t_sigma = -2
t_pi = 0.2
K = 1.44
a = 3.2341
q = 2 * np.pi / (3 * a)
fill = 2/3
beta = 40 # RT

g_sigma = 0.6
g_pi = g_sigma

t_al = 0.1
t_st = 0.05

# -----------------------------------------------------------------------------
#    PLD parameters, layer shifts and PBC
# -----------------------------------------------------------------------------

# Monolayer
#alpha = np.array([0.05])
#theta_x = np.array([0 * np.pi / 3])
#theta_y = np.array([2 * np.pi / 3])
#pattern = None
#PBC = None

# Bilayer
#                    1st layer,      2nd layer 
#alpha   = np.array([0.05,           0.05]) 
#theta_x = np.array([0. * np.pi / 3, 2. * np.pi / 3])
#theta_y = np.array([0. * np.pi / 3, 4. * np.pi / 3])
#pattern = [0, 0]
#PBC = None

# Trilayer
#                   1st layer,      2nd layer,     3rd layer
alpha   = np.array([0.05,          0.05,          0.05]) 
theta_x = np.array([0 * np.pi / 3, 2 * np.pi / 3, 0 * np.pi / 3])
theta_y = np.array([2 * np.pi / 3, 0 * np.pi / 3, 2 * np.pi / 3])
pattern = [0, 1, 1]
PBC = False

# Trilayer - ground state
#                   1st layer,  2nd layer,  3rd layer
#alpha   = np.array([0.04983735, 0.04957625, 0.04990018]) 
#theta_x = np.array([-0.1161348,  2.27881456, -0.06421723])
#theta_y = np.array([2.2105299,  -0.18441946,  2.15861234])
#pattern = [0, 1, 1]
#PBC = False

# Trilayer - metastable state
#                   1st layer,  2nd layer,  3rd layer
#alpha   = np.array([0.04979517,  0.04976127, -0.0499363 ]) 
#theta_x = np.array([-0.1212882,  2.23013255,  5.22434768])
#theta_y = np.array([2.2156833,  -0.13573745,  3.15323273])
#pattern = [0, 1, 1]
#PBC = False

# -----------------------------------------------------------------------------
#    Using EOM to find a stable state
# -----------------------------------------------------------------------------

# Build k-space
k_x, k_y = k_space_new(N_k, a)

# Iteration/convergence parameters
max_iter = 1000
iter = 0
threshold = 1e-3

# Iteration data
alpha_hist = [alpha]
theta_x_hist = [theta_x]
theta_y_hist = [theta_y]
norm_hist = []

# Iteration loop
converged = False
while not converged and iter < max_iter:
    iter += 1
    
    # Construct electronic Hamiltonian
    h_x, h_y = electronic_hamiltonian(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, 
                                    t_al, t_st, a, alpha, theta_x, theta_y, 
                                    PBC=PBC, pattern=pattern)
    
    # Diagonalise electronic Hamiltonian
    D_x, U_x, D_y, U_y = diagonalise(h_x, h_y)

    # Update chemical potential
    mu = chemical_potential(2.0, D_x, D_y, U_x, U_y, beta, fill)
    
    # Compute density matrix
    rho_x, rho_y = density_matrix(D_x, D_y, U_x, U_y, mu, beta)

    # Update alpha, theta_x, theta_y
    param = np.concatenate([alpha, theta_x, theta_y])
    param_new = root(update_all, param, args=(rho_x, rho_y, k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, K), tol=threshold).x
    alpha, theta_x, theta_y = param_new.reshape(3, -1)

    # Check for convergence
    norm = np.abs(param_new - param).max()
    converged = norm < threshold

    # Store history
    alpha_hist.append(alpha)
    theta_x_hist.append(theta_x)
    theta_y_hist.append(theta_y)
    norm_hist.append(norm)

    print(f"Iteration {iter:>3}: {alpha}, {theta_x}, {theta_y}, norm = {norm:5e}")

# Plot change of norm vs. iteration
if False:
    plt.semilogy(np.array(norm_hist), 'o-')
    plt.show()

# Print out results in a ready-to-use (copyable) format
print("Stable configuration found:")
print(80 * '-')
print(f"alpha   = np.{repr(alpha)}")
print(f"theta_x = np.{repr(theta_x)}")
print(f"theta_y = np.{repr(theta_y)}")
print(80 * '-')

# -----------------------------------------------------------------------------
#    Energies
# -----------------------------------------------------------------------------

# Compute energies
E_el = electronic_energy(h_x, rho_x) + electronic_energy(h_y, rho_y)
E_lat = lattice_energy(K, a, alpha)
E_tot = E_el + E_lat

print(f"Electronic energy:  {E_el :.8f}")
print(f"Lattice energy:     {E_lat:.8f}")
print(f"Total energy:       {E_tot:.8f}")
print(f"Chemical potential: {mu:.8f}")

