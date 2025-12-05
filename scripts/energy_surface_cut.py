"""Drawing the energy surface for switching of either the monolayer 
by itself or in the unit cell"""

# -----------------------------------------------------------------------------
#    Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as  plt
from tqdm import tqdm
from scipy.optimize import root

# Project library
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.arrays import  *
from lib.observables import *

# -----------------------------------------------------------------------------
#    Input parameters
# ----------------------------------------------------------------------------

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

pattern = [0, 1, 1]
PBC = False

setup = 'unit_cell' # 'monolayer', 'unit_cell'


# -----------------------------------------------------------------------------
#    Monolayer alpha, theta
# -----------------------------------------------------------------------------

if setup == 'monolayer':
    # Initial
    #                   1st layer
    alpha_start   = np.array([0.05001337])
    theta_x_start = np.array([-0.000173495])
    theta_y_start = np.array([2.09456859])

    # Final
    #                   1st layer
    alpha_final   = np.array([-0.05001337])
    theta_x_final = np.array([5.23616125])
    theta_y_final = np.array([3.14141916])

# -----------------------------------------------------------------------------
#    Unit cell - alpha, theta
# -----------------------------------------------------------------------------

if setup == 'unit_cell':
    # Initial - OX
    #                   1st layer,  2nd layer,  3rd layer
    alpha_start   = np.array([0.04983735, 0.04957625, 0.04990018]) 
    theta_x_start = np.array([-0.1161348,  2.27881456, -0.06421723])
    theta_y_start = np.array([2.2105299,  -0.18441946,  2.15861234])

    # Final - OB
    #                   1st layer,  2nd layer,  3rd layer
    alpha_final   = np.array([0.04979517,  0.04976127, -0.0499363 ]) 
    theta_x_final = np.array([-0.1212882,  2.23013255,  5.22434768])
    theta_y_final = np.array([2.2156833,  -0.13573745,  3.15323273])

# -----------------------------------------------------------------------------
#    Get ranges
# -----------------------------------------------------------------------------

N_in = 24
N_out = 12

alpha_diff   = (alpha_final   - alpha_start  ) / N_in
theta_x_diff = (theta_x_final - theta_x_start) / N_in
theta_y_diff = (theta_y_final - theta_y_start) / N_in

E = []
alphas = []
thetas_x = []
thetas_y = []

# -----------------------------------------------------------------------------
#    Computation
# -----------------------------------------------------------------------------

# Build k-space
k_x, k_y = k_space_new(N_k, a)

# Loop over energy surface points
for i in tqdm(range(-N_out, N_in + N_out)):
    
    alpha = alpha_start + i * alpha_diff
    theta_x = theta_x_start + i * theta_x_diff
    theta_y = theta_y_start + i * theta_y_diff

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
    
    # Compute energies
    E_el = electronic_energy(h_x, rho_x) + electronic_energy(h_y, rho_y)
    E_lat = lattice_energy(K, a, alpha)
    E_tot = E_el + E_lat

    E.append(E_tot)
    alphas.append(alpha[-1])
    thetas_x.append(theta_x[-1])
    thetas_y.append(theta_y[-1])


# -----------------------------------------------------------------------------
#    Save data
# -----------------------------------------------------------------------------

path = f"data/{setup}"

np.savetxt(f"{path}_alpha.txt", alphas)
np.savetxt(f"{path}_energy.txt", E)
np.savetxt(f"{path}_theta_x.txt", thetas_x)
np.savetxt(f"{path}_theta_y.txt", thetas_y)
