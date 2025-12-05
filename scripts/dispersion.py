"""Compute the dispersion relation of a given configuration along the high-symmetry path"""

# -----------------------------------------------------------------------------
#    Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as  plt

# Project library
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.arrays import  electronic_hamiltonian, diagonalise

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

geometry = 'meta' # 'ground', 'meta'

if geometry == 'ground':
    # Ground state
    #                   1st layer,  2nd layer,  3rd layer
    alpha   = np.array([0.04983735, 0.04957625, 0.04990018]) 
    theta_x = np.array([-0.1161348,  2.27881456, -0.06421723])
    theta_y = np.array([2.2105299,  -0.18441946,  2.15861234])
    pattern = [0, 1, 1]
    PBC = False
    mu = 2.04383382

elif geometry == 'meta':
    # Metastable state
    #                   1st layer,  2nd layer,  3rd layer
    alpha   = np.array([0.04979517,  0.04976127, -0.0499363 ]) 
    theta_x = np.array([-0.1212882,  2.23013255,  5.22434768])
    theta_y = np.array([2.2156833,  -0.13573745,  3.15323273])
    pattern = [0, 1, 1]
    PBC = False
    mu = 1.97929938 

# -----------------------------------------------------------------------------
#   Build k-space along high-symmetry path
# -----------------------------------------------------------------------------

k_x, k_y = np.array([]), np.array([])
ticks = [0]

# S - Gamma
k_x = np.concatenate((k_x, np.full(int(N_k * np.sqrt(2)), 0)))
k_y = np.concatenate((k_y, np.linspace(0, np.pi / a, int(N_k * np.sqrt(2)), endpoint=False)[::-1]))
ticks.append(len(k_x) - 1)
# Gamma - X
k_x = np.concatenate((k_x, np.linspace(0, np.pi / (2 * a), N_k, endpoint=False)))
k_y = np.concatenate((k_y, np.linspace(0, np.pi / (2 * a), N_k, endpoint=False)))
ticks.append(len(k_x) - 1)
# X - S
k_x = np.concatenate((k_x, np.linspace(0, np.pi / (2 * a), N_k, endpoint=False)[::-1]))
k_y = np.concatenate((k_y, np.linspace(np.pi / (2 * a), np.pi / a, N_k, endpoint=False)))
ticks.append(len(k_x) - 1)
# S - Y
k_x = np.concatenate((k_x, np.linspace(-np.pi / (2 * a), 0, N_k, endpoint=False)[::-1]))
k_y = np.concatenate((k_y, np.linspace(np.pi / (2 * a), np.pi / a, N_k, endpoint=False)[::-1]))
ticks.append(len(k_x) - 1)
# Y - Gamma
k_x = np.concatenate((k_x, np.linspace(-np.pi / (2 * a), 0, N_k, endpoint=False)))
k_y = np.concatenate((k_y, np.linspace(0, np.pi / (2 * a), N_k, endpoint=False)[::-1]))
ticks.append(len(k_x) - 1)

# Plot k-space path
#plt.scatter(k_x, k_y, c=np.arange(len(k_x)), cmap='jet')
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()

# Backfolded k-space
k_x_bf, k_y_bf = np.array([]), np.array([])
# S - Gamma
k_x_bf = np.concatenate((k_x_bf, np.full(int(N_k * np.sqrt(2)), -np.pi / a)))
k_y_bf = np.concatenate((k_y_bf, np.linspace(-np.pi / a, 0, int(N_k * np.sqrt(2)), endpoint=False)[::-1]))
# Gamma - X
k_x_bf = np.concatenate((k_x_bf, np.linspace(-np.pi/ a, -np.pi / (2 * a), N_k, endpoint=False)))
k_y_bf = np.concatenate((k_y_bf, np.linspace(-np.pi/ a, -np.pi / (2 * a), N_k, endpoint=False)))
# X - S
k_x_bf = np.concatenate((k_x_bf, np.linspace(-np.pi / a, -np.pi / (2 * a), N_k, endpoint=False)[::-1]))
k_y_bf = np.concatenate((k_y_bf, np.linspace(-np.pi / (2 * a), 0, N_k, endpoint=False)))
# S - Y
k_x_bf = np.concatenate((k_x_bf, np.linspace(np.pi / (2 * a), np.pi / a, N_k, endpoint=False)[::-1]))  
k_y_bf = np.concatenate((k_y_bf, np.linspace(-np.pi / (2 * a), 0, N_k, endpoint=False)[::-1]))
# Y - Gamma
k_x_bf = np.concatenate((k_x_bf, np.linspace(np.pi / (2 * a), np.pi / a, N_k, endpoint=False)))
k_y_bf = np.concatenate((k_y_bf, np.linspace(-np.pi / a, -np.pi / (2 * a), N_k, endpoint=False)[::-1]))

# Plot backfolded k-space
#plt.scatter(k_x_bf, k_y_bf, c=np.arange(len(k_x_bf)), cmap='jet')
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()

# -----------------------------------------------------------------------------
#    Computation
# -----------------------------------------------------------------------------

# Construct electronic Hamiltonian
h_x, h_y = electronic_hamiltonian(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, 
                                t_al, t_st, a, alpha, theta_x, theta_y, 
                                PBC=PBC, pattern=pattern)

# Diagonalise electronic Hamiltonian
D_x, U_x, D_y, U_y = diagonalise(h_x, h_y)


# Do the same for backfolded k-points
h_x_bf, h_y_bf = electronic_hamiltonian(k_x_bf, k_y_bf, t_sigma, t_pi, g_sigma, g_pi, 
                                t_al, t_st, a, alpha, theta_x, theta_y, 
                                PBC=PBC, pattern=pattern)
D_x_bf, U_x_bf, D_y_bf, U_y_bf = diagonalise(h_x_bf, h_y_bf)

# -----------------------------------------------------------------------------
#    Save data
# -----------------------------------------------------------------------------

np.save(f"data/D_x_{geometry}.npy", D_x)
np.save(f"data/D_y_{geometry}.npy", D_y)
np.save(f"data/D_x_bf_{geometry}.npy", D_x_bf)
np.save(f"data/D_y_bf_{geometry}.npy", D_y_bf)
np.save(f"data/mu_{geometry}.npy", mu)
np.save(f"data/idx_{geometry}.npy", np.array(ticks))

# -----------------------------------------------------------------------------
#   Plotting
# -----------------------------------------------------------------------------

if True:
    plt.figure(figsize=(8, 6))
    for i in range(D_x.shape[-1]):
        
        # Original bands - blue
        plt.plot(D_x[..., i].real - mu, c='b')
        plt.plot(D_y[..., i].real - mu, c='b')

        # Backfolded bands - red
        plt.plot(D_x_bf[..., i].real - mu, c='r', linestyle='dashed')
        plt.plot(D_y_bf[..., i].real - mu, c='r', linestyle='dashed')

    # Grid and ticks
    plt.hlines(0, 0, len(k_x), colors='k', linestyles='dashed')
    plt.vlines(ticks, ymin=-7, ymax=3, colors='k', linestyles='dashed')
    plt.xticks(ticks, ['S', r'$\Gamma$', 'X', 'S', 'Y', r'$\Gamma$'])
    
    # Limits
    plt.xlim(0, len(k_x) - 1)
    plt.ylim(-2, 2)
    plt.show()

