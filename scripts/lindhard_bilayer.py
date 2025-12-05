"""Compute Lindhard response of hybridised bilayer"""

# -----------------------------------------------------------------------------
#    Imports
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm

# Project library
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.arrays import  diagonalise, k_space, epsilon_x, epsilon_y, inter_aligned, inter_staggered
from lib.auxiliary import fermi

# -----------------------------------------------------------------------------
#    Input parameters
# -----------------------------------------------------------------------------

t_sigma = -2.0 
t_pi = 0.2
fill = 2/3
a = 3.2341
beta = 40 # RT

# k-space size (2D FBZ)
Nk = 300

# Interlayer hybridisation
t_al = 0.3
t_st = 0.1
aligned = False

# q-space (step = pi / a / Nq)
C = 1 # Nk / Nq ratio
Nq = Nk // C // 2 # q: 0 -> pi/a

# Broadening
eta = 1e-8

# -----------------------------------------------------------------------------
#    2x2 Hamiltonian matrix
# -----------------------------------------------------------------------------

# k-space grid
kx, ky = k_space(Nk, a)

# Hamiltonian matrix
Hx = np.zeros((Nk, Nk, 2, 2), dtype=complex)
Hy = np.zeros((Nk, Nk, 2, 2), dtype=complex)

# Diagonal elements - TB dispersion
Ex = epsilon_x(kx, ky, t_sigma, t_pi, a)
Ey = epsilon_y(kx, ky, t_sigma, t_pi, a)
for i in range(2):
    Hx[..., i, i] = Ex
    Hy[..., i, i] = Ey

# Off-diagonal elements - interlayer hybridisation
if aligned:
    Hx[..., 0, 1] = inter_aligned(t_al)
    Hx[..., 1, 0] = inter_aligned(t_al)
    Hy[..., 0, 1] = inter_aligned(t_al)
    Hy[..., 1, 0] = inter_aligned(t_al)
else:
    Hx[..., 0, 1] = inter_staggered(kx, ky, t_st, a)
    Hx[..., 1, 0] = inter_staggered(kx, ky, t_st, a).conj()
    Hy[..., 0, 1] = inter_staggered(kx, ky, t_st, a)
    Hy[..., 1, 0] = inter_staggered(kx, ky, t_st, a).conj()

print("Hamiltonian built")

# -----------------------------------------------------------------------------
#    Diagonalisation, density matrix, chemical potential, Fermi distribution
# -----------------------------------------------------------------------------

Dx, Ux, Dy, Uy = diagonalise(Hx, Hy)
Udx = Ux.conj().transpose(0, 1, 3, 2)
Udy = Uy.conj().transpose(0, 1, 3, 2)
print("Hamiltonian diagonalised")

# The functions below are the same as in arrays.py, but those are adapted for 
# the CDW backfolded problems, while this is for a simpler 2x2 problem.
def density_matrix(Dx, Dy, Ux, Uy, mu, beta):
    rhox = fermi(Dx, mu, beta)
    rhoy = fermi(Dy, mu, beta)
    rhox = (Ux * rhox[..., None, :]) @ Ux.conj().transpose(0, 1, 3, 2)
    rhoy = (Uy * rhoy[..., None, :]) @ Uy.conj().transpose(0, 1, 3, 2)
    return rhox / 2, rhoy / 2

def filling(rho):
    return np.sum(np.trace(rho, axis1=-2, axis2=-1)).real / (rho.shape[0]**2)

def chemical_potential(mu, D_x, D_y, U_x, U_y, beta, fill):
    def _f(mu, D_x, D_y, U_x, U_y, beta, fill):
        rho_x, rho_y = density_matrix(D_x, D_y, U_x, U_y, mu, beta)
        return (filling(rho_x) + filling(rho_y)) / 2 - fill
    return fsolve(_f, mu, args=(D_x, D_y, U_x, U_y, beta, fill))[0]

mu = chemical_potential(2.0, Dx, Dy, Ux, Uy, beta, fill)
rhox, rhoy = density_matrix(Dx, Dy, Ux, Uy, mu, beta)
print(f"Chemical potential:: {mu:.3f}")
print(f"Filling p_x: {filling(rhox):.3f}")
print(f"Filling p_y: {filling(rhoy):.3f}")

# Fermi distribution
Fx, Fy = np.zeros_like(Dx), np.zeros_like(Dy)
Fx = fermi(Dx, mu, beta)
Fy = fermi(Dy, mu, beta)

# -----------------------------------------------------------------------------
#    Lindhard near q=0 (roll only by dq index in each direction)
# -----------------------------------------------------------------------------

Lx0 = np.zeros((2, 2), dtype=complex)
Ly0 = np.zeros((2, 2), dtype=complex)

# Overlap integral
UUx = np.roll(np.roll(Udx, 1, axis=0), 1, axis=1) @ Ux
UUx *= UUx.conj()
UUy = np.roll(np.roll(Udy, 1, axis=0), 1, axis=1) @ Uy
UUy *= UUy.conj()

# Loop over bands
for m in range(2):
    for n in range(2):
        num = np.roll(np.roll(Fx[..., m], 1, axis=0), 1, axis=1) - Fx[..., n]
        den = np.roll(np.roll(Dx[..., m], 1, axis=0), 1, axis=1) - Dx[..., n] + eta * 1j
        Lx0[m, n] = np.sum(num / den * UUx[..., m, n]) / (Nk**2)

        num = np.roll(np.roll(Fy[..., m], 1, axis=0), 1, axis=1) - Fy[..., n]
        den = np.roll(np.roll(Dy[..., m], 1, axis=0), 1, axis=1) - Dy[..., n] + eta * 1j
        Ly0[m, n] = np.sum(num / den * UUy[..., m, n]) / (Nk**2)     

L0 = np.sum(Lx0 + Ly0)
print("L(q≈0) = ", L0)

# -----------------------------------------------------------------------------
#    Lindhard on WINDOWED q-grid
# -----------------------------------------------------------------------------

# WINDOWED q-grid
q_left = 6 * Nq // 10
q_right = 18 * Nq // 25
q_window = range(q_left, q_right)

# Lindhard placeholder
Lx = np.zeros((len(q_window), len(q_window), 2, 2), dtype=complex)
Ly = np.zeros((len(q_window), len(q_window), 2, 2), dtype=complex)

# Loop over q-grid
for a, i in enumerate(tqdm(q_window)):
    for b, j in enumerate(q_window):
        
        # Overlap integrals
        UUx = np.roll(np.roll(Udx, i*C, axis=0), j*C, axis=1) @ Ux
        UUx *= UUx.conj()

        UUy = np.roll(np.roll(Udy, i*C, axis=0), j*C, axis=1) @ Uy
        UUy *= UUy.conj()

        # Loop over bands
        for m in range(2):
            for n in range(2):
                num = np.roll(np.roll(Fx[..., m], i*C, axis=0), j*C, axis=1) - Fx[..., n]
                den = np.roll(np.roll(Dx[..., m], i*C, axis=0), j*C, axis=1) - Dx[..., n] + eta * 1j
                Lx[a, b, m, n] = np.sum(num / den * UUx[..., m, n]) / (Nk**2)

                num = np.roll(np.roll(Fy[..., m], i*C, axis=0), j*C, axis=1) - Fy[..., n]
                den = np.roll(np.roll(Dy[..., m], i*C, axis=0), j*C, axis=1) - Dy[..., n] + eta * 1j
                Ly[a, b, m, n] = np.sum(num / den * UUy[..., m, n]) / (Nk**2)     

L = np.sum(Lx + Ly, axis=(2, 3))

# Normalise 
L /= L0

# -----------------------------------------------------------------------------
#    Save data
# -----------------------------------------------------------------------------

# Create path
filename = "lindhard_bilayer_"
if aligned:
    filename += "aligned_"
    filename += f"{t_al}"
else:
    filename += "staggered_"
    filename += f"{t_st}"

with open(f"data/{filename}.npy", "wb") as f:
    np.save(f, L)

# -----------------------------------------------------------------------------
#   Visualization
# -----------------------------------------------------------------------------

if False:
    plt.imshow(L.real, extent=(q_left / Nq, q_right / Nq, q_left / Nq, q_right / Nq), cmap='RdBu_r', origin='lower')
    plt.hlines(2/3, q_left/Nq, q_right/Nq, ls='--', lw=2, color='lime')
    plt.vlines(2/3, q_left/Nq, q_right/Nq, ls='--', lw=2, color='lime')
    plt.colorbar()
    plt.show()
