"""Hellmann-Feynman based equations of motion (EOM) for
alpha and theta parameters."""

import numpy as np
from .arrays import Delta_x, Delta_y, Delta_term

# -----------------------------------------------------------------------------
#       EOM for alpha
# -----------------------------------------------------------------------------

def dh_dalpha(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, theta_x, theta_y):
    """Derivative of Hamiltonian w.r.t. alpha parameter.
    
    k_x, k_y ... arrays of k-points
    t_sigma, t_pi ... intralayer hopping parameters
    g_sigma, g_pi ... intralayer coupling parameters
    a ... lattice constant
    theta_x, theta_y ... arrays of phase parameters
    """
    q = 2 * np.pi / (3 * a)
    N_k = k_x.shape[0]  
    dh_xdA, dh_ydA = [], []
    for i in range(len(theta_x)):
        dh_xi_dA = np.zeros((N_k, N_k, 3, 3), dtype='complex')
        dh_yi_dA = np.zeros_like(dh_xi_dA)
        # Derivative of Delta w.r.t. alpha is just Delta itself with alpha=1
        for j in range(3):
            dh_xi_dA[..., (j+1)%3, j] = Delta_x(k_x + q * j, k_y + q * j, t_sigma, t_pi, g_sigma, g_pi, a, 1, theta_x[i], theta_y[i])       
            dh_xi_dA[..., j, (j+1)%3] = dh_xi_dA[..., (j+1)%3, j].conj()
            dh_yi_dA[..., (j+1)%3, j] = Delta_y(k_x + q * j, k_y + q * j, t_sigma, t_pi, g_sigma, g_pi, a, 1, theta_x[i], theta_y[i])       
            dh_yi_dA[..., j, (j+1)%3] = dh_yi_dA[..., (j+1)%3, j].conj()
        dh_xdA.append(dh_xi_dA)
        dh_ydA.append(dh_yi_dA)
    return dh_xdA, dh_ydA

def update_alpha(alpha, rho_x, rho_y, k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, theta_x, theta_y, K):
    """Residual for alpha parameter update.
    
    alpha ... array of alpha parameters
    rho_x, rho_y ... single-particle density matrices for x and y polarizations
    k_x, k_y ... arrays of k-points
    t_sigma, t_pi ... intralayer hopping parameters
    g_sigma, g_pi ... intralayer coupling parameters
    a ... lattice constant
    theta_x, theta_y ... arrays of phase parameters
    K ... stiffness constant
    """
    alpha_new = np.zeros_like(alpha, dtype='complex')
    N_k = k_x.shape[0]
    dh_x_dA, dh_ydA = dh_dalpha(k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, theta_x, theta_y)
    for i in range(len(alpha)):
        dh_xi_dA = dh_x_dA[i]
        dh_yi_dA = dh_ydA[i]
        rho_xi = rho_x[..., 3*i:3*(i+1), 3*i:3*(i+1)]
        rho_yi = rho_y[..., 3*i:3*(i+1), 3*i:3*(i+1)]
        alpha_new[i] = np.sum(np.trace(dh_xi_dA @ rho_xi, axis1=-2, axis2=-1) 
                              + np.trace(dh_yi_dA @ rho_yi, axis1=-2, axis2=-1))
        alpha_new[i] /= -(2 * N_k**2 * K * a**2)
        alpha_new[i] *= len(alpha)
    return alpha_new.real - alpha # Subtract alpha to get the residual

# -----------------------------------------------------------------------------
#       EOM for theta
# -----------------------------------------------------------------------------

def dh_dtheta(k_x, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x):
    """Derivative of Hamiltonian w.r.t. theta parameter.
    
    k_x ... array of k-points
    t_sigma, t_pi ... intralayer hopping parameters
    g_sigma, g_pi ... intralayer coupling parameters
    a ... lattice constant
    alpha ... array of alpha parameters
    theta_x ... array of relevant theta parameters

    For derivative w.r.t. theta_x use: k_x, t_sigma, t_pi   , g_sigma, g_pi   , a, alpha, theta_x
    For derivative w.r.t. theta_y use: k_y, t_pi   , t_sigma, g_pi   , g_sigma, a, alpha, theta_y   
    """
    q = 2 * np.pi / (3 * a)
    N_k = k_x.shape[0]  
    dh_xdT, dh_ydT = [], []
    for i in range(len(theta_x)):
        dh_xi_dT = np.zeros((N_k, N_k, 3, 3), dtype='complex')
        dh_yi_dT = np.zeros_like(dh_xi_dT)
        for j in range(3):
            dh_xi_dT[..., (j+1)%3, j] = 1j * Delta_term(k_x + q * j, t_sigma, g_sigma, a, alpha[i], theta_x[i])
            dh_xi_dT[..., j, (j+1)%3] = dh_xi_dT[..., (j+1)%3, j].conj()
            dh_yi_dT[..., (j+1)%3, j] = 1j * Delta_term(k_x + q * j, t_pi   , g_pi   , a, alpha[i], theta_x[i])
            dh_yi_dT[..., j, (j+1)%3] = dh_yi_dT[..., (j+1)%3, j].conj()
        dh_xdT.append(dh_xi_dT)
        dh_ydT.append(dh_yi_dT)
    return dh_xdT, dh_ydT

def update_theta(theta_x, rho_x, rho_y, k_x, t_sigma, t_pi, g_sigma, g_pi, a, alpha):
    """Residual for theta parameter update.

    theta_x ... array of relevant theta parameters
    rho_x, rho_y ... single-particle density matrices for x and y polarizations
    k_x ... array of k-points
    t_sigma, t_pi ... intralayer hopping parameters
    g_sigma, g_pi ... intralayer coupling parameters
    a ... lattice constant
    alpha ... array of alpha parameters
    """
    residual = np.zeros_like(theta_x, dtype='complex')
    dh_x_dT, dh_ydT = dh_dtheta(k_x, t_sigma, t_pi, g_sigma, g_pi, a, alpha, theta_x)
    for i in range(len(theta_x)):
        dh_xi_dT = dh_x_dT[i]
        dh_yi_dT = dh_ydT[i]
        rho_xi = rho_x[..., 3*i:3*(i+1), 3*i:3*(i+1)]
        rho_yi = rho_y[..., 3*i:3*(i+1), 3*i:3*(i+1)]
        residual[i] = np.sum(np.trace(dh_xi_dT @ rho_xi, axis1=-2, axis2=-1)) + np.sum(np.trace(dh_yi_dT @ rho_yi, axis1=-2, axis2=-1))
    return 2 * residual.real

# -----------------------------------------------------------------------------
#       Combined EOM
# -----------------------------------------------------------------------------

def update_all(param, rho_x, rho_y, k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, K):
    """Residual for all parameters update."""
    alpha, theta_x, theta_y = param.reshape(3, -1)
    alpha_new = update_alpha(alpha, rho_x, rho_y, k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, theta_x, theta_y, K)
    theta_x_new = update_theta(theta_x, rho_x, rho_y, k_x, t_sigma, t_pi, g_sigma, g_pi, a, alpha)
    theta_y_new = update_theta(theta_y, rho_x, rho_y, k_y, t_pi, t_sigma, g_pi, g_sigma, a, alpha)
    return np.array([alpha_new, theta_x_new, theta_y_new]).ravel()

def update_both_theta(param, rho_x, rho_y, k_x, k_y, t_sigma, t_pi, g_sigma, g_pi, a, alpha):
    """Residual for both theta parameters update."""
    theta_x, theta_y = param.reshape(2, -1)
    theta_x_new = update_theta(theta_x, rho_x, rho_y, k_x, t_sigma, t_pi, g_sigma, g_pi, a, alpha)
    theta_y_new = update_theta(theta_y, rho_x, rho_y, k_y, t_pi, t_sigma, g_pi, g_sigma, a, alpha)
    return np.array([theta_x_new, theta_y_new]).ravel()
