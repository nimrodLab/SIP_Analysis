# @Author: charles
# @Date:   2021-09-08 13:09:44
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 13:09:13

import numpy as np

# Constante
epsilon_0 = 8.854e-12 # permittivité du vide

# Exmeple d'entrée ---> forward_spherical(w, mixture**)
#mixture = { # en log10 
    #'log_a_i': np.log10(200e-4), # rayon caractéristique inclusion
    #'log_phi_i': np.log10(0.05), # % minéraux conducteur inclusion
    #'log_D_i': np.log10(1e-6), # Coefficient diffusion inclusion 
    #'log_sigma_i': np.log10(20), # conductivité inclusion
    #'log_epsilon_i': np.log10(10*epsilon_0), # permittivité diélectrique inclusion
    #'log_D_h': np.log10(1e-9), # Coefficient diffusion milieu
    #'log_sigma_h': np.log10(0.1), # conductivité milieu
    #'log_epsilon_h': np.log10(80*epsilon_0), # permittivité diélectrique milieu
#}

# Modèle sphérique
def forward_spherical(w, log_a_i, log_phi_i,
                      log_D_i, log_sigma_i, log_epsilon_i,
                      log_D_h, log_sigma_h, log_epsilon_h) :

    # Calcul vectoriel en tenseur pytorch
    w = w.unsqueeze(-1)
    log_a_i = log_a_i.unsqueeze(0)
    log_phi_i = log_phi_i.unsqueeze(0)
    log_D_i = log_D_i.unsqueeze(0)
    log_sigma_i = log_sigma_i.unsqueeze(0)
    log_epsilon_i = log_epsilon_i.unsqueeze(0)
    log_D_h = log_D_h.unsqueeze(0)
    log_sigma_h = log_sigma_h.unsqueeze(0)
    log_epsilon_h = log_epsilon_h.unsqueeze(0)

    a_i = 10**log_a_i # rayon caractéristique inclusion
    phi_i = 10**log_phi_i # % minéraux conducteur inclusion
    D_h = 10**log_D_h # Coefficient diffusion milieu 
    D_i = 10**log_D_i # Coefficient diffusion inclusion
    sigma_h = 10**log_sigma_h # conductivité milieu
    sigma_i = 10**log_sigma_i # conductivité inclusion
    epsilon_h = 10**log_epsilon_h # permittivité diélectrique milieu
    epsilon_i = 10**log_epsilon_i # permittivité diélectrique inclusion

    n = 2 # facteur de forme pour sphérique

    K_h = sigma_h + 1j*w*epsilon_h # conductivité complexe milieu
    K_i = sigma_i + 1j*w*epsilon_i # conductivité complexe inclusion

    gamma_h = np.sqrt(1j*w/D_h + sigma_h/(epsilon_h*D_h)) # eq (18)
    gamma_i = np.sqrt(1j*w/D_i + sigma_i/(epsilon_i*D_i)) # eq (18)

    ag_i = a_i*gamma_i # produit utile dans eq (C-16)
    ag_h = a_i*gamma_h # produit utile dans eq (C-16)

    F_i_over_H_i = a_i*(ag_i - np.tanh(ag_i)) / \
        (2*ag_i - ag_i**2 * np.tanh(ag_i) - 2*np.tanh(ag_i)) # eq (C-16) et (C-17)
    E_h_over_G_h = a_i*(ag_h + 1) / (ag_h**2 + 2*ag_h + 2) # eq (C-16) et (C-17)

    numerator = 3*1j*w
    term1 = (2 * sigma_h * E_h_over_G_h) / (a_i * epsilon_h)
    term2 = (2 * K_h * sigma_i * F_i_over_H_i) / (a_i * K_i * epsilon_i)
    term3 = 1j*w*(2 * K_h / K_i + 1)
    denominator = 2 * (term1 - term2 + term3)
    f_w = -0.5 + numerator/denominator # eq (26) ^^
    K_eff = K_h * (phi_i*f_w*n + 1) / (1 - phi_i*f_w) # eq (22) dans eq (26) pour isoler K_eff

    return K_eff # Conductivité complexe effective du milieu (roche)