import numpy as np

# Importation des fonctions de Bessel modifiées (I_0, I_1) 1st kind et (K_0, K_1) 2nd kind
from scipy.special import iv, kv # Version usuelle instable pour un grand ag_i / ag_h
from scipy.special import ive, kve # Version scaled pour éviter instabilité numérique 

# Constante
epsilon_0 = 8.854e-12 # permittivité du vide

# Exemple d'entrée ---> forward_rod(w, mixture**)
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

# Modèle en cylindres
def forward_rod(w, log_a_i, log_phi_i,
                      log_D_i, log_sigma_i, log_epsilon_i,
                      log_D_h, log_sigma_h, log_epsilon_h) : # en log10
    
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

    # Retour vers 10**log10(x)
    a_i = 10**log_a_i # rayon caractéristique inclusion
    phi_i = 10**log_phi_i # % minéraux conducteur inclusion
    D_h = 10**log_D_h # Coefficient diffusion milieu 
    D_i = 10**log_D_i # Coefficient diffusion inclusion
    sigma_h = 10**log_sigma_h # conductivité milieu
    sigma_i = 10**log_sigma_i # conductivité inclusion
    epsilon_h = 10**log_epsilon_h # permittivité diélectrique milieu
    epsilon_i = 10**log_epsilon_i # permittivité diélectrique inclusion

    n = 1 # direction radiale n_x = n_z = 1

    # Variables initiales
    K_h = sigma_h + 1j*w*epsilon_h # conductivité complexe milieu
    K_i = sigma_i + 1j*w*epsilon_i # conductivité complexe inclusion

    gamma_h = np.sqrt(1j*w/D_h + sigma_h/(epsilon_h*D_h)) # eq (18)
    gamma_i = np.sqrt(1j*w/D_i + sigma_i/(epsilon_i*D_i)) # eq (18)

    ag_i = a_i*gamma_i # produit utile dans eq (B-17 / B-18)
    ag_h = a_i*gamma_h # produit utile dans eq (B-17 / B-18)

    # Formulation de f_rod(w)
    K_h_over_K_i = np.divide(K_h, K_i)

    sigma_i_over_eps_i = np.divide(sigma_i, epsilon_i)

    sigma_h_over_eps_h = np.divide(sigma_h, epsilon_h)

    F_i_over_H_i = np.divide(1, gamma_i * np.divide(ive(int(0), ag_i), ive(int(1), ag_i)) - np.divide(1, a_i))

    E_h_over_G_h = np.divide(1, -1 * gamma_h * np.divide(kve(int(0), ag_h), kve(int(1), ag_h)) - np.divide(1, a_i))

    f_rod = -1 + np.divide(2j*w, np.divide(1, a_i) * K_h_over_K_i * sigma_i_over_eps_i * F_i_over_H_i \
                           - np.divide(1, a_i) * sigma_h_over_eps_h * E_h_over_G_h \
                            + 1j*w * (K_h_over_K_i + 1))
    
    # Formulation de K_eff
    K_eff = K_h * np.divide(1 + (n * phi_i * f_rod), 1 - (phi_i * f_rod))         

    return K_eff

