import numpy as np

# Constante
epsilon_0 = 8.854e-12 # permittivité du vide

# Exmeple d'entrée ---> forward_sheet(w, mixture**)
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

# Modèle en feuillets
def forward_sheet(w, log_a_i, log_phi_i,
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

    n_z = 0 # direction perpendiculaire au feuillet (axe z)
    
    # Variables initiales
    K_h = sigma_h + 1j*w*epsilon_h # conductivité complexe milieu
    K_i = sigma_i + 1j*w*epsilon_i # conductivité complexe inclusion

    gamma_h = np.sqrt(1j*w/D_h + sigma_h/(epsilon_h*D_h)) # eq (18)
    gamma_i = np.sqrt(1j*w/D_i + sigma_i/(epsilon_i*D_i)) # eq (18)

    ag_i = a_i*gamma_i # produit utile dans eq (A-10 / A-11)
    ag_h = a_i*gamma_h # produit utile dans eq (A-10 / A-11)

    # Formulation de f_sheet(w)
    K_h_over_K_i = np.divide(K_h, K_i)

    sigma_i_over_eps_i = np.divide(sigma_i, epsilon_i)

    sigma_h_over_eps_h = np.divide(sigma_h, epsilon_h)

    F_i_over_H_i = np.divide(np.tanh(ag_i), gamma_i)

    E_h_over_G_h = np.divide(-1, gamma_i)

    f_sheet = (1 - K_h_over_K_i) + np.divide(1j, w * a_i) * \
        ((K_h_over_K_i * sigma_i_over_eps_i * F_i_over_H_i) - (sigma_h_over_eps_h * E_h_over_G_h))

    # Formulation de K_eff
    K_eff = np.divide(K_h, 1 - (phi_i * f_sheet))

    return K_eff