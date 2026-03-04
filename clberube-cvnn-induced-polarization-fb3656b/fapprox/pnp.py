import cmath

import torch
from torch.quasirandom import SobolEngine


# Définir les constantes et paramètres du modèle.
a = torch.tensor(0.1e-6)  # 0.1 #10 # m
epsilon_r = 80.0
epsilon_0 = 8.85e-12
mu = 5e-8  #  m^2/Vs
n_1 = 1.0  # mol/m^3
n_2 = 1.0
n_3 = 0.0
E_0 = 1.0  # V/m
k = 1.380649e-23  # SI
T = 293.0  # K
omega = 3e6  # rad/s
elec = 1.602e-19  # SI
alpha = 1e-10  # m^2/Vs
beta = 1e-2  # m/s
F = 96485  # SI


D = mu * k * T / elec  # coefficient de diffusion

z_1 = -1  # valence espèce 1
z_2 = 1  # valence espèce 2
z_3 = 1  # valence espèce 3

kappa = cmath.sqrt(2 * n_1 * elec * F / (epsilon_0 * epsilon_r * k * T))

lambda_1 = cmath.sqrt(1j * omega / D + kappa**2)
lambda_2 = cmath.sqrt(1j * omega / D)

f_2 = (lambda_1**2 * a**2 + 2 * lambda_1 * a + 2) / (lambda_1 * a + 1)
f_1 = f_2 * 1j * omega / (D * kappa**2)
f_3 = (lambda_2 * a + 1) / (lambda_2**2 * a**2 + 2 * lambda_2 * a + 2)

# Normalisation de certaines valeurs
L = 3 * a
U_0 = k * T / elec  # pas encore utilisé pour le moment, à venir...

e_over_k_T = elec / (k * T)
L2_over_D = L**2 / D
eps_over_F_L2 = epsilon_0 * epsilon_r / (F * L**2)

max_u = torch.tensor(
    [
        [
            2.8839304e-06,  # dn1_r
            2.6886275e-06,  # dn1_i
            2.8839299e-06,  # dn2_r
            2.6886275e-06,  # dn2_i
            1.9603998e-07,  # dU_r
            4.6415156e-08,  # dU_i
        ]
    ]
)


# Fonctions de Bessel modifiées
def k1(lam, r):
    lambda_r = lam * r
    ratio = 1 / lambda_r
    k1_ = (torch.pi / 2) * torch.exp(-lambda_r) * (ratio + ratio**2)
    return k1_


def k1_prime(lam, r):
    lambda_r = lam * r
    k1_ = -lam * k1(lam, r) - ((torch.pi * torch.exp(-lambda_r)) / 2 * r) * (
        1 / (lambda_r) + 2 / (lambda_r**2)
    )
    return k1_


# Coefficients
def E_omega():
    terme_E_1 = 3 * (1 + beta * a * f_3 / D) + (3 * n_3 / (n_3 - 2 * n_1)) * (
        alpha / mu - 1
    )
    terme_E_2 = (
        n_3
        / (n_3 - 2 * n_1)
        * (f_1 + alpha / mu * (f_2 - 2) + (beta * a * lambda_1**2 / (D * kappa**2)) + 2)
    )
    terme_E_3 = (2 + f_1) * (1 + beta * a * f_3 / D)
    E_w = E_0 * (1 + terme_E_1 / (terme_E_2 - terme_E_3))
    return E_w


def A_omega(E_w):
    k_1 = k1(lambda_1, a)
    terme_A_1 = E_0 * a - E_w * a
    terme_A_2 = -2 * F / (lambda_1**2 * epsilon_0 * epsilon_r) * k_1
    A_w = terme_A_1 / terme_A_2
    return A_w


def B_omega(E_w):
    terme_B_1 = (E_0 - E_w) / k1_prime(lambda_2, a)
    terme_B_2 = (lambda_1**2 * epsilon_0 * epsilon_r / (2 * F)) * f_2
    terme_B_3 = (mu / D) * n_1
    terme_B_4 = f_2 + (E_0 + 2 * E_w) / (E_0 - E_w)
    B_w = terme_B_1 * (terme_B_2 - terme_B_3 * terme_B_4)
    return B_w


def M_omega(E_w):
    terme_M_1 = 1 + (beta * a * f_3) / D
    terme_M_2 = -(n_3 / n_1) * (E_0 - E_w) / (k1_prime(lambda_2, a) * terme_M_1)
    terme_M_3 = lambda_1**2 * epsilon_0 * epsilon_r / (2 * F)
    terme_M_4 = f_2 + beta * a / D
    terme_M_5 = (n_1 / D) * (mu - alpha)
    terme_M_6 = f_2 + (E_0 + 2 * E_w) / (E_0 - E_w)
    M_w = terme_M_2 * (terme_M_3 * terme_M_4 - terme_M_5 * terme_M_6)
    return M_w


def Delta_n_1(A_w, B_w, theta, r):
    D_n_1 = (-A_w * k1(lambda_1, r) + B_w * k1(lambda_2, r)) * torch.cos(theta)
    return D_n_1


def Delta_n_2(A_w, B_w, M_w, theta, r):
    D_n_2 = (
        (n_2 / n_1) * (A_w * k1(lambda_1, r)) + (B_w - M_w) * k1(lambda_2, r)
    ) * torch.cos(theta)
    return D_n_2


def Delta_n_3(A_w, M_w, theta, r):
    D_n_3 = ((n_3 / n_1) * A_w * k1(lambda_1, r) + M_w * k1(lambda_2, r)) * torch.cos(
        theta
    )
    return D_n_3


def Delta_U(r, theta, A_w, E_w):
    terme_P_1 = -2 * F / (lambda_1**2 * epsilon_0 * epsilon_r)
    terme_P_2 = -E_0 * r + E_w * (a**3 / r**2)
    D_U_values = (terme_P_1 * A_w * k1(lambda_1, r) + terme_P_2) * torch.cos(theta)
    return D_U_values


def solution_PNP(x, y):

    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)

    r = r * L

    E_w = E_omega()
    A_w = A_omega(E_w)
    M_w = M_omega(E_w)
    B_w = B_omega(E_w)

    dU = Delta_U(r, theta, A_w, E_w)
    dn1 = Delta_n_1(A_w, B_w, theta, r)
    dn2 = Delta_n_2(A_w, B_w, M_w, theta, r)
    dn3 = Delta_n_3(A_w, M_w, theta, r)  # On voudra éventuellement calculer n3 aussi

    sol = torch.stack([dn1, dn2, dU], -1)

    return sol


soboleng = SobolEngine(2, scramble=True)


def generate_data(n_c, n_b):
    """Génère du data sur demande

    Args:
        n_c (int): nombre de points de collocation
        n_b (int): nombre de points sur les frontières

    Returns:
        tensor: tenseur de coordonnées
    """

    theta = torch.empty(n_c + 2 * n_b).uniform_(0, 2 * torch.pi)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    r_c = torch.empty(n_c).uniform_(a / L, 1)
    x_c, y_c = r_c * cos_t[:n_c], r_c * sin_t[:n_c]

    # Boundary points on surface of particle
    r_b = (a / L) * torch.ones(n_b)
    x_b1, y_b1 = r_b * cos_t[n_c:-n_b], r_b * sin_t[n_c:-n_b]

    # Boundary points at edges of domain
    x_b2, y_b2 = cos_t[-n_b:], sin_t[-n_b:]

    # Je retourne une concaténation des points de collocation et boundary
    # Note ici on a pas besoin des points sur les frontières alors
    # ils ne sont pas inclus
    x, y = torch.cat(
        [
            x_c,
        ]
    ), torch.cat(
        [
            y_c,
        ]
    )
    return x, y
