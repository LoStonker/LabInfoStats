from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def calculating_cn (N, function_in, phi_n, a, b, f_params={}, phi_params={}):
    c_n = []
    for n in range(1, N+1):
        #Per i cn serve il coniugato alla phi
        def integranda_reale(x):
            phi_val = phi_n(x, n, **phi_params)
            f_val = function_in(x, **f_params)
            prodotto = np.conj(phi_val) * f_val
            return np.real(prodotto)
        
        def integranda_immaginaria(x):
            phi_val = phi_n(x, n, **phi_params)
            f_val = function_in(x, **f_params)
            prodotto = np.conj(phi_val) * f_val
            return np.imag(prodotto)

        cn_real, _ = quad(integranda_reale, a, b)
        cn_imag, _ = quad(integranda_immaginaria, a, b)

        cn = cn_real + 1j*cn_imag
        c_n.append(cn)
    
    return c_n


# Psi(x, t) = sum[ c_n * (e^-iE_n t/h) * phi_n]
def psi(x, t, c_n, energy, phi, **params):
    total_psi = 0.0 + 0.0j

    for n_idx, cn in enumerate(c_n):
        n = n_idx + 1

        energia = energy(n, **params)
        phi_val = phi(x, n, **params)
        fase = np.exp(-1j * t * energia)   #con h = 1
        
        total_psi += cn * fase * phi_val
    
    return total_psi
