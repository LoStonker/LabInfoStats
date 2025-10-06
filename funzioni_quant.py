from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


#Queste sono per calcolare coefficienti e PSi totale nel caso conosca la forma analitica della distribuzione al tempo 0 e delle autofunzioni
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





#Queste sono per il metodo numerico
#Devo inserire un array x_grid = dominio, e un array potential che vada bene. potential deve essere funzione di x_grid.
#Phi_initial_array è l'array della distribuzione al tempo 0, cioè o scrivo la funzione con def e inserisco successivamente il dominio x_grid ad esempio oppure suca
def calculating_cn_sum (phi_initial_array, eigen_matrix, dx):
    #Phi initial array è la distribuzione al tempo 0 
    #Devo ricordarmi di normalizzare sia phi_initial che le eigen_functions
    numero_eigenfunctions = eigen_matrix.shape[1]          #questa mi dice il numero di colonne/autovettori
    c_n = np.zeros(numero_eigenfunctions, dtype=np.complex128)

    for n in range(numero_eigenfunctions):
        phi = eigen_matrix[:, n]                #questa estrae la colonna n-esima
        integrale = np.vdot(phi, phi_initial_array) * dx
        c_n[n] = integrale
    return c_n


def calculating_psi_sum (t, eigen_functions, cn_array, energies_array, h=1):
    fasi = np.exp(-1j * t * (energies_array/h))
    cn_fasi = cn_array * fasi

    #psi_t = eigen_functions @ cn_fasi   #Questo è il prodotto matrice vettore
    psi_t = np.matmul(eigen_functions, cn_fasi)
    return psi_t


def matrice_hamiltoniana(potential_array, x_grid, hbar=1.0, m=1.0):
    N = len(x_grid)
    if N != len(potential_array):
        raise ValueError("Gli array del potenziale e della griglia devono avere la stessa lunghezza.")
        
    dx = x_grid[1] - x_grid[0]

    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    T_mat = - (hbar**2 / (2 * m)) * \
            (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)) / dx**2
            
    V_mat = np.diag(potential_array)
    
    H = T_mat + V_mat
    return H
