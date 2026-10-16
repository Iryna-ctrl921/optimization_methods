import numpy as np
from src.operators import energy, energy_grad

def gradient_descent(noisy, lam, L, max_iter=200, tol=1e-6):
    h = 1.0 / L
    I = noisy.copy()
    n_f = n_grad = 0
    for k in range(1, max_iter + 1):
        g = energy_grad(I, noisy, lam)
        n_grad += 1
        g_norm = float(np.linalg.norm(g))
        I_new = I - h * g
        e_k = energy(I, noisy, lam)
        n_f += 1
        if g_norm <= tol:
            break
        I = I_new
    return {"I": I, "iterations": k, "n_f": n_f, "n_grad": n_grad, "E": energy(I, noisy, lam), "grad": g_norm}