import numpy as np
from src.operators import energy, energy_grad

def project(I, lower=0.0, upper=1.0):
    return np.clip(I, lower, upper)

def projected_gradient(raw, lam, L, max_iter=200, tol=1e-6):
    h = 1.0 / L
    I = project(raw.copy())
    n_f = n_grad = 0
    for k in range(1, max_iter + 1):
        g = energy_grad(I, raw, lam)
        n_grad += 1
        I_new = project(I - h * g)
        g_map = float(np.linalg.norm(I_new - I) / h)
        e_k = energy(I, raw, lam)
        n_f += 1
        if g_map <= tol:
            break
        I = I_new
    return {"I": I, "iterations": k, "n_f": n_f, "n_grad": n_grad, "E": energy(I, raw, lam), "G": g_map}