import numpy as np

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])
ALPHA, GAMMA, RHO, SIGMA = 1.0, 2.0, 0.5, 0.5

def nelder_mead(oracle, x0, step, tol=1e-7, max_iter=200):
    x0 = np.asarray(x0, dtype=float)
    simplex = [x0,
               np.clip(x0 + np.array([step, 0.0]), LO, HI),
               np.clip(x0 + np.array([0.0, 0.4 * step]), LO, HI)]
    values = [oracle.value(p) for p in simplex]
    
    for k in range(1, max_iter + 1):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        best, middle, worst = simplex
        f_best, f_middle, f_worst = values
        
        centroid = (best + middle) / 2.0
        x_r = np.clip(centroid + ALPHA * (centroid - worst), LO, HI)
        f_r = oracle.value(x_r)
        
        if f_r < f_best:
            x_e = np.clip(centroid + GAMMA * (x_r - centroid), LO, HI)
            f_e = oracle.value(x_e)
            if f_e < f_r:
                simplex[2], values[2] = x_e, f_e
            else:
                simplex[2], values[2] = x_r, f_r
        elif f_r < f_middle:
            simplex[2], values[2] = x_r, f_r
        else:
            if f_r < f_worst:
                x_c = np.clip(centroid + RHO * (x_r - centroid), LO, HI)
            else:
                x_c = np.clip(centroid + RHO * (worst - centroid), LO, HI)
            f_c = oracle.value(x_c)
            if f_c < min(f_r, f_worst):
                simplex[2], values[2] = x_c, f_c
            else:
                for i in (1, 2):
                    simplex[i] = np.clip(best + SIGMA * (simplex[i] - best), LO, HI)
                    values[i] = oracle.value(simplex[i])
                    
        i_best = int(np.argmin(values))
        size = max(float(np.linalg.norm(p - simplex[i_best])) for p in simplex)
        if size <= tol and (max(values) - min(values)) <= tol:
            break
            
    i_best = int(np.argmin(values))
    return {"x": simplex[i_best], "f": values[i_best], "n_f": oracle.n_f}