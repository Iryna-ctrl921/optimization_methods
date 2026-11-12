import numpy as np
from functools import partial
from penalty import fletcher_reeves_inner

def penalty_general(F, gradF, r, ineq=(), eq=()):
    def Phi(x):
        v = F(x)
        for g, _ in ineq:
            v += r * max(0.0, g(x)) ** 2
        for h, _ in eq:
            v += r * h(x) ** 2
        return v

    def gradPhi(x):
        v = np.asarray(gradF(x), dtype=float).copy()
        for g, gg in ineq:
            s = max(0.0, g(x))
            if s > 0.0:
                v += 2.0 * r * s * np.asarray(gg(x), dtype=float)
        for h, gh in eq:
            v += 2.0 * r * h(x) * np.asarray(gh(x), dtype=float)
        return v

    return Phi, gradPhi

def run_sumt_general(F, gradF, x0, ineq=(), eq=(), r0=1.0, c=10.0, eps=0.01,
                     eps_inner=None, inner=fletcher_reeves_inner,
                     max_iter_inner=50, max_outer=40):
    """SUMT; стоп: r*P <= eps. Теплий старт. inner - внутрішній метод."""
    if eps_inner is None:
        eps_inner = eps
    r, x_curr = r0, np.array(x0, dtype=float)
    protocol, tot_f, tot_g = [], 0, 0

    for k in range(1, max_outer + 1):
        Phi, gradPhi = penalty_general(F, gradF, r, ineq, eq)
        eps_line, retries = 1e-4, 0
        x, it, nf, ng, hist, conv = inner(Phi, gradPhi, x_curr, eps=eps_inner,
                                          max_iter=max_iter_inner, eps_line=eps_line)
        while (not conv) and retries < 3:
            retries += 1
            eps_line *= 0.01
            x, it2, nf2, ng2, hist, conv = inner(Phi, gradPhi, x, eps=eps_inner,
                                                 max_iter=max_iter_inner, eps_line=eps_line)
            it += it2; nf += nf2; ng += ng2
        tot_f += nf
        tot_g += ng
        s = [max(0.0, g(x)) for g, _ in ineq]
        hv = [h(x) for h, _ in eq]
        P = sum(v * v for v in s) + sum(v * v for v in hv)
        viol = max([0.0] + s + [abs(v) for v in hv])
        protocol.append({
            "k": k, "r": r, "x": x.copy(), "F": F(x), "iters": it, "nf": nf, "ng": ng,
            "conv": conv, "retries": retries, "eps_line": eps_line, "s": s, "h": hv, "rP": r * P, "viol": viol,
            "lam": [2.0 * r * v for v in s] + [2.0 * r * v for v in hv],
            "tot_f": tot_f, "tot_g": tot_g, "hist": hist})
        if r * P <= eps:
            break
        r *= c
        x_curr = x.copy()
    return protocol

def barrier(F, gradF, cons, mu):
    def Phi(x):
        gs = [g(x) for g, _ in cons]
        if max(gs) >= 0:
            return float("inf")
        return F(x) - mu * sum(np.log(-v) for v in gs)

    def gradPhi(x):
        v = np.asarray(gradF(x), dtype=float).copy()
        for g, gg in cons:
            v += -mu / g(x) * np.asarray(gg(x), dtype=float)
        return v

    return Phi, gradPhi

def run_interior(F, gradF, cons, x0, mu0=10.0, c=10.0, eps=0.01, eps_inner=None,
                 max_iter_inner=2000, max_outer=40):
    """Стоп: mu*m <= eps (m - кількість обмежень). Старт строго всередині Omega."""
    if eps_inner is None:
        eps_inner = eps
    x_curr, mu = np.array(x0, dtype=float), mu0
    assert max(g(x_curr) for g, _ in cons) < 0, "старт має бути строго всередині Omega"
    inner = partial(fletcher_reeves_inner, guard_inf=True)
    protocol, tot_f, tot_g = [], 0, 0

    for k in range(1, max_outer + 1):
        Phi, gradPhi = barrier(F, gradF, cons, mu)
        x, it, nf, ng, hist, conv = inner(Phi, gradPhi, x_curr, eps=eps_inner,
                                          max_iter=max_iter_inner)
        tot_f += nf
        tot_g += ng
        gs = [g(x) for g, _ in cons]
        protocol.append({"k": k, "mu": mu, "x": x.copy(), "F": F(x), "g": gs,
                         "lam": [-mu / v for v in gs], "iters": it, "nf": nf, "ng": ng,
                         "conv": conv, "tot_f": tot_f, "tot_g": tot_g})
        if mu * len(cons) <= eps:
            break
        mu /= c
        x_curr = x.copy()
    return protocol

def sphere_descent(Phi, gradPhi, x_start, eps=0.01, max_iter=2000, h0=1.0):
    x = np.array(x_start, dtype=float)
    Fx = Phi(x)
    h = h0
    n_f, n_g = 1, 0
    for k in range(1, max_iter):
        g = gradPhi(x)
        n_g += 1
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
            return x, k - 1, n_f, n_g, [], True
        d = -g / norm_g
        h_curr = h
        while True:
            xb = x + h_curr * d
            Fb = Phi(xb)
            n_f += 1
            if Fb < Fx:
                break
            h_curr /= 2.0
            if h_curr < 1e-10:         
                return x, k - 1, n_f, n_g, [], False
        x, Fx, h = xb, Fb, h_curr
    return x, max_iter - 1, n_f, n_g, [], False