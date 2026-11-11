import numpy as np

def g_constraint(x):
    """Обмеження g(x) <= 0."""
    return 4.0 - x[0] - x[1]


def grad_g(x):
    """Градієнт обмеження: (-1, -1)."""
    return np.array([-1.0, -1.0])


def penalty_wrappers(F, gradF, r):
    """
    Зовнішній квадратичний штраф:
        Phi_r(x)      = F(x) + r * [max(0, g(x))]^2
        grad Phi_r(x) = gradF(x) + 2 r max(0, g(x)) grad g(x)
    """
    def Phi(x):
        s = max(0.0, g_constraint(x))
        return F(x) + r * s ** 2

    def gradPhi(x):
        s = max(0.0, g_constraint(x))
        return np.asarray(gradF(x), dtype=float) + 2.0 * r * s * grad_g(x)

    return Phi, gradPhi

def svenn_search(f, x, d, delta0=0.01):
    """Свенн уздовж d; повертає [a, b], що містить мінімум."""
    h0 = 0.0
    f0 = f(x + h0 * d)
    h1 = h0 + delta0
    f1 = f(x + h1 * d)

    if f1 < f0:
        step, h_prev, h_cur, f_cur = delta0, h0, h1, f1
    else:
        step = -delta0
        h_prev = h0
        h_cur = h0 + step
        f_cur = f(x + h_cur * d)
        if f_cur >= f0:
            return -delta0, delta0

    k = 1
    h_new = h_cur
    while k < 50:
        h_new = h_cur + (2.0 ** k) * step
        f_new = f(x + h_new * d)
        if f_new >= f_cur:
            break
        h_prev, h_cur, f_cur = h_cur, h_new, f_new
        k += 1
    return min(h_prev, h_new), max(h_prev, h_new)


def golden_section_search(f, x, d, a, b, eps=1e-4):
    tau = (np.sqrt(5.0) - 1.0) / 2.0
    h1 = b - tau * (b - a)
    h2 = a + tau * (b - a)
    f1 = f(x + h1 * d)
    f2 = f(x + h2 * d)
    while (b - a) > eps:
        if f1 <= f2:
            b = h2
            h2, f2 = h1, f1
            h1 = b - tau * (b - a)
            f1 = f(x + h1 * d)
        else:
            a = h1
            h1, f1 = h2, f2
            h2 = a + tau * (b - a)
            f2 = f(x + h2 * d)
    return (a + b) / 2.0


def fletcher_reeves_inner(Phi, gradPhi, x_start, eps=0.01, max_iter=100):
    """
    Метод Флетчера-Рівса з ЛР 9 для мінімізації Phi.
    Повертає: x, k (число кроків), n_f, n_grad, history, converged
    """
    calls = {"f": 0, "g": 0}

    def f(z):
        calls["f"] += 1
        return Phi(z)

    def grad(z):
        calls["g"] += 1
        return gradPhi(z)

    x = np.array(x_start, dtype=float)
    n = x.size
    g = grad(x)
    d = -g
    history = []
    converged = False
    k = 0

    while True:
        g_norm = np.linalg.norm(g)
        row = {"iter": k, "x": x.copy(), "Phi": f(x), "grad_norm": g_norm,
               "d": None, "h": None, "beta": None}
        history.append(row)

        if g_norm < eps:
            converged = True
            break
        if k >= max_iter:
            break

        a, b = svenn_search(f, x, d, delta0=0.01)
        h = golden_section_search(f, x, d, a, b, eps=1e-4)

        x_next = x + h * d
        g_next = grad(x_next)

        if (k + 1) % n == 0:
            beta = 0.0
        else:
            beta = np.dot(g_next, g_next) / np.dot(g, g)
        d_next = -g_next + beta * d

        row["d"], row["h"], row["beta"] = d.copy(), h, beta

        x, g, d = x_next, g_next, d_next
        k += 1

    return x, k, calls["f"], calls["g"], history, converged

def run_sumt(F, gradF, x0, r0=1.0, c=10.0, eps=0.01, eps_inner=None,
             max_iter_inner=500, max_outer=20):
    """
    Зовнішні ітерації: r_k = r0 * c^(k-1), теплий старт з x(r_{k-1}).
    Зупинка: r * [max(0, g)]^2 <= eps.
    """
    if eps_inner is None:
        eps_inner = eps

    r = r0
    x_curr = np.array(x0, dtype=float)
    protocol = []
    total_n_f = 0
    total_n_grad = 0
    r1_full_history = None

    for k in range(1, max_outer + 1):
        Phi, gradPhi = penalty_wrappers(F, gradF, r)
        x_next, fr_iters, n_f, n_grad, history_inner, conv = fletcher_reeves_inner(
            Phi, gradPhi, x_curr, eps=eps_inner, max_iter=max_iter_inner)

        if k == 1:
            r1_full_history = history_inner

        total_n_f += n_f
        total_n_grad += n_grad

        val_g = g_constraint(x_next)
        s = max(0.0, val_g)
        r_P = r * s ** 2

        protocol.append({
            "k": k, "r": r, "start_x": x_curr.copy(),
            "fr_iters": fr_iters, "n_f": n_f, "n_grad": n_grad,
            "converged": conv,
            "x_r": x_next.copy(), "F": F(x_next), "g": val_g,
            "r_P": r_P, "lam_est": 2.0 * r * s,
            "total_n_f": total_n_f, "total_n_grad": total_n_grad,
        })

        if r_P <= eps:
            break

        r *= c
        x_curr = x_next.copy()

    return protocol, r1_full_history


def project_to_boundary(x):
    """Ортогональна проєкція на межу x1 + x2 = 4."""
    g_val = g_constraint(x)
    return np.asarray(x, dtype=float) + np.array([g_val / 2.0, g_val / 2.0])