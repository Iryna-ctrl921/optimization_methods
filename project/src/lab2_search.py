import math

TAU = (math.sqrt(5.0) - 1.0) / 2.0

def dichotomy(oracle, a, b, eps=1e-4, delta=None):
    if delta is None:
        delta = eps / 3.0
    k = 0
    while (b - a) > eps:
        x1 = (a + b - delta) / 2.0
        x2 = (a + b + delta) / 2.0
        f1 = oracle.value(x1)
        f2 = oracle.value(x2)
        if f1 <= f2:
            b = x2
        else:
            a = x1
        k += 1
    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "iterations": k, "n_f": oracle.n_f}

def golden_section(oracle, a, b, eps=1e-4):
    x1 = b - TAU * (b - a)
    x2 = a + TAU * (b - a)
    f1 = oracle.value(x1)
    f2 = oracle.value(x2)
    k = 0
    while (b - a) > eps:
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1
            x1 = b - TAU * (b - a)
            f1 = oracle.value(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + TAU * (b - a)
            f2 = oracle.value(x2)
        k += 1
    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "iterations": k, "n_f": oracle.n_f}