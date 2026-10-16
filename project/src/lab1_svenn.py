def svenn(oracle, x0, delta0, lo=0.05, hi=8.0, max_iter=40):
    rows = []
    def ask(x, k, step, p, comment):
        value = oracle.value(x)
        rows.append({"k": k, "x": x, "f": value, "delta": step, "p": p, "comment": comment})
        return value

    f0 = ask(x0, 0, delta0, 0, "початкова точка x0")
    xr = min(x0 + delta0, hi)
    fr = ask(xr, 0, delta0, 0, "пробний крок праворуч")
    xl = max(x0 - delta0, lo)
    fl = ask(xl, 0, -delta0, 1, "пробний крок ліворуч")
    
    if fl >= f0 <= fr:
        return {"a": min(xl, xr), "b": max(xl, xr), "p": 0, "iterations": 0, "n_f": oracle.n_f, "rows": rows}
    if fr <= f0:
        p, step, x_cur, f_cur = 0, delta0, xr, fr
    else:
        p, step, x_cur, f_cur = 1, -delta0, xl, fl

    x_prev = x0
    k = 1
    while k <= max_iter:
        x_new = min(max(x_cur + (2.0 ** k) * step, lo), hi)
        if x_new == x_cur:
            break
        f_new = ask(x_new, k, (2.0 ** k) * step, p, f"розгін")
        if f_new >= f_cur:
            break
        x_prev, x_cur, f_cur = x_cur, x_new, f_new
        k += 1

    a, b = min(x_prev, x_new), max(x_prev, x_new)
    return {"a": max(a, lo), "b": min(b, hi), "p": p, "iterations": k, "n_f": oracle.n_f, "rows": rows}