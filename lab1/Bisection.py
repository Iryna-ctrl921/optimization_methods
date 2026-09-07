def f(x):
    """Цільова функція f(x) = 1.5 * (x + 3) * (x + 1.5) * (x - 1.1)^2 + 1.5"""
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5


def fmt(val, dec=4):
    """Безпечне форматування: для чисел додає коми та знаки після коми, рядки залишає без змін."""
    if isinstance(val, (int, float)):
        return f"{val:.{dec}f}".replace('.', ',')
    return str(val)


def bisection_method(f, a0, b0, eps=0.05, max_iter=10):
    """Метод половинного ділення (ділення навпіл)"""
    table = []
    a, b = a0, b0
    L = b - a

    table.append({
        'k': 0,
        'x1': '-', 'xm': '-', 'x2': '-',
        'fx1': '-', 'fxm': '-', 'fx2': '-',
        'Lk': L
    })

    k = 1
    while L > eps and k <= max_iter:
        xm = (a + b) / 2.0
        x1 = a + L / 4.0
        x2 = b - L / 4.0

        fx1 = f(x1)
        fxm = f(xm)
        fx2 = f(x2)

        L_new = L / 2.0

        table.append({
            'k': k,
            'x1': x1,
            'xm': xm,
            'x2': x2,
            'fx1': fx1,
            'fxm': fxm,
            'fx2': fx2,
            'Lk': L_new
        })

        if fx1 < fxm:
            b = xm
        elif fx2 < fxm:
            a = xm
        else:
            a = x1
            b = x2

        L = b - a
        k += 1

    return table


def print_bisection_table(table):
    """Форматоване виведення таблиці в консоль"""
    print("\nМетод половинного ділення")
    header = f"{'k':^4} | {'x₁':^10} | {'xₘ':^10} | {'x₂':^10} | {'f(x₁)':^10} | {'f(xₘ)':^10} | {'f(x₂)':^10} | {'Lₖ':^8}"
    line = "-" * len(header)

    print(line)
    print(header)
    print(line)

    for row in table:
        print(
            f"{row['k']:^4} | "
            f"{fmt(row['x1']):^10} | "
            f"{fmt(row['xm']):^10} | "
            f"{fmt(row['x2']):^10} | "
            f"{fmt(row['fx1']):^10} | "
            f"{fmt(row['fxm']):^10} | "
            f"{fmt(row['fx2']):^10} | "
            f"{fmt(row['Lk']):^8}"
        )

    print(line)

a0 = -2.7
b0 = -2.1
eps = 0.05

table_data = bisection_method(f, a0, b0, eps=eps, max_iter=5)

print_bisection_table(table_data)