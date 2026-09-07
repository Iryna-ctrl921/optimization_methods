def f(x):
    """Цільова функція f(x) = 1.5 * (x + 3) * (x + 1.5) * (x - 1.1)^2 + 1.5"""
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5


def sven_method(f, x0, delta0):
    """
    Метод Свена для знаходження інтервалу невизначеності (одномодальності).
    
    :param f: цільова функція
    :param x0: початкова точка
    :param delta0: початковий крок (величина кроку)
    :return: список рядків для таблиці та підсумковий інтервал [a, b]
    """
    table = []

    f_left = f(x0 - delta0)
    f_curr = f(x0)
    f_right = f(x0 + delta0)

    table.append({
        'k': 0,
        'delta0': delta0,
        'xk': x0,
        'fxk': f_curr,
        'cond': '',
        'interval': ''
    })

    if f_left >= f_curr and f_curr >= f_right:
        delta = delta0
    elif f_left <= f_curr and f_curr <= f_right:
        delta = -delta0
    elif f_left >= f_curr and f_right >= f_curr:
        a, b = min(x0 - delta0, x0 + delta0), max(x0 - delta0, x0 + delta0)
        table[0]['interval'] = f"[{a:.4f} ; {b:.4f}]"
        return table, (a, b)
    else:
        raise ValueError("Точка x0 є точкою локального максимуму.")

    points = [x0]
    f_values = [f_curr]

    k = 1
    while True:
        step = (2 ** (k - 1)) * delta
        xk = points[-1] + step
        fxk = f(xk)

        points.append(xk)
        f_values.append(fxk)

        is_decreasing = fxk < f_values[-2]

        if is_decreasing:
            table.append({
                'k': k,
                'delta0': delta0,
                'xk': xk,
                'fxk': fxk,
                'cond': '+',
                'interval': ''
            })
            k += 1
        else:
            a = min(points[-3], points[-1])
            b = max(points[-3], points[-1])

            table.append({
                'k': k,
                'delta0': delta0,
                'xk': xk,
                'fxk': fxk,
                'cond': '-',
                'interval': f"[{a:.4f} ; {b:.4f}]"
            })
            return table, (a, b)


def print_table(table_data):
    """Форматоване виведення таблиці в консоль"""
    print("\nМетод Свена")
    header = f"{'№ ітерації k':^14} | {'Δ₀':^8} | {'xₖ':^10} | {'f(xₖ)':^12} | {'f(xₖ) < f(xₖ₋₁)?':^18} | {'[a , b]':^18}"
    line = "-" * len(header)
    
    print(line)
    print(header)
    print(line)

    for row in table_data:
        delta_str = f"{row['delta0']:.2f}".replace('.', ',')
        xk_str = f"{row['xk']:.4f}".replace('.', ',')
        fxk_str = f"{row['fxk']:.4f}".replace('.', ',')
        interval_str = row['interval'].replace('.', ',')

        print(f"{row['k']:^14} | {delta_str:^8} | {xk_str:^10} | {fxk_str:^12} | {row['cond']:^18} | {interval_str:^18}")
    
    print(line)

x0 = -2.40    
delta0 = 0.01   

table_data, interval = sven_method(f, x0, delta0)
print_table(table_data)