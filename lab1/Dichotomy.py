def f(x):
    """Цільова функція f(x) = 1.5 * (x + 3) * (x + 1.5) * (x - 1.1)^2 + 1.5"""
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5


def dichotomy_method(f, a0, b0, delta=0.01, eps=0.05, max_iter=10):
    """
    Метод дихотомії для пошуку мінімуму функції f(x).
    
    :param f: цільова функція
    :param a0: ліва межа початкового інтервалу
    :param b0: права межа початкового інтервалу
    :param delta: мале значення рознесення точок (x1 = mid - delta/2, x2 = mid + delta/2)
    :param eps: задана точність (алгоритм зупиняється, коли L_k < eps)
    :param max_iter: гранична кількість ітерацій
    :return: список рядків для таблиці
    """
    table = []
    a, b = a0, b0
    L = b - a

    table.append({
        'k': 0,
        'x1': '-',
        'x2': '-',
        'fx1': '-',
        'fx2': '-',
        'ak': a,
        'bk': b,
        'L': L
    })

    k = 1
    while L > eps and k <= max_iter:
        mid = (a + b) / 2.0

        x1 = mid - delta / 2.0
        x2 = mid + delta / 2.0

        fx1 = f(x1)
        fx2 = f(x2)

        if fx1 <= fx2:
            b = x2 
        else:
            a = x1  

        L = b - a

        table.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'fx1': fx1,
            'fx2': fx2,
            'ak': a,
            'bk': b,
            'L': L
        })

        k += 1

    return table


def print_dichotomy_table(table):
    """Форматоване виведення таблиці в консоль"""
    print("\nМетод дихотомії")
    header = f"{'k':^5} | {'x₁':^10} | {'x₂':^10} | {'f(x₁)':^12} | {'f(x₂)':^12} | {'[aₖ , bₖ]':^22} | {'Lₖ':^10}"
    line = "-" * len(header)

    print(line)
    print(header)
    print(line)

    for row in table:
        if row['k'] == 0:
            x1_str = row['x1']
            x2_str = row['x2']
            fx1_str = row['fx1']
            fx2_str = row['fx2']
        else:
            x1_str = f"{row['x1']:.4f}".replace('.', ',')
            x2_str = f"{row['x2']:.4f}".replace('.', ',')
            fx1_str = f"{row['fx1']:.4f}".replace('.', ',')
            fx2_str = f"{row['fx2']:.4f}".replace('.', ',')

        interval_str = f"[{row['ak']:.4f} ; {row['bk']:.4f}]".replace('.', ',')
        L_str = f"{row['L']:.4f}".replace('.', ',')

        print(f"{row['k']:^5} | {x1_str:^10} | {x2_str:^10} | {fx1_str:^12} | {fx2_str:^12} | {interval_str:^22} | {L_str:^10}")

    print(line)

a0 = -2.7
b0 = -2.1

delta = 0.01  
eps = 0.05    

table_data = dichotomy_method(f, a0, b0, delta=delta, eps=eps, max_iter=5)

print_dichotomy_table(table_data)