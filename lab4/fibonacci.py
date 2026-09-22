import math

def f(x):
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5

def get_fibonacci_sequence(n=50):
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

F = get_fibonacci_sequence()

x_exact = -2.40 
a0 = -2.41 
b0 = -2.39
eps = 0.01
delta = eps / 10  

L0 = b0 - a0
R = L0 / eps

N = 1
while F[N] < R:
    N += 1

print(f"=== 1. ПОРАХОВАНО N ===")
print(f"L0 = {L0}, eps = {eps}, R = L0/eps = {R:.4f}")
print(f"Найменше N = {N} (F_{N} = {F[N]} >= {R:.4f})\n")

print("=== 2. ТАБЛИЦЯ ІТЕРАЦІЙ МЕТОДУ ФІБОНАЧЧІ ===")
headers = [
    "k", "m", "F_{m-2}/F_m", "F_{m-1}/F_m", "a_k", "b_k", 
    "x1", "x2", "f(x1)", "f(x2)", "L_k", "Рішення"
]
header_line = f"{headers[0]:<3} | {headers[1]:<3} | {headers[2]:<11} | {headers[3]:<11} | {headers[4]:<7} | {headers[5]:<7} | {headers[6]:<7} | {headers[7]:<7} | {headers[8]:<7} | {headers[9]:<7} | {headers[10]:<7} | {headers[11]}"
print(header_line)
print("-" * len(header_line))

a_k, b_k = a0, b0
f_evals = 0

# Якщо N = 3, виконується лише один крок з delta
if N == 3:
    m = 3
    L_k = b_k - a_k
    x1 = (a_k + b_k) / 2 - delta
    x2 = (a_k + b_k) / 2 + delta
    fx1, fx2 = f(x1), f(x2)
    f_evals += 2
    
    if fx1 < fx2:
        decision = "a_{k+1}=a_k, b_{k+1}=x2"
        a_k, b_k = a_k, x2
    else:
        decision = "a_{k+1}=x1, b_{k+1}=b_k"
        a_k, b_k = x1, b_k

    print(
        f"{1:<3} | {3:<3} | {1/3:<11.4f} | {1/3:<11.4f} | {a0:<7.4f} | {b0:<7.4f} | {x1:<7.4f} | {x2:<7.4f} | {fx1:<7.4f} | {fx2:<7.4f} | {L_k:<7.4f} | {decision}"
    )

else:
    # Загальний випадок для N > 3
    x1 = a_k + (F[N - 2] / F[N]) * (b_k - a_k)
    x2 = a_k + (F[N - 1] / F[N]) * (b_k - a_k)
    fx1, fx2 = f(x1), f(x2)
    f_evals += 2

    for k in range(1, N - 1):
        m = N - k + 1
        L_k = b_k - a_k
        frac1 = F[m - 2] / F[m]
        frac2 = F[m - 1] / F[m]

        if fx1 < fx2:
            decision = "a_{k+1}=a_k, b_{k+1}=x2"
            a_k, b_k = a_k, x2
            x2 = x1
            fx2 = fx1
            if m > 4:
                x1 = a_k + (F[m - 3] / F[m - 1]) * (b_k - a_k)
                fx1 = f(x1)
                f_evals += 1
            elif m == 4:
                x1 = (a_k + b_k) / 2 - delta
                fx1 = f(x1)
                f_evals += 1
        else:
            decision = "a_{k+1}=x1, b_{k+1}=b_k"
            a_k, b_k = x1, b_k
            x1 = x2
            fx1 = fx2
            if m > 4:
                x2 = a_k + (F[m - 2] / F[m - 1]) * (b_k - a_k)
                fx2 = f(x2)
                f_evals += 1
            elif m == 4:
                x2 = (a_k + b_k) / 2 + delta
                fx2 = f(x2)
                f_evals += 1

        print(
            f"{k:<3} | {m:<3} | {frac1:<11.4f} | {frac2:<11.4f} | {a_k:<7.4f} | {b_k:<7.4f} | {x1:<7.4f} | {x2:<7.4f} | {fx1:<7.4f} | {fx2:<7.4f} | {L_k:<7.4f} | {decision}"
        )

x_star = (a_k + b_k) / 2
L_final = b_k - a_k
error = abs(x_star - x_exact)

print("\n" + "=" * 80 + "\n")
print("=== 3. РЕЗУЛЬТАТИ ДЛЯ ПОРІВНЯННЯ З ЛР 3 ===")
print(f"N_f (кількість обчислень f(x)): {f_evals}")
print(f"Кінцева довжина L_кін:          {L_final:.6f}")
print(f"Точка мінімуму x*:              {x_star:.6f}")
print(f"Похибка |x* - x*_точн|:          {error:.6f}")

guarantee = L0 / F[N]
print(f"\n=== 4. ПЕРЕВІРКА ГАРАНТІЇ ===")
print(f"Перевірка умови L_кін <= L0 / F_N:")
print(f"{L_final:.6f} <= {L0} / {F[N]} = {guarantee:.6f}")
if L_final <= guarantee + 1e-9:
    print("Умова гарантії СПРАВДЖУЄТЬСЯ.")
else:
    print("Умова гарантії НЕ справджується.")