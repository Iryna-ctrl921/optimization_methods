import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5

def df(x):
    return 1.5 * (4 * x**3 + 6.9 * x**2 - 8.38 * x - 4.455)

x_prev = -2.41  
x_curr = -2.39  
eps = 0.001

df_prev = df(x_prev)
df_curr = df(x_curr)

print("=== ПЕРЕВІРКА ПОЧАТКОВИХ УМОВ ===")
print(f"f'({x_prev}) = {df_prev:.5f}")
print(f"f'({x_curr}) = {df_curr:.5f}")
print(f"Добуток f'(x0) * f'(x1) = {df_prev * df_curr:.5f}")
if df_prev * df_curr < 0:
    print("Умова f'(x0) * f'(x1) < 0 виконується успішно!\n")
else:
    print("Увага: умова не виконується, змініть інтервал!\n")

secant_lines = [(x_prev, df_prev, x_curr, df_curr)]

print("=== ТАБЛИЦЯ 4.11: МЕТОД СІЧНИХ ===")
print(f"{'k':<3} | {'xk-1':<8} | {'f\'(xk-1)':<10} | {'xk':<8} | {'f\'(xk)':<10} | {'xk+1':<8} | {'f\'(xk+1)':<10} | {'f(xk+1)':<9} | {'Заміна'}")
print("-" * 105)

k = 1
n_df = 2  

while True:
    x_next = x_curr - df_curr * (x_curr - x_prev) / (df_curr - df_prev)
    df_next = df(x_next)
    n_df += 1
    f_next = f(x_next)

    if df_next * df_prev < 0:
        replacement = "xk -> xk+1"
        x_prev_next = x_prev
        df_prev_next = df_prev
        x_curr_next = x_next
        df_curr_next = df_next
    else:
        replacement = "xk-1 -> xk+1"
        x_prev_next = x_next
        df_prev_next = df_next
        x_curr_next = x_curr
        df_curr_next = df_curr

    print(f"{k:<3} | {x_prev:<8.4f} | {df_prev:<10.4f} | {x_curr:<8.4f} | {df_curr:<10.4f} | {x_next:<8.4f} | {df_next:<10.4f} | {f_next:<9.4f} | {replacement}")

    if k <= 2:
        secant_lines.append((x_curr, df_curr, x_next, df_next))

    if abs(df_next) <= eps:
        print("-" * 105)
        print(f"Знайдено рішення: x* = {x_next:.5f}, f(x*) = {f_next:.5f}")
        print(f"Кількість ітерацій (n) = {k}")
        print(f"Кількість звернень до похідної N_df = {n_df} (Перевірка N_df = n + 2: {n_df == k + 2})")
        break

    x_prev, df_prev = x_prev_next, df_prev_next
    x_curr, df_curr = x_curr_next, df_curr_next
    k += 1

x_vals = np.linspace(-2.45, -2.35, 200)
phi_vals = [df(v) for v in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, phi_vals, label="φ(x) = f'(x)", color='blue', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

colors = ['red', 'green']
for idx, (xa, ya, xb, yb) in enumerate(secant_lines[:2]):
    slope = (yb - ya) / (xb - xa)
    line_y = [ya + slope * (v - xa) for v in x_vals]
    plt.plot(x_vals, line_y, linestyle=':', color=colors[idx], label=f"Січна {idx+1}")
    plt.scatter([xa, xb], [ya, yb], color=colors[idx], zorder=5)

plt.title("Графік φ(x) = f'(x) з першими двома січними")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()

print("=== ЕКСПЕРИМЕНТ: МЕТОД «ЧИСТИХ» СІЧНИХ ===")
x_prev_p = -2.41
x_curr_p = -2.39
df_prev_p = df(x_prev_p)
df_curr_p = df(x_curr_p)

k_p = 1
n_df_p = 2

print(f"{'k':<3} | {'xk-1':<8} | {'f\'(xk-1)':<10} | {'xk':<8} | {'f\'(xk)':<10} | {'xk+1':<8} | {'f\'(xk+1)':<10}")
print("-" * 75)

while True:
    x_next_p = x_curr_p - df_curr_p * (x_curr_p - x_prev_p) / (df_curr_p - df_prev_p)
    df_next_p = df(x_next_p)
    n_df_p += 1
    
    print(f"{k_p:<3} | {x_prev_p:<8.4f} | {df_prev_p:<10.4f} | {x_curr_p:<8.4f} | {df_curr_p:<10.4f} | {x_next_p:<8.4f} | {df_next_p:<10.4f}")
    
    if abs(df_next_p) <= eps:
        print("-" * 75)
        print(f"Результат (чисті січні): x* = {x_next_p:.5f}, ітерацій k = {k_p}, звернень до похідної N_df = {n_df_p}\n")
        break
        
    x_prev_p, df_prev_p = x_curr_p, df_curr_p
    x_curr_p, df_curr_p = x_next_p, df_next_p
    k_p += 1