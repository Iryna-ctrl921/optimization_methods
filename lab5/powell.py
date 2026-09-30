import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5

a0 = -2.41              
b0 = -2.39             
dx = (b0 - a0) / 10      
sigma_f = 0.01        
sigma_x = 0.001         

x1 = a0
x2 = x1 + dx
f1, f2 = f(x1), f(x2)
n_f = 2 

if f1 > f2:
    x3 = x1 + 2 * dx
else:
    x3 = x1 - dx

f3 = f(x3)
n_f += 1

x = [x1, x2, x3]
fx = [f1, f2, f3]

plot_x = list(x)
plot_fx = list(fx)
plot_a1, plot_a2 = None, None

print(f"{'k':<3} | {'x1':<7} | {'x2':<7} | {'x3':<7} | {'f(x1)':<7} | {'f(x2)':<7} | {'f(x3)':<7} | {'a1':<8} | {'a2':<8} | {'x~':<7} | {'f(x~)':<7} | {'dx<=σx':<6} | {'df<=σf':<6}")
print("-" * 115)

k = 1
while True:
    a1 = (fx[1] - fx[0]) / (x[1] - x[0])
    a2 = ((fx[2] - fx[0]) / (x[2] - x[0]) - a1) / (x[2] - x[1])

    if k == 1:
        plot_a1 = a1
        plot_a2 = a2

    if a2 <= 0:
        best_idx = fx.index(min(fx))
        x_tilde = x[best_idx] + dx  
    else:
        x_tilde = (x[0] + x[1]) / 2 - a1 / (2 * a2)

    for xi in x:
        if abs(x_tilde - xi) < 1e-9:
            x_tilde += sigma_x / 2  
            break

    f_tilde = f(x_tilde)
    n_f += 1

    min_old_idx = fx.index(min(fx))
    x_min_old = x[min_old_idx]
    f_min_old = fx[min_old_idx]

    check_x = abs(x_tilde - x_min_old) <= sigma_x
    check_f = abs(f_tilde - f_min_old) <= sigma_f

    print(f"{k:<3} | {x[0]:<7.4f} | {x[1]:<7.4f} | {x[2]:<7.4f} | {fx[0]:<7.4f} | {fx[1]:<7.4f} | {fx[2]:<7.4f} | {a1:<8.4f} | {a2:<8.4f} | {x_tilde:<7.4f} | {f_tilde:<7.4f} | {str(check_x):<6} | {str(check_f):<6}")

    if check_x and check_f:
        break

    max_idx = fx.index(max(fx))
    x[max_idx] = x_tilde
    fx[max_idx] = f_tilde

    k += 1

print("-" * 115)
print(f"Кількість ітерацій (n) = {k}")
print(f"Кількість обчислень функції N_f = {n_f} (Перевірка N_f = n + 3: {n_f == k + 3})")
print(f"Оптимальне рішення x* = {x_tilde:.5f}, f(x*) = {f_tilde:.5f}")


x_vals = np.linspace(min(plot_x) - 0.05, max(plot_x) + 0.05, 200)
y_vals = [f(v) for v in x_vals]

p_vals = [plot_fx[0] + plot_a1 * (v - plot_x[0]) + plot_a2 * (v - plot_x[0]) * (v - plot_x[1]) for v in x_vals]

plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_vals, label='Функція f(x)', color='blue', linewidth=2)

plt.plot(x_vals, p_vals, label='Парабола (1 ітерація)', color='orange', linestyle='--')

plt.scatter(plot_x, plot_fx, color='red', zorder=5, label='Стартові точки (x1, x2, x3)')

plt.title("Метод Пауелла: Функція, парабола та стартові точки")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()