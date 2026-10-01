import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5

def df(x):
    return 1.5 * (4 * x**3 + 6.9 * x**2 - 8.38 * x - 4.455)

def d2f(x):
    return 1.5 * (12 * x**2 + 13.8 * x - 8.38)

def check_derivatives(x_test, h=1e-5):
    df_num = (f(x_test + h) - f(x_test - h)) / (2 * h)
    d2f_num = (df(x_test + h) - df(x_test - h)) / (2 * h)
    
    print("=== 1. ПЕРЕВІРКА ПОХІДНИХ (x = {}) ===".format(x_test))
    print("f'(x)  аналітична: {:.6f} | чисельна: {:.6f} | diff: {:.2e}".format(df(x_test), df_num, abs(df(x_test) - df_num)))
    print("f''(x) аналітична: {:.6f} | чисельна: {:.6f} | diff: {:.2e}\n".format(d2f(x_test), d2f_num, abs(d2f(x_test) - d2f_num)))

a0, b0 = -2.41, -2.39
eps = 0.001
alpha = 0.1  

check_derivatives(a0)

def newton_classic(x0, eps):
    print("=== ЧАСТИНА А: Класичний метод Ньютона (Таблиця 4.8) ===")
    print("{:<3} | {:<9} | {:<9} | {:<9} | {:<9} | {:<9}".format("k", "xk", "f(xk)", "f'(xk)", "|f'(xk)|", "f''(xk)"))
    print("-" * 65)
    
    x = x0
    k = 0
    n_f, n_df, n_d2f = 0, 0, 0
    history = []
    
    while True:
        fx = f(x); n_f += 1
        dfx = df(x); n_df += 1
        d2fx = d2f(x); n_d2f += 1
        abs_dfx = abs(dfx)
        
        history.append((x, dfx))
        print("{:<3} | {:<9.5f} | {:<9.5f} | {:<9.5f} | {:<9.5f} | {:<9.5f}".format(k, x, fx, dfx, abs_dfx, d2fx))

        if abs_dfx <= eps:
            break

        if d2fx <= 0:
            print("Увага: f''(x) <= 0! Класичний Ньютон втрачає збіжність.")
            break
            
        x = x - dfx / d2fx
        k += 1
        
    print("Резюме Частини А: x* = {:.5f}, k = {}, звернень (f: {}, f': {}, f'': {})\n".format(x, k, n_f, n_df, n_d2f))
    return x, k, n_f, n_df, n_d2f, history

def newton_raphson_regulated(x0, eps, alpha):
    print("=== ЧАСТИНА Б: Ньютон-Рафсон із регулюванням кроку (Таблиця 4.9) ===")
    print("{:<3} | {:<8} | {:<8} | {:<8} | {:<8} | {:<5} | {:<8} | {:<8} | {:<8}".format(
        "k", "xk", "f(xk)", "f'(xk)", "f''(xk)", "t", "x_bar", "f(x_bar)", "f'(x_bar)"
    ))
    print("-" * 85)
    
    x = x0
    k = 0
    n_f, n_df, n_d2f = 0, 0, 0
    
    while True:
        fx = f(x); n_f += 1
        dfx = df(x); n_df += 1
        
        if abs(dfx) <= eps:
            d2fx = d2f(x); n_d2f += 1
            print("{:<3} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<5} | {:<8} | {:<8} | {:<8}".format(
                k, x, fx, dfx, d2fx, "-", "-", "-", "-"
            ))
            break
            
        d2fx = d2f(x); n_d2f += 1
        
        if d2fx <= 0:
            print("f''(x) <= 0! Регулювання змінює напрямок на антиградієнт.")
            S = -dfx
        else:
            S = -dfx / d2fx
            
        t = 1.0
        while True:
            x_bar = x + t * S
            f_bar = f(x_bar); n_f += 1
            df_bar = df(x_bar); n_df += 1  

            condition = f_bar <= fx - alpha * t * (dfx**2) / (d2fx if d2fx > 0 else 1.0)
            
            print("{:<3} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<8.5f} | {:<5.3f} | {:<8.5f} | {:<8.5f} | {:<8.5f}".format(
                k, x, fx, dfx, d2fx, t, x_bar, f_bar, df_bar
            ))
            
            if condition or t < 1e-4:
                x = x_bar
                break
            t /= 2.0  
            
        k += 1
        
    print("Резюме Частини Б: x* = {:.5f}, k = {}, звернень (f: {}, f': {}, f'': {})\n".format(x, k, n_f, n_df, n_d2f))
    return x, k, n_f, n_df, n_d2f

x_star_a, k_a, nf_a, ndf_a, nd2f_a, history_a = newton_classic(a0, eps)
x_star_b, k_b, nf_b, ndf_b, nd2f_b = newton_raphson_regulated(a0, eps, alpha)

print("=== 7. ЕКСПЕРИМЕНТ: Запуск із різних стартових точок ===")
starts = [a0, b0, 1.0, -0.5]
for x_start in starts:
    print("\n--- Старт x0 = {} ---".format(x_start))
    print("Класичний Ньютон:")
    newton_classic(x_start, eps)
    print("Ньютон-Рафсон із регулюванням:")
    newton_raphson_regulated(x_start, eps, alpha)

x_vals = np.linspace(-2.45, -2.35, 200)
phi_vals = [df(v) for v in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, phi_vals, label="φ(x) = f'(x)", color='blue', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

colors = ['red', 'green']
for idx in range(min(2, len(history_a))):
    xk, dfk = history_a[idx]
    d2fk = d2f(xk)
    tangent_y = [dfk + d2fk * (v - xk) for v in x_vals]
    plt.plot(x_vals, tangent_y, linestyle=':', color=colors[idx], label="Дотична на ітерації {}".format(idx+1))
    plt.scatter([xk], [dfk], color=colors[idx], zorder=5)

plt.title("Графік φ(x) = f'(x) з дотичними першої та другої ітерацій")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()