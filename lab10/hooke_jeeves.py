import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. ЦІЛЬОВА ФУНКЦІЯ ВАРІАНТА №3 ТА ЛІЧИЛЬНИК N_F
# ==========================================
n_f = 0

def F_var3(x):
    global n_f
    n_f += 1
    x1, x2 = x[0], x[1]
    return 2.0 * x1**2 + 2.0 * x2**2 - 3.0 * x1 * x2 - 15.0 * x1 + 13.0 * x2

def reset_nf():
    global n_f
    n_f = 0


# ==========================================
# 2. ДОСЛІДНИЙ ПОШУК ТА ПОШУК ЗА ЗРАЗКОМ
# ==========================================
def explore(func, y, Fy, delta):
    """Дослідний пошук: обмацування околиць уздовж осей координат"""
    z = y.copy()
    Fz = Fy
    for i in range(len(z)):
        t = z.copy()
        t[i] += delta
        Ft = func(t)
        if Ft < Fz:
            z, Fz = t, Ft
            continue
        t = z.copy()
        t[i] -= delta
        Ft = func(t)
        if Ft < Fz:
            z, Fz = t, Ft
    return z, Fz


def hooke_jeeves(func, x0, delta0=1.0, eps=0.01, max_iter=500):
    """Метод Хука-Дживса (прямий пошук без похідних)"""
    base = np.array(x0, dtype=float)
    Fb = func(base)
    y, Fy, d = base.copy(), Fb, delta0
    from_pattern = False
    
    history_bases = [base.copy()]
    protocol = []
    
    for k in range(max_iter):
        z, Fz = explore(func, y, Fy, d)
        
        if Fz < Fb:
            y_pattern = z + (z - base)
            Fy_pattern = func(y_pattern)
            base, Fb = z, Fz
            history_bases.append(base.copy())
            y, Fy = y_pattern, Fy_pattern
            from_pattern = True
        elif from_pattern:
            y, Fy = base.copy(), Fb
            from_pattern = False
        else:
            d /= 2.0
            if d < eps:
                break
                
    return base, Fb, history_bases


# ==========================================
# 5. НЕГЛАДКА ФУНКЦІЯ
# ==========================================
def non_smooth_func(x):
    global n_f
    n_f += 1
    x1, x2 = x[0], x[1]
    return 2.0 * abs(x1 - 1.0) + 3.0 * abs(x2 + 2.0) + 0.5 * (x1 - 1.0)**2


# ==========================================
# 6. ФУНКЦІЯ РОЗЕНБРОКА
# ==========================================
def rosenbrock(x):
    global n_f
    n_f += 1
    x1, x2 = x[0], x[1]
    return (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2


# ==========================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ ТА АНАЛІЗ
# ==========================================
if __name__ == "__main__":
    print("=== ЛАБОРАТОРНА РОБОТА №10 (МЕТОД ХУКА-ДЖИВСА) ===")

    reset_nf()
    x_start = np.array([8.0, 2.0])
    x_opt_1, f_opt_1, bases_1 = hooke_jeeves(F_var3, x_start, delta0=1.0, eps=0.01)
    nf_1 = n_f
    
    print(f"\n[Пункт 3] Прогін з delta_0 = 1.0, eps = 0.01:")
    print(f" Знайдений мінімум: x* = {x_opt_1}, F(x*) = {f_opt_1:.6f}")
    print(f" Кількість базових точок: {len(bases_1)}")
    print(f" Загальна кількість звернень до F (N_F): {nf_1}")
    print(f" Похибка за мінімумом: ||x* - x_analyt|| = {np.linalg.norm(x_opt_1 - np.array([3.0, -1.0])):.4f}")

    reset_nf()
    x_opt_2, f_opt_2, bases_2 = hooke_jeeves(F_var3, x_start, delta0=0.7, eps=0.01)
    nf_2 = n_f
    
    print(f"\n[Пункт 4] Прогін з delta_0 = 0.7, eps = 0.01:")
    print(f" Знайдений мінімум: x* = {x_opt_2}, F(x*) = {f_opt_2:.6f}")
    print(f" Кількість базових точок: {len(bases_2)}")
    print(f" Загальна кількість звернень до F (N_F): {nf_2}")
    print(f" Похибка за мінімумом: ||x* - x_analyt|| = {np.linalg.norm(x_opt_2 - np.array([3.0, -1.0])):.4f}")
    print(" Пояснення різниці (Ефект ґратки): При delta0 = 1.0 початковий сітковий крок ідеально збігся з кроком до мінімуму, тому метод 'влучив' точно. При delta0 = 0.7 вузли ґратки не містять точного центру, тому метод зупиняється на останньому кроці дроблення з невеликою залишковою похибкою.")

    reset_nf()
    ns_start = np.array([5.0, -3.0])
    ns_opt, ns_f, ns_bases = hooke_jeeves(non_smooth_func, ns_start, delta0=1.0, eps=0.01)
    print(f"\n[Пункт 5] Негладка функція з кутами (мінімум в (1, -2)):")
    print(f" Знайдено точку: {ns_opt}, G = {ns_f:.6f}, N_F = {n_f}")
    print(" Висновок: Градієнтні методи тут не працюють через відсутність похідної на лініях зламу, а Хук-Дживс оперує лише значеннями і успішно знаходить мінімум.")

    reset_nf()
    ros_start = np.array([-1.2, 1.0])
    ros_opt_1, ros_f_1, _ = hooke_jeeves(rosenbrock, ros_start, delta0=1.0, eps=0.01)
    nf_ros_1 = n_f
    
    reset_nf()
    ros_opt_2, ros_f_2, _ = hooke_jeeves(rosenbrock, ros_start, delta0=1.0, eps=1e-4)
    nf_ros_2 = n_f
    
    print(f"\n[Пункт 6] Функція Розенброка (пастка критерію зупинки):")
    print(f" З eps = 0.01: точка = {ros_opt_1}, f = {ros_f_1:.5f}, N_F = {nf_ros_1} (пастка пологого яру!)")
    print(f" З eps = 10^-4: точка = {ros_opt_2}, f = {ros_f_2:.2e}, N_F = {nf_ros_2} (успішне досягнення (1, 1))")

    x1_vals = np.linspace(-1, 9, 150)
    x2_vals = np.linspace(-3, 3, 150)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 2.0 * X1**2 + 2.0 * X2**2 - 3.0 * X1 * X2 - 15.0 * X1 + 13.0 * X2

    plt.figure(figsize=(9, 7))
    contours = plt.contour(X1, X2, Z, levels=25, cmap='coolwarm')
    plt.clabel(contours, inline=True, fontsize=8)

    bases_arr = np.array(bases_1)
    plt.plot(bases_arr[:, 0], bases_arr[:, 1], 'ro-', label='Хук-Дживс (delta0=1.0)', linewidth=2, markersize=6)
    plt.scatter([3.0], [-1.0], color='green', marker='*', s=250, label='Аналітичний мінімум (3, -1)')
    
    plt.title('Траєкторія базових точок методу Хука-Дживса')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()