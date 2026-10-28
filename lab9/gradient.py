import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. АНАЛІТИЧНІ ДАНІ ТА ЦІЛЬОВА ФУНКЦІЯ (Варіант №3)
# ==========================================
def F(x):
    x1, x2 = x[0], x[1]
    return 2.0 * x1**2 + 2.0 * x2**2 - 3.0 * x1 * x2 - 15.0 * x1 + 13.0 * x2

def grad_F(x):
    x1, x2 = x[0], x[1]
    df1 = 4.0 * x1 - 3.0 * x2 - 15.0
    df2 = -3.0 * x1 + 4.0 * x2 + 13.0
    return np.array([df1, df2], dtype=float)

Hessian = np.array([[4.0, -3.0], 
                    [-3.0, 4.0]], dtype=float)

n_f = 0
n_grad = 0

def oracle_F(x):
    global n_f
    n_f += 1
    return F(x)

def oracle_grad(x):
    global n_grad
    n_grad += 1
    return grad_F(x)

def reset_counters():
    global n_f, n_grad
    n_f = 0
    n_grad = 0

# ==========================================
# 2. ТОЧНИЙ ЛІНІЙНИЙ ПОШУК (Свенн + Золотий переріз)
# ==========================================
def svenn_search(x, d, delta0=0.01):
    """Алгоритм Свенна для пошуку початкового інтервалу [a, b] уздовж напрямку d"""
    h0 = 0.0
    f0 = oracle_F(x + h0 * d)

    h1 = h0 + delta0
    f1 = oracle_F(x + h1 * d)
    
    if f1 < f0:
        step = delta0
        h_prev = h0
        h_cur = h1
        f_cur = f1
    else:
        step = -delta0
        h_prev = h0
        h_cur = h0 + step
        f_cur = oracle_F(x + h_cur * d)
        if f_cur >= f0:
            return min(h0, h_cur), max(h0, h_cur)

    k = 1
    while k < 50:
        h_new = h_cur + (2.0 ** k) * step
        f_new = oracle_F(x + h_new * d)
        if f_new >= f_cur:
            break
        h_prev, h_cur, f_cur = h_cur, h_new, f_new
        k += 1
        
    return min(h_prev, h_new), max(h_prev, h_new)

def golden_section_search(x, d, a, b, eps=1e-4):
    """Метод золотого перерізу для точного пошуку кроку h"""
    tau = (np.sqrt(5.0) - 1.0) / 2.0
    h1 = b - tau * (b - a)
    h2 = a + tau * (b - a)
    f1 = oracle_F(x + h1 * d)
    f2 = oracle_F(x + h2 * d)
    
    while (b - a) > eps:
        if f1 <= f2:
            b = h2
            h2, f2 = h1, f1
            h1 = b - tau * (b - a)
            f1 = oracle_F(x + h1 * d)
        else:
            a = h1
            h1, f1 = h2, f2
            h2 = a + tau * (b - a)
            f2 = oracle_F(x + h2 * d)
    return (a + b) / 2.0

def exact_line_search(x, d):
    """Загальна функція лінійного пошуку (Свенн -> Золотий переріз)"""
    a, b = svenn_search(x, d, delta0=0.01)
    h_opt = golden_section_search(x, d, a, b, eps=1e-4)
    return h_opt


# ==========================================
# 3 & 4. МЕТОД ФЛЕТЧЕРА-РІВЗА З ПЕРЕВІРКАМИ
# ==========================================
def fletcher_reeves_optimization(x0, max_iter=7, tol=0.01):
    print("\n--- МЕТОД ФЛЕТЧЕРА-РІВЗА (ЛР №9) ---")
    print(f"{'k':<3} | {'x_k':<18} | {'F(x)':<10} | {'||grad||':<10} | {'beta':<8} | {'h_opt':<8}")
    print("-" * 75)
    
    x = np.array(x0, dtype=float)
    g = oracle_grad(x)
    d = -g
    history = [x.copy()]
    
    table_data = []
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        f_val = oracle_F(x)
        
        if g_norm < tol:
            print(f"Збіжність досягнута на ітерації {k}!")
            break

        h = exact_line_search(x, d)

        h_analyt = -np.dot(g, d) / np.dot(d, np.dot(Hessian, d))
        
        x_next = x + h * d
        g_next = oracle_grad(x_next)

        if (k + 1) % 2 == 0:
            beta = 0.0 
        else:
            beta = np.dot(g_next, g_next) / np.dot(g, g)
            
        d_next = -g_next + beta * d
        
        table_data.append({
            'k': k, 'x': x.copy(), 'F': f_val, 'grad_norm': g_norm, 
            'd': d.copy(), 'h': h, 'beta': beta
        })
        
        print(f"{k:<3} | [{x[0]:.4f}, {x[1]:.4f}] | {f_val:<10.4f} | {g_norm:<10.4f} | {beta:<8.4f} | {h:<8.4f}")

        if k == 0:
            check_1 = np.dot(g_next, d)          
            check_2 = np.dot(d_next, np.dot(Hessian, d))
            check_3 = np.dot(g_next, d_next)       
            
            print("\n  [Пункт 4] Перевірки теорії спряжених градієнтів:")
            print(f"   1. Ортогональність (<g1, d0>): {check_1:.2e} (очікується ~0)")
            print(f"   2. Кон'югатність (d1^T H d0): {check_2:.2e} (очікується ~0)")
            print(f"   3. Напрямок спадання (<g1, d1>): {check_3:.4f} (очікується < 0)\n")
            
        x, g, d = x_next, g_next, d_next
        history.append(x.copy())
        
    return x, np.array(history), table_data


# ==========================================
# 5. АНТИГРАДІЄНТ + ТОЧНИЙ ПОШУК (beta = 0)
# ==========================================
def steepest_descent(x0, max_iter=20, tol=0.01):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = oracle_grad(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g
        h = exact_line_search(x, d)
        x = x + h * d
        history.append(x.copy())
    return np.array(history)


# ==========================================
# 7. ФУНКЦІЯ РОЗЕНБРОКА (для перевірки п. 7)
# ==========================================
def rosenbrock(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

def rosenbrock_grad(x):
    df1 = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0])
    df2 = 200.0 * (x[1] - x[0]**2)
    return np.array([df1, df2], dtype=float)

def optimize_rosenbrock(with_restart=True):
    x = np.array([-1.2, 1.0], dtype=float)
    g = rosenbrock_grad(x)
    d = -g
    history = [x.copy()]
    
    for k in range(25):
        if np.linalg.norm(g) < 1e-3:
            break
        h = 0.001 
        x_next = x + h * d
        g_next = rosenbrock_grad(x_next)
        
        beta = 0.0 if (with_restart and (k + 1) % 2 == 0) else np.dot(g_next, g_next) / np.dot(g, g)
        d_next = -g_next + beta * d
        x, g, d = x_next, g_next, d_next
        history.append(x.copy())
    return np.array(history)


# ==========================================
# 6. ПОБУДОВА ГРАФІКІВ ЛІНІЙ РІВНЯ ТА ТРАЄКТОРІЙ
# ==========================================
def plot_trajectories(path_fr, path_sd):
    x1 = np.linspace(-1, 9, 150)
    x2 = np.linspace(-3, 4, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 2.0 * X1**2 + 2.0 * X2**2 - 3.0 * X1 * X2 - 15.0 * X1 + 13.0 * X2

    plt.figure(figsize=(10, 8))
    contours = plt.contour(X1, X2, Z, levels=25, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.plot(path_sd[:, 0], path_sd[:, 1], 'r--o', label='Антиградієнт (beta=0)', linewidth=2, markersize=5)
    plt.plot(path_fr[:, 0], path_fr[:, 1], 'b-s', label='Флетчер-Рівз (рестарт n=2)', linewidth=2, markersize=6)

    plt.scatter([3.0], [-1.0], color='green', marker='*', s=200, label='Аналітичний мінімум (3, -1)')
    plt.title('Порівняння траєкторій оптимізації')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()  

# ==========================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# ==========================================
if __name__ == "__main__":
    print("=== ЗАПУСК ЛАБОРАТОРНОЇ РОБОТИ №9 (ВАРІАНТ №3) ===")

    print("\n[Пункт 1] Аналітичні дані:")
    print(" Цільова функція: F(x1, x2) = 2x1^2 + 2x2^2 - 3x1x2 - 15x1 + 13x2")
    print(" Градієнт grad F(x) = [4x1 - 3x2 - 15, -3x1 + 4x2 + 13]")
    print(" Гесіан H =\n", Hessian)
    print(" Точний мінімум: x* = (3, -1), F(x*) = -29")
    
    x_start = np.array([8.0, 2.0])

    reset_counters()
    x_opt, fr_path, fr_table = fletcher_reeves_optimization(x_start)
    nf_fr, ng_fr = n_f, n_grad
    print(f"\n Результат Флетчера-Рівза: x* = {x_opt}, F(x*) = {F(x_opt):.4f}")

    reset_counters()
    sd_path = steepest_descent(x_start)
    nf_sd, ng_sd = n_f, n_grad

    print("\n" + "="*60)
    print(f"{'Пункт 8. ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ':^60}")
    print("="*60)
    print(f"{'Метод':<30} | {'Ітерації (n)':<12} | {'Виклики F (N_f)':<15} | {'Виклики grad (N_g)':<18}")
    print("-" * 80)
    print(f"{'Антиградієнт + точний пошук':<30} | {len(sd_path)-1:<12} | {nf_sd:<15} | {ng_sd:<18}")
    print(f"{'Флетчер-Рівз (рестарт n=2)':<30} | {len(fr_path)-1:<12} | {nf_fr:<15} | {ng_fr:<18}")
    print("="*60)

    plot_trajectories(fr_path, sd_path)

    print("\n[Пункт 7] Тест на функції Розенброка з точки (-1.2, 1.0):")
    path_rosen_restarts = optimize_rosenbrock(with_restart=True)
    path_rosen_no_restart = optimize_rosenbrock(with_restart=False)
    print(f" Кінцева точка (з рестартами): {path_rosen_restarts[-1]}, F = {rosenbrock(path_rosen_restarts[-1]):.5f}")
    print(f" Кінцева точка (без рестартів): {path_rosen_no_restart[-1]}, F = {rosenbrock(path_rosen_no_restart[-1]):.5f}")
    print(" Висновок: Версія з періодичним перезапуском кожні n=2 кроки ефективніше долає нелінійні вигини («жолоби») функції Розенброка.")