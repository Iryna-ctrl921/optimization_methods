import numpy as np
import matplotlib.pyplot as plt

def F(x):
    x1, x2 = x[0], x[1]
    return 2 * x1**2 + 2 * x2**2 - 3 * x1 * x2 - 15 * x1 + 13 * x2

def grad_F(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 4 * x1 - 3 * x2 - 15
    df_dx2 = 4 * x2 - 3 * x1 + 13
    return np.array([df_dx1, df_dx2], dtype=float)

def get_direction(g, metric):
    if metric == "sphere":
        norm_g = np.linalg.norm(g)
        return -g / norm_g if norm_g > 0 else np.zeros_like(g)
    elif metric == "cube":
        return -np.sign(g)
    elif metric == "octa":
        j = np.argmax(np.abs(g))
        d = np.zeros_like(g)
        d[j] = -np.sign(g[j])
        return d
    else:
        raise ValueError("Невідома метрика")

def optimize_gradient(x0, h0, eps, metric):
    x = np.array(x0, dtype=float)
    Fx = F(x)
    h = h0
    
    history = []
    n_F = 1      
    n_grad = 0   
    
    for k in range(1, 2000):
        g = grad_F(x)
        n_grad += 1
        norm_g = np.linalg.norm(g)
        
        if norm_g <= eps:
            break
            
        d = get_direction(g, metric)
        
        divisions = 0
        h_curr = h
        
        while True:
            xb = x + h_curr * d
            Fb = F(xb)
            n_F += 1
            
            if Fb < Fx:
                break
            else:
                divisions += 1
                h_curr /= 2.0
                if h_curr < 1e-10:
                    raise RuntimeError("Крок здрібнів до нуля!")
                    
        history.append({
            'k': k,
            'x': x.copy(),
            'F': Fx,
            'grad': g.copy(),
            'norm_g': norm_g,
            'd': d.copy(),
            'h': h_curr,
            'divisions': divisions,
            'F_next': Fb
        })
        
        x = xb
        Fx = Fb
        h = h_curr  
        
    return x, k, n_F, n_grad, history

x0_test = np.array([8.0, 2.0])
eps = 0.01
h_variants = [1.0, 0.7]
x_exact = np.array([3.0, -1.0])

summary_results = []

for h0 in h_variants:
    print(f"\n" + "="*95)
    print(f"РЕЗУЛЬТАТИ ДЛЯ ПОЧАТКОВОГО КРОКУ h0 = {h0}")
    print("="*95)
    
    for metric in ["sphere", "cube", "octa"]:
        x_star, k_iter, nF, nG, hist = optimize_gradient(x0_test, h0, eps, metric)
        error = np.linalg.norm(x_star - x_exact)
        
        summary_results.append({
            'metric': metric,
            'h0': h0,
            'n': k_iter,
            'nF': nF,
            'nG': nG,
            'total': nF + nG,
            'error': error,
            'x_star': x_star
        })

        metric_names = {"sphere": "СФЕРИЧНА", "cube": "КУБІЧНА", "octa": "ОКТАЕДРИЧНА"}
        print(f"\n--- Метрика: {metric_names[metric]} (h0 = {h0}) ---")
        print(f"{'k':<3} | {'x^k':<18} | {'F(x^k)':<9} | {'grad F':<18} | {'||grad||':<8} | {'d^k':<12} | {'h':<6} | {'дробл':<5} | {'F(x^(k+1))'}")
        print("-" * 105)
        
        for step in hist:
            x_str = f"({step['x'][0]:.2f}, {step['x'][1]:.2f})"
            grad_str = f"({step['grad'][0]:.1f}, {step['grad'][1]:.1f})"
            d_str = f"({step['d'][0]:.2f}, {step['d'][1]:.2f})"
            print(f"{step['k']:<3} | {x_str:<18} | {step['F']:<9.2f} | {grad_str:<18} | {step['norm_g']:<8.2f} | {d_str:<12} | {step['h']:<6.3f} | {step['divisions']:<5} | {step['F_next']:.2f}")
        
        print(f"Висновок: x* = ({x_star[0]:.4f}, {x_star[1]:.4f}), ітерацій: {k_iter}, N_F: {nF}, N_grad: {nG}, похибка: {error:.2e}\n")

print("\n" + "="*80)
print("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
print("="*80)
print(f"{'Метрика':<12} | {'h0':<4} | {'n':<4} | {'N_F':<4} | {'N_grad':<6} | {'Разом':<6} | {'Похибка ||x* - x_точн||'}")
print("-" * 75)
for res in summary_results:
    print(f"{res['metric']:<12} | {res['h0']:<4} | {res['n']:<4} | {res['nF']:<4} | {res['nG']:<6} | {res['total']:<6} | {res['error']:.2e}")
print("="*80)

plt.figure(figsize=(10, 8))
x1_vals = np.linspace(-1, 9, 200)
x2_vals = np.linspace(-4, 3, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = 2 * X1**2 + 2 * X2**2 - 3 * X1 * X2 - 15 * X1 + 13 * X2

levels = [ -29, -20, 0, 20, 50, 100, 180 ]
cs = plt.contour(X1, X2, Z, levels=levels, cmap='coolwarm')
plt.clabel(cs, inline=True, fontsize=8)

colors = {'sphere': 'blue', 'cube': 'orange', 'octa': 'green'}
labels = {'sphere': 'Сферична', 'cube': 'Кубічна', 'octa': 'Октаедрична'}

for metric in ["sphere", "cube", "octa"]:
    _, _, _, _, hist = optimize_gradient(x0_test, 1.0, eps, metric)
    traj_x = [hist[0]['x'][0]] + [step['x_next'] if 'x_next' in step else step['x'] for step in hist] 
    pts = [hist[0]['x']]
    for step in hist:
        pts.append(step['x'] + step['h'] * step['d']) 
    pts = np.array(pts)
    plt.plot(pts[:, 0], pts[:, 1], marker='o', color=colors[metric], label=labels[metric], linewidth=2)

plt.scatter([x0_test[0]], [x0_test[1]], color='black', s=100, zorder=5, label='Старт x0 (8, 2)')
plt.scatter([x_exact[0]], [x_exact[1]], color='red', s=100, zorder=5, label='Мінімум x* (3, -1)')

plt.title("Траєкторії градіентних методів для трьох метрик (h0 = 1.0)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.show()