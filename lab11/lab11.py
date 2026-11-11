import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from penalty import (run_sumt, project_to_boundary, g_constraint,
                     penalty_wrappers, fletcher_reeves_inner)

os.makedirs("outputs", exist_ok=True)

def F_opt(x):
    return 2.0 * x[0] ** 2 + 3.0 * x[1] ** 2 + x[0] * x[1]

def grad_F_opt(x):
    return np.array([4.0 * x[0] + x[1], x[0] + 6.0 * x[1]])

def exact_x(r):
    x2 = 24.0 * r / (23.0 + 16.0 * r)
    return np.array([5.0 * x2 / 3.0, x2])

def exact_g(r):
    return 5.75 / (r + 1.4375)

x0 = [5.0, -3.0]
protocol, r1_history = run_sumt(F_opt, grad_F_opt, x0, r0=1.0, c=10.0, eps=0.01)

print("=" * 100)
print("ПРОТОКОЛ ЗОВНІШНІХ ІТЕРАЦІЙ (ЛР 11, ЧАСТИНА 1)")
print("=" * 100)
print(f"{'k':<3} | {'r':<8} | {'ітер ФР':<7} | {'N_Phi':<6} | {'x(r)':<23} | "
      f"{'F(x(r))':<8} | {'g(x(r))':<9} | {'r*P':<9} | {'2r*g':<7} | збіж.")
print("-" * 110)
for p in protocol:
    x_str = f"({p['x_r'][0]:.5f}; {p['x_r'][1]:.5f})"
    print(f"{p['k']:<3} | {p['r']:<8.0f} | {p['fr_iters']:<7} | {p['n_f']:<6} | {x_str:<23} | "
          f"{p['F']:<8.4f} | {p['g']:<9.5f} | {p['r_P']:<9.5f} | {p['lam_est']:<7.3f} | "
          f"{'так' if p['converged'] else 'НІ'}")

print(f"\nЗагалом: N_Phi = {protocol[-1]['total_n_f']}, N_grad = {protocol[-1]['total_n_grad']}")

print("\nПеревірка проти аналітичного x(r):")
for p in protocol:
    err = np.linalg.norm(p["x_r"] - exact_x(p["r"]))
    print(f"  r = {p['r']:<8.0f} |x(r) - x_exact(r)| = {err:.2e}")

print("\n" + "=" * 100)
print("ПОВНИЙ ПРОТОКОЛ ВНУТРІШНЬОЇ ЗАДАЧІ, r = 1 (ЛР 9 ФР)")
print("=" * 100)
print(f"{'k':<3} | {'x_k':<22} | {'Phi':<10} | {'||grad||':<10} | {'d_k':<22} | {'h':<10} | beta")
for h in r1_history:
    xs = f"({h['x'][0]:.5f}; {h['x'][1]:.5f})"
    if h['d'] is None:
        print(f"{h['iter']:<3} | {xs:<22} | {h['Phi']:<10.5f} | {h['grad_norm']:<10.5f} | {'-':<22} | {'-':<10} | -")
    else:
        ds = f"({h['d'][0]:.4f}; {h['d'][1]:.4f})"
        print(f"{h['iter']:<3} | {xs:<22} | {h['Phi']:<10.5f} | {h['grad_norm']:<10.5f} | {ds:<22} | {h['h']:<10.6f} | {h['beta']:.4f}")

x_final = protocol[-1]["x_r"]
x_hat = project_to_boundary(x_final)
print("\nДОШЛІФОВУВАННЯ ПРОЄКЦІЄЮ НА МЕЖУ:")
print(f"До проєкції:    ({x_final[0]:.6f}; {x_final[1]:.6f}), g = {g_constraint(x_final):.6f}, F = {F_opt(x_final):.4f}")
print(f"Після проєкції: ({x_hat[0]:.6f}; {x_hat[1]:.6f}), g = {g_constraint(x_hat):.6f}, F = {F_opt(x_hat):.4f}")
print("Точний розв'язок: (2.500000; 1.500000), g = 0, F = 23.0000")

Phi_huge, gradPhi_huge = penalty_wrappers(F_opt, grad_F_opt, r=1e6)
x_huge, it_huge, nf_huge, ng_huge, _, conv_huge = fletcher_reeves_inner(
    Phi_huge, gradPhi_huge, x0, eps=0.01, max_iter=2000)
print("\nОДРАЗУ r=10^6 ЗІ СТАРТУ (5, -3):")
print(f"Ітерацій ФР: {it_huge}, N_Phi: {nf_huge}, збіжність: {'так' if conv_huge else 'НІ (вичерпано max_iter)'}, "
      f"x = ({x_huge[0]:.5f}; {x_huge[1]:.5f}), g = {g_constraint(x_huge):.2e}")

x1_vals = np.linspace(-1, 6, 300)
x2_vals = np.linspace(-4, 4, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = 2.0 * X1 ** 2 + 3.0 * X2 ** 2 + X1 * X2

plt.figure(figsize=(9, 7))
plt.fill_between(x1_vals, 4.0 - x1_vals, 5.0, color="green", alpha=0.15,
                 label=r"$\Omega: x_1+x_2 \geq 4$")
plt.plot(x1_vals, 4.0 - x1_vals, "g-", linewidth=2, label=r"Межа $x_1+x_2=4$")
cs = plt.contour(X1, X2, Z, levels=[5, 10, 23, 45, 62, 80], colors="gray",
                 linestyles="--", alpha=0.6)
plt.clabel(cs, inline=True, fontsize=8)
plt.plot(0, 0, "ro", markersize=8, label="Безумовний мін (0, 0)")
plt.plot(2.5, 1.5, "go", markersize=8, label=r"Умовний мін $x^*=(2.5,\,1.5)$")
plt.plot(5, -3, "ko", markersize=7, label=r"Старт $x_{(0)}=(5,\,-3)$")
tx = [5.0] + [p["x_r"][0] for p in protocol]
ty = [-3.0] + [p["x_r"][1] for p in protocol]
plt.plot(tx, ty, "b-o", linewidth=1.5, markersize=5, label=r"Траєкторія $x(r)$")
plt.xlim(-1, 6); plt.ylim(-4, 4)
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.title("Траєкторія руху методу зовнішніх штрафів")
plt.legend(loc="lower left"); plt.grid(alpha=0.3)
plt.savefig("lab11/Графік1.png", dpi=150, bbox_inches="tight")
plt.close()

r_vals = np.array([p["r"] for p in protocol])
g_vals = np.array([p["g"] for p in protocol])
plt.figure(figsize=(8, 5))
plt.loglog(r_vals, g_vals, "bo-", label=r"Практичне порушення $g(x(r))$")
plt.loglog(r_vals, exact_g(r_vals), "r--", label=r"Точне $5.75/(r+1.4375)$")
plt.loglog(r_vals, 5.75 / r_vals, "k:", label=r"Асимптотика $\lambda^*/(2r)=5.75/r$")
plt.xlabel("Коефіцієнт штрафу $r$ (логарифмічна шкала)")
plt.ylabel("Порушення $g(x(r))$ (логарифмічна шкала)")
plt.title("Залежність порушення від параметра $r$")
plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
plt.savefig("lab11/Графік2.png", dpi=150, bbox_inches="tight")
plt.close()