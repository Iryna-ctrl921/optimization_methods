import numpy as np
from penalty import fletcher_reeves_inner
from penalty2 import (penalty_general, run_sumt_general, run_interior, sphere_descent)

def F(x):  return 2.0 * x[0] ** 2 + 3.0 * x[1] ** 2 + x[0] * x[1]
def gF(x): return np.array([4.0 * x[0] + x[1], x[0] + 6.0 * x[1]])

g1 = (lambda x: 4.0 - x[0] - x[1], lambda x: np.array([-1.0, -1.0]))

g2 = (lambda x: x[1] - 1.0, lambda x: np.array([0.0, 1.0]))

R = 3.0
gc = (lambda x: R ** 2 - x[0] ** 2 - x[1] ** 2,
      lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]))

h1 = (lambda x: 4.0 - x[0] - x[1], lambda x: np.array([-1.0, -1.0]))

X0 = [5.0, -3.0]
X_STAR = np.array([2.5, 1.5])
F_STAR = 23.0

def exact_x(r):  
    x2 = 24.0 * r / (23.0 + 16.0 * r)
    return np.array([5.0 * x2 / 3.0, x2])

def exact_mu(mu):  
    a, b = 184.0 / 9.0, -92.0 / 3.0
    x2 = (-b + np.sqrt(b * b + 4.0 * a * mu)) / (2.0 * a)
    return np.array([5.0 * x2 / 3.0, x2])

def fx(x):  return f"({x[0]:.4f}; {x[1]:.4f})"
def hdr(t): print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)
def warn(p):
    bad = [q for q in p if not q["conv"]]
    if bad:
        print("  !!! внутрішній метод НЕ зійшовся навіть після уточнення на k =", [q["k"] for q in bad])
    rt = [(q["k"], q.get("retries", 0)) for q in p if q.get("retries", 0)]
    if rt:
        print("  (потрібне уточнення золотого перерізу: (k, спроб) =", rt, ")")

summary = []   

hdr("ЗАВДАННЯ 1. Множник c (r0 = 1, eps = 0.01, внутрішній eps = 0.01)")
print(f"{'c':<5} | {'зовн.':<5} | {'кінцевий r':<10} | {'N_Phi':<6} | {'N_grad':<6} | {'порушення':<10} | x")
base = None
for c in (2, 4, 10, 100):
    p = run_sumt_general(F, gF, X0, ineq=[g1], r0=1.0, c=c, eps=0.01)
    q = p[-1]
    print(f"{c:<5} | {len(p):<5} | {q['r']:<10.0f} | {q['tot_f']:<6} | {q['tot_g']:<6} | "
          f"{q['viol']:<10.5f} | {fx(q['x'])}")
    warn(p)
    if c == 10:
        base = p
print("Читати так: порівняйте стовпці 'зовн.' і 'N_Phi'; на цій простій задачі c=10 і c=100 "
      "коштують однаково, c=2 і 4 дорожчі за рахунок кількості зовнішніх ітерацій.")
q = base[-1]
summary.append(("1. SUMT, c=10 (штраф, нерівність)", len(base), q['tot_f'], q['tot_g'], q['x'],
                q['viol'], q['F'], f"{q['lam'][0]:.3f}"))

# ============================================================================
hdr("ЗАВДАННЯ 2. Внутрішній eps (c = 10)")
print(f"{'eps_вн':<7} | {'зовн.':<5} | {'N_Phi':<6} | {'N_grad':<6} | {'x при r=10^4':<22} | "
      f"{'|x-x*|':<8} | {'порушення':<10} | x1 - x1*")
for e in (0.1, 0.01, 0.001):
    p = run_sumt_general(F, gF, X0, ineq=[g1], r0=1.0, c=10.0, eps=0.01, eps_inner=e)
    q = p[-1]
    print(f"{e:<7} | {len(p):<5} | {q['tot_f']:<6} | {q['tot_g']:<6} | {fx(q['x']):<22} | "
          f"{np.linalg.norm(q['x'] - X_STAR):<8.4f} | {q['viol']:<10.6f} | {q['x'][0]-2.5:+.4f}")
    warn(p)
print("Порушення задає r (≈ 5.75/r), положення вздовж межі залежить від внутрішнього eps.")

# ============================================================================
hdr("ЗАВДАННЯ 3. Рівність h(x) = 4 - x1 - x2 = 0, штраф r*h^2")
pe = run_sumt_general(F, gF, X0, eq=[h1], r0=1.0, c=10.0, eps=0.01)
pi = base
print(f"{'r':<7} | {'x(r) рівність':<22} | {'h(x(r))':<9} | {'F':<8} | {'iter':<4} | {'N_Phi':<5} | "
      f"{'|x_eq - x_ineq|':<15} | |x_eq - exact|")
for a, b in zip(pe, pi):
    print(f"{a['r']:<7.0f} | {fx(a['x']):<22} | {a['h'][0]:<9.5f} | {a['F']:<8.4f} | {a['iters']:<4} | "
          f"{a['nf']:<5} | {np.linalg.norm(a['x'] - b['x']):<15.2e} | {np.linalg.norm(a['x'] - exact_x(a['r'])):.2e}")
print(f"Рівність: зовн. ітерацій {len(pe)}, N_Phi = {pe[-1]['tot_f']}, N_grad = {pe[-1]['tot_g']}; "
      f"нерівність: {len(pi)}, {pi[-1]['tot_f']}, {pi[-1]['tot_g']}")
print("Висновок: x(r) збігаються, бо безумовний мінімум (0;0) по ту саму сторону межі, "
      "де зрізка й так не працювала (h>0). Для x1+x2=1 було б інакше (слайд 3).")
warn(pe)
q = pe[-1]
summary.append(("3. Штраф, рівність h=0", len(pe), q['tot_f'], q['tot_g'], q['x'],
                q['viol'], q['F'], f"{q['lam'][0]:.3f}"))

# ============================================================================
hdr("ЗАВДАННЯ 4. Лог-бар'єр, старт (5; 1) строго всередині, mu0 = 10, c = 10, стоп mu*m <= 0.01")
pb = run_interior(F, gF, [g1], [5.0, 1.0], mu0=10.0, c=10.0, eps=0.01)
print(f"{'k':<2} | {'mu':<6} | {'x(mu) ФР':<22} | {'x(mu) аналіт.':<22} | {'g':<10} | {'F':<9} | "
      f"{'F-F*':<8} | {'mu/(-g)':<8} | {'iter/N_Phi':<10} | N_Phi разом")
for t in pb:
    print(f"{t['k']:<2} | {t['mu']:<6g} | {fx(t['x']):<22} | {fx(exact_mu(t['mu'])):<22} | "
          f"{t['g'][0]:<10.6f} | {t['F']:<9.4f} | {t['F']-F_STAR:<8.4f} | {t['lam'][0]:<8.3f} | "
          f"{str(t['iters'])+'/'+str(t['nf']):<10} | {t['tot_f']}")
warn(pb)
tb = pb[-1]
assert all(t['g'][0] < 0 for t in pb), "точка вийшла з Omega!"
print(f"Усі g < 0: жодна точка не вийшла з Omega. N_Phi = {tb['tot_f']}, N_grad = {tb['tot_g']}")
print(f"ДВОСТОРОННЯ ОЦІНКА: F(x(r)) = {base[-1]['F']:.4f} <= F* <= F(x(mu)) = {tb['F']:.4f}  "
      f"(ширина {tb['F']-base[-1]['F']:.4f}; точне F* = 23)")
summary.append(("4. Лог-бар'єр, mu=10..0.01", len(pb), tb['tot_f'], tb['tot_g'], tb['x'],
                f"g={tb['g'][0]:.5f}", tb['F'], f"{tb['lam'][0]:.3f}"))

# ============================================================================
hdr("ЗАВДАННЯ 5. Два обмеження: g1 = 4 - x1 - x2 <= 0, g2 = x2 - 1 <= 0")
xs = np.array([3.0, 1.0])                   
A = np.array([g1[1](xs), g2[1](xs)]).T      
lam_kkt = np.linalg.solve(A, -gF(xs))
print(f"Куна-Таккер: x* = {fx(xs)}, F* = {F(xs):.4f}, gradF = {gF(xs)}, "
      f"lambda = ({lam_kkt[0]:.4f}; {lam_kkt[1]:.4f}), обидва >= 0 -> обидва активні")
p2 = run_sumt_general(F, gF, X0, ineq=[g1, g2], r0=1.0, c=10.0, eps=0.01)
print(f"{'r':<7} | {'x(r)':<22} | {'s1':<9} | {'s2':<9} | {'F':<8} | {'r*P':<7} | {'2r*s1':<7} | "
      f"{'2r*s2':<7} | iter/N_Phi")
for t in p2:
    print(f"{t['r']:<7.0f} | {fx(t['x']):<22} | {t['s'][0]:<9.5f} | {t['s'][1]:<9.5f} | {t['F']:<8.4f} | "
          f"{t['rP']:<7.4f} | {t['lam'][0]:<7.2f} | {t['lam'][1]:<7.2f} | {t['iters']}/{t['nf']}")
warn(p2)
q = p2[-1]
print(f"Зовн. ітерацій {len(p2)}, N_Phi = {q['tot_f']}, N_grad = {q['tot_g']}; "
      f"оцінки 2r*s_j -> ({q['lam'][0]:.2f}; {q['lam'][1]:.2f}) проти KKT ({lam_kkt[0]:.2f}; {lam_kkt[1]:.2f})")
summary.append(("5. Два обмеження", len(p2), q['tot_f'], q['tot_g'], q['x'], q['viol'], q['F'],
                f"({q['lam'][0]:.2f}; {q['lam'][1]:.2f})"))

# ============================================================================
hdr(f"ЗАВДАННЯ 6. Крива межа: x1^2 + x2^2 >= {R**2:g} (R = {R:g}); два старти")
lam_min = np.linalg.eigvalsh(np.array([[2.0, 0.5], [0.5, 3.0]]))[0]
v = np.array([0.5, -(2.0 - lam_min)])      # (A - lam_min I) v = 0
v = v / np.linalg.norm(v)
v = v if v[0] > 0 else -v
print(f"Аналітично: lambda = lambda_min = {lam_min:.4f}, F* = lambda_min * R^2 = {lam_min * R**2:.4f}, x*_1 = {fx(R*v)}, x*_2 = {fx(-R*v)}")
circ = {}
for name, x0 in (("старт (5; -3)", [5.0, -3.0]), ("старт (-5; 3)", [-5.0, 3.0])):
    pc = run_sumt_general(F, gF, x0, ineq=[gc], r0=1.0, c=10.0, eps=0.01)
    circ[name] = pc
    print(f"\n{name}:")
    print(f"  {'r':<6} | {'x(r)':<22} | {'||x||':<8} | {'s':<9} | {'F':<9} | {'r*P':<8} | iter/N_Phi")
    for t in pc:
        print(f"  {t['r']:<6.0f} | {fx(t['x']):<22} | {np.linalg.norm(t['x']):<8.4f} | {t['s'][0]:<9.5f} | "
              f"{t['F']:<9.4f} | {t['rP']:<8.4f} | {t['iters']}/{t['nf']}")
    warn(pc)
    q = pc[-1]
    print(f"  Зовн. ітерацій {len(pc)}, N_Phi = {q['tot_f']}, N_grad = {q['tot_g']}, F = {q['F']:.4f}")
    summary.append((f"6. Коло, {name}", len(pc), q['tot_f'], q['tot_g'], q['x'], q['viol'], q['F'],
                    f"{q['lam'][0]:.3f}"))
Fs = [circ[k][-1]['F'] for k in circ]
print(f"\nОбидва мінімуми мають однакове F ({Fs[0]:.4f} і {Fs[1]:.4f}) - F парна; "
      "Omega невипукла, тож порівнюйте F з кількох стартів.")

# ============================================================================
hdr("ЗАВДАННЯ 7. Внутрішній метод: сферична метрика (ЛР 8) проти Флетчера-Рівса (ЛР 9), зі старту (5; -3)")
print("Сферична метрика - код ЛР 8 (d = -grad/||grad||, h0 = 1, дроблення h/2, h переноситься між ітераціями).")
print(f"{'метод':<22} | {'r':<5} | {'ітер':<6} | {'N_Phi':<7} | {'N_grad':<6} | {'порушення':<10} | збіж.")
for r in (1, 10, 100):
    Phi, gPhi = penalty_general(F, gF, r, [g1])
    for name, fn in (("сферична метрика", sphere_descent), ("Флетчер-Рівс", fletcher_reeves_inner)):
        kw = {"max_iter": 20000} if fn is sphere_descent else {"max_iter": 2000}
        x, k, nf, ng, _, conv = fn(Phi, gPhi, X0, eps=0.01, **kw)
        print(f"{name:<22} | {r:<5} | {k:<6} | {nf:<7} | {ng:<6} | {max(0, g1[0](x)):<10.5f} | "
              f"{'так' if conv else 'НІ'}")
print("Флетчер-Рівс, поступове зростання r=1..10^4 (теплий старт): "
      f"{len(base)} зовн. ітерацій, N_Phi = {base[-1]['tot_f']}, N_grad = {base[-1]['tot_g']}")

# ============================================================================
hdr("ЗАВДАННЯ 8. ЗВЕДЕНА ТАБЛИЦЯ")
print(f"{'прогін':<34} | {'зовн.':<5} | {'N_Phi':<6} | {'N_grad':<6} | {'відповідь':<20} | "
      f"{'порушення':<10} | {'F':<8} | λ")
for name, it, nf, ng, x, viol, Fv, lam in summary:
    vs = f"{viol:.5f}" if not isinstance(viol, str) else viol
    print(f"{name:<34} | {it:<5} | {nf:<6} | {ng:<6} | {fx(x):<20} | {vs:<10} | {Fv:<8.4f} | {lam}")
print("Точні: нерівність/рівність x*=(2.5;1.5), F*=23, λ=11.5; два обмеження x*=(3;1), F*=24, "
      "λ=(13;4); коло F*=16.1360.")