import math

def f(x):
    return 1.5 * (x + 3) * (x + 1.5) * ((x - 1.1) ** 2) + 1.5 

x = -2.40 
a = -2.41
b = -2.39
eps = 0.01

a0 = a
b0 = b

tau = (math.sqrt(5) - 1) / 2
k = 0
L = b - a

x1 = a + (1 - tau) * L
x2 = a + tau * L

f1 = f(x1)
f2 = f(x2)
n_f = 2 

print(f"{'k':<3} | {'a':<8} | {'b':<8} | {'x1':<8} | {'x2':<8} | {'f(x1)':<9} | {'f(x2)':<9} | {'L_k':<8} | {'Рішення'}")
print("-" * 90)

while True:
    L_k = b - a

    decision = f"b = x2" if f1 <= f2 else f"a = x1"
    print(f"{k:<3} | {a:<8.4f} | {b:<8.4f} | {x1:<8.4f} | {x2:<8.4f} | {f1:<9.4f} | {f2:<9.4f} | {L_k:<8.4f} | {decision}")
  
    if L_k <= eps:
        break
        
    k += 1

    if f1 <= f2:
        b = x2
        x2 = x1
        f2 = f1
        L = b - a
        x1 = a + (1 - tau) * L
        f1 = f(x1)
        n_f += 1  
    else:
        a = x1
        x1 = x2
        f1 = f2
        L = b - a
        x2 = a + tau * L
        f2 = f(x2)
        n_f += 1  

x_star = (a + b) / 2
f_star = f(x_star)
eta = (b0 - a0) / (eps * n_f)

print("-" * 90)
print(f"Мінімум x* = {x_star:.4f}")
print(f"Кількість ітерацій k = {k}")
print(f"Кількість обчислень функції N_f = {n_f}")
print(f"Коефіцієнт ефективності η = {eta:.4f}")