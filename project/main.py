from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, EPS, LAMBDA0, SIMPLEX_EDGE, SIZE
from src.image import load_clean_image, add_noise, save_png
from src.oracle import Oracle
from src.filters import gaussian_blur, bilateral_filter
from src.metrics import mse, psnr
from src.lab1_svenn import svenn
from src.lab2_search import dichotomy, golden_section
from src.lab3_nelder_mead import nelder_mead
from src.lab4_gradient import gradient_descent
from src.lab5_constrained import projected_gradient
import numpy as np

print("=== ЗАПУСК ПРОЄКТУ (ВАРІАНТ №3) ===")

# Частина 0: Підготовка
clean = load_clean_image("Фото.jpg", SIZE)
raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
save_png(clean, "outputs/00_clean.png")
save_png(noisy, "outputs/00_noisy.png")
print(f"Шумне зображення: MSE = {mse(noisy, clean):.8f}, PSNR = {psnr(noisy, clean):.4f} dB")

# Частина 1: Свенн
LO, HI = 0.05, 8.0
f_obj1 = lambda s: mse(gaussian_blur(noisy, s), clean) + 1000.0 * (s - min(max(s, LO), HI)) ** 2
oracle1 = Oracle(f_obj1)
bracket = svenn(oracle1, X0, DELTA0, LO, HI)
print(f"\n[Лаб 1] Свенн знайдено відрізок: [{bracket['a']:.4f}, {bracket['b']:.4f}], Nf = {bracket['n_f']}")

# Частина 2: Дихотомія і Золотий переріз
oracle1.reset()
res_d = dichotomy(oracle1, bracket["a"], bracket["b"], EPS)
oracle1.reset()
res_g = golden_section(oracle1, bracket["a"], bracket["b"], EPS)
print(f"[Лаб 2] Дихотомія: sigma* = {res_d['x']:.6f}, Nf = {res_d['n_f']}")
print(f"[Лаб 2] Золотий переріз: sigma* = {res_g['x']:.6f}, Nf = {res_g['n_f']}")

restored_g = gaussian_blur(noisy, res_g['x'])
save_png(restored_g, "outputs/lab2_restored.png")
print(f"        PSNR після Лаб 2: {psnr(restored_g, clean):.4f} dB")

# Частина 3: Нелдер—Мід
theta0 = [min(max(2 * res_g['x'], 0.5), 4.0), 0.1591]
f_obj3 = lambda t: mse(bilateral_filter(noisy, t[0], t[1]), clean)
oracle3 = Oracle(lambda t: f_obj3(t) + 1000.0 * float(np.sum((np.array(t) - np.clip(t, [0.3, 0.01], [5.0, 1.0]))**2)))
res_nm = nelder_mead(oracle3, theta0, SIMPLEX_EDGE)
print(f"\n[Лаб 3] Нелдер—Мід: theta* = ({res_nm['x'][0]:.4f}, {res_nm['x'][1]:.4f}), Nf = {res_nm['n_f']}")

restored_bilat = bilateral_filter(noisy, res_nm['x'][0], res_nm['x'][1])
save_png(restored_bilat, "outputs/lab3_restored.png")
print(f"        PSNR після Лаб 3: {psnr(restored_bilat, clean):.4f} dB")

# Частина 4: Градієнтні методи
L_val = 1.0 + 8.0 * LAMBDA0
res_gd = gradient_descent(noisy, LAMBDA0, L_val)
save_png(res_gd['I'], "outputs/lab4_restored.png")
print(f"\n[Лаб 4] Градієнтний спуск: E = {res_gd['E']:.6f}, Nf = {res_gd['n_f']}")
print(f"        PSNR після Лаб 4: {psnr(res_gd['I'], clean):.4f} dB")

# Частина 5: Умовна оптимізація
res_proj = projected_gradient(raw, LAMBDA0, L_val)
save_png(res_proj['I'], "outputs/lab5_restored.png")
print(f"\n[Лаб 5] Проєкція градієнта: E = {res_proj['E']:.6f}, порушень = 0")
print(f"        PSNR після Лаб 5: {psnr(res_proj['I'], clean):.4f} dB")

print("\nУсі розрахунки для Варіанта №3 успішно завершено!")