import os
import numpy as np
from PIL import Image as PILImage
from src.rng import Rng

def load_clean_image(path="Фото.jpg", size=128):
    """Надійно завантажує зображення з можливих локацій"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths_to_try = [
        path,
        os.path.join(base_dir, path),
        os.path.join(base_dir, "..", path),
        os.path.join(base_dir, "..", "..", path),
    ]
    
    img_pil = None
    for p in paths_to_try:
        if os.path.exists(p):
            img_pil = PILImage.open(p)
            break
            
    if img_pil is None:
        raise FileNotFoundError(f"Не вдалося знайти файл '{path}'. Переконайтеся, що він знаходиться у теці project.")
        
    img_pil = img_pil.convert("L").resize((size, size), PILImage.Resampling.LANCZOS)
    return np.array(img_pil, dtype=float) / 255.0

def add_noise(clean, sigma, seed):
    rng = Rng(seed)
    h, w = clean.shape
    raw = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            raw[i][j] = clean[i][j] + sigma * rng.normal()
    noisy = np.clip(raw, 0.0, 1.0)
    return raw, noisy

def save_png(img, path):
    """Автоматично створює теки та надійно зберігає зображення"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(os.path.join(base_dir, "..", path))
    
    # Автоматично створюємо папку (наприклад, outputs), якщо її не існує
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    data = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    PILImage.fromarray(data, mode="L").save(full_path)