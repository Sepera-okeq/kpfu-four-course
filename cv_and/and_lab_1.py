import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Считываем цветное изображение как numpy-массив:
image = Image.open('image.png')
image_np = np.array(image)

# 2. Инвертируем изображение:
if image_np.shape[2] == 4:  # RGBA изображение
    inverted_image = 255 - image_np[:, :, :3]  # Инвертируем только RGB, альфа оставляем как есть
else:
    inverted_image = 255 - image_np

# 3. Переводим изображение в полутоновое (градации серого):
gray_image = np.mean(image_np, axis=2).astype(np.uint8)

# 4. Добавляем случайный шум с нормальным распределением:
mean = 0
stddev = 10  # Стандартное отклонение для шума
noise = np.random.normal(mean, stddev, gray_image.shape)
noisy_image = np.clip(gray_image + noise, 0, 255).astype(np.uint8)

# 5. Строим гистограмму полученного изображения:
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title('Оригинальное изображение')
plt.imshow(image)

plt.subplot(2, 2, 2)
plt.title('Инвертированное изображение')
plt.imshow(inverted_image)

plt.subplot(2, 2, 3)
plt.title('Изображение с шумом')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Гистограмма полутонового c шумом изображения')
hist, bins = np.histogram(noisy_image, bins=256, range=[0, 255])
plt.bar(bins[:-1], hist, width=1, color='black')
plt.xlabel('Интенсивность пикселя')
plt.ylabel('Частота')
plt.show()

# 6. Размытие изображения с использованием ядра Гаусса:

# Алярм!!! Мы для того, чтобы не долбится в угол, сделаем по центру:
# + ---  + --- + --- +
# | 0x0 | 1x0 | 2x0 |
# + --- + --- + --- +
# | 0x1 | 1x1 | 2x1 |
# + --- + --- + --- +
# | 0x2 | 1x2 | 2x2 |
# + --- + --- + --- +
# Если у меня фильтр 3x3, то центр - в (1,1)

def gaussian_kernel(size, sigma):
    """Создание 2D Гауссовского ядра фильтра."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
        np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def gaussian_blur(image, kernel_size, sigma):
    """Применение свертки изображения с ядром Гаусса."""
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = np.zeros_like(image)
    padded_image = np.pad(image, pad_width=((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='reflect')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blurred_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return np.clip(blurred_image, 0, 255)

# Эксперименты с различными размытиями (размеры фильтра размера ядра и дисперсия)
blurred_results = []
params = [
    (3, 0.5),  # Малый фильтр, малая дисперсия
    (3, 1.5),  # Малый фильтр, большая дисперсия
    (9, 0.5),  # Большой фильтр, малая дисперсия
    (9, 1.5)   # Большой фильтр, большая дисперсия
]

for kernel_size, sigma in params:
    blurred_img = gaussian_blur(noisy_image, kernel_size, sigma)
    blurred_results.append((blurred_img, kernel_size, sigma))

# Визуализация всех вариантов размытия на одном графике:
plt.figure(figsize=(12, 12))

for i, (blurred_img, kernel_size, sigma) in enumerate(blurred_results):
    plt.subplot(2, 2, i+1)
    plt.imshow(blurred_img, cmap='gray')
    plt.title(f'Размер ядра: {kernel_size}, Сигма: {sigma}')
    plt.axis('off')

plt.suptitle('Гауссово размытие с различными размерами ядра и сигмами', fontsize=16)
plt.tight_layout()
plt.show()

# 7. Эквализация гистограммы изображения:
hist, bins = np.histogram(blurred_img.flatten(), bins=256, range=[0, 256], density=True)
cdf = hist.cumsum()
cdf_min = cdf[np.nonzero(cdf)].min()  # минимальное значение CDF, отличное от нуля
cdf = 255 * (cdf - cdf_min) / (cdf[-1] - cdf_min)
equalized_image = np.interp(blurred_img.flatten(), bins[:-1], cdf).reshape(blurred_img.shape)

# Визуализация эквализации
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Эквализованное изображение')
plt.imshow(equalized_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Гистограмма эквализованного изображения')
hist_eq, bins_eq = np.histogram(equalized_image, bins=256, range=[0, 256])
plt.bar(bins_eq[:-1], hist_eq, width=1, color='black')
plt.xlabel('Интенсивность пикселя')
plt.ylabel('Частота')
plt.show()