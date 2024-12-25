import numpy as np
import math
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma=0.5):
    """Создание 2D Гауссовского ядра фильтра."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
        np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)), 
        (size, size)
    )
    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, sigma):
    """Применение свертки изображения с гауссовским ядром."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)


def convolve(image, kernel):
    """Общая функция свертки."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    return output

# k - регулирует чувствительность к углам
def harris_detector(image, keypoints, k=0.05, threshold=1e-5):
    """Фильтрация особых точек с помощью метрики Харриса."""
    # Вычисление производных изображения (градиенты)
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # Гауссово сглаживание производных
    Ixx = gaussian_blur(Ixx, 5, 1)
    Iyy = gaussian_blur(Iyy, 5, 1)
    Ixy = gaussian_blur(Ixy, 5, 1)

    filtered_keypoints = []
    # Создание матрицы М в окрестности каждого ключевого пикселя
    for x, y in keypoints:
        # Структурная матрица M
        # Суммируется квадрат производных взвешенных с помощью гауссовского окна 5x5
        Sxx = Ixx[y, x]
        Syy = Iyy[y, x]
        Sxy = Ixy[y, x]
        
        # Вычисление детерминанта и следа для критерия
        det = Sxx * Syy - Sxy**2    # Определитель M
        trace = Sxx + Syy           # След (сумма диагональных элементов M)
        R = det - k * (trace ** 2)

        if R > threshold:
            filtered_keypoints.append((x, y, R))

    filtered_keypoints = sorted(filtered_keypoints, key=lambda k: k[2], reverse=True)
    return [(x, y) for x, y, _ in filtered_keypoints]


# Вычисляем ориентацию, чтобы дескрипторы могли быть инвариантными к повороту
def calculate_orientation(image, keypoints, patch_size=31):
    """Вычисление ориентации для каждой точки согласно её центроиду."""
    orientations = []
    radius = patch_size // 2

    for x, y in keypoints:
        patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
        if patch.shape != (patch_size, patch_size):
            orientations.append(0)
            continue
        
        # Момент по горизонтали
        m10 = np.sum(patch * np.arange(-radius, radius+1)[:, None])
        # Момент по вертикали
        m01 = np.sum(patch * np.arange(-radius, radius+1)[None, :])
        
        # Ориентация определяется углом между осями (на основе моментов m01 m10)
        angle = math.atan2(m01, m10)
        orientations.append(angle)

    return orientations

# Для проверки ключевых точек FAST используется яркость диаметральных точек (выше и ниже порога).
# Минимальная разность яркости threshold.
def detect_fast(image, threshold=30):
    """Реализация FAST для детектирования ключевых точек."""
    image = image.astype(np.int32)
    rows, cols = image.shape
    keypoints = []

    # Координаты круга Брезенхема радиуса 3
    bresenham_circle = [(0, -3), (1, -3), (2, -2), (3, -1),
                        (3, 0), (3, 1), (2, 2), (1, 3),
                        (0, 3), (-1, 3), (-2, 2), (-3, 1),
                        (-3, 0), (-3, -1), (-2, -2), (-1, -3)]

    # Центральная интенсивность пикселя сравнивается с 16 окружающими точками.
    for y in range(3, rows - 3):
        for x in range(3, cols - 3):
            center_intensity = image[y, x]
            brighter = 0
            darker = 0

            # Проверка пикселей 1, 9, 5, 13
            #           1
            #     16          2
            #   15              3
            # 13      *(center)    5
            #   12              6
            #     11          7
            #           9
            # P.S Для ускорения проверки сначала тестируются 4 точки (1, 9, 5, 13) на сильное различие яркости.
            # Накой и зачем? - каждая пара из них лежит по диаметру окружности (находятся на максимальном расстоянии друг от друга)
            # Поэтому, если яркость вокруг центрального пикселя существенно различна,
            # достаточно проверить эти диаметрально противоположные точки, чтобы уже предположить,
            # что текущий пиксель может быть ключевой точкой...
            # Если недостаточно точек яркости выше или ниже порога, центр пропускается.
            for idx in [0, 8, 4, 12]:
                dy, dx = bresenham_circle[idx]
                intensity = image[y + dy, x + dx]
                if intensity > center_intensity + threshold:
                    brighter += 1
                elif intensity < center_intensity - threshold:
                    darker += 1

            # Если 12 или больше пикселей на окружности ярче или темнее центрального пикселя — это ключевая точка.
            if brighter >= 3 or darker >= 3:
                sequential_count = 0
                for dy, dx in bresenham_circle:
                    intensity = image[y + dy, x + dx]
                    if intensity > center_intensity + threshold or intensity < center_intensity - threshold:
                        sequential_count += 1
                        if sequential_count >= 12:
                            keypoints.append((x, y))
                            break
                    else:
                        sequential_count = 0
    return keypoints


def brief_descriptor(image, keypoints, orientations, patch_size=31, n=256):
    """Генерация бинарных дескрипторов BRIEF."""
    descriptors = []
    radius = patch_size // 2
    # Генерируем N пар случайных точек в пределах патча (у нас нормально распределенные координаты центра)
    random_points = np.random.randint(-radius, radius, size=(n, 2))

    # У каждой ключевой точки берется небольшой патч, размером patch_size
    for (x, y), angle in zip(keypoints, orientations):
        patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
        if patch.shape != (patch_size, patch_size):  # Игнорируем неподходящие размеры
            continue

        # Поворот точек относительно ориентации (ранее матрицы вращения используем)
        rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_points = random_points @ rotation_mat.T
        rotated_points = np.round(rotated_points).astype(int)

        descriptor = []
        for dx, dy in rotated_points:
            px1, py1 = radius + dx, radius + dy
            px2, py2 = radius - dx, radius - dy
            if 0 <= px1 < patch_size and 0 <= py1 < patch_size and 0 <= px2 < patch_size and 0 <= py2 < patch_size:
                descriptor.append(1 if patch[py1, px1] > patch[py2, px2] else 0)
            else:
                descriptor.append(0)  # Добавляем значение, если точка вне области
        if len(descriptor) == n:  # Убедимся, что дескриптор имеет корректную длину
            descriptors.append(descriptor)

    return np.array(descriptors, dtype=np.uint8)


# Пример загрузки изображения
image = plt.imread("pizza.png")
if image.ndim == 3:  # Преобразование в градации серого
    image = np.mean(image, axis=2)
image = (image * 255).astype(np.uint8)

# Сглаживаем изображение
image_smoothed = gaussian_blur(image, kernel_size=5, sigma=1)

# 1. Детектирование FAST
keypoints = detect_fast(image_smoothed, threshold=20)

# 2. Фильтрация Харриса
keypoints = harris_detector(image_smoothed, keypoints)

# 3. Вычисляем ориентации
orientations = calculate_orientation(image_smoothed, keypoints)

# 4. Генерация дескрипторов BRIEF
descriptors = brief_descriptor(image_smoothed, keypoints, orientations)

# Отображаем ключевые точки
plt.imshow(image, cmap="gray")
for x, y in keypoints:
    plt.scatter(x, y, c="red", s=5)
plt.show()

with open("descriptors.txt", "w") as f:
    for descriptor in descriptors:
        f.write("".join(map(str, descriptor)) + "\n")