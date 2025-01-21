import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(file_path):
    """Загрузка изображения и преобразование в оттенки серого."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден!")
        return None
    return np.array(Image.open(file_path).convert('L'))

def gaussian_kernel(size, sigma=1):
    """Создание 2D ядра Гаусса для размытия.
    
    Аргументы:
        size: Размер ядра (size x size)
        sigma: Стандартное отклонение распределения Гаусса
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) *
        np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)), 
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_blur(image, kernel_size, sigma):
    """Применение размытия по Гауссу к изображению.
    
    Аргументы:
        image: Исходное изображение
        kernel_size: Размер ядра Гаусса
        sigma: Стандартное отклонение для ядра Гаусса
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = np.zeros_like(image, dtype=np.float32)
    padded_image = np.pad(image, pad_width=((kernel_size//2, kernel_size//2), 
                                           (kernel_size//2, kernel_size//2)), 
                         mode='reflect')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            blurred_image[i, j] = np.sum(region * kernel)
    
    return blurred_image

def get_bresenham_circle_points(radius=3):
    """Получение точек на окружности Брезенхема заданного радиуса.
    
    Алгоритм Брезенхема используется для построения окружности в целочисленных координатах.
    Возвращает список точек, упорядоченных по углу для удобства проверки последовательных точек.
    
    Аргументы:
        radius: Радиус окружности (по умолчанию 3 для алгоритма FAST)
    """
    points = []
    x, y = radius, 0
    decision = 1 - radius
    
    while y <= x:
        points.extend([
            (x, y), (y, x), (-x, y), (-y, x),
            (-x, -y), (-y, -x), (x, -y), (y, -x)
        ])
        y += 1
        if decision <= 0:
            decision += 2 * y + 1
        else:
            x -= 1
            decision += 2 * (y - x) + 1
    
    # Удаляем дубликаты и сортируем по углу
    points = list(set(points))
    points.sort(key=lambda p: np.arctan2(p[1], p[0]))
    return points

def fast_detector(image, threshold=30):
    """Детектор углов FAST (Features from Accelerated Segment Test).
    
    Реализует алгоритм FAST с окружностью Брезенхема радиуса 3 (16 точек).
    Использует оптимизацию с проверкой диаметрально противоположных точек.
    
    Аргументы:
        image: Входное изображение
        threshold: Пороговое значение разности яркости (20-40)
    """
    height, width = image.shape
    circle_points = get_bresenham_circle_points(3)  # радиус=3 для 16 точек
    keypoints = []
    
    image_float = image.astype(np.float32)
    padded = np.pad(image_float, pad_width=3, mode='reflect')
    
    for y in range(3, height+3):
        for x in range(3, width+3):
            center_val = padded[y, x]
            # Сначала проверяем диаметрально противоположные точки
            for i in range(8):
                p1 = circle_points[i]
                p2 = circle_points[i+8]
                val1 = padded[y + p1[1], x + p1[0]]
                val2 = padded[y + p2[1], x + p2[0]]
                
                if (abs(center_val - val1) > threshold and 
                    abs(center_val - val2) > threshold):
                    # Подсчет последовательных точек
                    brighter = darker = 0
                    for p in circle_points:
                        val = padded[y + p[1], x + p[0]]
                        if val > center_val + threshold:
                            brighter += 1
                        elif val < center_val - threshold:
                            darker += 1
                        else:
                            brighter = darker = 0
                        
                        # Если на окружности есть 12 последовательных точек,
                        # которые все ярче или все темнее центральной точки на величину threshold (у нас 40),
                        # то центральная точка считается особой...
                        if brighter >= 12 or darker >= 12:
                            keypoints.append((x-3, y-3))
                            break
                    break
    
    return np.array(keypoints)

def harris_response(image, keypoints, k=0.04):
    """Вычисление отклика Харриса для заданных точек.
    
    !!! Не все точки FAST одинаково полезны !!!
    
    Реализует критерий углов Харриса: R = det(M) - k*(trace(M))^2,
    где M - матрица вторых моментов градиента.
    
    Аргументы:
        image: Входное изображение
        keypoints: Список ключевых точек
        k: Параметр Харриса (обычно 0.04-0.06)
    """
    Ix = np.zeros_like(image, dtype=np.float32)
    Iy = np.zeros_like(image, dtype=np.float32)
    
    # Операторы Собеля для вычисления градиентов
    Ix[1:-1, 1:-1] = (image[1:-1, 2:] - image[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1]) / 2
    
    # Гауссово окно для взвешивания
    window = gaussian_kernel(5, 1.0)
    
    responses = []
    for x, y in keypoints:
        # Извлекаем локальное окно 5x5
        start_y, start_x = max(0, y-2), max(0, x-2)
        end_y, end_x = min(image.shape[0], y+3), min(image.shape[1], x+3)
        
        local_Ix = Ix[start_y:end_y, start_x:end_x]
        local_Iy = Iy[start_y:end_y, start_x:end_x]
        
        # Вычисляем компоненты матрицы M
        Ixx = local_Ix * local_Ix
        Ixy = local_Ix * local_Iy
        Iyy = local_Iy * local_Iy
        
        # Применяем гауссово окно
        window_crop = window[:end_y-start_y, :end_x-start_x]
        Sxx = np.sum(Ixx * window_crop)
        Sxy = np.sum(Ixy * window_crop)
        Syy = np.sum(Iyy * window_crop)
        
        # Вычисляем отклик Харриса R = det(M) - k*trace(M)^2
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        R = det_M - k * trace_M * trace_M
        
        responses.append(R)
    
    return np.array(responses)

def filter_keypoints(keypoints, responses, max_points=500):
    """Фильтрация ключевых точек на основе отклика Харриса.
    
    Отбирает top-N точек с наибольшим откликом Харриса.
    
    Аргументы:
        keypoints: Список ключевых точек
        responses: Список откликов Харриса
        max_points: Максимальное количество точек для сохранения
    """
    if len(keypoints) <= max_points:
        return keypoints
    
    # Сортируем по отклику и берем top-N точек
    indices = np.argsort(responses)[-max_points:]
    return keypoints[indices]

def compute_orientation(image, keypoints):
    """Вычисление ориентации ключевых точек с помощью моментов изображения.
    
    Использует моменты m01 и m10 для определения направления градиента:
    angle = arctan2(m01, m10)
    
    !!! Это нужно для инвариантности к повороту !!!
    
    Аргументы:
        image: Входное изображение
        keypoints: Список ключевых точек
    """
    orientations = []
    patch_size = 7  # Размер окна для вычисления моментов
    
    padded = np.pad(image, pad_width=patch_size//2, mode='reflect')
    
    for x, y in keypoints:
        # Извлекаем патч вокруг ключевой точки
        patch = padded[y:y+patch_size, x:x+patch_size]
        
        # Вычисляем центрированные координаты
        y_coords, x_coords = np.mgrid[:patch_size, :patch_size] - patch_size//2
        
        # Вычисляем моменты с учетом центрирования
        m00 = np.sum(patch)
        m10 = np.sum(x_coords * patch)  # используем x_coords для m10
        m01 = np.sum(y_coords * patch)  # используем y_coords для m01
        
        # Вычисляем ориентацию
        if m00 != 0:
            # Нормализуем моменты
            m10 = m10 / m00
            m01 = m01 / m00
            
            # Вычисляем угол, учитывая правильное направление градиента
            angle = np.arctan2(m01, m10)
        else:
            angle = 0
            
        orientations.append(angle)
    
    return np.array(orientations)

def generate_brief_pattern(patch_size=31, num_pairs=256):
    """Генерация случайных пар точек для дескриптора BRIEF.
    
    Генерирует пары точек по нормальному распределению для сравнения интенсивностей.
    
    Аргументы:
        patch_size: Размер патча (по умолчанию 31x31)
        num_pairs: Количество пар точек (по умолчанию 256)
    """
    np.random.seed(42)  # Для воспроизводимости
    sigma = patch_size * patch_size / 25.0
    
    # Генерируем случайные точки по нормальному распределению
    pattern = []
    for _ in range(num_pairs):
        p1 = np.random.normal(0, sigma, 2)
        p2 = np.random.normal(0, sigma, 2)
        
        # Обеспечиваем, чтобы точки были внутри патча
        p1 = np.clip(p1, -patch_size//2, patch_size//2)
        p2 = np.clip(p2, -patch_size//2, patch_size//2)
        
        pattern.append((p1, p2))
    
    return pattern

def rotate_pattern(pattern, angle):
    """Поворот паттерна сэмплирования на заданный угол.
    
    Аргументы:
        pattern: Список пар точек
        angle: Угол поворота в радианах
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    rotated = []
    for (x1, y1), (x2, y2) in pattern:
        # Поворачиваем первую точку
        rx1 = x1 * cos_theta - y1 * sin_theta
        ry1 = x1 * sin_theta + y1 * cos_theta
        
        # Поворачиваем вторую точку
        rx2 = x2 * cos_theta - y2 * sin_theta
        ry2 = x2 * sin_theta + y2 * cos_theta
        
        rotated.append(((rx1, ry1), (rx2, ry2)))
    
    return rotated

def compute_brief(image, keypoints, orientations, pattern):
    """Вычисление дескрипторов BRIEF.
    
    Реализует алгоритм BRIEF с учетом ориентации ключевых точек.
    Использует предварительно сглаженное изображение и поворот паттерна.
    
    Аргументы:
        image: Входное изображение
        keypoints: Список ключевых точек
        orientations: Список ориентаций точек
        pattern: Паттерн сэмплирования BRIEF
    """
    patch_size = 31
    half_patch = patch_size // 2
    descriptors = []
    
    # Предварительное сглаживание изображения
    smoothed = gaussian_blur(image, 5, 2.0)
    padded = np.pad(smoothed, pad_width=half_patch, mode='reflect')
    
    # Предвычисляем повернутые паттерны для 30 ориентаций
    angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
    rotated_patterns = [rotate_pattern(pattern, angle) for angle in angles]
    
    height, width = image.shape
    
    for kp, orientation in zip(keypoints, orientations):
        x, y = kp
        
        # Пропускаем точки слишком близко к границе
        if (x < half_patch or x >= width - half_patch or 
            y < half_patch or y >= height - half_patch):
            continue
        
        # Выбираем ближайший предвычисленный паттерн
        angle_idx = int((orientation + np.pi) * 30 / (2*np.pi)) % 30
        rot_pattern = rotated_patterns[angle_idx]
        
        # Вычисляем дескриптор
        desc = []
        valid_point = True
        
        for (x1, y1), (x2, y2) in rot_pattern:
            # Получаем координаты относительно ключевой точки
            sample1_x = int(x + x1)
            sample1_y = int(y + y1)
            sample2_x = int(x + x2)
            sample2_y = int(y + y2)
            
            # Проверяем, что точки сэмплирования внутри изображения
            if (0 <= sample1_x < width and 0 <= sample1_y < height and
                0 <= sample2_x < width and 0 <= sample2_y < height):
                # Сравниваем интенсивности
                if padded[sample1_y + half_patch, sample1_x + half_patch] < padded[sample2_y + half_patch, sample2_x + half_patch]:
                    desc.append(1)
                else:
                    desc.append(0)
            else:
                valid_point = False
                break
        
        if valid_point:
            descriptors.append(desc)
    
    return np.array(descriptors, dtype=np.uint8)

def save_descriptors(descriptors, filename='descriptors.txt'):
    """Сохранение дескрипторов в файл.
    
    Аргументы:
        descriptors: Массив дескрипторов
        filename: Имя файла для сохранения
    """
    with open(filename, 'w') as f:
        for desc in descriptors:
            f.write(''.join(map(str, desc)) + '\n')

def main():
    # Создаем директорию для результатов
    results_dir = 'and_lab_5_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Загружаем и обрабатываем изображение
    image = load_image('pizza.png')
    if image is None:
        return
    
    # 1. Детектируем углы FAST
    print("Шаг 1: Детектирование углов FAST...")
    keypoints = fast_detector(image, threshold=40)
    print(f"Найдено {len(keypoints)} начальных точек")
    
    # 2. Фильтруем с помощью отклика Харриса
    print("\nШаг 2: Фильтрация с помощью критерия Харриса...")
    responses = harris_response(image, keypoints, k=0.06)
    keypoints = filter_keypoints(keypoints, responses)
    print(f"Осталось {len(keypoints)} точек после фильтрации")
    
    # 3. Вычисляем ориентацию
    print("\nШаг 3: Вычисление ориентации точек...")
    orientations = compute_orientation(image, keypoints)
    
    # 4. Генерируем паттерн BRIEF
    print("\nШаг 4: Генерация паттерна BRIEF...")
    pattern = generate_brief_pattern()
    
    # 5. Вычисляем дескрипторы BRIEF
    print("\nШаг 5: Вычисление дескрипторов BRIEF...")
    descriptors = compute_brief(image, keypoints, orientations, pattern)
    print(f"Создано {len(descriptors)} дескрипторов")
    
    # Сохраняем дескрипторы
    print("\nСохранение дескрипторов в файл...")
    save_descriptors(descriptors)
    
    # Визуализируем результаты
    print("\nВизуализация результатов...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=3)
    
    # Рисуем линии ориентации
    for (x, y), orientation in zip(keypoints, orientations):
        dx = np.cos(orientation) * 10
        dy = np.sin(orientation) * 10
        plt.plot([x, x + dx], [y, y + dy], 'g-', linewidth=1)
    
    plt.title(f'Особые точки ORB (Найдено {len(keypoints)} точек)')
    plt.axis('off')
    plt.savefig(f'{results_dir}/orb_features.png')
    plt.close()
    
    print("\nГотово! Результаты сохранены в директории", results_dir)

if __name__ == "__main__":
    main()
