import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Константы
PATCH_SIZE = 31  # Размер патча для дескриптора
K = 0.06  # Константа для вычисления значений R в методе Харриса
LOWE_RATIO = 0.8  # Порог для теста Лоу
RANSAC_ITERATIONS = 5  # Количество итераций RANSAC
RANSAC_DISTANCE_THRESHOLD = 2  # Порог расстояния для RANSAC

# Создание папки для сохранения результатов
RESULTS_FOLDER = "and_lab_6_results"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def load_image(file_path):
    """
    Загружает изображение из файла и преобразует его в оттенки серого.

    Args:
        file_path: Путь к файлу изображения.

    Returns:
        NumPy массив, представляющий изображение в оттенках серого, или None, если произошла ошибка.
    """
    try:
        image = Image.open(file_path).convert("L")
        return np.array(image, dtype=int)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None

def display_image(image, title="", cmap="gray", save_path=None):
    """
    Отображает изображение с помощью matplotlib.

    Args:
        image: NumPy массив, представляющий изображение.
        title: Заголовок изображения.
        cmap: Цветовая карта matplotlib.
        save_path: Путь для сохранения изображения. Если None, изображение не сохраняется.
    """
    plt.figure()
    plt.imshow(image, cmap=cmap, vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")  # Отключаем оси

    if save_path:
        plt.savefig(save_path)

    #plt.show()

def resize_image(image, scale_factor):
    """
    Изменяет размер изображения с помощью усреднения пикселей.

    Args:
        image: NumPy массив, представляющий изображение.
        scale_factor: Коэффициент масштабирования.

    Returns:
        NumPy массив, представляющий изображение с измененным размером.
    """
    original_height, original_width = image.shape
    new_height = original_height // scale_factor
    new_width = original_width // scale_factor

    resized_image = np.zeros((new_height, new_width), dtype=int)

    for i in range(new_height):
        for j in range(new_width):
            # Усредняем значения пикселей в блоке scale_factor x scale_factor
            resized_image[i, j] = int(
                np.mean(
                    image[
                        i * scale_factor : scale_factor * (i + 1),
                        j * scale_factor : scale_factor * (j + 1),
                    ]
                )
            )

    return resized_image

def get_circle_pixels_intensities(image, center_y, center_x):
    """
    Возвращает интенсивности пикселей, лежащих на окружности вокруг заданной точки.

    Args:
        image: NumPy массив, представляющий изображение.
        center_y: Координата y центра окружности.
        center_x: Координата x центра окружности.

    Returns:
        Кортеж из трех массивов:
        - top_circle: Интенсивности пикселей верхней части окружности.
        - bottom_circle: Интенсивности пикселей нижней части окружности.
        - full_circle: Интенсивности пикселей всей окружности.
    """

    # Верхняя часть окружности
    top_circle = np.concatenate(
        (
            image[center_y : center_y + 2, center_x - 3],
            image[center_y + 2, center_x - 2 : center_x - 1],
            image[center_y + 3, center_x - 1 : center_x + 2],
            image[center_y + 2, center_x + 2 : center_x + 3],
            image[center_y : center_y + 2, center_x + 3],
        ),
        axis=None,
    )

    # Нижняя часть окружности
    bottom_circle = np.concatenate(
        (
            image[center_y + 3, center_x : center_x + 2],
            image[center_y + 2, center_x + 2 : center_x + 3],
            image[center_y - 1 : center_y + 2, center_x + 3],
            image[center_y - 2, center_x + 2 : center_x + 3],
            image[center_y - 3, center_x : center_x + 2],
        ),
        axis=None,
    )

    # Полная окружность
    full_circle = np.concatenate(
        (
            image[center_y - 1 : center_y + 2, center_x - 3],  # Верхние точки
            image[center_y + 2, center_x - 2 : center_x - 1],  # Правые верхние
            image[center_y + 3, center_x - 1 : center_x + 2],  # Правые точки
            image[center_y + 2, center_x + 2 : center_x + 3],  # Правые нижние
            image[center_y - 1 : center_y + 2, center_x + 3],  # Нижние точки
            image[center_y - 2, center_x + 2 : center_x + 3],  # Левые нижние
            image[center_y - 3, center_x - 1 : center_x + 2],  # Левые точки
            image[center_y - 2, center_x - 2 : center_x - 1],  # Левые верхние
        ),
        axis=None,
    )

    return top_circle, bottom_circle, full_circle

def fast_feature_detector(image, min_consecutive_pixels, intensity_threshold):
    """
    Реализует алгоритм FAST для обнаружения ключевых точек на изображении.

    Алгоритм:
    1. Для каждого пикселя изображения:
        - Выбирается окружность из 16 пикселей вокруг текущего пикселя.
        - Проверяется, есть ли на окружности n последовательных пикселей, которые все ярче или все темнее
          центрального пикселя на величину порога t.
        - Если такое условие выполняется, пиксель считается ключевой точкой.

    Args:
        image: NumPy массив, представляющий изображение в оттенках серого.
        min_consecutive_pixels: Минимальное количество последовательных пикселей (n),
                                 которые должны быть ярче или темнее центрального пикселя.
        intensity_threshold: Порог интенсивности (t).

    Returns:
        Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.
    """

    # Проверка корректности входных параметров
    if min_consecutive_pixels < 6 or min_consecutive_pixels > 16:
        print(
            "Ошибка: Минимальное количество последовательных пикселей должно быть в диапазоне от 6 до 16."
        )
        return []

    height, width = image.shape
    feature_points = []
    margin = PATCH_SIZE // 2

    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            # Центральный пиксель
            center_pixel = image[y, x]

            # Получаем интенсивности пикселей на окружности
            top, bottom, full = get_circle_pixels_intensities(image, y, x)

            # Проверяем условие для ярких точек
            if all(
                q > center_pixel + intensity_threshold for q in top
            ) or all(q > center_pixel + intensity_threshold for q in bottom):
                consecutive_count = 0
                # Дублируем массив, чтобы обеспечить непрерывность при переходе с конца на начало
                for pixel in np.concatenate((full, full)):
                    if pixel > center_pixel + intensity_threshold:
                        consecutive_count += 1
                    else:
                        consecutive_count = 0
                    if consecutive_count >= min_consecutive_pixels:
                        feature_points.append((y, x))
                        break

            # Проверяем условие для темных точек
            elif all(
                q < center_pixel - intensity_threshold for q in top
            ) or all(q < center_pixel - intensity_threshold for q in bottom):
                consecutive_count = 0
                for pixel in np.concatenate((full, full)):
                    if pixel < center_pixel - intensity_threshold:
                        consecutive_count += 1
                    else:
                        consecutive_count = 0
                    if consecutive_count >= min_consecutive_pixels:
                        feature_points.append((y, x))
                        break

    return feature_points

def create_gaussian_kernel(sigma, size):
    """
    Создает ядро Гаусса заданного размера и стандартного отклонения.

    Args:
        sigma: Стандартное отклонение распределения Гаусса.
        size: Размер ядра (должно быть нечетным).

    Returns:
        NumPy массив, представляющий ядро Гаусса.
    """
    # Создаем ось x от -size//2 до size//2
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    # Создаем сетку координат xx, yy
    xx, yy = np.meshgrid(ax, ax)
    # Вычисляем ядро Гаусса по формуле
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    # Нормализуем ядро, чтобы сумма его элементов равнялась 1
    kernel = kernel / (2 * np.pi * sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, kernel):
    """
    Применяет гауссово размытие к изображению.

    Args:
        image: NumPy массив, представляющий изображение.
        kernel: Ядро Гаусса.

    Returns:
        NumPy массив, представляющий размытое изображение.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_y, padding_x = kernel_height // 2, kernel_width // 2

    # Создаем пустой массив для результата
    blurred_image = np.zeros_like(image, dtype=np.float32)

    # Применяем свертку
    for i in range(image_height):
        for j in range(image_width):
            sum_value = 0.0

            for m in range(kernel_height):
                for n in range(kernel_width):
                    # Индексы для области среза
                    x = i + m - padding_y
                    y = j + n - padding_x

                    # Проверяем, чтобы не выйти за границы изображения
                    if 0 <= x < image_height and 0 <= y < image_width:
                        sum_value += image[x, y] * kernel[m, n]

            blurred_image[i, j] = sum_value

    # Обрезаем значения, чтобы они оставались в диапазоне [0, 255]
    return np.clip(blurred_image, 0, 255).astype(np.uint8)

def apply_filter(image, kernel):
    """
    Применяет фильтр (свертку) к изображению.

    Args:
        image: NumPy массив, представляющий изображение.
        kernel: Ядро фильтра.

    Returns:
        NumPy массив, представляющий отфильтрованное изображение.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_y, padding_x = kernel_height // 2, kernel_width // 2

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            sum_value = 0.0

            for m in range(kernel_height):
                for n in range(kernel_width):
                    x = i + m - padding_y
                    y = j + n - padding_x

                    if 0 <= x < image_height and 0 <= y < image_width:
                        sum_value += image[x, y] * kernel[m, n]

            filtered_image[i, j] = sum_value

    return filtered_image

def compute_image_gradient(image):
    """
    Вычисляет градиент изображения с помощью фильтров Собеля.

    Args:
        image: NumPy массив, представляющий изображение в оттенках серого.

    Returns:
        NumPy массив, представляющий градиент изображения.
        Каждый элемент массива - это вектор (dx, dy), где dx - градиент по оси x,
        dy - градиент по оси y.
    """

    # Фильтры Собеля для вычисления градиента по осям x и y
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Применяем фильтры Собеля к изображению
    gradient_x = apply_filter(image, sobel_x)
    gradient_y = apply_filter(image, sobel_y)

    height, width = image.shape
    # Создаем массив для хранения градиента
    gradient = np.zeros((height, width, 2), dtype=float)

    # Записываем значения градиента в массив
    gradient[:, :, 0] = gradient_x
    gradient[:, :, 1] = gradient_y

    return gradient

def compute_structure_tensor_for_points(image, gradient, feature_points):
    """
    Вычисляет матрицу вторых моментов (тензор структуры) для каждой ключевой точки.

    Матрица M (тензор структуры) для каждой точки (x, y) определяется следующим образом:
    M = [[Σ(w*dx*dx), Σ(w*dx*dy)],
         [Σ(w*dx*dy), Σ(w*dy*dy)]]
    где:
        - dx, dy - компоненты градиента в точке (x, y).
        - w - вес, определяемый ядром Гаусса.
        - Σ - сумма по окрестности точки (x, y).

    Args:
        image: NumPy массив, представляющий изображение в оттенках серого.
        gradient: NumPy массив, представляющий градиент изображения.
        feature_points: Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.

    Returns:
        Список NumPy массивов, где каждый массив представляет собой матрицу M для соответствующей ключевой точки.
    """
    w, h = image.shape

    # Добавляем отступы к изображению для корректной работы с окном
    padded_image = np.zeros((w + 4, h + 4), dtype=int)
    padded_image[2 : w + 2, 2 : h + 2] = image

    # Используем ядро Гаусса размером 5x5
    W = create_gaussian_kernel(sigma=1, size=5)

    M_matrices = []

    for x, y in feature_points:
        M = np.zeros((2, 2), dtype=float)

        # Проходим по окну 5x5 вокруг ключевой точки
        for i in range(x - 2, x + 3):
            for j in range(y - 2, y + 3):
                Ix = gradient[i, j, 0]
                Iy = gradient[i, j, 1]

                # Вычисляем матрицу A
                A = np.array([[Ix**2, Ix * Iy], [Ix * Iy, Iy**2]], dtype=float)

                # Взвешиваем матрицу A с помощью ядра Гаусса и добавляем к матрице M
                new = np.sum(W * padded_image[i : i + 5, j : j + 5]) * A
                M = np.add(M, new)

        M_matrices.append(M)

    return M_matrices

def compute_harris_response_values(M_matrices, k):
    """
    Вычисляет значение R (отклик детектора Харриса) для каждой матрицы M.

    R = det(M) - k * (trace(M))^2

    Args:
        M_matrices: Список NumPy массивов, где каждый массив представляет собой матрицу M.
        k: Константа.

    Returns:
        Список значений R.
    """
    R_values = []
    for m in M_matrices:
        # Вычисляем R
        R = np.linalg.det(m) - k * np.trace(m) ** 2
        R_values.append(R)

    return R_values

def filter_points_by_harris_response(feature_points, R_values, max_points):
    """
    Фильтрует ключевые точки на основе значения R (отклика детектора Харриса).

    Алгоритм:
    1. Сортируем точки по убыванию значения R.
    2. Выбираем max_points точек с наибольшим значением R, которые больше 0.

    Args:
        feature_points: Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.
        R_values: Список значений R для каждой точки.
        max_points: Максимальное количество точек, которое нужно оставить.

    Returns:
        Список отфильтрованных ключевых точек.
    """
    # Объединяем точки и значения R
    points_with_R = list(zip(feature_points, R_values))

    # Сортируем точки по убыванию R
    sorted_points = sorted(points_with_R, key=lambda x: x[1], reverse=True)

    # Фильтруем точки
    filtered_points = []
    for point, r_value in sorted_points[:max_points]:
        if r_value < 0:
            break
        filtered_points.append(point)

    return filtered_points

def draw_features_on_image(image, points, color=[128, 0, 255]):
    """
    Рисует ключевые точки на изображении.

    Args:
        image: NumPy массив, представляющий изображение.
        points: Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.
        color: Цвет, которым будут нарисованы точки (по умолчанию синий).

    Returns:
        NumPy массив, представляющий изображение с нарисованными точками.
    """
    height, width = image.shape

    # Если изображение черно-белое, преобразуем его в RGB
    if len(image.shape) == 2:
        result_image = np.stack([image] * 3, axis=-1)
    else:
        result_image = image.copy()

    # Рисуем точки
    for center_y, center_x in points:
        for dy in range(-1, 2):
            result_image[center_y + dy, center_x - 3] = color
            result_image[center_y + dy, center_x + 3] = color

        result_image[center_y + 2, center_x - 2] = color
        result_image[center_y + 2, center_x + 2] = color

        for dx in range(-1, 2):
            result_image[center_y + 3, center_x + dx] = color
            result_image[center_y - 3, center_x + dx] = color

        result_image[center_y - 2, center_x + 2] = color
        result_image[center_y - 2, center_x - 2] = color

    return result_image

def calculate_image_moments(image, p, q, x, y, r):
    """
    Вычисляет центральный момент изображения I в точке (x, y) порядка (p, q) в окрестности радиуса r.

    Args:
        image: NumPy массив, представляющий изображение в оттенках серого.
        p: Порядок момента по x.
        q: Порядок момента по y.
        x: Координата x точки.
        y: Координата y точки.
        r: Радиус окрестности.

    Returns:
        Значение центрального момента.
    """
    w, h = image.shape

    # Координаты центральной точки изображения
    ci = w // 2
    cj = h // 2

    m = 0

    # Проходим по окну (max и min - границы окна)
    for i in range(max(0, y - r), min(w, y + r + 1)):
        for j in range(max(0, x - r), min(h, x + r + 1)):
            m += (i - ci) ** p * (j - cj) ** q * image[i, j]
    return m

def calculate_orientations(image, points, r):
    """
    Вычисляет ориентацию (угол) для каждой ключевой точки.

    Ориентация вычисляется как угол между осью x и вектором, соединяющим центр масс окрестности
    точки с центром изображения.

    Угол вычисляется по формуле: angle = arctan2(m01, m10), где m01 и m10 - центральные моменты.

    Args:
        image: NumPy массив, представляющий изображение в оттенках серого.
        points: Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.
        r: Радиус окрестности для вычисления моментов.

    Returns:
        Список углов (в радианах) для каждой ключевой точки.
    """
    angles = []
    for x, y in points:
        # Вертикальный момент m01 = Σ( (j-cj) * I[j, i] )
        m01 = calculate_image_moments(image, 0, 1, x, y, r)
        # Горизонтальный момент m10 = Σ( (i-ci) * I[j, i] )
        m10 = calculate_image_moments(image, 1, 0, x, y, r)
        # Угол в диапазоне [0, 2π] (без отрицательных значений)
        a = math.atan2(m01, m10) % (2 * np.pi)
        angles.append(a)

    return angles

def generate_rotation_matrices(angles):
    """
    Генерирует матрицы поворота для заданных углов.

    Args:
        angles: Список углов (в радианах).

    Returns:
        Список NumPy массивов, где каждый массив представляет собой матрицу поворота.
    """
    rotation_matrices = []
    for angle in angles:
        # Матрица поворота
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotation_matrices.append(rotation_matrix)
    return rotation_matrices

def generate_rotated_sampling_points(rotation_matrices, n, patch_size):
    """
    Генерирует повернутые пары точек для построения дескриптора.

    Args:
        rotation_matrices: Список матриц поворота.
        n: Количество случайных пар точек.
        patch_size: Размер патча.

    Returns:
        Список списков пар точек, повернутых на разные углы.
    """
    # Генерируем случайные пары точек
    random_points = (np.pi / 5) * np.random.randn(n, 2, 2)
    # Нормализуем и масштабируем
    random_points /= np.max(random_points)
    random_points *= patch_size // 2 - 2

    rotated_points_list = []
    for rotation_matrix in rotation_matrices:
        rotated_points = []
        for i in range(random_points.shape[0]):
            # Поворачиваем каждую пару точек
            rotated_point1 = rotation_matrix.dot(random_points[i, 0])
            rotated_point2 = rotation_matrix.dot(random_points[i, 1])
            rotated_points.append(
                [rotated_point1.astype(int), rotated_point2.astype(int)]
            )
        rotated_points_list.append(rotated_points)

    return rotated_points_list

def create_orb_descriptors(
    gaussian_image, keypoints, orientations, patch_size, rotated_sampling_points
):
    """
    Создает ORB дескрипторы для ключевых точек.

    Алгоритм:
    1. Для каждой ключевой точки:
        - Находим ближайший угол поворота из списка углов, используемых для построения дескриптора.
        - Выбираем соответствующие повернутые пары точек.
        - Смещаем пары точек относительно центра ключевой точки.
        - Сравниваем интенсивности пикселей в парах точек и формируем бинарный вектор дескриптора.

    Args:
        gaussian_image: Изображение, размытое фильтром Гаусса.
        keypoints: Список кортежей, где каждый кортеж содержит координаты (y, x) ключевой точки.
        orientations: Список ориентаций (углов) для каждой ключевой точки.
        patch_size: Размер патча.
        rotated_sampling_points: Список списков пар точек, повернутых на разные углы.

    Returns:
        Список NumPy массивов, где каждый массив представляет собой бинарный ORB дескриптор.
    """
    height, width = gaussian_image.shape
    # 30 равномерно распределенных углов от 0 до 2π
    angles_for_descriptor = [2 * k * np.pi / 30 for k in range(30)]

    def compare_intensities(image, point1, point2):
        """
        Сравнивает интенсивности двух точек на изображении.

        Args:
            image: NumPy массив, представляющий изображение.
            point1: Координаты первой точки (y, x).
            point2: Координаты второй точки (y, x).

        Returns:
            1, если интенсивность первой точки меньше интенсивности второй точки, иначе 0.
        """
        # Проверяем, чтобы точки не выходили за границы изображения
        if (
            0 <= point1[0] < height
            and 0 <= point1[1] < width
            and 0 <= point2[0] < height
            and 0 <= point2[1] < width
        ):
            intensity1 = image[point1[0], point1[1]]
            intensity2 = image[point2[0], point2[1]]
            return 1 if intensity1 < intensity2 else 0
        else:
            return 0  # Если точки выходят за границы, возвращаем 0

    descriptors = []

    # Ключевые точки
    for point_idx, center_point in enumerate(keypoints):
        # Находим индекс угла поворота
        rotation_idx = len(angles_for_descriptor) - 1
        # Ближайший меньший угол
        for angle_idx, angle in enumerate(angles_for_descriptor):
            if orientations[point_idx] < angle:
                rotation_idx = angle_idx - 1
                break

        # Повернутые пары точек для текущего угла
        sampling_points_for_angle = []
        for i in range(len(rotated_sampling_points[0])):
            # Смещение относительно центра
            offset_point1 = (
                rotated_sampling_points[rotation_idx][i][0] + center_point
            )
            offset_point2 = (
                rotated_sampling_points[rotation_idx][i][1] + center_point
            )
            sampling_points_for_angle.append([offset_point1, offset_point2])

        # Формирование дескриптора
        binary_tests = []
        for test_points in reversed(sampling_points_for_angle):
            result = compare_intensities(
                gaussian_image, test_points[0], test_points[1]
            )
            binary_tests.append(result)

        descriptors.append(np.array(binary_tests))

    return descriptors

def calculate_hamming_distances(descriptors1, descriptors2):
    """
    Вычисляет расстояние Хэмминга между двумя наборами дескрипторов.

    Args:
        descriptors1: Список дескрипторов первого изображения.
        descriptors2: Список дескрипторов второго изображения.

    Returns:
        NumPy массив, представляющий матрицу расстояний Хэмминга.
        Каждый элемент матрицы (i, j) - это расстояние Хэмминга между i-м дескриптором
        из descriptors1 и j-м дескриптором из descriptors2.
    """
    num_descriptors1 = len(descriptors1)
    num_descriptors2 = len(descriptors2)

    hamming_distances = np.zeros(
        (num_descriptors1, num_descriptors2), dtype=int
    )

    for i in range(num_descriptors1):
        for j in range(num_descriptors2):
            # Вычисляем расстояние Хэмминга как сумму поэлементных XOR
            distance = sum(np.absolute(descriptors1[i] - descriptors2[j]))
            hamming_distances[i, j] = distance

    return hamming_distances
def perform_lowe_test(keypoints1, keypoints2, hamming_distances):
    """
    Выполняет тест Лоу для фильтрации сопоставлений ключевых точек.

    Алгоритм:
    1. Для каждой точки первого изображения находим два ближайших сопоставления на втором изображении.
    2. Вычисляем отношение расстояний между этими двумя сопоставлениями.
    3. Если отношение меньше порога LOWE_RATIO, считаем сопоставление хорошим.
    4. Проделываем то же самое в обратном направлении (для каждой точки второго изображения).

    Args:
        keypoints1: Список ключевых точек первого изображения.
        keypoints2: Список ключевых точек второго изображения.
        hamming_distances: Матрица расстояний Хэмминга.

    Returns:
        Список списков пар точек:
        - matches_forward: Сопоставления, найденные при поиске от первого изображения ко второму.
        - matches_backward: Сопоставления, найденные при поиске от второго изображения к первому.
    """
    height, width = hamming_distances.shape

    matches_forward = []  # прямой поиск (1→2)
    matches_backward = []  # обратный поиск (2→1)

    # Прямой поиск (для каждой точки первого изображения)
    for i in range(height):
        # Сортируем расстояния по возрастанию
        distances = sorted(hamming_distances[i])

        # Проверяем первые два ближайших соответствия
        for t in range(len(distances) - 1):
            dist1, dist2 = distances[t : t + 2]  # два ближайших расстояния
            ratio = dist1 / dist2

            if ratio < LOWE_RATIO:
                # Находим индекс точки с минимальным расстоянием
                match_idx = list(hamming_distances[i]).index(dist1)
                # Пара точек -> соответствие
                matches_forward.append([keypoints1[i], keypoints2[match_idx]])
                break

    # Обратный поиск (для каждой точки второго изображения)
    for i in range(width):
        # Сортируем расстояния по возрастанию
        distances = sorted(hamming_distances[:, i])
        for t in range(len(distances) - 1):
            dist1, dist2 = distances[t : t + 2]
            ratio = dist1 / dist2

            if ratio < LOWE_RATIO:
                match_idx = list(hamming_distances[:, i]).index(dist1)
                matches_backward.append([keypoints1[match_idx], keypoints2[i]])
                break

    return [matches_forward, matches_backward]

def perform_cross_check(lowe_matches):
    """
    Выполняет перекрестную проверку (cross-check) для фильтрации сопоставлений, полученных после теста Лоу.

    Алгоритм:
    1. Для каждой пары точек (p1, p2) в списке прямых сопоставлений (matches_forward):
        - Ищем пару (p1, p2) в списке обратных сопоставлений (matches_backward).
        - Если находим, добавляем p1 и p2 в списки отфильтрованных точек.

    Args:
        lowe_matches: Список списков пар точек, полученный после теста Лоу.

    Returns:
        Список списков пар точек:
        - filtered_points1: Отфильтрованные точки первого изображения.
        - filtered_points2: Отфильтрованные точки второго изображения.
    """
    filtered_points1 = []  # первое изображение
    filtered_points2 = []  # второе изображение

    # Верхний цикл - прямые соответствия
    for i in range(len(lowe_matches[0])):
        point1, point2 = lowe_matches[0][i]  # точки прямого соответствия

        # Эта же пара в обратных соответствиях
        for j in range(len(lowe_matches[1])):
            point1_reverse, point2_reverse = lowe_matches[1][j]

            # В обоих направлениях есть эта пара
            if point1 == point1_reverse and point2 == point2_reverse:
                filtered_points1.append(point1)
                filtered_points2.append(point2)
                break

    return [filtered_points1, filtered_points2]

def draw_line_bresenham(x0, y0, x1, y1, max_size):
    """
    Рисует линию между двумя точками по алгоритму Брезенхема.

    Args:
        x0: Координата x первой точки.
        y0: Координата y первой точки.
        x1: Координата x второй точки.
        y1: Координата y второй точки.
        max_size: Максимальный размер изображения (для проверки границ).

    Returns:
        Список координат точек, составляющих линию.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1

    error = dx - dy

    line_points = []
    x, y = x0, y0

    while True:
        if 0 <= x < max_size and 0 <= y < max_size:
            line_points.append([x, y])

        if x == x1 and y == y1:
            break

        error2 = error * 2

        if error2 > -dy:
            error -= dy
            x += step_x

        if error2 < dx:
            error += dx
            y += step_y

    return line_points

def visualize_matches(image1, image2, points1, points2):
    """
    Визуализирует сопоставления между двумя изображениями, соединяя соответствующие точки линиями.

    Args:
        image1: NumPy массив, представляющий первое изображение.
        image2: NumPy массив, представляющий второе изображение.
        points1: Список точек первого изображения.
        points2: Список точек второго изображения.

    Returns:
        NumPy массив, представляющий объединенное изображение с нарисованными линиями сопоставлений.
    """
    img1_with_points = draw_features_on_image(image1, points1)
    img2_with_points = draw_features_on_image(image2, points2)

    # Размеры для объединенного изображения
    total_width = img1_with_points.shape[1] + img2_with_points.shape[1]
    height = img2_with_points.shape[0]
    width2 = img2_with_points.shape[1]

    result = np.ones((height, total_width, 3), dtype=int) * 255

    result[0:img2_with_points.shape[0], 0:width2] = img2_with_points
    result[0:img1_with_points.shape[0], width2:] = img1_with_points

    color = [128, 0, 255]
    for i in range(len(points1)):
        x1, y1 = points1[i]
        y1 += width2
        x2, y2 = points2[i]

        line_points = draw_line_bresenham(x1, y1, x2, y2, total_width)

        for x, y in line_points:
            result[x, y] = color

    return result

def calculate_affine_transform_parameters(src_points, dst_points):
    """
    Вычисляет параметры аффинного преобразования (матрицу M и вектор смещения T) по трем парам соответствующих точек.

    Уравнения аффинного преобразования:
    x' = m1*x + m2*y + tx
    y' = m3*x + m4*y + ty

    где:
        (x, y) - координаты исходной точки.
        (x', y') - координаты преобразованной точки.
        m1, m2, m3, m4 - элементы матрицы аффинного преобразования M.
        tx, ty - компоненты вектора смещения T.

    Args:
        src_points: Список из трех точек исходного изображения [(x1, y1), (x2, y2), (x3, y3)].
        dst_points: Список из трех соответствующих точек целевого изображения [(x'1, y'1), (x'2, y'2), (x'3, y'3)].

    Returns:
        Кортеж (M, T), где:
        - M: NumPy массив 2x2, представляющий матрицу аффинного преобразования.
        - T: NumPy массив 2x1, представляющий вектор смещения.
        Если не удалось вычислить параметры (например, из-за линейной зависимости точек), возвращает (None, None).
    """
    src = np.array(src_points)  # исходные точки
    dst = np.array(dst_points)  # целевые точки

    x1, y1 = src[0]
    x2, y2 = src[1]
    x3, y3 = src[2]

    u1, v1 = dst[0]
    u2, v2 = dst[1]
    u3, v3 = dst[2]

    # определитель первой матрицы
    det1 = (x1 - x3) * (y1 - y2) - (x1 - x2) * (y1 - y3)
    # определитель второй матрицы
    det2 = y1 - y2

    if abs(det1) < 1e-10 or abs(det2) < 1e-10:
        return None, None

    try:
        m1 = ((u1 - u3) * (y1 - y2) - (u1 - u2) * (y1 - y3)) / det1
        m2 = ((u1 - u2) - m1 * (x1 - x2)) / det2
        tx = u1 - m1 * x1 - m2 * y1

        m3 = ((v1 - v3) * (y1 - y2) - (v1 - v2) * (y1 - y3)) / det1
        m4 = ((v1 - v2) - m3 * (x1 - x2)) / det2
        ty = v1 - m3 * x1 - m4 * y1

        M = np.array([[m1, m2], [m3, m4]])
        T = np.array([tx, ty])

        return M, T

    except:
        return None, None

def apply_affine_transformation(points, M, T):
    """
    Применяет аффинное преобразование к списку точек.

    Args:
        points: Список точек [(x1, y1), (x2, y2), ...].
        M: NumPy массив 2x2, представляющий матрицу аффинного преобразования.
        T: NumPy массив 2x1, представляющий вектор смещения.

    Returns:
        NumPy массив, содержащий преобразованные точки.
    """

    M = np.array(M)
    T = np.array(T)
    points = np.array(points)

    x = points[:, 0]
    y = points[:, 1]

    # x' = m11*x + m12*y + tx
    # y' = m21*x + m22*y + ty
    x_new = M[0][0] * x + M[0][1] * y + T[0]
    y_new = M[1][0] * x + M[1][1] * y + T[1]

    # Преобразованные координаты -> массив точек
    return np.column_stack((x_new, y_new))

def calculate_affine_transform_mnk(points):
    """
    Вычисляет параметры аффинного преобразования (матрицу M и вектор смещения T) методом наименьших квадратов (МНК).

    Система уравнений для каждой пары точек (x,y) -> (u,v):
    u = m1*x + m2*y + tx
    v = m3*x + m4*y + ty

    Матричная форма A*X = B
    A = [x y 0 0 1 0] для u
        [0 0 x y 0 1] для v
    X = [m1 m2 m3 m4 tx ty]^T
    B = [u v]^T

    Args:
        points: Список пар точек, где каждая пара содержит исходную точку и соответствующую ей целевую точку.
                Например: [[(x1, y1), (u1, v1)], [(x2, y2), (u2, v2)], ...].

    Returns:
        Кортеж (M, T), где:
        - M: NumPy массив 2x2, представляющий матрицу аффинного преобразования.
        - T: NumPy массив 2x1, представляющий вектор смещения.
        Если не удалось вычислить параметры (например, недостаточно точек), возвращает (None, None).
    """
    if len(points) < 3:
        print("Недостаточно точек для вычисления аффинного преобразования методом МНК.")
        return None, None

    A = []  # матрица коэффициентов
    B = []  # вектор значений
    for (x, y), (u, v) in points:
        # Уравнение для u
        A.append([x, y, 0, 0, 1, 0])
        B.append(u)
        # Уравнение для v
        A.append([0, 0, x, y, 0, 1])
        B.append(v)

    A = np.array(A, dtype=np.float64)
    # Массив [u1,v1,u2,v2,...] -> столбец [[u1],[v1],[u2],[v2],...]
    B = np.array(B, dtype=np.float64).reshape(-1, 1)

    # Решение МНК: X = (A^T * A)^(-1) * A^T * B
    try:
        X = np.linalg.inv(A.T @ A) @ A.T @ B
    except np.linalg.LinAlgError:
        print("Ошибка: Невозможно вычислить обратную матрицу. Возможно, матрица сингулярна.")
        return None, None

    # матрица M и вектор T
    M = X[0:4].reshape(2, 2)
    T = X[4:6].reshape(2, 1)

    return M, T

def ransac_affine_transform(points1, points2, num_iterations, distance_threshold):
    """
    Вычисляет параметры аффинного преобразования с помощью алгоритма RANSAC.

    Алгоритм:
    1. Случайным образом выбираем минимальное количество точек (3 для аффинного преобразования) из points1 и points2.
    2. Вычисляем параметры аффинного преобразования по этим точкам.
    3. Применяем преобразование ко всем точкам из points1.
    4. Подсчитываем количество inlier'ов - точек, для которых расстояние между преобразованной точкой и
       соответствующей точкой из points2 меньше distance_threshold.
    5. Повторяем шаги 1-4 num_iterations раз.
    6. Выбираем параметры, для которых количество inlier'ов максимально.
    7. (Опционально) Уточняем параметры, используя все inlier'ы, методом наименьших квадратов.

    Args:
        points1: Список точек первого изображения.
        points2: Список соответствующих точек второго изображения.
        num_iterations: Количество итераций RANSAC.
        distance_threshold: Порог расстояния для определения inlier'ов.

    Returns:
        Кортеж (M, T), где:
        - M: NumPy массив 2x2, представляющий матрицу аффинного преобразования.
        - T: NumPy массив 2x1, представляющий вектор смещения.
        Если не удалось вычислить параметры (например, недостаточно точек), возвращает (None, None).
    """
    min_points = 3  # минимальное количество точек для аффинного преобразования

    # Проверка количества точек
    if len(points1) < min_points:
        print("Недостаточно точек для RANSAC.")
        return None, None

    # Инициализация лучших параметров
    best_inliers_count = -1
    total_points = len(points1)
    best_matching_points = []
    best_transform_matrix = None
    best_translation = None

    try:
        # Цикл N итераций
        for _ in range(num_iterations):
            # Случайное множество сопоставленных точек (по минимальному количеству)
            random_indices = random.sample(range(len(points1)), min_points)
            sample_points1 = [points1[i] for i in random_indices]
            sample_points2 = [points2[i] for i in random_indices]

            # Оценка параметров через СЛАУ
            transform_matrix, translation = calculate_affine_transform_parameters(
                sample_points1, sample_points2
            )
            if transform_matrix is None or translation is None:
                continue

            # Преобразование точек
            transformed_points = apply_affine_transformation(
                points1, transform_matrix, translation
            )

            # Подсчет верных соответствий (inliers)
            current_matching_points = []
            inliers_count = 0

            # Нахождение не-выбросов (inliers) по порогу расстояния
            for idx in range(len(points2)):
                x_transformed, y_transformed = transformed_points[idx]
                x_target, y_target = points2[idx]

                # Проверка расстояния
                if (
                    abs(x_transformed - x_target) < distance_threshold
                    and abs(y_transformed - y_target) < distance_threshold
                ):
                    x_source, y_source = points1[idx]
                    inliers_count += 1
                    # Пара точек исходная -> целевая
                    current_matching_points.append(
                        [[x_source, y_source], [x_target, y_target]]
                    )

            # Обновление лучшего результата
            # Все точки совпали - идеальный случай
            if inliers_count == total_points:
                best_matching_points = current_matching_points
                best_transform_matrix = transform_matrix
                best_translation = translation
                break

            # Обновление лучшего результата (если больше inliers)
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_matching_points = current_matching_points
                best_transform_matrix = transform_matrix
                best_translation = translation

        # Финальное уточнение методом наименьших квадратов
        if best_matching_points:
            best_transform_matrix, best_translation = calculate_affine_transform_mnk(
                best_matching_points
            )

        return best_transform_matrix, best_translation

    except Exception as e:
        print(f"Ошибка RANSAC: {str(e)}")
        return None, None

def detect_object_with_transform(box_image, scene_image, transform_matrix, translation):
    """
    Обнаруживает объект на изображении сцены, соединяя 4 трансформированных вершины в полигон.

    Args:
        box_image: Изображение объекта.
        scene_image: Изображение сцены.
        transform_matrix: Матрица аффинного преобразования.
        translation: Вектор смещения.

    Returns:
        Изображение сцены с обведенным полигоном.
    """
    height, width = box_image.shape
    # Углы объекта (y, x)
    object_corners = np.array([
        [0, 0],
        [0, width - 1],
        [height - 1, width - 1],
        [height - 1, 0]
    ])
    # Применяем аффинное преобразование
    transformed_corners = apply_affine_transformation(object_corners, transform_matrix, translation)

    # Создаем копию сцены
    if len(scene_image.shape) == 2:
        result_scene = np.stack([scene_image] * 3, axis=-1)
    else:
        result_scene = scene_image.copy()

    # Соединяем уголки в полигон
    polygon_points = [
        (int(transformed_corners[i, 0]), int(transformed_corners[i, 1]))
        for i in range(len(transformed_corners))
    ]

    for i in range(len(polygon_points)):
        y0, x0 = polygon_points[i]
        y1, x1 = polygon_points[(i + 1) % len(polygon_points)]
        line_pts = draw_line_bresenham(y0, x0, y1, x1, max(result_scene.shape))
        for (ly, lx) in line_pts:
            if 0 <= ly < result_scene.shape[0] and 0 <= lx < result_scene.shape[1]:
                result_scene[ly, lx] = [255, 0, 0]

    return result_scene

# Загрузка изображений
box_image = load_image("box.png")
box_in_scene_image = load_image("box_in_scene.png")

# Отображение исходных изображений
display_image(box_image, title="Исходное изображение объекта", save_path=os.path.join(RESULTS_FOLDER, "original_box.png"))
display_image(box_in_scene_image, title="Исходное изображение сцены", save_path=os.path.join(RESULTS_FOLDER, "original_box_in_scene.png"))

# Уменьшение размера изображения объекта
box_image = resize_image(box_image, 2)
display_image(box_image, title="Уменьшенное изображение объекта", save_path=os.path.join(RESULTS_FOLDER, "resized_box.png"))

# Применение алгоритма FAST для обнаружения ключевых точек
fast_keypoints_query = fast_feature_detector(
    box_image, min_consecutive_pixels=12, intensity_threshold=30
)
fast_keypoints_test = fast_feature_detector(
    box_in_scene_image, min_consecutive_pixels=12, intensity_threshold=30
)

print(
    f"Количество ключевых точек FAST (box): {len(fast_keypoints_query)}, {fast_keypoints_query[:5]}"
)
print(
    f"Количество ключевых точек FAST (box_in_scene): {len(fast_keypoints_test)}, {fast_keypoints_test[:5]}"
)

# Вычисление градиента изображений
gradient_query = compute_image_gradient(box_image)
gradient_test = compute_image_gradient(box_in_scene_image)

# Вычисление матрицы вторых моментов для каждой ключевой точки
M_matrices_query = compute_structure_tensor_for_points(
    box_image, gradient_query, fast_keypoints_query
)
M_matrices_test = compute_structure_tensor_for_points(
    box_in_scene_image, gradient_test, fast_keypoints_test
)

# Вычисление значений R (отклика детектора Харриса)
R_values_query = compute_harris_response_values(M_matrices_query, K)
R_values_test = compute_harris_response_values(M_matrices_test, K)

# Фильтрация точек по значению R
filtered_keypoints_query = filter_points_by_harris_response(
    fast_keypoints_query, R_values_query, max_points=500
)
filtered_keypoints_test = filter_points_by_harris_response(
    fast_keypoints_test, R_values_test, max_points=500
)

print(
    f"Количество отфильтрованных точек (box): {len(filtered_keypoints_query)}, {filtered_keypoints_query[:5]}"
)
print(
    f"Количество отфильтрованных точек (box_in_scene): {len(filtered_keypoints_test)}, {filtered_keypoints_test[:5]}"
)

# Отрисовка ключевых точек на изображениях
box_with_features = draw_features_on_image(box_image, filtered_keypoints_query)
box_in_scene_with_features = draw_features_on_image(
    box_in_scene_image, filtered_keypoints_test
)

display_image(box_with_features, title="Ключевые точки на изображении объекта", save_path=os.path.join(RESULTS_FOLDER, "box_with_features.png"))
display_image(box_in_scene_with_features, title="Ключевые точки на изображении сцены", save_path=os.path.join(RESULTS_FOLDER, "box_in_scene_with_features.png"))

# Вычисление ориентации ключевых точек
orientations_query = calculate_orientations(
    box_image, filtered_keypoints_query, PATCH_SIZE
)
orientations_test = calculate_orientations(
    box_in_scene_image, filtered_keypoints_test, PATCH_SIZE
)

# Размытие изображений фильтром Гаусса
gaussian_box = apply_gaussian_blur(
    box_image, create_gaussian_kernel(sigma=10, size=5)
)
gaussian_scene = apply_gaussian_blur(
    box_in_scene_image, create_gaussian_kernel(sigma=10, size=5)
)

# Отображение размытых изображений
display_image(gaussian_box, title="Размытое изображение объекта", save_path=os.path.join(RESULTS_FOLDER, "gaussian_box.png"))
display_image(gaussian_scene, title="Размытое изображение сцены", save_path=os.path.join(RESULTS_FOLDER, "gaussian_scene.png"))

# Генерация повернутых пар точек для построения дескриптора
rotation_matrices = generate_rotation_matrices(
    [2 * k * np.pi / 30 for k in range(30)]
)
rotated_sampling_points = generate_rotated_sampling_points(
    rotation_matrices, n=256, patch_size=PATCH_SIZE
)

# Создание ORB дескрипторов
descriptors_query = create_orb_descriptors(
    gaussian_box,
    filtered_keypoints_query,
    orientations_query,
    PATCH_SIZE,
    rotated_sampling_points,
)
descriptors_test = create_orb_descriptors(
    gaussian_scene,
    filtered_keypoints_test,
    orientations_test,
    PATCH_SIZE,
    rotated_sampling_points,
)

print(f"Длина дескриптора (box): {len(descriptors_query)}")
print(f"Длина дескриптора (box_in_scene): {len(descriptors_test)}")

# Вычисление расстояний Хэмминга между дескрипторами
hamming_distances = calculate_hamming_distances(descriptors_query, descriptors_test)

# Выполнение теста Лоу
lowe_matches = perform_lowe_test(
    filtered_keypoints_query, filtered_keypoints_test, hamming_distances
)
print(
    f"Количество сопоставлений после теста Лоу: прямой поиск - {len(lowe_matches[0])}, обратный поиск - {len(lowe_matches[1])}"
)

# Выполнение перекрестной проверки
cross_checked_matches = perform_cross_check(lowe_matches)
print(
    f"Количество сопоставлений после перекрестной проверки: {len(cross_checked_matches[0])}"
)

# Визуализация сопоставлений
matched_image = visualize_matches(
    box_image,
    box_in_scene_image,
    cross_checked_matches[0],
    cross_checked_matches[1],
)
display_image(matched_image, title="Сопоставления ключевых точек", save_path=os.path.join(RESULTS_FOLDER, "matched_image.png"))

# Вычисление аффинного преобразования с помощью RANSAC
M, T = ransac_affine_transform(
    cross_checked_matches[0],
    cross_checked_matches[1],
    RANSAC_ITERATIONS,
    RANSAC_DISTANCE_THRESHOLD,
)

print("Матрица аффинного преобразования (M):\n", M)
print("Вектор смещения (T):\n", T)

# Обнаружение объекта на изображении сцены
detected_object_image = detect_object_with_transform(
    box_image, box_in_scene_image, M, T
)
display_image(detected_object_image, title="Обнаруженный объект", save_path=os.path.join(RESULTS_FOLDER, "detected_object.png"))

print("Результаты сохранены в папку:", RESULTS_FOLDER)