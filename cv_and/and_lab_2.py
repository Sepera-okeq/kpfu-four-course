import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import deque

def load_image(file_path):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден!")
        return None
    return Image.open(file_path).convert('RGB')  # Конвертируем в RGB

# Функция для медианного фильтра
def median_filter(image_array, kernel_size=3):
    """Быстрый медианный фильтр с использованием numpy, поддерживает цветные и серые изображения."""
    pad_size = kernel_size // 2
    if image_array.ndim == 2:  # Градации серого
        padded_image = np.pad(image_array, pad_size, mode='edge')
        # Создаем представление с использованием скользящего окна
        shape = (image_array.shape[0], image_array.shape[1], kernel_size, kernel_size)
        strides = (padded_image.strides[0], padded_image.strides[1], padded_image.strides[0], padded_image.strides[1])
        patches = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
        patches = patches.reshape(-1, kernel_size*kernel_size)
        medians = np.median(patches, axis=1)
        return medians.reshape(image_array.shape)
    elif image_array.ndim == 3:  # Цветное изображение
        filtered_image = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            filtered_image[:, :, c] = median_filter(image_array[:, :, c], kernel_size)
        return filtered_image
    else:
        raise ValueError("Unsupported image dimensionality")

def remove_salt_and_pepper(image):
    """Удаляет шум типа соль и перец, анализируя окрестность 3x3"""
    height, width = image.shape
    result = np.copy(image)
    
    # Добавляем отступы для обработки краев
    padded = np.pad(image, pad_width=1, mode='edge')
    
    # Шаблоны для поиска
    salt_pattern = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
    
    pepper_pattern = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
    
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            # Получаем окрестность 3x3
            neighborhood = padded[i-1:i+2, j-1:j+2]
            
            # Проверяем на соответствие шаблону "соли"
            if np.array_equal(neighborhood, salt_pattern):
                result[i-1, j-1] = 0
            
            # Проверяем на соответствие шаблону "перца"
            elif np.array_equal(neighborhood, pepper_pattern):
                result[i-1, j-1] = 1
                    
    return result

# Реализация алгоритма Otsu
def otsu_binarization(image):
    """Бинаризация методом Оцу."""
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))
    total_pixels = image.size
    current_max_variance = 0
    threshold = 0
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0
    weight_background = 0
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        inter_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if inter_class_variance > current_max_variance:
            current_max_variance = inter_class_variance
            threshold = t
    binary_image = image > threshold
    return binary_image.astype(np.uint8)  # Returns 0 and 1

# Функция для получения соседей (4-связность)
def get_neighbors(y, x, height, width):
    neighbors = []
    if y > 0:
        neighbors.append((y - 1, x))
    if y < height - 1:
        neighbors.append((y + 1, x))
    if x > 0:
        neighbors.append((y, x - 1))
    if x < width - 1:
        neighbors.append((y, x + 1))
    return neighbors

# Алгоритм выращивания семян
def seed_growing(image, mask, similarity_threshold, feature_extractor):
    """Алгоритм выращивания семян."""
    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=int)
    visited = np.zeros((height, width), dtype=bool)
    label = 1

    # Создаем список семян
    seeds = np.argwhere(mask)
    seeds_list = deque(map(tuple, seeds))

    while seeds_list:
        y, x = seeds_list.popleft()
        if visited[y, x]:
            continue
        # Начинаем новый регион
        region_pixels = [(y, x)]
        region_features = [feature_extractor(image, y, x)]
        visited[y, x] = True
        queue = deque()
        queue.append((y, x))
        while queue:
            cy, cx = queue.popleft()
            current_feature = feature_extractor(image, cy, cx)
            neighbors = get_neighbors(cy, cx, height, width)
            for ny, nx in neighbors:
                if mask[ny, nx] and not visited[ny, nx]:
                    neighbor_feature = feature_extractor(image, ny, nx)
                    similarity = np.linalg.norm(neighbor_feature - current_feature)
                    if similarity <= similarity_threshold:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                        region_pixels.append((ny, nx))
        # Присваиваем метку всем пикселям региона
        for ry, rx in region_pixels:
            labels[ry, rx] = label
        label += 1
    return labels

# Функция извлечения признаков для цветных изображений
def color_feature_extractor(image, y, x):
    return image[y, x].astype(float)

# Функция извлечения признаков для полутоновых изображений
def grayscale_feature_extractor(image, y, x):
    return np.array([image[y, x]], dtype=float)

# Функция раскраски сегментов
def color_segments(labeled_image):
    """Раскрашиваем сегменты в случайные цвета."""
    h, w = labeled_image.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(labeled_image)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(unique_labels.max()+1, 3))
    for label in unique_labels:
        if label != 0:
            colored_image[labeled_image == label] = colors[label]
    return colored_image

def main():
    # Создаем директорию для результатов, если она не существует
    results_dir = 'and_lab_2_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Загружаем изображения
    image_paths = ['image1.jpg', 'image2.jpg']
    images = []
    for path in image_paths:
        img = load_image(path)
        if img:
            images.append(img)
        else:
            print(f"Не удалось загрузить {path}")
    if len(images) == 0:
        exit(0)

    # Обработка изображений
    for idx, image in enumerate(images):
        print(f"\nОбработка изображения {idx+1}")

        # Алгоритм 1
        print("Алгоритм 1:")
        # Шаг 1: Удаление шума медианным фильтром
        image_array = np.array(image)
        denoised_array = image_array
        #plt.figure(figsize=(6,6))
        #plt.title('Удаление шума')
        #plt.imshow(denoised_array.astype(np.uint8))
        #plt.axis('off')
        #plt.savefig(f'{results_dir}/img{idx+1}_alg1_step1_denoised.png')
        #plt.close()

        # Шаг 2: Конвертация в оттенки серого
        gray_image = Image.fromarray(denoised_array.astype(np.uint8)).convert('L')
        gray_array = np.array(gray_image)
        plt.figure(figsize=(6,6))
        plt.title('Оттенки серого')
        plt.imshow(gray_array, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg1_step2_grayscale.png')
        plt.close()

        # Шаг 3: Бинаризация методом Оцу
        binary_image = otsu_binarization(gray_array)
        plt.figure(figsize=(6,6))
        plt.title('Бинаризация (Оцу)')
        plt.imshow(binary_image, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg1_step3_binary.png')
        plt.close()

        # Удаляем шум соль и перец
        binary_image = remove_salt_and_pepper(binary_image)

        # Шаг 4: Выделение сегментов (выращивание семян)
        mask = binary_image > 0
        # Вычисляем адаптивный порог на основе стандартного отклонения
        std_dev = np.std(denoised_array)
        similarity_threshold = std_dev * 0.5  # Используем 50% от стандартного отклонения
        labels = seed_growing(denoised_array.astype(np.uint8), mask, similarity_threshold, color_feature_extractor)
        plt.figure(figsize=(6,6))
        plt.title('Выращивание семян')
        plt.imshow(labels)
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg1_step4_seeds.png')
        plt.close()

        # Шаг 5: Раскраска сегментов
        colored_segments_image = color_segments(labels)
        plt.figure(figsize=(6,6))
        plt.title('Раскраска сегментов')
        plt.imshow(colored_segments_image)
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg1_step5_colored.png')
        plt.close()

        # Алгоритм 2
        print("Алгоритм 2:")
        # Шаг 1: Удаление шума медианным фильтром
        gray_image2 = image.convert('L')
        gray_array2 = np.array(gray_image2)
        denoised_array2 = median_filter(gray_array2)
        #plt.figure(figsize=(6,6))
        #plt.title('Удаление шума')
        #plt.imshow(denoised_array2.astype(np.uint8), cmap='gray')
        #plt.axis('off')
        #plt.savefig(f'{results_dir}/img{idx+1}_alg2_step1_denoised.png')
        #plt.close()

        # Шаг 2: Гистограммный метод
        # Нормализация гистограммы
        min_val = np.min(denoised_array2)
        max_val = np.max(denoised_array2)
        normalized_array = ((denoised_array2 - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        
        hist, bins = np.histogram(normalized_array.ravel(), bins=256, range=(0, 256))
        # Сглаживание гистограммы
        hist = np.convolve(hist, np.ones(5)/5, mode='same')
        
        plt.figure(figsize=(6,4))
        plt.title('Гистограмма изображения (нормализованная)')
        plt.plot(hist)
        plt.xlabel('Уровень серого')
        plt.ylabel('Частота')
        plt.savefig(f'{results_dir}/img{idx+1}_alg2_step2_histogram.png')
        plt.close()

        # Удаляем шум соль и перец
        denoised_array2 = remove_salt_and_pepper(normalized_array)

        # Находим пороговые значения для отделения пиков
        # Здесь мы используем метод Оцу для нахождения оптимального порога
        threshold_otsu = otsu_binarization(denoised_array2)
        threshold_value = np.mean(denoised_array2[threshold_otsu > 0])
        thresholds = [0, threshold_value, 255]

        # Создаем маски для каждого диапазона интенсивностей
        masks = []
        for i in range(len(thresholds)-1):
            lower = thresholds[i]
            upper = thresholds[i+1]
            mask = (denoised_array2 >= lower) & (denoised_array2 < upper)
            masks.append(mask)

        # Шаг 3: Выращивание семян для каждого порога
        labels_total = np.zeros_like(denoised_array2, dtype=int)
        label_offset = 1
        for mask in masks:
            if np.any(mask):
                labels = seed_growing(denoised_array2, mask, similarity_threshold=15, feature_extractor=grayscale_feature_extractor)
                labels[labels > 0] += label_offset
                labels_total += labels
                label_offset = labels_total.max() + 1

        plt.figure(figsize=(6,6))
        plt.title('Выращивание семян')
        plt.imshow(labels_total)
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg2_step3_seeds.png')
        plt.close()

        # Шаг 4: Раскраска сегментов
        colored_segments_image2 = color_segments(labels_total)
        plt.figure(figsize=(6,6))
        plt.title('Раскраска сегментов')
        plt.imshow(colored_segments_image2)
        plt.axis('off')
        plt.savefig(f'{results_dir}/img{idx+1}_alg2_step4_colored.png')
        plt.close()

if __name__ == "__main__":
    main()
