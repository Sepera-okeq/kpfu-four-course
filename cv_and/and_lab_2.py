import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def load_image(file_path):
    """Загрузка изображения с проверкой наличия файла."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден!")
        return None
    return Image.open(file_path)


# Загрузка входных изображений
image_1 = load_image('image1.jpg')  # Укажите ваш путь к файлу
image_2 = load_image('image2.jpg')  # Укажите ваш путь к файлу
if image_1 is None or image_2 is None:
    exit(0)

# Преобразование к numpy-массивам
image_1_np = np.array(image_1)
image_2_np = np.array(image_2)


def otsu_binarization(image):
    """Бинаризация метода Оцу."""
    if len(image.shape) == 3:  # Если изображение цветное
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image  # Если изображение уже в градациях серого
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Уменьшение шумов
    _, binary = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def color_segments(binary_image):
    """Раскрашивает сегменты в случайные цвета."""
    # Выделение меток (связные компоненты)
    num_labels, labels = cv2.connectedComponents(binary_image)
    
    # Создаем цветное изображение
    h, w = binary_image.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Назначаем случайные цвета каждому сегменту
    np.random.seed(42)  # Для воспроизводимости
    colors = np.random.randint(0, 255, size=(num_labels, 3))

    # Закрасить каждый сегмент
    for label in range(1, num_labels):  # Пропускаем фон (метка 0)
        colored_image[labels == label] = colors[label]

    return colored_image


def histogram_and_seed_growth(image):
    """Сегментация через гистограмму и выращивание семян."""
    if len(image.shape) == 3:  # HSV нужно только для цветных изображений
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
    else:
        print("Ожидалось цветное изображение для гистограммного подхода.")
        return None

    # Находим диапазон по гистограмме насыщенности (saturation)
    hist = cv2.calcHist([s], [0], None, [256], [0, 256])
    dominant_s = np.argmax(hist)

    lower_bound = np.array([0, max(0, dominant_s - 40), 50])  # Границы "доминирующего" цвета
    upper_bound = np.array([180, min(255, dominant_s + 40), 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Далее разметка семян
    num_labels, labels = cv2.connectedComponents(mask)

    # Создаем цветное изображение для сегментов
    h, w = mask.shape
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_labels, 3))
    for label in range(1, num_labels):  # Пропускаем фон
        colored_image[labels == label] = colors[label]

    return colored_image

# Запускаем сегментацию для каждого изображения

# Для изображения 1
binary_1 = otsu_binarization(image_1_np)
colored_segments_1a = color_segments(binary_1)
colored_segments_1b = histogram_and_seed_growth(image_1_np)

# Для изображения 2
binary_2 = otsu_binarization(image_2_np)
colored_segments_2a = color_segments(binary_2)
colored_segments_2b = histogram_and_seed_growth(image_2_np)

# Визуализация результатов
plt.figure(figsize=(12, 12))

# Результаты для первого изображения
plt.subplot(2, 3, 1)
plt.imshow(image_1)
plt.title("Изображение 1 (оригинальное)")

plt.subplot(2, 3, 2)
plt.imshow(colored_segments_1a)
plt.title("Метод 1: Оцу + Раскраска")

plt.subplot(2, 3, 3)
plt.imshow(colored_segments_1b)
plt.title("Метод 2: Гистограмма + Раскраска")

# Результаты для второго изображения
plt.subplot(2, 3, 4)
plt.imshow(image_2)
plt.title("Изображение 2 (оригинальное)")

plt.subplot(2, 3, 5)
plt.imshow(colored_segments_2a)
plt.title("Метод 1: Оцу + Раскраска")

plt.subplot(2, 3, 6)
plt.imshow(colored_segments_2b)
plt.title("Метод 2: Гистограмма + Раскраска")

plt.tight_layout()
plt.show()