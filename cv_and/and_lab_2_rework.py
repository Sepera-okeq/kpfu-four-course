import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os 

# Функция для загрузки изображения
def load_image(file_path):
    if not os.path.exists(file_path):  # Проверяем, существует ли файл
        print(f"Ошибка: Файл '{file_path}' не найден!")  # Сообщение об ошибке
        return None  # Возвращаем None, если файл отсутствует
    return Image.open(file_path)  # Открываем изображение через PIL


# Загрузка входных изображений
image_1 = load_image('image1.jpg')  # Загружаем первое изображение
image_2 = load_image('image2.jpg')  # Загружаем второе изображение
if image_1 is None or image_2 is None:  # Если хотя бы одно изображение не удалось загрузить
    exit(0)  # Завершаем выполнение программы с выходным кодом 0

# Преобразуем изображения в массивы numpy для обработки
image_1_np = np.array(image_1)  # Конвертация первого изображения в numpy массив
image_2_np = np.array(image_2)  # Конвертация второго изображения в numpy массив

# Реализация метода Оцу
def otsu_binarization(image):
    """Бинаризация методом Оцу."""
    # Убедимся, что изображение преобразовано в numpy массив
    if isinstance(image, Image.Image):  # Если это объект типа PIL.Image
        image = np.array(image)  # Конвертируем в numpy-массив

    if len(image.shape) == 3:  # Если изображение цветное (3 канала)
        # Конвертируем в оттенки серого через усреднение по каналам
        gray_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray_image = image  # Если изображение уже в градациях серого

    # Рассчитываем гистограмму значений от 0 до 255
    hist, bins = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))

    # Считаем общее количество пикселей
    total_pixels = gray_image.size

    # Инициализируем переменные для метода Оцу
    current_max_variance = 0  # Текущая максимальная межклассовая дисперсия
    threshold = 0  # Пороговое значение Оцу

    weight_background = 0  # Вес фона (заполнен 0 начального веса)
    sum_background = 0  # Сумма интенсивностей фона
    sum_total = np.dot(hist, np.arange(256))  # Полная сумма всех значений интенсивностей

    # Перебираем все возможные значения порогов (0-255)
    for t in range(256):
        weight_background += hist[t]  # Увеличиваем вес фона
        if weight_background == 0:  # Если пикселей нет, пропускаем
            continue

        weight_foreground = total_pixels - weight_background  # Вес переднего плана
        if weight_foreground == 0:  # Если пикселей переднего фона нет, пропускаем
            break

        sum_background += t * hist[t]  # Сумма интенсивностей фона
        mean_background = sum_background / weight_background  # Средняя интенсивность фона
        mean_foreground = (sum_total - sum_background) / weight_foreground  # Средняя интенсивность переднего плана

        # Вычисляем межклассовую дисперсию
        inter_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )

        # Если нашли большую дисперсию, обновляем параметры
        if inter_class_variance > current_max_variance:
            current_max_variance = inter_class_variance
            threshold = t

    # Применяем порог бинаризации
    binary_image = gray_image > threshold  # Бинаризация, пиксели больше порога = True
    binary_image = binary_image.astype(np.uint8) * 255  # Конвертация в 0 и 255 для маски
    return binary_image  # Возвращаем двоичное изображение


# Реализация функции окраски сегментов
def color_segments(binary_image):
    """Раскрашивает сегменты в случайные цвета."""
    from scipy.ndimage import label  # Для получения меток (аналог connectedComponents)
    h, w = binary_image.shape  # Получаем высоту и ширину изображения

    # Генерация меток, каждая связь получает уникальное число
    structure = np.ones((3, 3), dtype=int)  # Структурный элемент для связей (соседство 8 пикселей)
    labeled_image, num_labels = label(binary_image, structure=structure)

    # Создаем цветное изображение
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)  # Массив для цветного изображения
    np.random.seed(42)  # Устанавливаем фиксированный seed для воспроизводимости
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3))  # Генерируем случайные цвета

    # Применяем цвета к каждому сегменту
    for label_idx in range(1, num_labels + 1):  # Пропускаем фон (метка 0)
        colored_image[labeled_image == label_idx] = colors[label_idx]  # Закрашиваем сегмент

    return colored_image  # Возвращаем результат


# Функция для создания сегментации на основе гистограммы
def histogram_and_seed_growth(image):
    """Сегментация через гистограмму и выращивание семян."""
    # Убедимся, что изображение преобразовано в numpy массив
    if isinstance(image, Image.Image):  # Если это объект типа PIL.Image
        image = np.array(image)  # Преобразуем в numpy-массив

    if len(image.shape) == 3:  # Проверяем, цветное ли изображение (3-канальное)
        # Конвертируем в HSV (Hue, Saturation, Value)
        hsv_image = image  # В данном случае оставляем numpy-массив, если это цветное изображение
        hsv_np = hsv_image  # Переименование для удобства
        h, s, v = hsv_np[:, :, 0], hsv_np[:, :, 1], hsv_np[:, :, 2]  # Разделяем каналы H, S, V
    else:
        print("Ожидалось цветное изображение для гистограммного подхода.")
        return None  # Если изображение не цветное, возвращаем None

    # Строим гистограмму для значения насыщенности (saturation)
    hist, bins = np.histogram(s.ravel(), bins=256, range=(0, 256))
    dominant_s = np.argmax(hist)  # Находим доминирующее значение насыщенности

    # Определяем границы диапазона для выделения объектов
    lower_bound = np.maximum(0, dominant_s - 15)  # Нижняя граница
    upper_bound = np.minimum(255, dominant_s + 15)  # Верхняя граница
    mask = (s >= lower_bound) & (s <= upper_bound)  # Создаем маску по диапазону насыщенности

    h, w = mask.shape  # Определяем размеры маски изображения
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)  # Подготавливаем цветное изображение

    # Генерация цвета сегментов (аналог `connectedComponents`)
    from scipy.ndimage import label  # Для меток связных областей
    labeled_mask, num_labels = label(mask)  # Метки для сегментированных объектов

    np.random.seed(42)  # Seed для воспроизводимости
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3))  # Генерация цветов объектов
    for label_idx in range(1, num_labels + 1):  # Раскрашивание сегментов
        colored_image[labeled_mask == label_idx] = colors[label_idx]

    return colored_image  # Возвращаем цветное изображение


# Выполняем сегментацию для каждого изображения

# Для изображения 1
binary_1 = otsu_binarization(image_1)  # Бинаризация методом Оцу
colored_segments_1a = color_segments(binary_1)  # Раскраска сегментов
colored_segments_1b = histogram_and_seed_growth(image_1)  # Гистограммы + раскраска

# Для изображения 2
binary_2 = otsu_binarization(image_2)  # Бинаризация методом Оцу
colored_segments_2a = color_segments(binary_2)  # Раскраска сегментов
colored_segments_2b = histogram_and_seed_growth(image_2)  # Гистограммы + раскраска

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