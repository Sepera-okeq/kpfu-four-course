import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os

results_dir = 'and_lab_4_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


image = Image.open('image.png').convert('L')
image_np = np.array(image, dtype=np.float32)

plt.imshow(image_np, cmap='gray')
plt.title('Исходное изображение в градациях серого')
plt.axis('off')
plt.savefig(f'{results_dir}/result0.png')

# Шаг 1: Гауссово размытие

def gaussian_kernel(size, sigma=1):
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
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            blurred_image[i, j] = np.sum(region * kernel)
    return np.clip(blurred_image, 0, 255)

def convolve(image, kernel):
    """Общая функция свертки, используемая для различных ядер."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'reflect')
    
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i,j] = np.sum(region * kernel)
    return output

kernel_size = 5
sigma = 2

# Применяем Гауссово размытие к изображению
blurred_image = gaussian_blur(image_np, kernel_size, sigma)

# Отображаем размазанное изображение
plt.imshow(blurred_image, cmap='gray')
plt.title('После Гауссова размытия')
plt.axis('off')
plt.savefig(f'{results_dir}/result1.png')

# Шаг 2: Градиенты Собеля

def sobel_filters(image):
    """Применение фильтров Собеля для вычисления градиентов."""
    # Определяем ядра Собеля для осей X и Y
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)  # Фильтр для горизонтальных изменений (градиент по X)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)  # Фильтр для вертикальных изменений (градиент по Y)
    
    Ix = convolve(image, Kx)  # Применяем фильтр Kx для получения градиента по X
    Iy = convolve(image, Ky)  # Применяем фильтр Ky для получения градиента по Y
    
    G = np.hypot(Ix, Iy)      # Вычисляем магнитуду градиента (G = sqrt(Ix^2 + Iy^2))
    G = G / G.max() * 255     # Нормализуем магнитуду к диапазону 0-255
    
    theta = np.arctan2(Iy, Ix)  # Вычисляем направление градиента (угол в радианах)
    
    return (G, theta)  # Возвращаем магнитуду и направление градиента

# Применяем фильтры Собеля к размытому изображению
G, theta = sobel_filters(blurred_image)

# Отображаем магнитуду градиента
plt.imshow(G, cmap='gray')
plt.title('Магнитуда градиента после фильтра Собеля')
plt.axis('off')
plt.savefig(f'{results_dir}/result2.png')

# Шаг 3: Направление градиента (уже вычислено в theta)

# Шаг 4: Округление направления градиента

def angle_rounding(theta):
    """Округление направления градиента до 0°, 45°, 90°, 135°."""
    theta = np.rad2deg(theta)  # Конвертируем углы из радиан в градусы
    theta[theta < 0] += 180    # Приводим углы к диапазону от 0 до 180 градусов
    rounded_theta = np.zeros_like(theta)  # Инициализируем массив для округленных углов
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            angle = theta[i,j]
            # Округляем угол к ближайшему из 4 направлений
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                rounded_theta[i,j] = 0       # Горизонтальное направление
            elif (22.5 <= angle < 67.5):
                rounded_theta[i,j] = 45      # Диагональное направление (/)
            elif (67.5 <= angle < 112.5):
                rounded_theta[i,j] = 90      # Вертикальное направление
            elif (112.5 <= angle < 157.5):
                rounded_theta[i,j] = 135     # Диагональное направление (\)
    return rounded_theta

# Применяем функцию округления к направлениям градиента
rounded_theta = angle_rounding(theta)

# Шаг 5: Подавление немаксимумов
# Оставить только локальные максимумы вдоль направления градиента, подавив остальные пиксели.
def non_max_suppression(G, theta):
    """Подавление немаксимумов в направлении градиента."""
    M, N = G.shape  # Размеры изображения
    Z = np.zeros((M,N), dtype=np.float32)  # Инициализируем выходное изображение
    angle = theta  # Используем округленное направление
    
    # Проходимся по каждому пикселю изображения, исключая краевые пиксели
    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 128 # Значения соседних пикселей в направлении градиента
            r = 255
            angle_ = angle[i,j]
            
            # В зависимости от направления, выбираем соседние пиксели для сравнения
            if angle_ == 0:
                q = G[i, j+1]    # Пиксель справа
                r = G[i, j-1]    # Пиксель слева
            elif angle_ == 45:
                q = G[i+1, j-1]  # Пиксель снизу слева
                r = G[i-1, j+1]  # Пиксель сверху справа
            elif angle_ == 90:
                q = G[i+1, j]    # Пиксель снизу
                r = G[i-1, j]    # Пиксель сверху
            elif angle_ == 135:
                q = G[i-1, j-1]  # Пиксель сверху слева
                r = G[i+1, j+1]  # Пиксель снизу справа
            
            # Если текущий пиксель имеет максимальную магнитуду в направлении градиента, сохраняем его
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0  # Иначе подавляем (зануляем) его
    return Z

# Применяем подавление немаксимумов
nms_image = non_max_suppression(G, rounded_theta)

# Отображаем изображение после подавления немаксимумов
plt.imshow(nms_image, cmap='gray')
plt.title('После подавления немаксимумов')
plt.axis('off')
plt.savefig(f'{results_dir}/result5.png')

# Шаг 6: Пороговая обработка и гистерезис
# Разделить границы на сильные, слабые и несущественные.
def threshold(nms_image, lowThreshold, highThreshold):
    """Применение двойной пороговой обработки к изображению."""
    M, N = nms_image.shape  # Размеры изображения
    res = np.zeros((M,N), dtype=np.int32)  # Инициализируем результирующее изображение
    
    # Определяем значения для сильных и слабых пикселей
    strong = 255  # Сильные пиксели (границы)
    weak = 100     # Слабые пиксели (потенциальные границы)
    
    # Находим позиции сильных и слабых пикселей на основе пороговых значений
    strong_i, strong_j = np.where(nms_image >= highThreshold / 10)
    zeros_i, zeros_j = np.where(nms_image < lowThreshold / 10)
    weak_i, weak_j = np.where((nms_image >= lowThreshold / 10) & (nms_image < highThreshold / 10))
    
    # Присваиваем соответствующие значения пикселям в результирующем изображении
    res[strong_i, strong_j] = strong  # Сильные пиксели
    res[weak_i, weak_j] = weak        # Слабые пиксели
    
    return res

def trace_edge(img, i, j, visited):
    """Рекурсивная функция для трассировки границ."""
    M, N = img.shape
    strong = 255
    weak = 100
    
    # Проверяем границы изображения и посещенные пиксели
    if (i < 0 or i >= M or j < 0 or j >= N or visited[i, j]):
        return
    
    # Помечаем текущий пиксель как посещенный
    visited[i, j] = True
    
    # Если текущий пиксель слабый
    if img[i, j] == weak:
        # Проверяем 8 соседних пикселей
        neighbors = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),             (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        
        # Если хотя бы один сосед сильный, усиливаем текущий пиксель
        for ni, nj in neighbors:
            if (0 <= ni < M and 0 <= nj < N and img[ni, nj] == strong):
                img[i, j] = strong
                # Рекурсивно проверяем соседей текущего пикселя
                for ni2, nj2 in neighbors:
                    if (0 <= ni2 < M and 0 <= nj2 < N and not visited[ni2, nj2]):
                        trace_edge(img, ni2, nj2, visited)
                break
        
        # Если пиксель не был усилен, подавляем его
        if img[i, j] == weak:
            img[i, j] = 0

def hysteresis(img):
    """Применение гистерезиса для соединения границ с использованием рекурсии."""
    M, N = img.shape
    strong = 255
    
    # Создаем массив для отслеживания посещенных пикселей
    visited = np.zeros((M, N), dtype=bool)
    
    # Находим все сильные пиксели и начинаем с них трассировку
    strong_points = np.argwhere(img == strong)
    for i, j in strong_points:
        if not visited[i, j]:
            trace_edge(img, i, j, visited)
    
    return img

# Устанавливаем пороговые значения
# Нижний порог. Увеличение этого значения уменьшит количество слабых границ.
lowThreshold = 120
# Верхний порог. Увеличение этого значения уменьшит количество сильных границ, возможно пропуская некоторые важные границы.
highThreshold = 200 
# Применяем пороговую обработку
thresholded_image = threshold(nms_image, lowThreshold, highThreshold)

# Применяем процедуру гистерезиса
final_image = hysteresis(thresholded_image)



# Отображаем итоговое изображение с обнаруженными границами
plt.imshow(final_image, cmap='gray')
plt.title('Обнаруженные границы (Алгоритм Кэнни)')
plt.axis('off')
plt.savefig(f'{results_dir}/result6.png')
plt.close()

# Отображаем найденные линии на изображении (Для 8 этапа)
def plot_detected_lines(image, lines):
    """Отображение исходного изображения с наложенными найденными линиями."""
    height, width = image.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    for rho, theta in lines:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0, y0 = cos_t * rho, sin_t * rho
        
        # Находим точки на краях изображения
        # Уравнение линии: x*cos(theta) + y*sin(theta) = rho
        
        # Точки на левом и правом краях
        x1, y1 = 0, rho / sin_t if sin_t != 0 else 0
        x2, y2 = width - 1, (rho - (width-1)*cos_t) / sin_t if sin_t != 0 else 0
        
        # Точки на верхнем и нижнем краях
        x3, y3 = (rho - (height-1)*sin_t) / cos_t if cos_t != 0 else 0, height - 1
        x4, y4 = rho / cos_t if cos_t != 0 else 0, 0
        
        # Фильтруем точки, находящиеся в пределах изображения
        points = []
        for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
        
        # Рисуем линию, если есть две точки
        if len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            plt.plot([x1, x2], [y1, y2], 'r')
    
    plt.axis('off')
    plt.title(f'Обнаруженные прямые')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/result8_lines.png')
    plt.close()

# Шаг 7: Реализация преобразования Хафа
def hough_transform_fixed(edge_image):
    """Реализация преобразования Хафа."""
    # Размеры изображения
    M, N = edge_image.shape
    
    # Диагональ изображения – диапазон ro
    diag_len = int(np.sqrt(M ** 2 + N ** 2))
    thetas = np.deg2rad(np.arange(-90, 90, 1))  # Последовательность углов (от -90 до 90 градусов с шагом 1°)
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)  # Значения ro (от минус диагонали до плюс диагонали)
    
    # Инициализируем кумулятивный массив
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # Находим координаты всех пикселей-границ
    edge_points = np.argwhere(edge_image > 0)
    
    # Преобразование Хафа для каждой точки границы
    for y, x in edge_points:
        # Вычисляем ro для каждого theta
        for theta_idx, theta in enumerate(thetas):
            ro = int(x * cos_t[theta_idx] + y * sin_t[theta_idx])
            ro_index = np.argmin(np.abs(rhos - ro))  # Находим соответствующий индекс в массиве ro
            accumulator[ro_index, theta_idx] += 1  # Увеличиваем значение в ячейке
    
    return accumulator, rhos, thetas

# Применение преобразования Хафа
hough_accumulator, rhos, thetas = hough_transform_fixed(final_image)

# Отображение кумулятивного массива
plt.figure(figsize=(10, 10))
plt.imshow(
    hough_accumulator, cmap='hot',
    aspect='auto',
)
plt.title('Кумулятивный массив (пространство Хафа)')
plt.xlabel('Theta (градусы)')
plt.ylabel('Rho (пиксели)')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{results_dir}/result7_hough.png')
plt.close()

def gaussian_blur_on_accumulator(accumulator, kernel_size=2, sigma=0.5):
    """Применение фильтра Гаусса для сглаживания кумулятивного массива."""
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_accumulator = convolve(accumulator, kernel)
    return smoothed_accumulator

# Применение фильтра Гаусса
smoothed_hough_accumulator = gaussian_blur_on_accumulator(hough_accumulator, kernel_size=3, sigma=0.1)

# Отображение сглаженного кумулятивного массива
plt.figure(figsize=(10, 10))
plt.imshow(
    smoothed_hough_accumulator, cmap='hot',
    aspect='auto'
)
plt.title('Сглаженный кумулятивный массив (пространство Хафа)')
plt.xlabel('Theta (градусы)')
plt.ylabel('Rho (пиксели)')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{results_dir}/result7_hough_smoothed.png')
plt.close()

# Шаг 8.1: Подавление немаксимумов и извлечение линий из кумулятивного массива
def detect_lines(accumulator, rhos, thetas, threshold=120):
    """Извлечение прямых из кумулятивного массива."""
    M, N = accumulator.shape
    detected_lines = []
    
    for i in range(1, M - 1):  # Проходим по всем ячейкам, кроме краевых
        for j in range(1, N - 1):
            if accumulator[i, j] > threshold:
                # Проверяем, является ли текущая ячейка локальным максимумом
                local_max = np.max(accumulator[i - 1:i + 2, j - 1:j + 2])
                if accumulator[i, j] == local_max:
                    rho = rhos[i]
                    theta = thetas[j]
                    detected_lines.append((rho, theta)) # Сохраняем значение
    return detected_lines

# Экстракция линий из кумулятивного массива
lines = detect_lines(smoothed_hough_accumulator, rhos, thetas)

# Шаг 8.2: Извлечение линий с ограничением по количеству и выводим крутые сначала!111
def detect_top_lines(accumulator, rhos, thetas, threshold=120, max_lines=15):
    """Извлечение топ-N линий из сглаженного кумулятивного массива."""
    M, N = accumulator.shape
    detected_lines = []
    
    for i in range(1, M - 1):  # Проходим по всем ячейкам, кроме краевых
        for j in range(1, N - 1):
            if accumulator[i, j] > threshold:
                # Проверяем, является ли текущая ячейка локальным максимумом
                local_max = np.max(accumulator[i - 1:i + 2, j - 1:j + 2])
                if accumulator[i, j] == local_max:
                    rho = rhos[i]
                    theta = thetas[j]
                    detected_lines.append((rho, theta, accumulator[i, j]))  # Сохраняем значение из аккумулятора

    # Сортируем линии по значению в аккумуляторе (самые сильные линии в начале)
    detected_lines = sorted(detected_lines, key=lambda x: x[2], reverse=True)
    
    # Оставляем только top-N линий
    top_lines = [(rho, theta) for rho, theta, _ in detected_lines[:max_lines]]
    return top_lines

top_lines = detect_top_lines(smoothed_hough_accumulator, rhos, thetas, max_lines=15)

# Шаг 8.3: Отображение линий на исходном изображении
plot_detected_lines(image_np, top_lines)

# Выводим количество обнаруженных линий в консоль
if len(top_lines) > 0:
    print(f"Выведено {len(top_lines)} прямых из {len(lines)} найденных!")
else:
    print("Линии не найдены!")