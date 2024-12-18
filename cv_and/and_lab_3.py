import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open('image.png').convert('L')
image_np = np.array(image, dtype=np.float32)

plt.imshow(image_np, cmap='gray')
plt.title('Исходное изображение в градациях серого')
plt.axis('off')
plt.show()

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
sigma = 4

# Применяем Гауссово размытие к изображению
blurred_image = gaussian_blur(image_np, kernel_size, sigma)

# Отображаем размазанное изображение
plt.imshow(blurred_image, cmap='gray')
plt.title('После Гауссова размытия')
plt.axis('off')
plt.show()

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
plt.show()

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
            q = 255  # Значения соседних пикселей в направлении градиента
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
plt.show()

# Шаг 6: Пороговая обработка и гистерезис
# Разделить границы на сильные, слабые и несущественные.
def threshold(nms_image, lowThreshold, highThreshold):
    """Применение двойной пороговой обработки к изображению."""
    M, N = nms_image.shape  # Размеры изображения
    res = np.zeros((M,N), dtype=np.int32)  # Инициализируем результирующее изображение
    
    # Определяем значения для сильных и слабых пикселей
    strong = 255  # Сильные пиксели (границы)
    weak = 75     # Слабые пиксели (потенциальные границы)
    
    # Находим позиции сильных и слабых пикселей на основе пороговых значений
    strong_i, strong_j = np.where(nms_image >= highThreshold)
    zeros_i, zeros_j = np.where(nms_image < lowThreshold)
    weak_i, weak_j = np.where((nms_image >= lowThreshold) & (nms_image < highThreshold))
    
    # Присваиваем соответствующие значения пикселям в результирующем изображении
    res[strong_i, strong_j] = strong  # Сильные пиксели
    res[weak_i, weak_j] = weak        # Слабые пиксели
    
    return res

# Оставить только те слабые границы, которые связаны с сильными.
# TODO: РЕКРУСИВНООООООО, ИБО ВСЕ СЛАБОЕЕЕЕ
def hysteresis(img):
    """Применение гистерезиса для соединения границ."""
    M, N = img.shape  # Размеры изображения
    strong = 255      # Значение сильного пикселя
    weak = 75         # Значение слабого пикселя
    
    # Проходимся по каждому пикселю изображения, исключая границы
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i,j] == weak:
                # Если хотя бы один из 8 соседних пикселей является сильным, то пиксель считается границей
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i,j] = strong  # Укрепляем слабый пиксель до сильного
                else:
                    img[i,j] = 0  # Иначе подавляем пиксель (не считаем его границей)
    return img

# Устанавливаем пороговые значения
lowThreshold = 10   # Нижний порог. Увеличение этого значения уменьшит количество слабых границ.
highThreshold = 40  # Верхний порог. Увеличение этого значения уменьшит количество сильных границ, возможно пропуская некоторые важные границы.

# Применяем пороговую обработку
thresholded_image = threshold(nms_image, lowThreshold, highThreshold)

# Применяем процедуру гистерезиса
final_image = hysteresis(thresholded_image)

# Отображаем итоговое изображение с обнаруженными границами
plt.imshow(final_image, cmap='gray')
plt.title('Обнаруженные границы (Алгоритм Кэнни)')
plt.axis('off')
plt.show()