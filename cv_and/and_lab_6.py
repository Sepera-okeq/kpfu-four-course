from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import random

box=np.array(Image.open('box.png'), dtype=int)
plt.figure()
plt.imshow(box, cmap='gray', vmin=0, vmax=255)
plt.show()

def scale(image, s_resized):
    """ 
    img = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])

    array([[3, 5],
           [11, 13]])
    """
    w = image.shape[0]  
    h = image.shape[1]  

    newI = np.zeros((w//s_resized, h//s_resized), dtype=int)

    for i in range(0, w//s_resized):
        for j in range(0, h//s_resized):
            newI[i, j] = int(np.mean(image[i*s_resized:s_resized*(i+1), j*s_resized:s_resized*(j+1)]))

    return newI

box = scale(box, 2)
plt.figure()
plt.imshow(box, cmap='gray', vmin=0, vmax=255)
plt.show()

box_in_scene=np.array(Image.open('box_in_scene.png'), dtype=int)
plt.figure()
plt.imshow(box_in_scene, cmap='gray', vmin=0, vmax=255)
plt.show()

patch_size=31

def get_circle_pixels(image, center_x, center_y):
    # Верхняя часть окружности
    top_circle = np.concatenate((
        image[center_x:center_x+2, center_y-3],      
        image[center_x+2, center_y-2:center_y-1],    
        image[center_x+3, center_y-1:center_y+2],    
        image[center_x+2, center_y+2:center_y+3],    
        image[center_x:center_x+2, center_y+3]       
    ), axis=None)
    
    # Нижняя часть окружности
    bottom_circle = np.concatenate((
        image[center_x+3, center_y:center_y+2],      
        image[center_x+2, center_y+2:center_y+3],     
        image[center_x-1:center_x+2, center_y+3],    
        image[center_x-2, center_y+2:center_y+3],    
        image[center_x-3, center_y:center_y+2]       
    ), axis=None)
    
    # Полная окружность
    full_circle = np.concatenate((
        image[center_x-1:center_x+2, center_y-3],    # Верхние точки
        image[center_x+2, center_y-2:center_y-1],    # Правые верхние 
        image[center_x+3, center_y-1:center_y+2],    # Правые точки
        image[center_x+2, center_y+2:center_y+3],    # Правые нижние 
        image[center_x-1:center_x+2, center_y+3],    # Нижние точки
        image[center_x-2, center_y+2:center_y+3],    # Левые нижние 
        image[center_x-3, center_y-1:center_y+2],    # Левые точки
        image[center_x-2, center_y-2:center_y-1]     # Левые верхние
    ), axis=None)
    
    # 9 - 9 - 16 
    return top_circle, bottom_circle, full_circle

def FAST(image, min_consecutive_pixels, intensity_threshold):
    """
    min_consecutive_pixels - минимальное число последовательных пикселей (n)
    intensity_threshold - порог интенсивности (t)
    """
    if min_consecutive_pixels < 6 or min_consecutive_pixels > 16:
        return []
    
    height, width = image.shape
    feature_points = []
    margin = patch_size // 2
    
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            
            # Центральный пиксель
            center_pixel = image[y, x]
            
            top, bottom, full = get_circle_pixels(image, y, x)
            
            # Яркие точки
            if (all(q > center_pixel + intensity_threshold for q in top) or 
                all(q > center_pixel + intensity_threshold for q in bottom)):
                consecutive_count = 0
                for pixel in np.concatenate((full, full)):
                    if pixel > center_pixel + intensity_threshold:
                        consecutive_count += 1
                    else:
                        consecutive_count = 0
                    if consecutive_count >= min_consecutive_pixels:
                        feature_points.append((y, x))
                        break
                        
            # Темные точки
            elif (all(q < center_pixel - intensity_threshold for q in top) or 
                  all(q < center_pixel - intensity_threshold for q in bottom)):
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

fast_query = FAST(box, 12, 30)
fast_test = FAST(box_in_scene, 12, 30)
print(len(fast_query), fast_query)
print(len(fast_test), fast_test)

# Гаусс
def gaussian_kernel(sigma, size):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)  
    kernel = kernel / np.sum(kernel)  # Нормализуем ядро
    return kernel

def apply_gaussian_blur(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_y, padding_x = kernel_height // 2, kernel_width // 2

    # Создаем пустой массив для результата
    blurred_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            sum_value = 0.0 

            for m in range(kernel_height):
                for n in range(kernel_width):
                    # Индексы для области среза
                    x = i + m - padding_y
                    y = j + n - padding_x

                    if 0 <= x < image_height and 0 <= y < image_width:
                        sum_value += image[x, y] * kernel[m, n]

            blurred_image[i, j] = sum_value

    return np.clip(blurred_image, 0, 255).astype(np.uint8)

def apply_filter(image, kernel):
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

def get_gradient(image):
    # Фильтры Собеля
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])

    sobel_y = np.array([[ 1,  2,  1], 
                        [ 0,  0,  0], 
                        [-1, -2, -1]])
    
    grad_x = apply_filter(image, sobel_x)
    grad_y = apply_filter(image, sobel_y)
    
    height, width = image.shape
    gradient = np.zeros((height, width, 2), dtype=float)
    
    gradient[:,:,0] = grad_x
    gradient[:,:,1] = grad_y
    
    return gradient

def visualize_gradients(gradient_query, gradient_test):
    plt.figure(figsize=(15, 10))
    
    # Градиенты для первого изображения
    plt.subplot(221)
    plt.imshow(gradient_query[:,:,0], cmap='gray')
    plt.title('Box: Градиент по X')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(gradient_query[:,:,1], cmap='gray')
    plt.title('Box: Градиент по Y')
    plt.axis('off')
    
    # Градиенты для второго изображения
    plt.subplot(223)
    plt.imshow(gradient_test[:,:,0], cmap='gray')
    plt.title('Scene: Градиент по X')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(gradient_test[:,:,1], cmap='gray')
    plt.title('Scene: Градиент по Y')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

gradient_query = get_gradient(box)
gradient_test = get_gradient(box_in_scene)
visualize_gradients(gradient_query, gradient_test)

def compute_M_for_points(image, gradient, feature_points):
    """
    # Матрица M:
    # [
    #     [Σ(w*dx*dx), Σ(w*dx*dy)],
    #     [Σ(w*dx*dy), Σ(w*dy*dy)]
    # ]
    """
    w, h = image.shape

    padded_image = np.zeros((w+4, h+4), dtype=int)
    padded_image[2:w+2, 2:h+2] = image
    
    W = gaussian_kernel(sigma=1, size=5)  
    
    M_matrices = []
    
    for x, y in feature_points:
        M = np.zeros((2, 2), dtype=float)
        
        # 5x5 окно
        for i in range(x-2, x+3):
            for j in range(y-2, y+3):

                Ix = gradient[i, j, 0]
                Iy = gradient[i, j, 1]

                A = np.array([[Ix**2, Ix*Iy], 
                            [Ix*Iy, Iy**2]], dtype=float)
                
                new = np.sum(W * padded_image[i:i+5, j:j+5]) * A
                
                # M и точка
                M = np.add(M, new)
                
        M_matrices.append(M)
    
    return M_matrices

matrix_M_query = compute_M_for_points(box, gradient_query, fast_query)
matrix_M_test = compute_M_for_points(box_in_scene, gradient_test, fast_test)

def compute_R_values(M_matrices, k):
    R_values = []
    for m in M_matrices:
        # R = det(M) - k * (trace(M))^2
        R = np.linalg.det(m) - k * np.trace(m)**2
        R_values.append(R)
    
    return R_values

k = 0.04
R_query= compute_R_values(matrix_M_query, k)
R_test = compute_R_values(matrix_M_test, k)
print(R_query)
print(R_test)

def Harris_filter(feature_points, R_values, max_points):
    # (точка, значение R)
    points_with_R = list(zip(feature_points, R_values))
    
    # По убыванию R
    sorted_points = sorted(points_with_R, key=lambda x: x[1], reverse=True)
    
    # Точки с положительным R
    filtered_points = []
    for point, r_value in sorted_points[:max_points]:
        if r_value < 0:
            break
        filtered_points.append(point)
        
    return filtered_points

points_query = Harris_filter(fast_query, R_query, 500)
points_test = Harris_filter(fast_test, R_test, 500)
print(len(points_query), points_query)
print(len(points_test), points_test)

def draw(image, points, color=[0, 0, 255]):
    height, width = image.shape
    
    result_image = np.zeros((height, width, 3), dtype=int)
    
    for i in range(height):
        for j in range(width):
            result_image[i, j, :] = image[i, j]
            
    for center_x, center_y in points:
        
        result_image[center_x-1:center_x+2, center_y-3] = color
        
        result_image[center_x+2, center_y-2] = color
        
        result_image[center_x+3, center_y-1:center_y+2] = color
        
        result_image[center_x+2, center_y+2] = color
        
        result_image[center_x-1:center_x+2, center_y+3] = color
        
        result_image[center_x-2, center_y+2] = color
        
        result_image[center_x-3, center_y-1:center_y+2] = color
        
        result_image[center_x-2, center_y-2] = color
        
    return result_image

boxWithFeatures = draw(box, points_query)
plt.figure()
plt.imshow(boxWithFeatures, vmin=0, vmax=255)
plt.show()

boxInSceneWithFeatures = draw(box_in_scene, points_test)
plt.figure()
plt.imshow(boxInSceneWithFeatures, vmin=0, vmax=255)
plt.show()

# Момент изображения I в точке (x, y) 
def moments_getter(I, p, q, x, y, r):
    # I - полутоновое изображение
    # p, q - порядки центрального момента m, соответствующие координатам x, y
    # x, y - координаты
    # r - радиус окрестности точки (x, y)
  
    w = I.shape[0]
    h = I.shape[1]

    # Координаты центральной точки
    ci = w//2
    cj = h//2

    m = 0

    # Проход по окну (max и min - границы окна)
    for i in range(max(0, x-r), min(w, x+r+1)):
        for j in range(max(0, y-r), min(h, y+r+1)):
            m += (i-ci)**p * (j-cj)**q * I[i, j]
    return m

# Получение углов (ориентаций) для ключевых точек
# angle = arctan2(m01, m10)
def angles_getter(I, points, r):
    angles = []
    for x, y in points:
        # Вертикальный момент m01 = Σ( (j-cj) * I[j, i] )
        m01 = moments_getter(I, 0, 1, x, y, r)
        # Горизонтальный момент m10 = Σ( (i-ci) * I[j, i] )
        m10 = moments_getter(I, 1, 0, x, y, r)
        # Угол в диапазоне [0, 2π] (без отрицательных значений)
        a = math.atan2(m01, m10) % (2*np.pi)
        angles.append(a)

    return angles

angles_query = angles_getter(box, points_query, patch_size)
angles_test = angles_getter(box_in_scene, points_test, patch_size)
print(angles_query)
print(angles_test)

def Gaussian_filtering(image_e):
    kernel = gaussian_kernel(sigma=10, size=5)  
    
    gaussImage = apply_gaussian_blur(image_e, kernel)
    
    return gaussImage

gaussBox = Gaussian_filtering(box)
plt.figure()
plt.imshow(gaussBox, cmap='gray', vmin=0, vmax=255)

gaussScene = Gaussian_filtering(box_in_scene)
plt.figure()
plt.imshow(gaussScene, cmap='gray', vmin=0, vmax=255)

# 30 равномерно распределенных углов от 0 до 2π
angles_for_descriptor = [2*k*np.pi/30 for k in range(30)]
print(len(angles_for_descriptor), angles_for_descriptor)

# Матрица поворота для каждого угла
# |cos(θ) -sin(θ)|
# |sin(θ)  cos(θ)|
rotate_matrix = []
for t in angles_for_descriptor:
    rt = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rotate_matrix.append(rt)

# Случайные точки
n = 256 
random_points= (np.pi/5) * np.random.randn(n, 2, 2)

# Нормализация
random_points/= np.max(random_points)
# масштабирование до размера окна
random_points*= (patch_size//2 - 2)
print(random_points)

# Ориентации (повернутые точки)
rotated_angles = []
for rt in rotate_matrix:
    st = []
    for i in range(random_points.shape[0]):
        st.append(list(np.array([rt.dot(random_points[i, 0]), rt.dot(random_points[i, 1])], dtype=int)))
    rotated_angles.append(st)

print(rotated_angles)

def get_descriptor(gaussImage, points, angles, patch_size):
    height, width = gaussImage.shape
    
    def compare_intensities(image, point1, point2):
        """
        1 если первая точка темнее, 0 иначе
        """
        intensity1 = image[point1[0], point1[1]]
        intensity2 = image[point2[0], point2[1]]
        return 1 if intensity1 < intensity2 else 0
    
    descriptors = []
    
    # Ключевые точки
    for point_idx, center_point in enumerate(points):
        # Находим индекс угла поворота
        rotation_idx = len(angles_for_descriptor) - 1
        # Ближайший меньший угол
        for angle_idx, angle in enumerate(angles_for_descriptor):
            if angles[point_idx] < angle:
                rotation_idx = angle_idx - 1
                break
        
        # Повернутые пары точек для текущего угла
        sampling_points = []
        for i in range(n):
            # смещение относительно центра
            offset_point1 = rotated_angles[rotation_idx][i][0] + center_point
            offset_point2 = rotated_angles[rotation_idx][i][1] + center_point
            sampling_points.append([offset_point1, offset_point2])
        
        # Формирование дескриптора
        binary_tests = []
        for test_points in reversed(sampling_points):
            result = compare_intensities(gaussImage, 
                                      test_points[0], 
                                      test_points[1])
            binary_tests.append(result)
            
        descriptors.append(np.array(binary_tests))
    
    return descriptors

descriptor_query = get_descriptor(gaussBox, points_query, angles_query, patch_size)
descriptor_test = get_descriptor(gaussScene, points_test, angles_test, patch_size)

print("Длина дескриптора query:", len(descriptor_query))
print("Длина дескриптора test:", len(descriptor_test))

def compare_descriptors(descriptor_query, descriptor_test):
    num_query_descriptors = len(descriptor_query)
    num_test_descriptors = len(descriptor_test)
    
    hamming_distances = np.zeros((num_query_descriptors, num_test_descriptors), dtype=int)
    
    for i in range(num_query_descriptors):
        for j in range(num_test_descriptors):
            
            # Дескриптор первой точки
            descriptor1 = descriptor_query[i]  

            # Дескриптор второй точки
            descriptor2 = descriptor_test[j]   
            
            distance = sum(np.absolute(descriptor1 - descriptor2))
            
            hamming_distances[i, j] = distance
            
    return hamming_distances

h_dist = compare_descriptors(descriptor_query, descriptor_test)
print(h_dist)

def Lowe_test(points1, points2, hamming_distances):
    LOWE_RATIO = 0.8 
    height, width = hamming_distances.shape
    
    matches_forward = []  # прямой поиск (1→2)
    matches_backward = [] # обратный поиск (2→1)
    
    # Прямой поиск (для каждой точки первого изображения)
    for i in range(height):
        # Порядок возврастания расстояний Хэмминга
        # строка
        distances = sorted(hamming_distances[i])
        
        # Проверяем первые два ближайших соответствия
        for t in range(len(distances)-1):
            dist1, dist2 = distances[t:t+2]  # два ближайших расстояния
            ratio = dist1/dist2  
            
            if ratio < LOWE_RATIO:
                # Находим индекс точки с минимальным расстоянием
                match_idx = list(hamming_distances[i]).index(dist1)
                # Пара точек -> соответствие 
                matches_forward.append([points1[i], points2[match_idx]])
                break
    
    # Обратный поиск (для каждой точки второго изображения)
    for i in range(width):
        # столбец
        distances = sorted(hamming_distances[:, i])
        for t in range(len(distances)-1):
            dist1, dist2 = distances[t:t+2]
            ratio = dist1/dist2
            
            if ratio < LOWE_RATIO:
                match_idx = list(hamming_distances[:, i]).index(dist1)
                matches_backward.append([points1[match_idx], points2[i]])
                break
    
    return [matches_forward, matches_backward]

Lowe_distances = Lowe_test(points_query, points_test, h_dist)
print(len(Lowe_distances[0]), len(Lowe_distances[1]))
print(Lowe_distances)

def cross_check(lowe_matches):
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

cross_distances = cross_check(Lowe_distances)
print(len(cross_distances[0]), len(cross_distances[1]))
print(cross_distances)

def bresenham(x0, y0, x1, y1, max_size):
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

points1, points2 = cross_distances

def visualize_matches(image1, image2, points1, points2):
    img1_with_points = draw(image1, points1)
    img2_with_points = draw(image2, points2)
    
    # Размеры для объединенного изображения
    total_width = img1_with_points.shape[1] + img2_with_points.shape[1]
    height = img2_with_points.shape[0]
    width2 = img2_with_points.shape[1]
    
    result = np.ones((height, total_width, 3), dtype=int) * 255
    
    result[0:img2_with_points.shape[0], 0:width2] = img2_with_points
    result[0:img1_with_points.shape[0], width2:] = img1_with_points
    
    color = [0, 0, 255]  
    for i in range(len(points1)):
        x1, y1 = points1[i]
        y1 += width2  
        x2, y2 = points2[i]
        
        line_points = bresenham(x1, y1, x2, y2, total_width)
        
        for x, y in line_points:
            result[x, y] = color
            
    return result

result = visualize_matches(box, box_in_scene, points1, points2)
plt.figure(figsize=(10, 8))
plt.imshow(result, vmin=0, vmax=255)
plt.show()

# Нахождение матрицы афинного преобразования и вектора смещения
# Оценка параметров через СЛАУ
def get_affine_transformation_parameters(src_points, dst_points):
    """
    src_points - три точки исходного изображения [(x1,y1), (x2,y2), (x3,y3)]
    dst_points - три соответствующие точки целевого изображения [(u1,v1), (u2,v2), (u3,v3)]
    
    T - вектор переноса
    M - матрица поворота и масштаба
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
    det1 = (x1-x3)*(y1-y2) - (x1-x2)*(y1-y3)
    # определитель второй матрицы
    det2 = y1-y2                               
    
    if abs(det1) < 1e-10 or abs(det2) < 1e-10:
        return None, None
    
    try:
        m1 = ((u1-u3)*(y1-y2) - (u1-u2)*(y1-y3)) / det1
        m2 = ((u1-u2) - m1*(x1-x2)) / det2
        tx = u1 - m1*x1 - m2*y1  
        
        m3 = ((v1-v3)*(y1-y2) - (v1-v2)*(y1-y3)) / det1
        m4 = ((v1-v2) - m3*(x1-x2)) / det2
        ty = v1 - m3*x1 - m4*y1  
        
        M = np.array([[m1, m2], [m3, m4]])  
        T = np.array([tx, ty])             
        
        return M, T
        
    except:
        return None, None

def Affine_transformation(points, M, T):

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

# Оценка параметров афинного преобразования методом наименьших квадратов
def MNK(points):
    """    
    Система уравнений для каждой пары точек (x,y) -> (u,v):
    u = m1*x + m2*y + tx
    v = m3*x + m4*y + ty
    
    Матричная форма A*X = B
    A = [x y 0 0 1 0] для u
        [0 0 x y 0 1] для v
    X = [m1 m2 m3 m4 tx ty]^T
    B = [u v]^T
    """
    if len(points) < 3:
        print("Недостаточно точек")
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
    X = np.linalg.inv(A.T @ A) @ A.T @ B

    # матрица M и вектор T
    M = X[0:4].reshape(2, 2)  
    T = X[4:6].reshape(2, 1)  
    
    return M, T

import random


def RANSAC(points1, points2, num_iterations, distance_threshold):
    """    
    points1, points2 - соответствующие точки двух изображений
    num_iterations: количество итераций
    distance_threshold: порог
    """
    min_points = 3  # минимальное количество точек для аффинного преобразования
    
    # Проверка количества точек
    if len(points1) < min_points:
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
            transform_matrix, translation = get_affine_transformation_parameters(
                sample_points1, sample_points2
            )
            if transform_matrix is None or translation is None:
                continue
            
            # Преобразование точек
            transformed_points = Affine_transformation(points1, transform_matrix, translation)
            
            # Подсчет верных соответствий (inliers)
            current_matching_points = []
            inliers_count = 0
            
            # Нахождение не-выбросов (inliers) по порогу расстояния
            for idx in range(len(points2)):
                x_transformed, y_transformed = transformed_points[idx]
                x_target, y_target = points2[idx]
                
                # Проверка расстояния
                if (abs(x_transformed - x_target) < distance_threshold and 
                    abs(y_transformed - y_target) < distance_threshold):
                    x_source, y_source = points1[idx]
                    inliers_count += 1
                    # Пара точек исходная -> целевая
                    current_matching_points.append([[x_source, y_source], 
                                                 [x_target, y_target]])
            
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
            best_transform_matrix, best_translation = MNK(best_matching_points)
            
        return best_transform_matrix, best_translation
    
    except Exception as e:
        print(f"Ошибка RANSAC: {str(e)}")
        return None, None
    
M, T = RANSAC(cross_distances[0], cross_distances[1], 5, 2)
print(M)
print(T)

def detect_query(box, scene, transform_matrix, translation):
    height, width = box.shape
    
    object_points = []
    for y in range(height):
        for x in range(width):
            object_points.append([y, x])
    
    transformed_points = Affine_transformation(
        object_points, 
        transform_matrix, 
        translation
    )

    polygon_points = np.array(transformed_points, dtype=int)
    
    if len(scene.shape) == 2:
        # Приведение к RGB
        result_scene = np.stack([scene] * 3, axis=-1)
    else:
        result_scene = scene.copy()
    
    scene_height, scene_width = result_scene.shape[:2]
    for point in polygon_points:
        y, x = point
        if 0 <= y < scene_height and 0 <= x < scene_width:
            result_scene[y, x] = [255, 0, 0]  
            
    return result_scene

detected_query = detect_query(box, box_in_scene, M, T)

print("Найденный объект")

plt.figure()
plt.imshow(detected_query, cmap='gray', vmin=0, vmax=255)
plt.show()