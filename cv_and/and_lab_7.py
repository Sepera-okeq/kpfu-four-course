import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загрузка набора данных MNIST
def load_data():
    """
    Загружает набор данных цифр из sklearn
    Возвращает:
    - digits: объект с данными
    - X: массив изображений (1797x64)
    - y: метки классов (1797)
    """
    digits = load_digits()
    X = digits.data  # Получаем массив "развернутых" изображений
    y = digits.target  # Получаем метки классов
    return digits, X, y

def extract_features(images, feature_type='raw'):
    """
    Извлекает признаки из изображений
    Параметры:
    - images: массив изображений
    - feature_type: тип извлекаемых признаков ('raw', 'histogram', 'horizontal_proj', 'vertical_proj', 'gradient')
    Возвращает:
    - features: массив признаков
    """
    # Преобразуем развернутые изображения обратно в матрицы 8x8
    images_reshaped = images.reshape(-1, 8, 8)
    
    if feature_type == 'raw':
        # Используем сами пиксели как признаки
        return images
    elif feature_type == 'histogram':
        # Создаем гистограммы интенсивности (16 бинов)
        features = np.array([np.histogram(img, bins=16, range=(0, 16))[0] 
                           for img in images])
        return features
    elif feature_type == 'horizontal_proj':
        # Суммируем значения пикселей по горизонтали
        features = np.array([np.sum(img_matrix, axis=1) for img_matrix in images_reshaped])
        return features
    elif feature_type == 'vertical_proj':
        # Суммируем значения пикселей по вертикали
        features = np.array([np.sum(img_matrix, axis=0) for img_matrix in images_reshaped])
        return features
    elif feature_type == 'gradient':
        # Вычисляем градиенты по x и y направлениям
        features = []
        for img in images_reshaped:
            gradient_x = np.gradient(img, axis=1)
            gradient_y = np.gradient(img, axis=0)
            # Вычисляем магнитуду градиента
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            # Усредняем значения магнитуды по 4x4 блокам для уменьшения размерности
            magnitude_reduced = magnitude.reshape(2, 4, 2, 4).mean(axis=(1,3)).flatten()
            features.append(magnitude_reduced)
        return np.array(features)

def evaluate_clustering(X, kmeans, y_true):
    """
    Оценивает качество кластеризации
    Параметры:
    - X: входные данные
    - kmeans: обученная модель KMeans
    - y_true: истинные метки классов
    Возвращает:
    - intra_dist: среднее внутрикластерное расстояние
    - inter_dist: среднее межкластерное расстояние
    - conf_matrix: матрица ошибок
    """
    # Вычисляем внутрикластерное расстояние
    intra_dist = kmeans.inertia_ / X.shape[0]
    
    # Вычисляем центры кластеров
    centers = kmeans.cluster_centers_
    
    # Вычисляем межкластерное расстояние
    n_clusters = len(centers)
    inter_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centers[i] - centers[j])
            inter_distances.append(dist)
    inter_dist = np.mean(inter_distances)
    
    # Вычисляем матрицу ошибок
    y_pred = kmeans.labels_
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return intra_dist, inter_dist, conf_matrix

def plot_results(digits, kmeans, feature_type):
    """
    Визуализирует результаты кластеризации
    Параметры:
    - digits: объект с данными
    - kmeans: обученная модель KMeans
    - feature_type: тип использованных признаков
    """
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Для каждого кластера
    for i in range(10):
        # Находим индексы изображений в текущем кластере
        cluster_images = digits.images[kmeans.labels_ == i]
        
        if len(cluster_images) > 0:
            # Вычисляем среднее изображение кластера
            mean_image = np.mean(cluster_images, axis=0)
            
            # Отображаем среднее изображение
            axes[i].imshow(mean_image, cmap='gray')
            axes[i].set_title(f'Кластер {i}')
        axes[i].axis('off')
    
    feature_descriptions = {
        'raw': 'исходные пиксели (64 признака)',
        'histogram': 'гистограмма интенсивности (16 признаков)',
        'horizontal_proj': 'горизонтальные проекции (8 признаков)',
        'vertical_proj': 'вертикальные проекции (8 признаков)',
        'gradient': 'градиентные характеристики (4 признака)'
    }
    plt.suptitle(f'Средние изображения кластеров\nПризнаки: {feature_descriptions[feature_type]}')
    plt.tight_layout()
    plt.show()

def main():
    """
    Основная функция программы
    """
    # Загружаем данные
    digits, X, y = load_data()
    
    # Список типов признаков для сравнения
    feature_types = ['raw', 'histogram', 'horizontal_proj', 'vertical_proj', 'gradient']
    
    for feature_type in feature_types:
        print(f"\nИспользуем признаки типа: {feature_type}")
        
        # Извлекаем признаки
        features = extract_features(X, feature_type)
        
        # Нормализуем данные
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Создаем и обучаем модель k-means
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(features_scaled)
        
        # Оцениваем качество кластеризации
        intra_dist, inter_dist, conf_matrix = evaluate_clustering(features_scaled, kmeans, y)
        # !!!! Среднее внутрикластерное расстояние (чем меньше, тем лучше) !!!!!
        print(f"Среднее внутрикластерное расстояние: {intra_dist:.4f}")
        # !!!! Среднее межкластерное расстояние (чем больше, тем лучше) !!!!!
        print(f"Среднее межкластерное расстояние: {inter_dist:.4f}")
        print("\nМатрица ошибок:")
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(conf_matrix):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')

        plt.xlabel('Пред')
        plt.ylabel('Истина')
        plt.title('Матрица ошибок')
        plt.show()
        
        # Визуализируем результаты
        plot_results(digits, kmeans, feature_type)

if __name__ == "__main__":
    main()
