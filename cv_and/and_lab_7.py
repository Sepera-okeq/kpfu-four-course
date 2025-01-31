import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
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

def map_clusters_to_labels(cluster_labels, true_labels):
    """
    Каждому кластеру присваивается тот класс (цифра), которая встречается в нем чаще всего
    """
    n_clusters = len(np.unique(cluster_labels))
    n_classes = len(np.unique(true_labels))
    mapping_matrix = np.zeros((n_clusters, n_classes))

    for i in range(len(true_labels)):
        mapping_matrix[cluster_labels[i], true_labels[i]] += 1
    
    cluster_to_class = {}
    for cluster in range(n_clusters):
        true_class = np.argmax(mapping_matrix[cluster])
        cluster_to_class[cluster] = true_class
    return np.array([cluster_to_class[label] for label in cluster_labels])

def calculate_metrics(y_true, y_pred):
    """
    Вычисляет различные метрики качества классификации
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    accuracy = np.zeros(n_classes)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    alpha = np.zeros(n_classes)
    beta = np.zeros(n_classes)
    
    for i, class_label in enumerate(classes):
        true_bin = (y_true == class_label)
        pred_bin = (y_pred == class_label)
        
        TP = np.sum((true_bin) & (pred_bin))
        FP = np.sum((y_true != class_label) & (pred_bin))
        TN = np.sum((y_true != class_label) & (y_pred != class_label))
        FN = np.sum((true_bin) & (y_pred != class_label))
        
        accuracy[i] = (TP + TN) / (TP + TN + FP + FN)
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        alpha[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        beta[i] = FN / (TP + FN) if (TP + FN) > 0 else 0
    
    return {
        'accuracy': np.mean(accuracy),
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1),
        'alpha': np.mean(alpha),
        'beta': np.mean(beta)
    }

def plot_roc_curves(y_true, y_pred, title="ROC кривые"):
    """
    Строит ROC-кривые для каждого класса
    """
    n_classes = 10
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC класс {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.show()

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
    - metrics: словарь с дополнительными метриками
    - predicted_labels: предсказанные метки после отображения кластеров на классы
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
    
    # Получаем метки кластеров и отображаем их на реальные классы
    cluster_labels = kmeans.labels_
    predicted_labels = map_clusters_to_labels(cluster_labels, y_true)
    
    # Вычисляем матрицу ошибок
    conf_matrix = confusion_matrix(y_true, predicted_labels)
    
    # Вычисляем дополнительные метрики
    metrics = calculate_metrics(y_true, predicted_labels)
    
    return intra_dist, inter_dist, conf_matrix, metrics, predicted_labels

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
        intra_dist, inter_dist, conf_matrix, metrics, predicted_labels = evaluate_clustering(features_scaled, kmeans, y)
        # !!!! Среднее внутрикластерное расстояние (чем меньше, тем лучше) !!!!!
        print(f"Среднее внутрикластерное расстояние: {intra_dist:.4f}")
        # !!!! Среднее межкластерное расстояние (чем больше, тем лучше) !!!!!
        print(f"Среднее межкластерное расстояние: {inter_dist:.4f}")
        print(f"\nДополнительные метрики:")
        # Доля правильно классифицированных образцов.
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        # Точность (из всех образцов, предсказанных как принадлежащих к классу, какая доля действительно принадлежит этому классу)!!!!
        print(f"Precision: {metrics['precision']:.3f}")
        # Полнота (из всех образцов, принадлежащих к классу, какая доля была правильно классифицирована)!!!
        print(f"Recall: {metrics['recall']:.3f}")
        # Гармоническое среднее precision и recall.
        print(f"F1 Score: {metrics['f1']:.3f}")
        # Ошибка I рода (false positive rate) - доля образцов, не принадлежащих к классу, но классифицированных как принадлежащие.
        print(f"Ошибка I рода (alpha): {metrics['alpha']:.3f}")
        # Ошибка II рода (false negative rate) - доля образцов, принадлежащих к классу, но классифицированных как не принадлежащие.
        print(f"Ошибка II рода (beta): {metrics['beta']:.3f}")
        
        # Строим ROC-кривые (Receiver Operating Characteristic) для каждого класса. 
        # ROC-кривая показывает зависимость между True Positive Rate (TPR) и False Positive Rate (FPR) при изменении порога классификации.
        # Площадь под ROC-кривой (AUC - Area Under the Curve) является показателем качества классификации: чем ближе AUC к 1, тем лучше.
        plot_roc_curves(y, predicted_labels, f"ROC-кривые для признаков типа {feature_type}")
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
