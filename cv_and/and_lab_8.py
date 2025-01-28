import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Установка случайного зерна для воспроизводимости результатов
torch.manual_seed(42)
np.random.seed(42)

class SimpleVGG(nn.Module):
    """
    Упрощенная архитектура VGG для классификации рукописных цифр
    Архитектура:
    input -> conv3x3(8) -> conv3x3(8) -> maxpool -> conv3x3(10) -> conv3x3(16) -> 
    conv3x3(16) -> maxpool -> fc(100) -> fc(10) -> softmax
    """
    def __init__(self):
        super(SimpleVGG, self).__init__()
        
        # Первый блок свёрточных слоев (2 слоя по 8 фильтров)
        self.conv_block1 = nn.Sequential(
            # Первый свёрточный слой (1 канал -> 8 каналов)
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            # Второй свёрточный слой (8 каналов -> 8 каналов)
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            # Слой подвыборки (максимальный пулинг 2x2)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Второй блок свёрточных слоев
        self.conv_block2 = nn.Sequential(
            # Третий свёрточный слой (8 каналов -> 10 каналов)
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            # Четвертый свёрточный слой (10 каналов -> 16 каналов)
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Пятый свёрточный слой (16 каналов -> 16 каналов)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Слой подвыборки
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            # Преобразование в одномерный вектор
            nn.Flatten(),
            # Первый полносвязный слой (16 * 2 * 2 -> 100)
            nn.Linear(16 * 2 * 2, 100),
            nn.ReLU(),
            # Выходной слой (100 -> 10 классов)
            nn.Linear(100, 10)
        )
        
    def forward(self, x):
        # Прямой проход через сеть
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_layers(x)
        return x

def prepare_data():
    """
    Подготовка данных: загрузка, нормализация и разделение на выборки
    """
    # Загрузка набора данных
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Разделение на обучающую и временную выборки (80% / 20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    # Разделение временной выборки на валидационную и тестовую (50% / 50%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Преобразование данных в тензоры PyTorch и изменение формы для свёрточной сети
    X_train = torch.FloatTensor(X_train).view(-1, 1, 8, 8)
    X_val = torch.FloatTensor(X_val).view(-1, 1, 8, 8)
    X_test = torch.FloatTensor(X_test).view(-1, 1, 8, 8)
    
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train, X_val, y_train, y_val, num_epochs=50, batch_size=32):
    """
    Обучение модели с использованием мини-батчей и валидации
    """
    # Создание загрузчиков данных
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Списки для хранения истории обучения
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()  # Режим обучения
        total_train_loss = 0
        
        # Обучение на мини-батчах
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()  # Обнуление градиентов
            outputs = model(batch_X)  # Прямой проход
            loss = criterion(outputs, batch_y)  # Вычисление функции потерь
            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Обновление весов
            total_train_loss += loss.item()
        
        # Вычисление среднего значения функции потерь на эпохе
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация модели
        model.eval()  # Режим оценки
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Отключение вычисления градиентов
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Вычисление метрик на валидационной выборке
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        # Вывод прогресса обучения
        if (epoch + 1) % 5 == 0:
            print(f'Эпоха [{epoch+1}/{num_epochs}], '
                  f'Потери на обучении: {avg_train_loss:.4f}, '
                  f'Потери на валидации: {avg_val_loss:.4f}, '
                  f'Точность на валидации: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, X_test, y_test):
    """
    Оценка модели на тестовой выборке
    """
    model.eval()
    with torch.no_grad():
        # Получение предсказаний модели
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
        # Вычисление точности
        accuracy = accuracy_score(y_test, predicted)
        # Построение матрицы ошибок
        conf_matrix = confusion_matrix(y_test, predicted)
        
        return accuracy, conf_matrix, predicted

def plot_confusion_matrix(conf_matrix):
    """
    Визуализация матрицы ошибок
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.show()

def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    Визуализация процесса обучения
    """
    plt.figure(figsize=(12, 4))
    
    # График функции потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Обучающая выборка')
    plt.plot(val_losses, label='Валидационная выборка')
    plt.title('Динамика функции потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Точность на валидации')
    plt.title('Динамика точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Подготовка данных
    print("Подготовка данных...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    # Создание и обучение модели
    print("\nСоздание модели...")
    model = SimpleVGG()
    
    print("\nНачало обучения...")
    train_losses, val_losses, val_accuracies = train_model(
        model, X_train, X_val, y_train, y_val,
        num_epochs=50,
        batch_size=32
    )
    
    # Оценка модели
    print("\nОценка модели на тестовой выборке...")
    accuracy, conf_matrix, predictions = evaluate_model(model, X_test, y_test)
    print(f"\nТочность на тестовой выборке: {accuracy * 100:.2f}%")
    
    # Визуализация результатов
    print("\nПостроение графиков...")
    plot_training_history(train_losses, val_losses, val_accuracies)
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()
