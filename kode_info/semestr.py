import sys
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QWidget, QGraphicsView, 
                             QGraphicsScene, QLineEdit, QRadioButton, QButtonGroup,
                             QDockWidget, QTabWidget)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtCore import Qt, QPointF

class TextConverter:
    @staticmethod
    def text_to_binary(text):
        return ''.join(format(ord(char), '08b') for char in text)
    
    @staticmethod
    def binary_to_text(binary):
        binary_chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
        return ''.join(chr(int(char, 2)) for char in binary_chars)

class HammingCode:
    @staticmethod
    def encode(data):
        """
        Кодирование по методу Хемминга
        """
        r = 1
        while (2**r < len(data) + r + 1):
            r += 1
        
        encoded = ['0'] * (len(data) + r)
        
        data_index = 0
        for i in range(1, len(encoded) + 1):
            if (i & (i-1)) != 0:  # Не позиции степеней 2
                encoded[i-1] = data[data_index]
                data_index += 1
        
        for i in range(r):
            pos = 2**i - 1
            parity = 0
            for j in range(pos, len(encoded), 2**(i+1)):
                for k in range(j, min(j + 2**i, len(encoded))):
                    if encoded[k] == '1':
                        parity ^= 1
            encoded[pos] = str(parity)
        
        return ''.join(encoded)

    @staticmethod
    def decode(encoded_data):
        """
        Декодирование и исправление ошибок
        """
        r = 1
        while 2**r < len(encoded_data) + 1:
            r += 1
        
        error_pos = 0
        for i in range(r):
            pos = 2**i - 1
            parity = 0
            for j in range(pos, len(encoded_data), 2**(i+1)):
                for k in range(j, min(j + 2**i, len(encoded_data))):
                    if encoded_data[k] == '1':
                        parity ^= 1
            
            if parity != 0:
                error_pos += pos + 1
        
        if error_pos > 0:
            corrected_data = list(encoded_data)
            corrected_data[error_pos-1] = '0' if encoded_data[error_pos-1] == '1' else '1'
            encoded_data = ''.join(corrected_data)
        
        decoded = []
        for i in range(1, len(encoded_data) + 1):
            if (i & (i-1)) != 0:  # Не позиции степеней 2
                decoded.append(encoded_data[i-1])
        
        return ''.join(decoded), error_pos

class HammingCodeVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Код Хемминга')
        self.setGeometry(100, 100, 1600, 900)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Верхняя панель управления
        control_layout = QHBoxLayout()
        
        # Выбор типа ввода
        input_type_group = QButtonGroup()
        self.text_radio = QRadioButton("Текст")
        self.binary_radio = QRadioButton("Бинарный")
        input_type_group.addButton(self.text_radio)
        input_type_group.addButton(self.binary_radio)
        self.text_radio.setChecked(True)
        
        # Поле ввода
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите данные")
        
        # Кнопка обработки
        process_btn = QPushButton('Обработать')
        process_btn.clicked.connect(self.process_input)
        
        # Добавление элементов в layout
        control_layout.addWidget(self.text_radio)
        control_layout.addWidget(self.binary_radio)
        control_layout.addWidget(self.input_field)
        control_layout.addWidget(process_btn)
        
        # Создаем TabWidget для графики и логов
        self.tab_widget = QTabWidget()
        
        # Графическая сцена
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        
        # Логи
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Добавляем вкладки
        self.tab_widget.addTab(self.view, "Визуализация")
        self.tab_widget.addTab(self.log_text, "Логи")
        
        # Добавляем элементы в главный layout
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.tab_widget)
        
        # Создаем Dock виджет для дополнительной информации
        #self.info_dock = QDockWidget("Информация", self)
        #self.info_text = QTextEdit()
        #self.info_text.setReadOnly(True)
        #self.info_dock.setWidget(self.info_text)
        #self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)
        
        # Цвета
        self.soft_green = QColor(150, 200, 150)
        self.soft_red = QColor(200, 150, 150)
        self.soft_blue = QColor(150, 150, 200)
        
    def process_input(self):
        # Очистка сцены
        self.scene.clear()
        self.log_text.clear()
        #self.info_text.clear()
        
        # Получение входных данных
        input_text = self.input_field.text()
        
        # Конвертация если нужно
        if self.text_radio.isChecked():
            self.original_data = TextConverter.text_to_binary(input_text)
        else:
            self.original_data = input_text
        
        # Проверка корректности бинарного ввода
        if not all(bit in '01' for bit in self.original_data):
            self.log_text.append("Некорректный бинарный ввод!")
            return
        
        # Основной процесс кодирования
        self.encode_and_process()
        
    def encode_and_process(self):
        # Кодирование
        self.encoded_data = HammingCode.encode(self.original_data)
        
        # Внесение ошибки
        self.error_data = list(self.encoded_data)
        self.error_pos = random.randint(0, len(self.error_data) - 1)
        self.error_data[self.error_pos] = '0' if self.error_data[self.error_pos] == '1' else '1'
        
        # Декодирование
        self.corrected_data, detected_error_pos = HammingCode.decode(''.join(self.error_data))
        
        # Логирование
        self.log_messages()
        
        # Визуализация
        self.visualize_hamming_code()
        
    def log_messages(self):
        # Основные логи с подробным описанием
        self.log_text.append("🔍 ПРОЦЕСС КОДИРОВАНИЯ ПО ХЕМИНГУ 🔍")
        self.log_text.append("\n1. ТЕОРЕТИЧЕСКИЕ ОСНОВЫ:")
        self.log_text.append("   - Код Хемминга - метод помехоустойчивого кодирования")
        self.log_text.append("   - Позволяет обнаруживать и исправлять одиночные ошибки")
        self.log_text.append("   - Использует контрольные биты для проверки целостности")

        self.log_text.append("\n2. ПРИНЦИП КОНТРОЛЬНЫХ БИТОВ:")
        self.log_text.append("   - Позиции контрольных битов: степени 2 (1, 2, 4, 8...)")
        self.log_text.append("   - Каждый контрольный бит отвечает за определенную область")
        self.log_text.append("   - Четность битов используется для обнаружения ошибок")

        self.log_text.append(f"\n3. ИСХОДНЫЕ ДАННЫЕ:")
        self.log_text.append(f"   - Бинарный код: {self.original_data}")
        self.log_text.append(f"   - Длина: {len(self.original_data)} бит")

        self.log_text.append("\n4. ЭТАПЫ КОДИРОВАНИЯ:")
        self.log_text.append(f"   - Закодированные данные: {self.encoded_data}")
        self.log_text.append(f"   - Длина кодового слова: {len(self.encoded_data)} бит")

        # Детали внесения ошибки
        self.log_text.append(f"\n5. ВНЕСЕНИЕ ОШИБКИ:")
        self.log_text.append(f"   - Позиция ошибки: {self.error_pos}")
        self.log_text.append(f"   - Бит до изменения: {self.encoded_data[self.error_pos]}")
        self.log_text.append(f"   - Бит после изменения: {self.error_data[self.error_pos]}")

        self.log_text.append("\n6. МЕХАНИЗМ ПОИСКА И ИСПРАВЛЕНИЯ ОШИБКИ:")
        # Детальный алгоритм поиска
        self.log_error_detection_process()

        # Конвертация текста
        if self.text_radio.isChecked():
            original_text = TextConverter.binary_to_text(self.original_data)
            corrected_text = TextConverter.binary_to_text(self.corrected_data)
            self.log_text.append(f"\n7. ТЕКСТОВОЕ ПРЕДСТАВЛЕНИЕ:")
            self.log_text.append(f"   - Оригинальный текст: {original_text}")
            self.log_text.append(f"   - Исправленный текст: {corrected_text}")

    def log_error_detection_process(self):
        """Детальное логирование процесса обнаружения ошибки"""
        self.log_text.append("   a) Проверка контрольных битов:")
        
        # Симуляция процесса проверки
        r = 1
        error_syndrome = 0
        while 2**r < len(self.error_data) + 1:
            pos = 2**r - 1
            parity = 0
            
            # Детальная проверка каждого контрольного бита
            for j in range(pos, len(self.error_data), 2**(r)):
                for k in range(j, min(j + 2**r, len(self.error_data))):
                    if self.error_data[k] == '1':
                        parity ^= 1
            
            self.log_text.append(f"     - Контрольный бит {pos+1}: {'Ошибка' if parity != 0 else 'Корректен'}")
            
            if parity != 0:
                error_syndrome += pos + 1
            
            r += 1
        
        self.log_text.append(f"   b) Синдром ошибки: {error_syndrome}")
        
        # Объяснение синдрома
        if error_syndrome > 0:
            self.log_text.append(f"   c) Обнаружена ошибка в позиции: {error_syndrome}")
            self.log_text.append("      - Синдром указывает точную позицию искаженного бита")
        else:
            self.log_text.append("   c) Ошибок не обнаружено")
            
    def visualize_hamming_code(self):
        # Визуализация этапов кодирования
        y_offset = 50
        self.add_bit_row("Оригинальные данные", self.original_data, y_offset)
        
        y_offset += 100
        self.add_bit_row("Закодированные данные", self.encoded_data, y_offset)
        
        y_offset += 100
        self.add_bit_row("Данные с ошибкой", ''.join(self.error_data), y_offset, error_pos=self.error_pos)
        
        y_offset += 100
        self.add_bit_row("Исправленные данные", self.encoded_data, y_offset, error_pos=self.error_pos)
        
        y_offset += 100
        self.add_bit_row("Декодированные данные", self.corrected_data, y_offset)
        
    def add_bit_row(self, label, data, y_offset, error_pos=None):
        """Улучшенная визуализация с точным размещением ошибки"""
        # Метка
        text_item = self.scene.addText(label, QFont("Arial", 12, QFont.Bold))
        text_item.setPos(10, y_offset)
        
        # Определение букв и контекста
        letters = []
        if self.text_radio.isChecked() and label == "Оригинальные данные":
            letters = list(self.input_field.text())
        
        # Группировка по 8 бит
        grouped_data = [data[i:i+8] for i in range(0, len(data), 8)]
        
        # Динамическая ширина контейнера
        container_width = len(grouped_data) * 80
        container_height = 70
        horizontal_spacing = 10
        
        for group_index, group in enumerate(grouped_data):
            # Расчет точной позиции
            x_pos = 250 + group_index * (container_width + horizontal_spacing)
            
            # Стильный контейнер
            group_rect = self.scene.addRect(
                x_pos, 
                y_offset, 
                container_width, 
                container_height, 
                QPen(Qt.black, 1, Qt.SolidLine), 
                QBrush(QColor(250, 250, 250))
            )
            
            # Биты внутри контейнера с равномерным распределением
            bit_width = 24
            padding = (container_width - len(group) * bit_width) // (len(group) + 1)
            
            for bit_in_group, bit in enumerate(group):
                # Определение глобального индекса бита
                global_bit_index = group_index * 8 + bit_in_group
                
                # Цвет только для конкретных позиций с ошибкой
                if (label in ["Данные с ошибкой", "Исправленные данные"] 
                    and error_pos is not None 
                    and global_bit_index == error_pos):
                    color = self.soft_blue  # Цвет ошибки
                else:
                    color = QColor(100, 200, 100) if bit == '1' else QColor(200, 100, 100)
                
                rect = self.scene.addRect(
                    x_pos + padding + bit_in_group * (bit_width + padding), 
                    y_offset + 20, 
                    bit_width, 
                    25, 
                    QPen(Qt.black), 
                    QBrush(color)
                )
                
                # Текст бита
                bit_text = self.scene.addText(bit, QFont("Arial", 9, QFont.Bold))
                bit_text.setPos(
                    x_pos + padding + bit_in_group * (bit_width + padding) + 3, 
                    y_offset + 22
                )
            
            # Надпись буквы или значения
            if label == "Оригинальные данные" and letters and group_index < len(letters):
                letter_text = self.scene.addText(
                    letters[group_index], 
                    QFont("Arial", 10, QFont.Bold)
                )
                letter_text.setPos(
                    x_pos + container_width // 4, 
                    y_offset + container_height
                )
            

def main():
    app = QApplication(sys.argv)
    visualizer = HammingCodeVisualizer()
    visualizer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()