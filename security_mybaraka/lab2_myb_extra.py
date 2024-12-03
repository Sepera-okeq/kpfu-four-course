import sys
import string
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QTextEdit,
    QLineEdit, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox
)
from PyQt5.QtCore import QLocale, QTranslator
from collections import Counter
import itertools

# Алфавиты
RU_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# Частотный анализ
RU_FREQ_LETTERS = 'оеаинтсрвлк'  # Часто встречающиеся буквы русского языка
EN_FREQ_LETTERS = 'etaoinshrdlu'  # Часто встречающиеся буквы английского языка

class VigenereCipher:
    @staticmethod
    def extend_key(text, key):
        if len(key) == 0:
            raise ValueError("Ключ не должен быть пустым после фильтрации недопустимых символов.")
        repeat_times = len(text) // len(key) + 1
        return (key * repeat_times)[:len(text)]

    @staticmethod
    def encrypt(text, key, alphabet):
        # Фильтруем текст и ключ, оставляя только символы из алфавита
        text = ''.join([char for char in text if char in alphabet])
        key = ''.join([char for char in key if char in alphabet])
        
        if len(key) == 0:
            raise ValueError("Ключ не содержит символов из выбранного алфавита.")
        
        key = VigenereCipher.extend_key(text, key)
        encrypted_text = []
        for t_char, k_char in zip(text, key):
            t_index = alphabet.index(t_char)
            k_index = alphabet.index(k_char)
            encrypted_char = alphabet[(t_index + k_index) % len(alphabet)]
            encrypted_text.append(encrypted_char)
        return ''.join(encrypted_text)

    @staticmethod
    def decrypt(text, key, alphabet):
        # Фильтруем текст и ключ, оставляя только символы из алфавита
        text = ''.join([char for char in text if char in alphabet])
        key = ''.join([char for char in key if char in alphabet])
        
        if len(key) == 0:
            raise ValueError("Ключ не содержит символов из выбранного алфавита.")
        
        key = VigenereCipher.extend_key(text, key)
        decrypted_text = []
        for t_char, k_char in zip(text, key):
            t_index = alphabet.index(t_char)
            k_index = alphabet.index(k_char)
            decrypted_char = alphabet[(t_index - k_index) % len(alphabet)]
            decrypted_text.append(decrypted_char)
        return ''.join(decrypted_text)

    @staticmethod
    def get_factors(n):
        factors = set()
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                factors.add(i)
                if n // i <= 20:
                    factors.add(n // i)
        if n <= 20:
            factors.add(n)
        return factors

    @staticmethod
    def kasiski_examination(text, alphabet):
        # Ищем повторяющиеся триграммы
        sequences = {}
        for i in range(len(text) - 2):
            seq = text[i:i+3]
            if seq in sequences:
                sequences[seq].append(i)
            else:
                sequences[seq] = [i]
        # Находим расстояния между повторами
        distances = []
        seq_distances = {}  # Для подробной информации
        for seq, indexes in sequences.items():
            if len(indexes) > 1:
                seq_distances[seq] = []
                for i in range(len(indexes) - 1):
                    distance = indexes[i+1] - indexes[i]
                    distances.append(distance)
                    seq_distances[seq].append(distance)
        # Находим возможные длины ключа
        key_lengths = []
        factors = {}  # Для подсчета количества делителей
        for distance in distances:
            # Факторизация расстояния
            factor_list = VigenereCipher.get_factors(distance)
            for factor in factor_list:
                if 2 <= factor <= 20:
                    key_lengths.append(factor)
                    if factor in factors:
                        factors[factor] += 1
                    else:
                        factors[factor] = 1
        if not key_lengths:
            return 0, sequences, seq_distances, distances, factors  # Не удалось определить длину ключа
        # Возвращаем наиболее вероятную длину ключа
        probable_key_length = max(set(key_lengths), key=key_lengths.count)
        return probable_key_length, sequences, seq_distances, distances, factors

    @staticmethod
    def index_of_coincidence(text, alphabet):
        # Рассчитываем индекс совпадений для разных сдвигов
        key_length = 0
        max_ic = 0
        for m in range(1, 21):  # Проверяем ключи длиной от 1 до 20
            ic_values = []
            for i in range(m):
                seq = text[i::m]
                freq = Counter(seq)
                n = len(seq)
                ic = sum([freq[char] * (freq[char] - 1) for char in alphabet]) / (n * (n - 1)) if n > 1 else 0
                ic_values.append(ic)
            avg_ic = sum(ic_values) / len(ic_values)
            if avg_ic > max_ic:
                max_ic = avg_ic
                key_length = m
        return key_length

    @staticmethod
    def frequency_analysis(text, key_length, freq_letters, alphabet):
        key = ''
        analytical_steps = []  # Для детализации процесса

        for i in range(key_length):
            seq = text[i::key_length]
            freq = Counter(seq)
            most_common_char = freq.most_common(1)[0][0]

            # Предполагаем, что наиболее частая буква в секции соответствует одной из частых букв языка
            possible_keys = {}
            step_info = f"Блок {i+1} среза текста:\nЧастоты символов: {freq}\n"
            step_info += f"Наиболее частый символ: '{most_common_char}'\n"

            for common_letter in freq_letters:
                shift = (alphabet.index(most_common_char) - alphabet.index(common_letter)) % len(alphabet)
                possible_char = alphabet[shift]
                possible_keys[possible_char] = possible_keys.get(possible_char, 0) + 1

                step_info += f"Предполагаем, что '{most_common_char}' соответствует '{common_letter}'. Тогда символ ключа: '{possible_char}'\n"

            # Выбираем символ ключа с наибольшей вероятностью
            key_char = max(possible_keys, key=possible_keys.get)
            key += key_char

            step_info += f"Выбранный символ ключа для этого блока: '{key_char}'\n"
            analytical_steps.append(step_info)

        return key, analytical_steps

class CipherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Шифр Виженера")
        self.setGeometry(300, 300, 600, 400)

        self.tabs = QTabWidget()
        self.tab1 = Tab1()
        self.tab2 = Tab2()
        self.tab3 = Tab3()

        self.tabs.addTab(self.tab1, "Шифрование Виженера")
        self.tabs.addTab(self.tab2, "Шифрование файла")
        self.tabs.addTab(self.tab3, "Взлом шифра")

        self.setCentralWidget(self.tabs)

    def handle_tab_change(self, index):
        current_tab = self.tabs.widget(self.previous_index)

        if isinstance(current_tab, (Tab2, Tab3)) and current_tab.file_changed:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Несохраненные изменения')
            msg_box.setText('Вы внесли изменения на этой вкладке, но не сохранили их. Сохранить изменения перед переходом?')

            save_button = msg_box.addButton('Сохранить', QMessageBox.AcceptRole)
            discard_button = msg_box.addButton('Сбросить', QMessageBox.DestructiveRole)
            cancel_button = msg_box.addButton('Отмена', QMessageBox.RejectRole)

            msg_box.setDefaultButton(save_button)
            msg_box.exec_()

            if msg_box.clickedButton() == save_button:
                current_tab.save_to_file()
            elif msg_box.clickedButton() == discard_button:
                current_tab.reset_changes()
            elif msg_box.clickedButton() == cancel_button:
                # Возвращаемся на предыдущую вкладку
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(self.previous_index)
                self.tabs.blockSignals(False)
                return

        self.previous_index = index  # Обновляем предыдущий индекс

class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(self.alphabet_box)

        self.input_text = QTextEdit()
        layout.addWidget(QLabel("Введите текст:"))
        layout.addWidget(self.input_text)

        self.key_text = QLineEdit()
        self.key_text.setPlaceholderText("Введите ключевое слово")
        layout.addWidget(self.key_text)

        self.encrypt_btn = QPushButton('Зашифровать')
        self.decrypt_btn = QPushButton('Расшифровать')
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        layout.addWidget(self.encrypt_btn)
        layout.addWidget(self.decrypt_btn)
        layout.addWidget(QLabel("Результат:"))
        layout.addWidget(self.result_output)

        self.encrypt_btn.clicked.connect(self.encrypt)
        self.decrypt_btn.clicked.connect(self.decrypt)

        self.setLayout(layout)

    def get_alphabet(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET
        else:
            return EN_ALPHABET

    def filter_text(self, text, alphabet):
        return ''.join([char.lower() for char in text if char.lower() in alphabet])

    def encrypt(self):
        alphabet = self.get_alphabet()
        text = self.input_text.toPlainText()
        key = self.key_text.text()

        if not text or not key:
            QMessageBox.warning(self, 'Ошибка', 'Введите текст и ключевое слово.')
            return

        text = text.lower()
        key = key.lower()

        # Проверяем текст на наличие недопустимых символов
        text = self.check_text_validity(text, alphabet)
        if text is None:
            # Пользователь отменил действие
            return

        # Проверяем ключ на наличие недопустимых символов
        key = self.check_key_validity(key, alphabet)
        if key is None:
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Ключ не может быть пустым после удаления недопустимых символов.')
            return

        try:
            result = VigenereCipher.encrypt(text, key, alphabet)
            self.result_output.setText(result)
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', str(e))

    def decrypt(self):
        alphabet = self.get_alphabet()
        text = self.input_text.toPlainText()
        key = self.key_text.text()

        if not text or not key:
            QMessageBox.warning(self, 'Ошибка', 'Введите текст и ключевое слово.')
            return

        text = text.lower()
        key = key.lower()

        # Проверяем текст на наличие недопустимых символов
        text = self.check_text_validity(text, alphabet)
        if text is None:
            return

        # Проверяем ключ на наличие недопустимых символов
        key = self.check_key_validity(key, alphabet)
        if key is None:
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Ключ не может быть пустым после удаления недопустимых символов.')
            return

        try:
            result = VigenereCipher.decrypt(text, key, alphabet)
            self.result_output.setText(result)
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', str(e))
                
    def check_key_validity(self, key, alphabet):
        """Проверяем ключ на наличие недопустимых символов."""
        invalid_chars = set([char for char in key if char.lower() not in alphabet])
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Недопустимые символы в ключе')
            invalid_symbols_list = ''.join(sorted(invalid_chars))  # Уникальные недопустимые символы
            msg_box.setText(f'Ключ содержит недопустимые символы: {invalid_symbols_list}. Удалить их?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            result = msg_box.exec_()
            if result == QMessageBox.Ok:
                # Удаляем недопустимые символы из ключа
                key = ''.join([char for char in key if char.lower() in alphabet])
                # Обновляем поле ввода ключа
                self.key_text.setText(key)
                return key
            else:
                # Отменяем действие
                return None
        else:
            # Если недопустимых символов нет, возвращаем исходный ключ
            return key

    def check_text_validity(self, text, alphabet):
        """Проверяем текст на наличие недопустимых символов."""
        invalid_chars = set([char for char in text if char.lower() not in alphabet])
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Недопустимые символы в тексте')
            invalid_symbols_list = ''.join(sorted(invalid_chars))
            msg_box.setText(f'Текст содержит недопустимые символы: {invalid_symbols_list}. Удалить их?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            result = msg_box.exec_()
            if result == QMessageBox.Ok:
                # Удаляем недопустимые символы из текста
                text = ''.join([char for char in text if char.lower() in alphabet])
                # Обновляем поле ввода текста
                self.input_text.setPlainText(text)
                return text
            else:
                # Отменяем действие
                return None
        else:
            # Если недопустимых символов нет, возвращаем исходный текст
            return text

class Tab2(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_changed = False  # Флаг для отслеживания изменений
        self.original_text = ''    # Исходный текст

    def init_ui(self):
        layout = QVBoxLayout()

        # Выбор алфавита
        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(QLabel("Выберите алфавит:"))
        layout.addWidget(self.alphabet_box)

        # Кнопка для выбора файла
        self.file_button = QPushButton('Выбрать файл')
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Поле для отображения содержимого файла
        self.file_input = QTextEdit()
        layout.addWidget(QLabel("Содержимое файла:"))
        layout.addWidget(self.file_input)

        # Поле для ввода ключа
        self.key_text = QLineEdit()
        self.key_text.setPlaceholderText("Введите ключевое слово")
        layout.addWidget(self.key_text)

        # Кнопки для шифрования, дешифрования и сохранения
        self.encrypt_btn = QPushButton('Зашифровать файл')
        self.decrypt_btn = QPushButton('Расшифровать файл')
        self.save_btn = QPushButton('Сохранить результат')

        layout.addWidget(self.encrypt_btn)
        layout.addWidget(self.decrypt_btn)
        layout.addWidget(self.save_btn)

        # Поле для отображения результата
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(QLabel("Результат:"))
        layout.addWidget(self.result_output)

        # Подключение сигналов к функциям
        self.encrypt_btn.clicked.connect(self.encrypt_file)
        self.decrypt_btn.clicked.connect(self.decrypt_file)
        self.save_btn.clicked.connect(self.save_file)

        self.setLayout(layout)

    def get_alphabet(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET
        else:
            return EN_ALPHABET

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_input.setText(content)
                self.original_text = content  # Сохраняем исходный текст
                self.result_output.clear()
            self.file_changed = False  # Сбрасываем флаг изменений

    def filter_text(self, text, alphabet):
        return ''.join([char.lower() for char in text if char.lower() in alphabet])

    def check_key_validity(self, key, alphabet):
        """Проверяем ключ на наличие недопустимых символов."""
        invalid_chars = set([char for char in key if char.lower() not in alphabet])
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Недопустимые символы в ключе')
            invalid_symbols_list = ''.join(sorted(invalid_chars))  # Уникальные недопустимые символы
            msg_box.setText(f'Ключ содержит недопустимые символы: {invalid_symbols_list}. Удалить их?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            result = msg_box.exec_()
            if result == QMessageBox.Ok:
                # Удаляем недопустимые символы из ключа
                key = ''.join([char for char in key if char.lower() in alphabet])
                # Обновляем поле ввода ключа
                self.key_text.setText(key)
                return key
            else:
                # Отменяем действие
                return None
        else:
            # Если недопустимых символов нет, возвращаем исходный ключ
            return key

    def check_text_validity(self, text, alphabet):
        """Проверяем текст на наличие недопустимых символов."""
        invalid_chars = set([char for char in text if char.lower() not in alphabet])
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Недопустимые символы в тексте')
            invalid_symbols_list = ''.join(sorted(invalid_chars))
            msg_box.setText(f'Текст содержит недопустимые символы: {invalid_symbols_list}. Удалить их?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            result = msg_box.exec_()
            if result == QMessageBox.Ok:
                # Удаляем недопустимые символы из текста
                text = ''.join([char for char in text if char.lower() in alphabet])
                # Обновляем поле ввода текста
                self.file_input.setPlainText(text)
                return text
            else:
                # Отменяем действие
                return None
        else:
            # Если недопустимых символов нет, возвращаем исходный текст
            return text

    def encrypt_file(self):
        alphabet = self.get_alphabet()
        text = self.file_input.toPlainText().strip()
        key = self.key_text.text()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Файл пуст или не выбран.')
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Введите ключевое слово.')
            return

        text = text.lower()
        key = key.lower()

        # Проверяем текст на наличие недопустимых символов
        text = self.check_text_validity(text, alphabet)
        if text is None:
            return

        # Проверяем ключ на наличие недопустимых символов
        key = self.check_key_validity(key, alphabet)
        if key is None:
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Ключ не может быть пустым после удаления недопустимых символов.')
            return

        try:
            result = VigenereCipher.encrypt(text, key, alphabet)
            self.result_output.setText(result)
            self.file_changed = True  # Устанавливаем флаг изменений
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', str(e))

    def decrypt_file(self):
        alphabet = self.get_alphabet()
        text = self.file_input.toPlainText().strip()
        key = self.key_text.text()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Файл пуст или не выбран.')
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Введите ключевое слово.')
            return

        text = self.filter_text(text, alphabet)
        key = key.lower()

        # Проверяем ключ на наличие недопустимых символов
        key = self.check_key_validity(key, alphabet)
        if key is None:
            return

        if not key:
            QMessageBox.warning(self, 'Ошибка', 'Ключ не может быть пустым после удаления недопустимых символов.')
            return

        try:
            result = VigenereCipher.decrypt(text, key, alphabet)
            self.result_output.setText(result)
            self.file_changed = True  # Устанавливаем флаг изменений
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', str(e))

    def save_file(self):
        text = self.result_output.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, 'Ошибка', 'Нет данных для сохранения.')
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            QMessageBox.information(self, 'Успех', 'Файл успешно сохранён.')
            self.file_changed = False  # Сбрасываем флаг изменений

    def reset_changes(self):
        # Сбрасываем внесенные изменения
        self.result_output.clear()
        self.file_input.setText(self.original_text)
        self.key_text.clear()
        self.file_changed = False

class Tab3(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_changed = False  # Для отслеживания изменений
        self.original_text = ''  # Для хранения исходного текста

    def init_ui(self):
        layout = QVBoxLayout()

        # Выбор алфавита
        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(QLabel("Выберите алфавит:"))
        layout.addWidget(self.alphabet_box)

        # Выбор файла
        self.file_button = QPushButton('Выбрать зашифрованный файл')
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Поле ввода зашифрованного сообщения
        self.file_input = QTextEdit()
        layout.addWidget(QLabel("Зашифрованное сообщение:"))
        layout.addWidget(self.file_input)

        # Выбор метода взлома
        self.method_box = QComboBox()
        self.method_box.addItems(['Метод Касиски']) # , 'Индекс совпадений'
        layout.addWidget(QLabel("Выберите метод взлома:"))
        layout.addWidget(self.method_box)

        # Кнопка для взлома
        self.break_btn = QPushButton('Взломать шифр')
        self.break_btn.clicked.connect(self.break_cipher)
        layout.addWidget(self.break_btn)

        # Кнопка для показа подробного анализа
        self.analysis_btn = QPushButton('Показать подробный анализ')
        self.analysis_btn.setEnabled(False)
        self.analysis_btn.clicked.connect(self.show_analysis)
        layout.addWidget(self.analysis_btn)

        # Поле для отображения найденного ключа (read-only)
        self.key_display = QLineEdit()
        self.key_display.setReadOnly(True)
        layout.addWidget(QLabel("Найденный ключ:"))
        layout.addWidget(self.key_display)

        # Поле для вывода расшифрованного сообщения
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(QLabel("Расшифрованное сообщение:"))
        layout.addWidget(self.result_output)

        # Кнопка для сохранения результата
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.clicked.connect(self.save_to_file)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def get_alphabet_and_freq_letters(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET, RU_FREQ_LETTERS
        else:
            return EN_ALPHABET, EN_FREQ_LETTERS

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_input.setText(content)

    def filter_text(self, text, alphabet):
        return ''.join([char.lower() for char in text if char.lower() in alphabet])

    def break_cipher(self):
        text = self.file_input.toPlainText().strip()
        alphabet, freq_letters = self.get_alphabet_and_freq_letters()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Загрузите зашифрованный текст.')
            return

        # Фильтруем текст, оставляя только буквы из алфавита
        text = self.filter_text(text, alphabet)

        # Выбор метода взлома
        if self.method_box.currentText() == 'Метод Касиски':
            result = VigenereCipher.kasiski_examination(text, alphabet)
            if result[0] == 0:
                QMessageBox.warning(self, 'Ошибка', 'Не удалось определить длину ключа.')
                return
            self.probable_key_length, self.sequences, self.seq_distances, self.distances, self.factors = result
        else:
            pass  # Если добавите другие методы, обработайте здесь

        # Частотный анализ для определения ключа
        key, self.frequency_steps = VigenereCipher.frequency_analysis(text, self.probable_key_length, freq_letters, alphabet)

        # Отображаем найденный ключ в поле только для чтения
        self.key_display.setText(key)

        # Расшифровываем текст с найденным ключом
        decrypted_text = VigenereCipher.decrypt(text, key, alphabet)

        # Показываем расшифрованное сообщение
        self.result_output.setText(decrypted_text)

        # Устанавливаем флаг, что изменения были внесены
        self.file_changed = True

        # Активируем кнопку для отображения подробного анализа
        self.analysis_btn.setEnabled(True)

        # Сохраняем данные для анализа
        self.text = text
        self.key = key
        self.decrypted_text = decrypted_text
        self.alphabet = alphabet

    def save_to_file(self):
        text = self.result_output.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, 'Ошибка', 'Нет данных для сохранения.')
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            QMessageBox.information(self, 'Успех', 'Файл успешно сохранён.')
            self.file_changed = False  # Сбрасываем флаг изменений

    def reset_changes(self):
        # Сброс изменений
        self.result_output.clear()
        self.key_display.clear()
        self.file_changed = False

    def show_analysis(self):
        self.analysis_window = AnalysisWindow(self.sequences, self.seq_distances, self.distances, self.factors, self.probable_key_length, self.frequency_steps, self.key, self.text, self.decrypted_text, self.alphabet)
        self.analysis_window.show()

class AnalysisWindow(QWidget):
    def __init__(self, sequences, seq_distances, distances, factors, probable_key_length, frequency_steps, key, text, decrypted_text, alphabet, parent=None):
        super().__init__(parent)
        self.sequences = sequences
        self.seq_distances = seq_distances
        self.distances = distances
        self.factors = factors
        self.probable_key_length = probable_key_length
        self.frequency_steps = frequency_steps
        self.key = key
        self.text = text
        self.decrypted_text = decrypted_text
        self.alphabet = alphabet
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Подробный анализ методом Касиски")
        layout = QVBoxLayout()

        # Создаем QTextEdit для отображения анализа
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)

        # Формируем подробный анализ
        analysis_str = ""

        analysis_str += "Шаг 1: Найдены следующие повторяющиеся последовательности (триграммы) и их позиции:\n"
        for seq, positions in self.sequences.items():
            if len(positions) > 1:
                analysis_str += f"'{seq}': позиции {positions}\n"

        analysis_str += "\nШаг 2: Вычислены расстояния между повторениями последовательностей:\n"
        for seq, dist_list in self.seq_distances.items():
            if len(dist_list) > 0:
                analysis_str += f"Для последовательности '{seq}': расстояния {dist_list}\n"

        analysis_str += "\nШаг 3: Найдены все расстояния и их факторизация (общие делители):\n"
        for distance in self.distances:
            factors = VigenereCipher.get_factors(distance)
            analysis_str += f"Расстояние {distance}: делители {sorted(factors)}\n"

        analysis_str += "\nШаг 4: Подсчитываем частоту появлений делителей (возможных длин ключа):\n"
        sorted_factors = sorted(self.factors.items(), key=lambda x: x[1], reverse=True)
        for factor, count in sorted_factors:
            analysis_str += f"Длина ключа {factor}: встречается {count} раз(а)\n"

        analysis_str += f"\nШаг 5: Наиболее вероятная длина ключа: {self.probable_key_length}\n"

        # Добавляем шаги частотного анализа
        analysis_str += "\nШаг 6: Выполняем частотный анализ для каждого блока:\n"
        for step_info in self.frequency_steps:
            analysis_str += step_info + "\n"

        analysis_str += f"\nШаг 7: Предполагаемый ключ: '{self.key}'\n"

        # Показываем процесс расшифровки
        analysis_str += "\nШаг 8: Расшифровываем текст с найденным ключом.\n"

        # (Опционально) Можно показать несколько первых строк исходного и расшифрованного текста
        analysis_str += "\nПервые 200 символов зашифрованного текста:\n"
        analysis_str += self.text[:200] + "\n"

        analysis_str += "\nПервые 200 символов расшифрованного текста:\n"
        analysis_str += self.decrypted_text[:200] + "\n"

        self.analysis_text.setPlainText(analysis_str)
        layout.addWidget(self.analysis_text)
        self.setLayout(layout)

def main():
    app = QApplication(sys.argv)
    window = CipherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()