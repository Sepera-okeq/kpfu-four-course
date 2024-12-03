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
        for seq, indexes in sequences.items():
            if len(indexes) > 1:
                for i in range(len(indexes) - 1):
                    distances.append(indexes[i+1] - indexes[i])
        # Находим возможные длины ключа
        key_lengths = []
        for distance in distances:
            for i in range(2, 21):  # Предполагаем, что длина ключа от 2 до 20
                if distance % i == 0:
                    key_lengths.append(i)
        if not key_lengths:
            return 0  # Не удалось определить длину ключа
        # Возвращаем наиболее вероятную длину ключа
        return max(set(key_lengths), key=key_lengths.count)

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
        for i in range(key_length):
            seq = text[i::key_length]
            freq = Counter(seq)
            most_common_char = freq.most_common(1)[0][0]

            # Предполагаем, что наиболее частая буква в секции соответствует одной из частых букв языка
            # Попробуем различные варианты и выберем наиболее вероятный
            possible_keys = {}
            for common_letter in freq_letters:
                shift = (alphabet.index(most_common_char) - alphabet.index(common_letter)) % len(alphabet)
                possible_char = alphabet[shift]
                # Подсчитываем частоту предполагаемого ключевого символа
                if possible_char in possible_keys:
                    possible_keys[possible_char] += 1
                else:
                    possible_keys[possible_char] = 1

            # Выбираем символ ключа с наибольшей вероятностью
            if possible_keys:
                key_char = max(possible_keys, key=possible_keys.get)
                key += key_char
            else:
                key += alphabet[0]  # Если не удалось определить, берем первый символ алфавита

        return key

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

    # Остальной код класса остается без изменений

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
            key_length = VigenereCipher.kasiski_examination(text, alphabet)
        else:
            key_length = VigenereCipher.index_of_coincidence(text, alphabet)

        if key_length == 0:
            QMessageBox.warning(self, 'Ошибка', 'Не удалось определить длину ключа.')
            return

        # Частотный анализ для определения ключа
        key = VigenereCipher.frequency_analysis(text, key_length, freq_letters, alphabet)

        # Отображаем найденный ключ в поле только для чтения
        self.key_display.setText(key)

        # Расшифровываем текст с найденным ключом
        decrypted_text = VigenereCipher.decrypt(text, key, alphabet)

        # Показываем расшифрованное сообщение
        self.result_output.setText(decrypted_text)

        # Устанавливаем флаг, что изменения были внесены
        self.file_changed = True

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

def main():
    app = QApplication(sys.argv)
    window = CipherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
