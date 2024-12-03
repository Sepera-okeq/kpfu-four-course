import sys
import string
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox
from PyQt5.QtCore import QLocale, QTranslator
from collections import Counter

# Алфавиты
RU_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789'
EN_ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'

# Частотный анализ основан на самых частых буквкахххххх
RU_FREQ_LETTERS = 'о'
EN_FREQ_LETTERS = 'e'
# Правда для некоторых текстов это может и не работать... так как метод по себе плох...

MAX_INT_SIZE = 2147483647
MIN_INT_SIZE = -2147483648

class CaesarCipher:
    @staticmethod
    def normalize_key(key, alphabet_length):
        """Нормализует ключ, приводя его к диапазону от 0 до длины алфавита-1"""
        return int(key) % alphabet_length

    @staticmethod
    def encrypt_decrypt(text, key, alphabet, encrypt=True):
        shifted_text = []
        key = CaesarCipher.normalize_key(key, len(alphabet))
        if not encrypt:
            key = -key
        for symbol in text:
            symbol_lower = symbol.lower()
            if symbol_lower in alphabet:
                index = alphabet.index(symbol_lower)
                shifted_index = (index + key) % len(alphabet)
                shifted_symbol = alphabet[shifted_index]
                shifted_text.append(shifted_symbol)
            else:
                continue
        return ''.join(shifted_text)


class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Выбор алфавита
        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(self.alphabet_box)

        self.input_text = QTextEdit()
        layout.addWidget(self.input_text)

        self.key_text = QLineEdit("0")  # Устанавливаем стартовое значение 0 для ключа
        self.key_text.setPlaceholderText("Введите ключ (целое число)")
        layout.addWidget(self.key_text)

        # Кнопки шифрования и дешифрования
        self.encrypt_btn = QPushButton('Зашифровать')
        self.decrypt_btn = QPushButton('Расшифровать')
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        layout.addWidget(self.encrypt_btn)
        layout.addWidget(self.decrypt_btn)
        layout.addWidget(self.result_output)

        # Подключаем события
        self.encrypt_btn.clicked.connect(self.encrypt)
        self.decrypt_btn.clicked.connect(self.decrypt)

        self.setLayout(layout)

    def get_alphabet(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET
        else:
            return EN_ALPHABET

    def filter_invalid_chars(self, text, alphabet):
        """Удаляет символы, которые не принадлежат текущему алфавиту. Возвращает только уникальные!"""
        invalid_chars = set([char for char in text if char.lower() not in alphabet and char not in string.whitespace])
        filter_spaces = set([char for char in invalid_chars if char in string.whitespace])  # ПРОБЕЛЫ
        return ''.join([char for char in text if char.lower() in alphabet]), invalid_chars, filter_spaces

    def show_invalid_chars_dialog(self, invalid_chars):
        """Предупреждение с предложением: Удалить символы или отменить действие. ОТОБРАЖАЕМ УНИКАЛЬНЫЕ!"""
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Ошибка')
            invalid_symbols_list = ''.join(sorted(invalid_chars))  # Сортируем для удобства чтения
            msg_box.setText(f'Недопустимые символы: {invalid_symbols_list}')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def show_whitespace_warning(self, filter_spaces):
        """Предупреждение перед удалением пробелов"""
        if filter_spaces:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Предупреждение')
            whitespace_symbols_list = ''.join(sorted(filter_spaces))  # Тоже уникальные пробелы
            msg_box.setText(f'Присутствуют пробелы после невалидных символов: {whitespace_symbols_list}. Удалить пробелы?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить пробелы')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def encrypt(self):
        alphabet = self.get_alphabet()
        text = self.input_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Введите текст для шифрования.')
            return

        # Фильтруем текст, недопустимые символы и пробелы
        filtered_text, invalid_chars, filter_spaces = self.filter_invalid_chars(text, alphabet)
        
        # Проверка на недопустимые символы
        if invalid_chars:
            user_choice = self.show_invalid_chars_dialog(invalid_chars)
            if user_choice == QMessageBox.Cancel:
                return

        # Проверка на пробелы
        if filter_spaces:
            space_user_choice = self.show_whitespace_warning(filter_spaces)
            if space_user_choice == QMessageBox.Cancel:
                return

        key = self.key_text.text().strip()
        if not key.lstrip('-').isdigit():
            QMessageBox.warning(self, 'Ошибка', 'Некорректный ключ. Пожалуйста, введите целое число.')
            return

        try:
            key = int(key)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", f"Ошибка в преобразовании ключа. Пожалуйста, убедитесь, что ключ целое число и не превышает размер INT.")
            return

        # Проверка на предельные значения ключа
        if key < MIN_INT_SIZE or key > MAX_INT_SIZE:
            QMessageBox.warning(self, "Ошибка", f"Ключ должен быть между {MIN_INT_SIZE} и {MAX_INT_SIZE}.")
            return

        if key == 0:
            QMessageBox.information(self, 'Предупреждение', 'Вы ввели ключ 0. Текст не будет изменён.')
            self.result_output.setText(filtered_text)
            return
        
        # Шифруем результат
        result = CaesarCipher.encrypt_decrypt(filtered_text, key, alphabet)
        self.result_output.setText(result)

    def decrypt(self):
        alphabet = self.get_alphabet()
        text = self.input_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Введите текст для дешифрования.')
            return

        # Фильтруем текст, недопустимые символы и пробелы
        filtered_text, invalid_chars, filter_spaces = self.filter_invalid_chars(text, alphabet)
        
        # Проверка на недопустимые символы
        if invalid_chars:
            user_choice = self.show_invalid_chars_dialog(invalid_chars)
            if user_choice == QMessageBox.Cancel:
                return

        # Проверка на пробелы
        if filter_spaces:
            space_user_choice = self.show_whitespace_warning(filter_spaces)
            if space_user_choice == QMessageBox.Cancel:
                return

        key = self.key_text.text().strip()
        if not key.lstrip('-').isdigit():
            QMessageBox.warning(self, 'Ошибка', 'Некорректный ключ. Пожалуйста, введите целое число.')
            return

        try:
            key = int(key)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", f"Ошибка в преобразовании ключа. Пожалуйста, убедитесь, что ключ целое число и не превышает размер INT.")
            return

        # Проверка на предельные значения ключа
        if key < MIN_INT_SIZE or key > MAX_INT_SIZE:
            QMessageBox.warning(self, "Ошибка", f"Ключ должен быть между {MIN_INT_SIZE} и {MAX_INT_SIZE}.")
            return

        if key == 0:
            QMessageBox.information(self, 'Предупреждение', 'Вы ввели ключ 0. Текст не будет изменён.')
            self.result_output.setText(filtered_text)
            return

        # Дешифруем результат
        result = CaesarCipher.encrypt_decrypt(filtered_text, key, alphabet, encrypt=False)
        self.result_output.setText(result)


#### Tab 2 (Шифрование файла)

class Tab2(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_changed = False
        self.original_text = ''  # Для хранения исходного текста

    def init_ui(self):
        layout = QVBoxLayout()

        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(self.alphabet_box)

        self.file_button = QPushButton('Выбрать файл')
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        self.file_input = QTextEdit()  # Открытый файл (исходный текст)
        layout.addWidget(QLabel("Исходный текст:"))
        layout.addWidget(self.file_input)

        self.file_output = QTextEdit()  # Изменения, внесённые в файл (итоговый текст)
        layout.addWidget(QLabel("Итоговый текст:"))
        layout.addWidget(self.file_output)
        self.file_output.setReadOnly(True)

        self.key_text = QLineEdit("0")  # Устанавливаем ключ 0 по умолчанию
        self.key_text.setPlaceholderText("Введите ключ (целое число)")
        layout.addWidget(self.key_text)

        self.encrypt_btn = QPushButton('Зашифровать файл')
        self.decrypt_btn = QPushButton('Расшифровать файл')
        self.save_btn = QPushButton("Сохранить файл")

        layout.addWidget(self.encrypt_btn)
        layout.addWidget(self.decrypt_btn)
        layout.addWidget(self.save_btn)

        self.encrypt_btn.clicked.connect(self.encrypt_file)
        self.decrypt_btn.clicked.connect(self.decrypt_file)
        self.save_btn.clicked.connect(self.save_to_file)

        self.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_input.setText(content)
                self.original_text = content  # Сохраняем исходный текст
            self.file_changed = False


    def get_alphabet(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET
        else:
            return EN_ALPHABET

    def filter_invalid_chars(self, text, alphabet):
        """Удаляет символы, которые не принадлежат текущему алфавиту"""
        invalid_chars = set([char for char in text if char.lower() not in alphabet and char not in string.whitespace])
        filter_spaces = set([char for char in invalid_chars if char in string.whitespace])
        return ''.join([char for char in text if char.lower() in alphabet]), invalid_chars, filter_spaces

    def show_invalid_chars_dialog(self, invalid_chars):
        """Предупреждение с предложением: Удалить символы или отменить действие. Показывает уникальные символы!"""
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Ошибка')
            invalid_symbols_list = ''.join(sorted(invalid_chars))  # Отображаем отсортированный набор уникальных символов.
            msg_box.setText(f'Недопустимые символы: {invalid_symbols_list}')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def show_whitespace_warning(self, filter_spaces):
        """Предупреждение перед удалением пробелов"""
        if filter_spaces:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Предупреждение')
            whitespace_symbols_list = ''.join(sorted(filter_spaces))  # Отображаем отсортированные уникальные пробелы.
            msg_box.setText(f'Присутствуют пробелы после невалидных символов: {whitespace_symbols_list}. Удалить пробелы?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить пробелы')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def encrypt_file(self):
        alphabet = self.get_alphabet()
        text = self.file_input.toPlainText().strip()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Файл пуст или не выбран.')
            return

        # Фильтруем текст, недопустимые символы и пробелы
        filtered_text, invalid_chars, filter_spaces = self.filter_invalid_chars(text, alphabet)
        
        # Проверка на недопустимые символы
        if invalid_chars:
            user_choice = self.show_invalid_chars_dialog(invalid_chars)
            if user_choice == QMessageBox.Cancel:
                return

        # Проверка на пробелы
        if filter_spaces:
            space_user_choice = self.show_whitespace_warning(filter_spaces)
            if space_user_choice == QMessageBox.Cancel:
                return

        key = self.key_text.text().strip()
        if not key.lstrip('-').isdigit():
            QMessageBox.warning(self, 'Ошибка', 'Некорректный ключ. Пожалуйста, введите целое число.')
            return

        try:
            key = int(key)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ошибка в преобразовании ключа. Пожалуйста, убедитесь, что ключ целое число.")
            return

        # Проверка на предельные значения ключа
        if key < MIN_INT_SIZE or key > MAX_INT_SIZE:
            QMessageBox.warning(self, "Ошибка", f"Ключ должен быть между {MIN_INT_SIZE} и {MAX_INT_SIZE}.")
            return

        if key == 0:
            QMessageBox.information(self, 'Предупреждение', 'Вы ввели ключ 0. Текст не будет изменён.')
            self.file_output.setText(filtered_text)  # Use self.file_output here
            return

        self.file_changed = True
        result = CaesarCipher.encrypt_decrypt(filtered_text, key, alphabet, encrypt=True)
        self.file_output.setText(result)  # Use self.file_output here

    def decrypt_file(self):
        alphabet = self.get_alphabet()
        text = self.file_input.toPlainText().strip()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Файл пуст или не выбран.')
            return

        # Фильтруем текст, недопустимые символы и пробелы
        filtered_text, invalid_chars, filter_spaces = self.filter_invalid_chars(text, alphabet)
        
        # Проверка на недопустимые символы
        if invalid_chars:
            user_choice = self.show_invalid_chars_dialog(invalid_chars)
            if user_choice == QMessageBox.Cancel:
                return

        # Проверка на пробелы
        if filter_spaces:
            space_user_choice = self.show_whitespace_warning(filter_spaces)
            if space_user_choice == QMessageBox.Cancel:
                return

        key = self.key_text.text().strip()
        if not key.lstrip('-').isdigit():
            QMessageBox.warning(self, 'Ошибка', 'Некорректный ключ. Пожалуйста, введите целое число.')
            return

        try:
            key = int(key)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ошибка в преобразовании ключа. Пожалуйста, убедитесь, что ключ целое число.")
            return

        # Проверка на предельные значения ключа
        if key < MIN_INT_SIZE or key > MAX_INT_SIZE:
            QMessageBox.warning(self, "Ошибка", f"Ключ должен быть между {MIN_INT_SIZE} и {MAX_INT_SIZE}.")
            return

        if key == 0:
            QMessageBox.information(self, 'Предупреждение', 'Вы ввели ключ 0. Текст не будет изменён.')
            self.file_output.setText(filtered_text)  # Use self.file_output here
            return

        self.file_changed = True
        result = CaesarCipher.encrypt_decrypt(filtered_text, key, alphabet, encrypt=False)
        self.file_output.setText(result)  # Use self.file_output here

    def save_to_file(self):
        text = self.file_output.toPlainText().strip()
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            self.file_changed = False
            
    def reset_changes(self):
        # Сбрасываем изменения в исходное состояние
        self.file_output.clear()
        self.file_input.setText(self.original_text)
        self.file_changed = False

#### Tab 3 (Взлом шифра)

class Tab3(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_changed = False
        self.original_text = ''  # Для хранения исходного текста

    def init_ui(self):
        layout = QVBoxLayout()

        # Элементы GUI для работы с файлом, алфавитами и дешифрованными данными
        self.alphabet_box = QComboBox()
        self.alphabet_box.addItems(['Русский', 'Английский'])
        layout.addWidget(self.alphabet_box)

        self.file_button = QPushButton('Выбрать файл')
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Поле для ввода зашифрованного текста (исходное сообщение)
        self.file_input = QTextEdit()  # Исходная информация
        layout.addWidget(QLabel("Зашифрованное сообщение:"))
        layout.addWidget(self.file_input)

        # Поле для отображения найденного ключа
        self.key_display = QLineEdit()
        self.key_display.setReadOnly(True)
        layout.addWidget(QLabel("Найденный ключ:"))
        layout.addWidget(self.key_display)

        # Поле для вывода дешифрованного сообщения
        self.result_output = QTextEdit()  # Дешифрованное сообщение
        self.result_output.setReadOnly(True)
        layout.addWidget(QLabel("Дешифрованное сообщение:"))
        layout.addWidget(self.result_output)

        # Кнопки для взлома и сохранения
        self.break_btn = QPushButton('Взломать шифр')
        self.save_btn = QPushButton("Сохранить файл")

        layout.addWidget(self.break_btn)
        layout.addWidget(self.save_btn)

        # Подключаем обработчики событий для кнопок
        self.break_btn.clicked.connect(self.break_cipher)
        self.save_btn.clicked.connect(self.save_to_file)

        # Устанавливаем layout для вкладки
        self.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите зашифрованный файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.file_input.setText(content)
                self.original_text = content  # Сохраняем исходный текст
                self.result_output.clear()
                self.key_display.clear()
            self.file_changed = False

    def get_alphabet(self):
        if self.alphabet_box.currentText() == 'Русский':
            return RU_ALPHABET, RU_FREQ_LETTERS
        else:
            return EN_ALPHABET, EN_FREQ_LETTERS

    def filter_invalid_chars(self, text, alphabet):
        """Удаляет символы, которые не принадлежат текущему алфавиту"""
        invalid_chars = set([char for char in text if char.lower() not in alphabet and char not in string.whitespace])
        filter_spaces = set([char for char in invalid_chars if char in string.whitespace])  # Уникальные пробелы
        return ''.join([char for char in text if char.lower() in alphabet]), invalid_chars, filter_spaces

    def show_invalid_chars_dialog(self, invalid_chars):
        """Предупреждение с предложением: Удалить символы или отменить действие"""
        if invalid_chars:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Ошибка')
            invalid_symbols_list = ''.join(sorted(invalid_chars))  # Уникальные символы!
            msg_box.setText(f'Недопустимые символы: {invalid_symbols_list}')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def show_whitespace_warning(self, filter_spaces):
        """Предупреждение перед удалением пробелов"""
        if filter_spaces:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Предупреждение')
            whitespace_symbols_list = ''.join(sorted(filter_spaces))  # Отображаем уникальные пробелы
            msg_box.setText(f'Присутствуют пробелы или другие неотображаемые символы после невалидных символов: {whitespace_symbols_list}. Удалить пробелы?')
            msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Cancel)
            msg_box.button(QMessageBox.Ok).setText('Удалить пробелы')
            msg_box.button(QMessageBox.Cancel).setText('Отмена')
            return msg_box.exec_()

    def break_cipher(self):
        text = self.file_input.toPlainText().strip()
        alphabet, freq_letter = self.get_alphabet()

        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Файл пуст или не выбран.')
            return

        # Фильтруем текст на недопустимые символы и пробелы
        filtered_text, invalid_chars, filter_spaces = self.filter_invalid_chars(text, alphabet)

        # Проверка на недопустимые символы
        if invalid_chars:
            user_choice = self.show_invalid_chars_dialog(invalid_chars)
            if user_choice == QMessageBox.Cancel:
                return

        # Проверка на пробелы
        if filter_spaces:
            space_user_choice = self.show_whitespace_warning(filter_spaces)
            if space_user_choice == QMessageBox.Cancel:
                return

        try:
            # Делаем частотный анализ и получаем ключ
            key = self.frequency_analysis(filtered_text, alphabet, freq_letter)

            # Показываем найденный ключ в поле для ключа
            self.key_display.setText(str(key))

            # Расшифровываем текст с найденным ключом
            decrypted_text = CaesarCipher.encrypt_decrypt(filtered_text, key, alphabet, encrypt=False)

            # Показываем расшифрованное сообщение
            self.result_output.setText(decrypted_text)

            self.file_changed = True  # Файл изменён

        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', str(e))
        except IndexError:
            QMessageBox.warning(self, 'Ошибка', "Ошибка при анализе частот символов.")

    def frequency_analysis(self, text, alphabet, most_freq_letter):
        # Частотный анализ символов
        frequencies = Counter(c.lower() for c in text if c.lower() in alphabet)
        if not frequencies:
            raise ValueError("Недостаточно данных для анализа")

        most_common_letter = frequencies.most_common(1)[0][0]

        # Попытаемся сопоставить частые буквы и найти ключ
        key = (alphabet.index(most_common_letter) - alphabet.index(most_freq_letter)) % len(alphabet)
        return key

    def save_to_file(self):
        text = self.result_output.toPlainText().strip()
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            self.file_changed = False

    # Добавляем метод для сброса изменений
    def reset_changes(self):
        # Сбрасываем изменения в исходное состояние
        self.result_output.clear()
        self.file_input.setText(self.original_text)
        self.key_display.clear()
        self.file_changed = False

class CipherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Шифрование Цезаря")
        self.setGeometry(300, 300, 600, 400)

        self.tabs = QTabWidget()
        self.tab1 = Tab1()
        self.tab2 = Tab2()
        self.tab3 = Tab3()

        self.tabs.addTab(self.tab1, "Шифр Цезаря")
        self.tabs.addTab(self.tab2, "Шифрование файла")
        self.tabs.addTab(self.tab3, "Взлом шифра")

        self.setCentralWidget(self.tabs)

        # Обработка смены вкладки с предупреждением
        self.tabs.currentChanged.connect(self.handle_tab_change)
        self.previous_index = 0  # Для отслеживания предыдущего индекса вкладки

    def handle_tab_change(self, index):
        current_tab = self.tabs.widget(self.previous_index)

        if isinstance(current_tab, (Tab2, Tab3)) and current_tab.file_changed:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Несохраненные изменения')
            msg_box.setText('Вы внесли изменения во вкладке, но не сохранили их. Сохранить итоговые изменения перед сменой вкладки?')

            save_button = msg_box.addButton('Сохранить', QMessageBox.AcceptRole)
            reset_button = msg_box.addButton('Сбросить', QMessageBox.DestructiveRole)
            cancel_button = msg_box.addButton('Отмена', QMessageBox.RejectRole)

            msg_box.setDefaultButton(save_button)

            msg_box.exec_()

            if msg_box.clickedButton() == save_button:
                current_tab.save_to_file()
            elif msg_box.clickedButton() == reset_button:
                current_tab.reset_changes()
            elif msg_box.clickedButton() == cancel_button:
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(self.previous_index)
                self.tabs.blockSignals(False)
                return

        self.previous_index = index  # Обновляем предыдущий индекс

def main():
    app = QApplication(sys.argv)
    window = CipherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()