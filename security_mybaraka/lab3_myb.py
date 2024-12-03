import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QTabWidget, QHBoxLayout, QTextEdit,
    QRadioButton, QButtonGroup, QMessageBox,
)

def string_to_binary(text):
    """
    Преобразование текста в двоичное представление.
    """
    return ''.join([f'{ord(char):016b}' for char in text])


def binary_to_string(binary):
    """
    Преобразование двоичного представления в текст.
    """
    chars = [binary[i:i + 16] for i in range(0, len(binary), 16)]
    return ''.join([chr(int(char, 2)) for char in chars])


def generate_random_binary(length):
    """Генерация случайной двоичной строки"""
    return ''.join(random.choice('01') for _ in range(length))


class CryptoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InfoSec — Гаммирование")
        self.setGeometry(100, 100, 1200, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Текущий алфавит
        self.allowed_chars = set("abcdefghijklmnopqrstuvwxyz")  # Default английский алфавит
        self.current_lang = "английский"

        self.add_xor_tab()
        self.add_otp_tab()
        self.add_block_chain_tab()
        self.add_language_switch()

    #### Общие вспомогательные функции ####
    def validate_text(self, text):
        """Проверка текста на допустимые символы"""
        invalid_chars = [char for char in text.lower() if char not in self.allowed_chars]
        if invalid_chars:
            QMessageBox.warning(self, "Ошибка", f"Текст содержит недопустимые символы: {''.join(invalid_chars)}")
            return False
        return True

    def add_language_switch(self):
        """Добавить переключатель языка"""
        language_layout = QHBoxLayout()
        english_button = QRadioButton("Английский")
        english_button.setChecked(True)
        russian_button = QRadioButton("Русский")

        self.language_group = QButtonGroup()
        self.language_group.addButton(english_button)
        self.language_group.addButton(russian_button)

        language_layout.addWidget(english_button)
        language_layout.addWidget(russian_button)

        # Подключить смену языка
        english_button.toggled.connect(self.switch_language)
        russian_button.toggled.connect(self.switch_language)

        language_widget = QWidget()
        language_widget.setLayout(language_layout)
        self.tabs.addTab(language_widget, "Язык")

    def switch_language(self):
        """Переключение языка"""
        if self.language_group.checkedButton().text() == "Английский":
            self.allowed_chars = set("abcdefghijklmnopqrstuvwxyz")
            self.current_lang = "английский"
        else:
            self.allowed_chars = set("абвгдежзийклмнопрстуфхцчшщъыьэюя")
            self.current_lang = "русский"

    #### Вкладка 1: XOR ####
    def add_xor_tab(self):
        """Первая вкладка: XOR"""
        tab = QWidget()
        layout = QVBoxLayout()

        self.xor_input = QLineEdit()
        self.xor_input.setPlaceholderText("Введите открытый текст")
        layout.addWidget(QLabel("Открытый текст"))
        layout.addWidget(self.xor_input)

        self.xor_key_input = QLineEdit()
        self.xor_key_input.setPlaceholderText("Введите ключ (текстовый)")
        layout.addWidget(QLabel("Ключ (текстовый)"))
        layout.addWidget(self.xor_key_input)

        self.xor_key_binary = QTextEdit()
        self.xor_key_binary.setReadOnly(True)
        layout.addWidget(QLabel("Двоичное представление ключа"))
        layout.addWidget(self.xor_key_binary)

        self.xor_output_binary = QTextEdit()
        self.xor_output_binary.setReadOnly(True)
        layout.addWidget(QLabel("Двоичное представление текста"))
        layout.addWidget(self.xor_output_binary)

        self.encrypted_output = QTextEdit()
        self.encrypted_output.setReadOnly(True)
        layout.addWidget(QLabel("Зашифрованный текст"))
        layout.addWidget(self.encrypted_output)

        self.decrypted_output = QTextEdit()
        self.decrypted_output.setReadOnly(True)
        layout.addWidget(QLabel("Расшифрованный текст"))
        layout.addWidget(self.decrypted_output)

        self.xor_buttons = QHBoxLayout()
        encrypt_button = QPushButton("Зашифровать")
        encrypt_button.clicked.connect(self.encrypt_xor)
        decrypt_button = QPushButton("Расшифровать")
        decrypt_button.clicked.connect(self.decrypt_xor)
        self.xor_buttons.addWidget(encrypt_button)
        self.xor_buttons.addWidget(decrypt_button)
        layout.addLayout(self.xor_buttons)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "XOR")

    def encrypt_xor(self):
        """Шифрование XOR"""
        text = self.xor_input.text()
        if not self.validate_text(text):
            return
        key_text = self.xor_key_input.text()
        if not key_text:
            QMessageBox.warning(self, "Ошибка", "Ключ отсутствует.")
            return

        binary_text = string_to_binary(text)
        binary_key = string_to_binary(key_text)  # Преобразуем ключ в двоичный
        self.xor_key_binary.setText(binary_key)  # Отобразить двоичный ключ

        #if len(binary_key) < len(binary_text):
        #    QMessageBox.warning(self, "Ошибка", "Ключ слишком короткий.")
        #    return

        encrypted_binary = ''.join(
            str(int(b1) ^ int(b2))
            for b1, b2 in zip(binary_text, binary_key[:len(binary_text)])
        )
        self.xor_output_binary.setText(binary_text)
        self.encrypted_output.setText(encrypted_binary)

    def decrypt_xor(self):
        """Расшифровка XOR"""
        encrypted_binary = self.encrypted_output.toPlainText()
        key_text = self.xor_key_input.text()
        if not encrypted_binary or not key_text:
            QMessageBox.warning(self, "Ошибка", "Нет зашифрованного текста или ключа.")
            return

        binary_key = string_to_binary(key_text)
        #if len(binary_key) < len(encrypted_binary):
        #    QMessageBox.warning(self, "Ошибка", "Ключ слишком короткий.")
        #    return

        decrypted_binary = ''.join(
            str(int(b1) ^ int(b2))
            for b1, b2 in zip(encrypted_binary, binary_key[:len(encrypted_binary)])
        )
        self.decrypted_output.setText(binary_to_string(decrypted_binary))

    #### Вкладка 2: Абсолютно криптостойкий шифр ####
    def add_otp_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.otp_input = QLineEdit()
        self.otp_input.setPlaceholderText("Введите открытый текст")
        layout.addWidget(QLabel("Открытый текст"))
        layout.addWidget(self.otp_input)

        self.otp_key_binary = QTextEdit()
        self.otp_key_binary.setReadOnly(True)
        layout.addWidget(QLabel("Случайный ключ (двоичный)"))
        layout.addWidget(self.otp_key_binary)

        self.otp_encrypted_binary = QTextEdit()
        self.otp_encrypted_binary.setReadOnly(True)
        layout.addWidget(QLabel("Зашифрованный текст (двоичный)"))
        layout.addWidget(self.otp_encrypted_binary)

        self.otp_decrypted_text = QTextEdit()
        self.otp_decrypted_text.setReadOnly(True)
        layout.addWidget(QLabel("Расшифрованный текст"))
        layout.addWidget(self.otp_decrypted_text)

        self.otp_buttons = QHBoxLayout()
        otp_generate_key_button = QPushButton("Сгенерировать ключ")
        otp_generate_key_button.clicked.connect(self.generate_otp_key)
        otp_encrypt_button = QPushButton("Зашифровать")
        otp_encrypt_button.clicked.connect(self.encrypt_otp)
        otp_decrypt_button = QPushButton("Расшифровать")
        otp_decrypt_button.clicked.connect(self.decrypt_otp)
        self.otp_buttons.addWidget(otp_generate_key_button)
        self.otp_buttons.addWidget(otp_encrypt_button)
        self.otp_buttons.addWidget(otp_decrypt_button)
        layout.addLayout(self.otp_buttons)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Абсолютно криптостойкий шифр")

    def generate_otp_key(self):
        """Генерация случайного ключа для OTP"""
        text = self.otp_input.text()
        if not self.validate_text(text):
            return
        binary_text = string_to_binary(text)
        self.otp_key = generate_random_binary(len(binary_text))
        self.otp_key_binary.setText(self.otp_key)

    def encrypt_otp(self):
        """Шифрование OTP"""
        text = self.otp_input.text()
        if not self.validate_text(text):
            return
        if not hasattr(self, "otp_key") or len(self.otp_key) != len(string_to_binary(text)):
            QMessageBox.warning(self, "Ошибка", "Ключ отсутствует или некорректной длины.")
            return
        encrypted_binary = ''.join(
            str(int(b1) ^ int(b2)) for b1, b2 in zip(string_to_binary(text), self.otp_key)
        )
        self.otp_encrypted_binary.setText(encrypted_binary)

    def decrypt_otp(self):
        """Расшифровка OTP"""
        encrypted_binary = self.otp_encrypted_binary.toPlainText()
        if not hasattr(self, "otp_key") or not encrypted_binary:
            QMessageBox.warning(self, "Ошибка", "Нет зашифрованного текста или ключа.")
            return
        decrypted_binary = ''.join(
            str(int(b1) ^ int(b2)) for b1, b2 in zip(encrypted_binary, self.otp_key)
        )
        self.otp_decrypted_text.setText(binary_to_string(decrypted_binary))

    #### Вкладка 3: Сцепление блоков ####
    def add_block_chain_tab(self):
        """Третья вкладка: Сцепление блоков"""
        tab = QWidget()
        layout = QVBoxLayout()

        self.block_input = QLineEdit()
        self.block_input.setPlaceholderText("Введите открытый текст")
        layout.addWidget(QLabel("Открытый текст"))
        layout.addWidget(self.block_input)

        self.block_manual_key = QLineEdit()
        self.block_manual_key.setPlaceholderText("Введите свой ключ (текстовый)")
        layout.addWidget(QLabel("Ручной ключ (текстовый)"))
        layout.addWidget(self.block_manual_key)

        self.block_manual_binary = QTextEdit()
        self.block_manual_binary.setReadOnly(True)
        layout.addWidget(QLabel("Двоичное представление ручного ключа"))
        layout.addWidget(self.block_manual_binary)

        self.block_generated_key = QTextEdit()
        self.block_generated_key.setReadOnly(True)
        layout.addWidget(QLabel("Сгенерированный ключ (двоичный)"))
        layout.addWidget(self.block_generated_key)

        self.block_output_binary = QTextEdit()
        self.block_output_binary.setReadOnly(True)
        layout.addWidget(QLabel("Результат цепления блоков"))
        layout.addWidget(self.block_output_binary)

        self.chain_buttons = QHBoxLayout()
        block_generate_key_button = QPushButton("Сгенерировать ключ")
        block_generate_key_button.clicked.connect(self.generate_block_key)
        block_encrypt_button = QPushButton("Шифровать")
        block_encrypt_button.clicked.connect(self.encrypt_block_chain)
        self.chain_buttons.addWidget(block_generate_key_button)
        self.chain_buttons.addWidget(block_encrypt_button)
        layout.addLayout(self.chain_buttons)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Сцепление блоков")

    def generate_block_key(self):
        """Генерация двоичного ключа для цепления блоков"""
        text = self.block_input.text()
        if not self.validate_text(text):
            return
        binary_length = len(string_to_binary(text))
        self.generated_key = generate_random_binary(binary_length)
        self.block_generated_key.setText(self.generated_key)

    def encrypt_block_chain(self):
        """Шифрование с цеплением блоков"""
        text = self.block_input.text()
        if not self.validate_text(text):
            return
        binary_text = string_to_binary(text)

        manual_key = self.block_manual_key.text()
        if not manual_key:
            QMessageBox.warning(self, "Ошибка", "Введите свой ключ (текстовый).")
            return
        binary_manual_key = string_to_binary(manual_key)
        self.block_manual_binary.setText(binary_manual_key)  # Отобразить двоичный ключ

        generated_key = self.generated_key
        if not generated_key:
            QMessageBox.warning(self, "Ошибка", "Сгенерируйте ключ.")
            return

        #if len(binary_manual_key) < len(binary_text):
        #    QMessageBox.warning(self, "Ошибка", "Ручной ключ слишком короткий.")
        #    return

        gamma = generated_key[:len(binary_text)]  # Берем начальный гамма-вектор
        encrypted_result = ""
        for i in range(len(binary_text)):
            encrypted_char = str(int(binary_text[i]) ^ int(gamma[i]) ^ int(binary_manual_key[i]))
            encrypted_result += encrypted_char

        self.block_output_binary.setText(encrypted_result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoApp()
    window.show()
    sys.exit(app.exec_())