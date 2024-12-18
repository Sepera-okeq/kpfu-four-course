import sys
import socket
import json
import random
import hashlib
import threading
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                           QLineEdit, QStackedWidget, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from crypto import CryptoUtils
from logger import CustomLogger

class SignalHandler(QObject):
    message_received = pyqtSignal(str)
    connection_status = pyqtSignal(str)
    auth_status = pyqtSignal(bool)
    register_status = pyqtSignal(bool, str)

class Client2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = CustomLogger("client2")
        self.crypto = CryptoUtils()
        self.signals = SignalHandler()
        self.socket = None
        self.session_key = None
        self.rsa_keys = None
        self.server_public_key = None
        
        self.init_ui()
        self.generate_rsa_keys()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Клиент 2')
        self.setGeometry(900, 100, 800, 600)
        
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создание стека виджетов для разных экранов
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Экран регистрации
        self.setup_registration_screen()
        
        # Экран входа
        self.setup_login_screen()
        
        # Экран чата
        self.setup_chat_screen()
        
        # Начинаем с экрана входа
        self.stacked_widget.setCurrentIndex(1)
        
        # Подключение сигналов
        self.signals.message_received.connect(self.handle_message)
        self.signals.connection_status.connect(self.handle_connection_status)
        self.signals.auth_status.connect(self.handle_auth_status)
        self.signals.register_status.connect(self.handle_register_status)
        
    def setup_registration_screen(self):
        """Настройка экрана регистрации"""
        register_widget = QWidget()
        layout = QVBoxLayout(register_widget)
        
        # Заголовок
        title = QLabel("Регистрация")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; margin: 10px;")
        layout.addWidget(title)
        
        # Поля ввода
        self.reg_login = QLineEdit()
        self.reg_login.setPlaceholderText("Введите логин")
        layout.addWidget(self.reg_login)
        
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Введите пароль")
        self.reg_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.reg_password)
        
        self.reg_confirm_password = QLineEdit()
        self.reg_confirm_password.setPlaceholderText("Подтвердите пароль")
        self.reg_confirm_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.reg_confirm_password)
        
        # Статус регистрации
        self.reg_status_label = QLabel()
        self.reg_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reg_status_label)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.register_button = QPushButton("Зарегистрироваться")
        self.register_button.clicked.connect(self.register)
        button_layout.addWidget(self.register_button)
        
        self.to_login_button = QPushButton("Уже есть аккаунт")
        self.to_login_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        button_layout.addWidget(self.to_login_button)
        
        layout.addLayout(button_layout)
        
        # Добавляем виджет в стек
        self.stacked_widget.addWidget(register_widget)
        
    def setup_login_screen(self):
        """Настройка экрана входа"""
        login_widget = QWidget()
        layout = QVBoxLayout(login_widget)
        
        # Заголовок
        title = QLabel("Вход")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; margin: 10px;")
        layout.addWidget(title)
        
        # Поля ввода
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Введите логин")
        layout.addWidget(self.login_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Введите пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.login_button = QPushButton("Войти")
        self.login_button.clicked.connect(self.login)
        button_layout.addWidget(self.login_button)
        
        self.to_register_button = QPushButton("Регистрация")
        self.to_register_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        button_layout.addWidget(self.to_register_button)
        
        layout.addLayout(button_layout)
        
        # Статус соединения
        self.connection_label = QLabel()
        self.connection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.connection_label)
        
        # Добавляем виджет в стек
        self.stacked_widget.addWidget(login_widget)
        
    def setup_chat_screen(self):
        """Настройка экрана чата"""
        chat_widget = QWidget()
        layout = QVBoxLayout(chat_widget)
        
        # Чат
        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        layout.addWidget(self.chat_text)
        
        # Поле ввода сообщения
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(50)
        layout.addWidget(self.message_input)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.send_button = QPushButton("Отправить сообщение")
        self.send_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_button)
        
        self.send_file_button = QPushButton("Отправить файл")
        self.send_file_button.clicked.connect(self.send_file)
        button_layout.addWidget(self.send_file_button)
        
        layout.addLayout(button_layout)
        
        # Добавляем виджет в стек
        self.stacked_widget.addWidget(chat_widget)
        
    def generate_rsa_keys(self):
        """Генерация ключей RSA"""
        try:
            self.rsa_keys = self.crypto.generate_rsa_keys()
            self.logger.info("Ключи RSA успешно сгенерированы")
        except Exception as e:
            self.logger.error(f"Ошибка при генерации ключей RSA: {str(e)}")
            
    def connect_to_server(self):
        """Подключение к серверу"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(('localhost', 5000))
            
            # Запуск прослушивания в отдельном потоке
            threading.Thread(target=self.listen_server, daemon=True).start()
            
            self.signals.connection_status.emit("Подключено к серверу")
            self.logger.info("Подключение к серверу установлено")
            return True
        except Exception as e:
            self.signals.connection_status.emit("Ошибка подключения к серверу")
            self.logger.error(f"Ошибка подключения к серверу: {str(e)}")
            return False
            
    def register(self):
        """Регистрация нового пользователя"""
        login = self.reg_login.text().strip()
        password = self.reg_password.text()
        confirm_password = self.reg_confirm_password.text()
        
        if not all([login, password, confirm_password]):
            self.signals.register_status.emit(False, "Все поля должны быть заполнены")
            return
            
        if password != confirm_password:
            self.signals.register_status.emit(False, "Пароли не совпадают")
            return
            
        if self.connect_to_server():
            try:
                # Отправка запроса на регистрацию
                request = {
                    'type': 'register',
                    'content': {
                        'login': login,
                        'password': self.crypto.hash_password(password)
                    }
                }
                self.socket.send(json.dumps(request).encode())
                self.logger.info(f"Отправлен запрос на регистрацию для {login}")
            except Exception as e:
                self.logger.error(f"Ошибка при регистрации: {str(e)}")
                self.signals.register_status.emit(False, f"Ошибка при регистрации: {str(e)}")
                
    def login(self):
        """Вход в систему"""
        self.current_login = self.login_input.text().strip()
        self.current_password = self.password_input.text()
        
        if not all([self.current_login, self.current_password]):
            QMessageBox.warning(self, "Ошибка", "Все поля должны быть заполнены")
            return
            
        if self.connect_to_server():
            try:
                # Отправка запроса на аутентификацию
                request = {
                    'type': 'auth_request',
                    'content': {
                        'login': self.current_login
                    }
                }
                self.socket.send(json.dumps(request).encode())
                self.logger.info(f"Отправлен запрос на аутентификацию для {self.current_login}")
            except Exception as e:
                self.logger.error(f"Ошибка при входе: {str(e)}")
                QMessageBox.warning(self, "Ошибка", "Ошибка при входе")
                
    def listen_server(self):
        """Прослушивание сервера"""
        try:
            while True:
                data = self.socket.recv(4096)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                self.process_message(message)
                
        except Exception as e:
            self.logger.error(f"Ошибка при прослушивании сервера: {str(e)}")
            self.signals.connection_status.emit("Соединение с сервером потеряно")
            
    def process_message(self, message):
        """Обработка входящих сообщений"""
        try:
            msg_type = message.get('type')
            content = message.get('content')
            
            if msg_type == 'register_response':
                self.handle_register_response(content)
            elif msg_type == 'auth_challenge':
                self.handle_auth_challenge(content)
            elif msg_type == 'auth_success':
                self.handle_auth_success(content)
            elif msg_type == 'auth_error':
                self.handle_auth_error(content)
            elif msg_type == 'dh_response':
                self.handle_dh_response(content)
            elif msg_type == 'chat_message':
                self.handle_chat_message(content)
            else:
                self.logger.warning(f"Получен неизвестный тип сообщения: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщения: {str(e)}")

    def handle_register_response(self, content):
        """Обработка ответа на регистрацию"""
        success = content.get('success')
        message = content.get('message')
        self.signals.register_status.emit(success, message)
        
        if success:
            self.logger.info("Регистрация успешна")
            # Переход на экран входа
            self.stacked_widget.setCurrentIndex(1)
        else:
            self.logger.warning(f"Ошибка регистрации: {message}")
            
    def handle_auth_challenge(self, content):
        """Обработка слова-вызова"""
        try:
            sw = content.get('sw')
            
            # Хэширование пароля и слова-вызова
            password_hash = self.crypto.hash_password(self.current_password)
            sw_hash = self.crypto.hash_sw(sw)
            response_hash = self.crypto.hash_password(password_hash + sw_hash)
            
            # Отправка ответа
            response = {
                'type': 'auth_response',
                'content': {
                    'login': self.current_login,
                    'hash': response_hash
                }
            }
            self.socket.send(json.dumps(response).encode())
            self.logger.info("Отправлен ответ на слово-вызов")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке слова-вызова: {str(e)}")
            
    def handle_auth_success(self, content):
        """Обработка успешной аутентификации"""
        try:
            self.server_public_key = content.get('rsa_public_key')
            self.signals.auth_status.emit(True)
            
            # Инициализация Диффи-Хеллмана
            self.init_diffie_hellman()
            
            # Переключение на экран чата
            self.stacked_widget.setCurrentIndex(2)
            self.logger.info("Аутентификация успешна")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке успешной аутентификации: {str(e)}")
            
    def handle_auth_error(self, content):
        """Обработка ошибки аутентификации"""
        message = content.get('message')
        self.signals.auth_status.emit(False)
        QMessageBox.warning(self, "Ошибка", f"Ошибка аутентификации: {message}")
        self.logger.warning(f"Ошибка аутентификации: {message}")
            
    def init_diffie_hellman(self):
        """Инициализация обмена ключами по Диффи-Хеллману"""
        try:
            # Генерация параметров
            self.dh_a = random.getrandbits(512) | 1
            self.dh_g = 2  # Генератор
            self.dh_p = self.crypto.generate_prime(512)  # Простое число
            
            # Вычисление открытого ключа A
            A = pow(self.dh_g, self.dh_a, self.dh_p)
            
            # Отправка параметров серверу
            request = {
                'type': 'dh_init',
                'content': {
                    'login': self.current_login,
                    'g': self.dh_g,
                    'p': self.dh_p,
                    'A': A
                }
            }
            self.socket.send(json.dumps(request).encode())
            self.logger.info("Отправлены параметры Диффи-Хеллмана")
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации Диффи-Хеллмана: {str(e)}")
            
    def handle_dh_response(self, content):
        """Обработка ответа на обмен ключами"""
        try:
            B = content.get('B')
            
            # Вычисление общего секрета
            self.session_key = str(pow(B, self.dh_a, self.dh_p))
            self.logger.info("Получен сеансовый ключ")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке ответа Диффи-Хеллмана: {str(e)}")
            
    def handle_chat_message(self, content):
        """Обработка сообщений чата"""
        try:
            login = content.get('login')
            encrypted_message = content.get('message')
            
            if self.session_key:
                # Расшифровка сообщения
                decrypted_message = self.crypto.rc4_encrypt(
                    self.session_key,
                    encrypted_message
                ).decode()
                
                self.signals.message_received.emit(f"{login}: {decrypted_message}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщения чата: {str(e)}")
            
    def send_message(self):
        """Отправка сообщения"""
        try:
            message = self.message_input.toPlainText().strip()
            if not message or not self.session_key:
                return
                
            # Шифрование сообщения
            encrypted_message = self.crypto.rc4_encrypt(self.session_key, message)
            
            # Отправка сообщения
            request = {
                'type': 'chat_message',
                'content': {
                    'login': self.current_login,
                    'message': encrypted_message
                }
            }
            self.socket.send(json.dumps(request).encode())
            
            self.message_input.clear()
            self.signals.message_received.emit(f"Вы: {message}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при отправке сообщения: {str(e)}")
            
    def send_file(self):
        """Отправка файла с ЭЦП"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
            if not file_path:
                return
                
            with open(file_path, 'rb') as f:
                file_data = f.read()
                
            # Создание ЭЦП
            file_hash = hashlib.sha256(file_data).hexdigest()
            signature = self.crypto.rsa_encrypt(
                int(file_hash, 16),
                self.rsa_keys[1]  # Используем закрытый ключ
            )
            
            # Шифрование файла
            encrypted_file = self.crypto.rc4_encrypt(self.session_key, file_data)
            
            # Отправка файла
            request = {
                'type': 'file_transfer',
                'content': {
                    'login': self.current_login,
                    'file': encrypted_file,
                    'signature': signature
                }
            }
            self.socket.send(json.dumps(request).encode())
            self.logger.info("Файл отправлен")
            
        except Exception as e:
            self.logger.error(f"Ошибка при отправке файла: {str(e)}")
            QMessageBox.warning(self, "Ошибка", "Ошибка при отправке файла")
            
    def handle_message(self, message):
        """Обработка сообщения для чата"""
        self.chat_text.append(message)
        
    def handle_connection_status(self, status):
        """Обработка статуса соединения"""
        self.connection_label.setText(status)
        
    def handle_auth_status(self, success):
        """Обработка статуса аутентификации"""
        if success:
            self.logger.info("Аутентификация успешна")
        else:
            self.logger.error("Ошибка аутентификации")
            QMessageBox.warning(self, "Ошибка", "Ошибка аутентификации")
            
    def handle_register_status(self, success, message):
        """Обработка статуса регистрации"""
        if success:
            self.reg_status_label.setText("Регистрация успешна")
            self.reg_status_label.setStyleSheet("color: green;")
        else:
            self.reg_status_label.setText(message)
            self.reg_status_label.setStyleSheet("color: red;")
            
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        if self.socket:
            self.socket.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = Client2()
    client.show()
    sys.exit(app.exec_())
