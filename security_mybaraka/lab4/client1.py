import sys
import socket
import json
import threading
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from database import Database
from crypto import CryptoUtils
from logger import CustomLogger

class SignalHandler(QObject):
    message_received = pyqtSignal(str)
    client_connected = pyqtSignal(str)

class Client1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = CustomLogger("client1")
        self.db = Database()
        self.crypto = CryptoUtils()
        self.signals = SignalHandler()
        self.session_keys = {}  # Хранение сеансовых ключей для каждого клиента
        self.rsa_keys = None
        self.connected_clients = {}  # Хранение подключенных клиентов
        
        self.init_ui()
        self.init_server()
        self.generate_rsa_keys()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Клиент 1 (Сервер)')
        self.setGeometry(100, 100, 800, 600)
        
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Лог сообщений
        self.log_label = QLabel("Лог событий:")
        layout.addWidget(self.log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Чат
        self.chat_label = QLabel("Чат:")
        layout.addWidget(self.chat_label)
        
        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        layout.addWidget(self.chat_text)
        
        # Поле ввода сообщения
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(50)
        layout.addWidget(self.message_input)
        
        # Кнопки
        self.send_button = QPushButton("Отправить сообщение")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)
        
        self.sign_file_button = QPushButton("Подписать и отправить файл")
        self.sign_file_button.clicked.connect(self.sign_and_send_file)
        layout.addWidget(self.sign_file_button)
        
        # Подключение сигналов
        self.signals.message_received.connect(self.handle_message)
        self.signals.client_connected.connect(self.handle_client_connected)
        
    def init_server(self):
        """Инициализация сервера"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(('localhost', 5000))
            self.server_socket.listen(5)
            
            # Запуск прослушивания в отдельном потоке
            threading.Thread(target=self.listen_for_connections, daemon=True).start()
            
            self.logger.info("Сервер запущен на порту 5000")
            self.log_message("Сервер запущен и ожидает подключений...")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации сервера: {str(e)}")
            self.log_message(f"Ошибка при запуске сервера: {str(e)}")
            
    def generate_rsa_keys(self):
        """Генерация ключей RSA"""
        try:
            self.rsa_keys = self.crypto.generate_rsa_keys()
            self.logger.info("Ключи RSA успешно сгенерированы")
            self.log_message("Ключи RSA сгенерированы")
        except Exception as e:
            self.logger.error(f"Ошибка при генерации ключей RSA: {str(e)}")
            self.log_message(f"Ошибка при генерации ключей RSA: {str(e)}")
            
    def listen_for_connections(self):
        """Прослушивание входящих подключений"""
        while True:
            try:
                client_socket, address = self.server_socket.accept()
                threading.Thread(target=self.handle_client,
                              args=(client_socket, address),
                              daemon=True).start()
            except Exception as e:
                self.logger.error(f"Ошибка при принятии подключения: {str(e)}")
                
    def handle_client(self, client_socket, address):
        """Обработка подключенного клиента"""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                self.process_message(message, client_socket)
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке клиента: {str(e)}")
        finally:
            client_socket.close()
            
    def process_message(self, message, client_socket):
        """Обработка входящих сообщений"""
        try:
            msg_type = message.get('type')
            content = message.get('content')
            
            if msg_type == 'register':
                self.handle_registration(content, client_socket)
            elif msg_type == 'auth_request':
                self.handle_auth_request(content, client_socket)
            elif msg_type == 'auth_response':
                self.handle_auth_response(content, client_socket)
            elif msg_type == 'dh_init':
                self.handle_dh_init(content, client_socket)
            elif msg_type == 'chat_message':
                self.handle_chat_message(content, client_socket)
            elif msg_type == 'file_transfer':
                self.handle_file_transfer(content, client_socket)
            else:
                self.logger.warning(f"Получен неизвестный тип сообщения: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщения: {str(e)}")

    def handle_registration(self, content, client_socket):
        """Обработка регистрации нового пользователя"""
        try:
            login = content.get('login')
            password = content.get('password')
            
            if self.db.add_user(login, password):
                response = {
                    'type': 'register_response',
                    'content': {
                        'success': True,
                        'message': 'Регистрация успешна'
                    }
                }
                self.log_message(f"Пользователь {login} успешно зарегистрирован")
            else:
                response = {
                    'type': 'register_response',
                    'content': {
                        'success': False,
                        'message': 'Пользователь уже существует'
                    }
                }
                self.log_message(f"Ошибка регистрации: пользователь {login} уже существует")
                
            client_socket.send(json.dumps(response).encode())
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке регистрации: {str(e)}")
            response = {
                'type': 'register_response',
                'content': {
                    'success': False,
                    'message': str(e)
                }
            }
            client_socket.send(json.dumps(response).encode())
            
    def handle_auth_request(self, content, client_socket):
        """Обработка запроса аутентификации"""
        try:
            login = content.get('login')
            if self.db.check_login(login):
                # Генерация слова-вызова
                sw = self.crypto.generate_sw()
                self.db.update_sw(login, sw)
                
                response = {
                    'type': 'auth_challenge',
                    'content': {
                        'sw': sw
                    }
                }
                client_socket.send(json.dumps(response).encode())
                self.log_message(f"Отправлено слово-вызов для {login}")
            else:
                response = {
                    'type': 'auth_error',
                    'content': {
                        'message': 'Пользователь не найден'
                    }
                }
                client_socket.send(json.dumps(response).encode())
                self.log_message(f"Ошибка аутентификации: пользователь {login} не найден")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса аутентификации: {str(e)}")
            
    def handle_auth_response(self, content, client_socket):
        """Обработка ответа на аутентификацию"""
        try:
            login = content.get('login')
            response_hash = content.get('hash')
            
            user_data = self.db.get_user_data(login)
            if not user_data:
                return
                
            # Проверка времени действия слова-вызова
            if datetime.now() > datetime.fromisoformat(str(user_data[3])):
                self.log_message(f"Время действия слова-вызова для {login} истекло")
                return
                
            # Проверка хэша
            password_hash = user_data[1]
            sw_hash = self.crypto.hash_sw(user_data[2])
            expected_hash = self.crypto.hash_password(password_hash + sw_hash)
            
            if response_hash == expected_hash:
                self.connected_clients[login] = client_socket
                self.signals.client_connected.emit(login)
                self.log_message(f"Клиент {login} успешно аутентифицирован")
                
                # Отправка публичного ключа RSA
                response = {
                    'type': 'auth_success',
                    'content': {
                        'rsa_public_key': self.rsa_keys[0]
                    }
                }
                client_socket.send(json.dumps(response).encode())
            else:
                self.log_message(f"Ошибка аутентификации для {login}: неверный хэш")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке ответа аутентификации: {str(e)}")
            
    def handle_dh_init(self, content, client_socket):
        """Обработка инициализации Диффи-Хеллмана"""
        try:
            login = content.get('login')
            g = content.get('g')
            p = content.get('p')
            A = content.get('A')
            
            # Генерация закрытого ключа b
            b = int.from_bytes(self.crypto.generate_sw().encode(), 'big') % p
            # Вычисление открытого ключа B
            B = pow(g, b, p)
            # Вычисление общего секрета
            K = pow(A, b, p)
            
            self.session_keys[login] = str(K)
            
            response = {
                'type': 'dh_response',
                'content': {
                    'B': B
                }
            }
            client_socket.send(json.dumps(response).encode())
            self.log_message(f"Обмен ключами DH с {login} завершен")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке DH: {str(e)}")
            
    def handle_chat_message(self, content, client_socket):
        """Обработка сообщений чата"""
        try:
            login = content.get('login')
            encrypted_message = content.get('message')
            
            if login in self.session_keys:
                # Расшифровка сообщения с помощью RC4
                decrypted_message = self.crypto.rc4_encrypt(
                    self.session_keys[login],
                    encrypted_message
                ).decode()
                
                self.signals.message_received.emit(f"{login}: {decrypted_message}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке сообщения чата: {str(e)}")
            
    def handle_file_transfer(self, content, client_socket):
        """Обработка передачи файла"""
        try:
            login = content.get('login')
            encrypted_file = content.get('file')
            signature = content.get('signature')
            
            if login in self.session_keys:
                # Расшифровка файла
                decrypted_file = self.crypto.rc4_encrypt(
                    self.session_keys[login],
                    encrypted_file
                )
                
                # Проверка подписи
                file_hash = hashlib.sha256(decrypted_file).hexdigest()
                if self.verify_signature(file_hash, signature, login):
                    self.log_message(f"Получен подписанный файл от {login}")
                    # Здесь можно добавить сохранение файла
                else:
                    self.log_message(f"Ошибка проверки подписи файла от {login}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке файла: {str(e)}")
            
    def verify_signature(self, file_hash, signature, login):
        """Проверка ЭЦП"""
        try:
            if login not in self.connected_clients:
                return False
                
            # Получение публичного ключа клиента
            client_public_key = self.connected_clients[login].get('public_key')
            if not client_public_key:
                return False
                
            # Проверка подписи
            decrypted_hash = self.crypto.rsa_decrypt(signature, client_public_key)
            return decrypted_hash == int(file_hash, 16)
            
        except Exception as e:
            self.logger.error(f"Ошибка при проверке подписи: {str(e)}")
            return False
            
    def send_message(self):
        """Отправка сообщения"""
        try:
            message = self.message_input.toPlainText().strip()
            if not message:
                return
                
            # Отправка сообщения всем подключенным клиентам
            for login, client_socket in self.connected_clients.items():
                if login in self.session_keys:
                    encrypted_message = self.crypto.rc4_encrypt(
                        self.session_keys[login],
                        message
                    )
                    
                    response = {
                        'type': 'chat_message',
                        'content': {
                            'login': 'Сервер',
                            'message': encrypted_message
                        }
                    }
                    client_socket.send(json.dumps(response).encode())
                    
            self.message_input.clear()
            self.signals.message_received.emit(f"Сервер: {message}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при отправке сообщения: {str(e)}")
            
    def sign_and_send_file(self):
        """Подписание и отправка файла"""
        # TODO: Реализовать выбор и отправку файла
        pass
            
    def log_message(self, message):
        """Добавление сообщения в лог"""
        self.log_text.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        
    def handle_message(self, message):
        """Обработка сообщения для чата"""
        self.chat_text.append(message)
        
    def handle_client_connected(self, login):
        """Обработка подключения клиента"""
        self.log_message(f"Клиент {login} подключился")
        
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.server_socket.close()
        self.db.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = Client1()
    client.show()
    sys.exit(app.exec_())
