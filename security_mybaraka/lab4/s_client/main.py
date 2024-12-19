# main.py
"""
Серверная часть защищенного чата.
Реализует функционал обработки подключений, аутентификации и обмена сообщениями.

Основные компоненты:
- ServerThread: Поток для асинхронной обработки подключений
- ClientO: Основной класс сервера для работы с сетью и криптографией
- ChatWindow: GUI окно чата
- MainWindow: GUI окно управления сервером
"""

import select
import socket
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QPushButton, QLabel, QMessageBox)
from PyQt5.QtGui import QColor, QTextCharFormat, QBrush
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal

# Импорты из utils
from utils import (
    Logger, RC4, RSA, UserDatabase,
    generate_sw, hash_md5, generate_odd_64bit,
    generate_prime_512bit, generate_generator, mod_exp
)

logger = Logger("server")

class ServerThread(QThread):
    """
    Поток для асинхронной обработки подключений клиентов.
    
    Сигналы:
        client_connected: Отправляется при подключении клиента
        chat_ready: Отправляется когда чат готов к работе
        message_received: Отправляется при получении сообщения
    """
    client_connected = pyqtSignal()
    chat_ready = pyqtSignal(int)
    message_received = pyqtSignal(str, str)
    
    def __init__(self, server):
        super().__init__()
        self.server = server

    def run(self):
        """Основной метод потока"""
        self.server.wait_for_client(self)
        self.client_connected.emit()

class MainWindow(QMainWindow):
    """
    Главное окно сервера.
    
    Отвечает за:
    - Отображение статуса сервера
    - Управление чатом
    """
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.chat_window = None
        
        self.setWindowTitle("Основной клиент")
        self.setGeometry(100, 100, 400, 200)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.status_label = QLabel("Ожидание клиента...")
        self.layout.addWidget(self.status_label)
        
        self.server_thread = ServerThread(server)
        self.server_thread.client_connected.connect(self.on_client_connected)
        self.server_thread.chat_ready.connect(self.show_chat)
        self.server_thread.message_received.connect(self.on_message_received)
        self.server.message_box_signal.connect(self.show_message_box)
        self.server.close_window_signal.connect(self.close)
        self.server_thread.start()
    
    def on_client_connected(self):
        """Обработка подключения клиента"""
        self.status_label.setText("Клиент подключен. Ожидание авторизации...")
    
    def show_chat(self, session_key):
        """Открытие окна чата"""
        self.chat_window = ChatWindow(rc4=self.server.rc4, 
                                    is_server=True, 
                                    socket=self.server.client_socket)
        self.chat_window.show()
        self.hide()
    
    def on_message_received(self, encrypted: str, decrypted: str):
        """Обработка полученного сообщения"""
        if self.chat_window:
            self.chat_window.message_received.emit(encrypted, decrypted)

    def show_message_box(self, title, message):
        """Отображение информационного сообщения"""
        QMessageBox.information(self, title, message)

class ChatWindow(QMainWindow):
    """
    Окно чата сервера.
    
    Отвечает за:
    - Отображение и отправку сообщений
    - Работу с RSA ключами
    - Проверку подписи файлов
    """
    message_received = pyqtSignal(str, str)
    
    def __init__(self, session_key=None, is_server: bool = True, socket=None, rc4=None):
        super().__init__()
        self.rc4 = rc4 if rc4 else RC4(str(session_key))
        self.is_server = is_server
        self.socket = socket
        self.e = None      # Открытый ключ RSA
        self.n = None      # Модуль RSA
        self.d = None      # Закрытый ключ RSA
        self.init_ui()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Чат (Основной клиент)" if self.is_server else "Чат (Клиент)")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)
        
        input_layout = QHBoxLayout()
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(50)
        self.message_input.textChanged.connect(self.limit_text_length)
        input_layout.addWidget(self.message_input)
        
        send_button = QPushButton("Отправить")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        layout.addLayout(input_layout)
        
        rsa_layout = QHBoxLayout()
        generate_button = QPushButton("Сгенерировать")
        generate_button.clicked.connect(self.generate_keys)
        rsa_layout.addWidget(generate_button)
        
        self.send_key_button = QPushButton("Отправить")
        self.send_key_button.setEnabled(False)
        self.send_key_button.clicked.connect(self.send_keys)
        rsa_layout.addWidget(self.send_key_button)
        
        layout.addLayout(rsa_layout)
        
        self.message_received.connect(self.display_received_message)

    def limit_text_length(self):
        """Ограничение длины сообщения"""
        text = self.message_input.toPlainText()
        if len(text) > 700:
            self.message_input.setPlainText(text[:700])
            cursor = self.message_input.textCursor()
            cursor.setPosition(700)
            self.message_input.setTextCursor(cursor)
        
    def send_message(self):
        """Отправка сообщения"""
        text = self.message_input.toPlainText().strip()
        if text:
            encrypted = self.rc4.encrypt(text)
            self.display_sent_message(encrypted, text)
            
            if self.socket:
                try:
                    message = (encrypted + "\n").encode('utf-8')
                    self.socket.send(message)
                except Exception as e:
                    logger.error(f"Ошибка отправки сообщения: {e}")
                    self.close()
            
            self.message_input.clear()

    def display_sent_message(self, encrypted: str, decrypted: str):
        """Отображение отправленного сообщения"""
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.End)
        
        cursor.insertHtml('''
            <div style="margin: 10px 0; display: flex; align-items: center;">
                <div style="color: gray; font-size: 12px; margin-right: 10px;">Отправлено</div>
                <div style="padding: 10px; border-radius: 10px;">
        ''')
        
        cursor.insertHtml(f'<div style="color: red; margin-bottom: 5px;">[{encrypted}]</div>')
        cursor.insertHtml(f'<div>{decrypted}</div>')
        cursor.insertHtml('</div></div><br>')
        
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def display_received_message(self, encrypted: str, decrypted: str):
        """Отображение полученного сообщения"""
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.End)
        
        cursor.insertHtml('''
            <div style="margin: 10px 0; display: flex; align-items: center;">
                <div style="color: gray; font-size: 12px; margin-right: 10px;">Получено</div>
                <div style="padding: 10px; border-radius: 10px;">
        ''')
        
        cursor.insertHtml(f'<div style="color: red; margin-bottom: 5px;">[{encrypted}]</div>')
        cursor.insertHtml(f'<div>{decrypted}</div>')
        cursor.insertHtml('</div></div><br>')
                
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def generate_keys(self):
        """Генерация ключей RSA"""
        (self.e, self.n), self.d = RSA.generate_keys()
        logger.info(f"Сгенерированы ключи RSA: e={self.e}, n={self.n}, d={self.d}")
        self.send_key_button.setEnabled(True)

    def send_keys(self):
        """Отправка открытого ключа RSA"""
        if self.e is not None and self.n is not None:
            public_key = f"KEYS|{self.e}|{self.n}"
            encrypted_key = self.rc4.encrypt(public_key)
            if self.socket:
                try:
                    self.socket.send((encrypted_key + "\n").encode())
                    logger.info(f"Отправлен открытый ключ: e={self.e}, n={self.n}")
                except Exception as e:
                    logger.error(f"Ошибка отправки ключа")
                    self.close()

class ClientO(QObject):
    """
    Основной класс сервера.
    
    Отвечает за:
    - Обработку подключений
    - Регистрацию и аутентификацию пользователей
    - Криптографические операции
    - Управление чатом
    """
    message_box_signal = pyqtSignal(str, str)
    close_window_signal = pyqtSignal()

    def __init__(self, host: str = 'localhost', port: int = 12345):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.client_socket = None
        self.db = UserDatabase('users.db')

        # Криптографические параметры
        self.a = None      # Закрытый ключ Диффи-Хеллмана
        self.g = None      # Генератор
        self.p = None      # Модуль
        self.A = None      # Открытый ключ
        self.K = None      # Сеансовый ключ
        self.client_e = None  # Открытый ключ RSA клиента
        self.client_n = None  # Модуль RSA клиента

        self.server_thread = None
        self.rc4 = None

    def wait_for_client(self, server_thread):
        """Ожидание подключения клиента"""
        self.server_thread = server_thread
        self.client_socket, _ = self.socket.accept()
        self.handle_client()

    def handle_client(self):
        """Обработка сообщений от клиента"""
        try:
            while True:
                try:
                    data = self.client_socket.recv(4096).decode()
                    if not data:
                        break
                    
                    if data.startswith("AUTH|"):
                        _, login = data.split("|")
                        
                        if self.authenticate_user(login):
                            if self.exchange_keys():
                                logger.info("Сеансовый ключ установлен")
                        continue
                    
                    if data.startswith("REGISTER|"):
                        _, login, password = data.split("|")
                        if self.register_user(login, password):
                            self.client_socket.send(b"SUCCESS")
                        else:
                            self.client_socket.send(b"EXISTS")
                        continue

                    if self.rc4:
                        decrypted = self.rc4.encrypt(data.strip())
                        
                        if decrypted.startswith("KEYS|"):
                            _, e, n = decrypted.split("|")
                            self.client_e = int(e)
                            self.client_n = int(n)
                            logger.info(f"Получен открытый ключ клиента: e={e}, n={n}")
                            continue
                            
                        if decrypted.startswith("ECP|"):
                            _, file_data, signature = decrypted.split("|")
                            self.verify_signature(file_data, int(signature))
                            continue
                            
                        if self.server_thread:
                            self.server_thread.message_received.emit(data.strip(), decrypted)
                            
                except socket.timeout:
                    continue
                        
        except Exception as e:
            pass
        finally:
            if self.client_socket:
                self.client_socket.close()

    def register_user(self, login: str, password: str) -> bool:
        """
        Регистрация нового пользователя
        
        Args:
            login: Логин пользователя
            password: Пароль пользователя
            
        Returns:
            bool: Успешность регистрации
        """
        try:
            if self.db.find_user(login):
                self.client_socket.send(b"EXISTS")
                logger.info(f"Пользователь {login} уже существует")
                self.close()
                self.close_window_signal.emit()
                logger.info("Соединение закрыто.")
                return False
                
            if self.db.add_user(login, password):
                logger.info(f"Пользователь {login} успешно зарегистрирован")
                self.client_socket.send(b"SUCCESS")
                return True
            else:
                self.client_socket.send(b"ERROR")
                logger.info(f"Ошибка при регистрации пользователя {login}")
                self.close()
                self.close_window_signal.emit()
                logger.info("Соединение закрыто.")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при регистрации: {e}")
            self.client_socket.send(b"ERROR")
            return False

    def authenticate_user(self, login: str) -> bool:
        """
        Аутентификация пользователя
        
        Args:
            login: Логин пользователя
            
        Returns:
            bool: Успешность аутентификации
        """
        try:
            logger.info(f"Аутентификация пользователя: {login}")
            user = self.db.find_user(login)
            if not user:
                logger.info("Пользователь не найден")
                self.client_socket.send(b"NOT_FOUND")
                self.close()
                self.close_window_signal.emit()
                logger.info("Соединение закрыто.")
                return False

            sw = generate_sw()
            time = datetime.now() + timedelta(hours=24)
            logger.info(f"SW: {sw}, Time: {time}")
            
            self.db.update_user_auth(login, sw, time)
            
            sw_hash = hash_md5(sw)
            logger.info(f"Хеш SW: {sw_hash}")  
            self.client_socket.send(sw_hash.encode())
            
            client_hash = self.client_socket.recv(1024).decode()
            logger.info(f"Хеш от клиента: {client_hash}")
            
            stored_password = user[2]
            server_hash = hash_md5(hash_md5(sw) + stored_password)
            logger.info(f"Вычисленный хеш: {server_hash}")
            
            if client_hash != server_hash:
                logger.info("Неверный пароль")
                self.client_socket.send(b"WRONG_PASSWORD")
                self.close()
                self.close_window_signal.emit()
                logger.info("Соединение закрыто.")
                return False
            
            logger.info("Аутентификация успешна")
            self.client_socket.send(b"SUCCESS")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            self.client_socket.send(b"ERROR")
            return False

    def exchange_keys(self) -> bool:
        """
        Обмен ключами по протоколу Диффи-Хеллмана
        
        Returns:
            bool: Успешность обмена ключами
        """
        try:
            self.a = generate_odd_64bit()
            self.g = generate_generator()
            self.p = generate_prime_512bit()
            self.A = mod_exp(self.g, self.a, self.p)

            logger.info(f"Сгенерировано число a: {self.a}")
            logger.info(f"Генератор g: {self.g}")
            logger.info(f"Простое число p: {self.p}")
            logger.info(f"Вычислено A: {self.A}")
            
            self.client_socket.settimeout(5.0)
            
            self.client_socket.send(f"{self.A}|{self.g}|{self.p}".encode())
            logger.info("Параметры отправлены клиенту")
            
            B = int(self.client_socket.recv(4096).decode())
            logger.info(f"Получено B от клиента: {B}")
            
            self.K = mod_exp(B, self.a, self.p)
            logger.info(f"Вычислен сеансовый ключ K: {self.K}")
            
            self.rc4 = RC4(str(self.K))
            
            self.client_socket.settimeout(None)
            
            self.server_thread.chat_ready.emit(self.K)
            
            return True
                
        except Exception as e:
            logger.error(f"Ошибка при обмене ключами: {e}")
            return False

    def verify_signature(self, file_data: str, signature: int):
        """
        Проверка подписи файла
        
        Args:
            file_data: Содержимое файла
            signature: Цифровая подпись
        """
        logger.info(f"Проверка подписи файла")
        logger.info(f"Подпись: {signature}")

        file_hash = hash_md5(file_data)
        logger.info(f"Хеш файла: {file_hash}")

        calculated_hash = mod_exp(signature, self.client_e, self.client_n)

        if int(file_hash, 16) == calculated_hash:
            self.message_box_signal.emit('Подпись верна', 'Подпись файла верна.')
            logger.info("Подпись файла верна")
        else:
            self.message_box_signal.emit('Подпись неверна', 'Подпись файла неверна.')
            logger.warning("Подпись файла неверна")

    def close(self):
        """Закрытие соединений"""
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()

def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    server = ClientO()
    window = MainWindow(server)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
