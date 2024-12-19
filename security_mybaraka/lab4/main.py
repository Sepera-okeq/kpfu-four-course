import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, 
                            QTextEdit, QVBoxLayout, QWidget, QTableWidget,
                            QTableWidgetItem, QHeaderView, QPushButton,
                            QLabel, QLineEdit, QMessageBox, QHBoxLayout,
                            QFileDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from c_client.main import ClientObject, ChatWindow as ClientChatWindow
from s_client.main import ServerObject, ChatWindow as ServerChatWindow
import sqlite3
from datetime import datetime
from utils import validate_credentials, Logger
import os

logger = Logger("main")

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

class LogViewer(QDockWidget):
    """Компонент для просмотра логов в реальном времени"""
    def __init__(self, parent=None):
        super().__init__("Логи", parent)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setWidget(self.text_edit)
        
        # Словарь для хранения последних позиций чтения каждого файла
        self.log_positions = {}
        
        # Список файлов логов для мониторинга
        self.log_files = ['main.log', 'client.log', 'server.log']
        
        # Таймер для обновления логов
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logs)
        self.timer.start(100)  # Обновление каждые 100мс для более быстрой реакции
        
    def update_logs(self):
        for log_file in self.log_files:
            try:
                if not os.path.exists(log_file):
                    continue
                    
                # Инициализация позиции для нового файла
                if log_file not in self.log_positions:
                    self.log_positions[log_file] = 0
                
                with open(log_file, "r", encoding='utf-8') as f:
                    # Переход к последней позиции чтения
                    f.seek(self.log_positions[log_file])
                    new_data = f.read()
                    
                    if new_data:
                        self.text_edit.append(new_data.rstrip())
                        self.log_positions[log_file] = f.tell()
                        
                        # Прокрутка вниз
                        scrollbar = self.text_edit.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())
                        
            except (FileNotFoundError, IOError):
                continue

class DatabaseViewer(QDockWidget):
    """Компонент для просмотра базы данных в реальном времени"""
    def __init__(self, parent=None):
        super().__init__("База данных", parent)
        self.table = QTableWidget()
        self.setWidget(self.table)
        
        # Настройка таблицы
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Логин", "Пароль", "SW", "Время"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Таймер для обновления данных
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # Обновление каждую секунду
        
    def update_data(self):
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users')
            data = cursor.fetchall()
            
            self.table.setRowCount(len(data))
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    if j == 4 and value:  # Форматирование времени
                        try:
                            dt = datetime.fromisoformat(value)
                            value = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Запрет редактирования
                    self.table.setItem(i, j, item)
            
            conn.close()
        except sqlite3.Error:
            pass

class ClientDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Клиент", parent)
        self.client = ClientObject()
        self.chat_widget = None
        self.init_ui()
        self.show_registration()

    def init_ui(self):
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.setWidget(self.content)

    def clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_registration(self):
        self.clear_layout()
        
        self.layout.addWidget(QLabel('Регистрация'))
        
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText('Логин')
        self.layout.addWidget(self.login_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Пароль')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_input)
        
        register_btn = QPushButton('Зарегистрироваться')
        register_btn.clicked.connect(self.register)
        self.layout.addWidget(register_btn)
        
        login_link = QPushButton('Уже есть аккаунт? Войти')
        login_link.clicked.connect(self.show_login)
        self.layout.addWidget(login_link)

    def show_login(self):
        self.clear_layout()
        
        self.layout.addWidget(QLabel('Вход'))
        
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText('Логин')
        self.layout.addWidget(self.login_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Пароль')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_input)
        
        login_btn = QPushButton('Войти')
        login_btn.clicked.connect(self.login)
        self.layout.addWidget(login_btn)
        
        register_link = QPushButton('Создать аккаунт')
        register_link.clicked.connect(self.show_registration)
        self.layout.addWidget(register_link)

    def show_chat(self):
        self.clear_layout()
        self.chat_widget = ClientChatWindow(rc4=self.client.rc4, is_server=False, socket=self.client.socket)
        self.layout.addWidget(self.chat_widget)

    def register(self):
        login = self.login_input.text()
        password = self.password_input.text()
        
        valid, message = validate_credentials(login, password)
        if not valid:
            QMessageBox.warning(self, 'Ошибка', message)
            logger.error(f"Ошибка валидации: {message}")
            return
            
        success, message = self.client.register(login, password)
        if success:
            QMessageBox.information(self, 'Успех', message)
            logger.info(f"Регистрация успешна: {login}")
            self.show_login()
        else:
            QMessageBox.warning(self, 'Ошибка', message)
            logger.error(f"Ошибка регистрации: {message}")

    def login(self):
        login = self.login_input.text()
        password = self.password_input.text()
            
        success, message = self.client.authenticate(login, password)
        
        if success:
            self.show_chat()
        else:
            QMessageBox.warning(self, 'Ошибка', message)

class ServerDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Сервер", parent)
        self.server = ServerObject()
        self.chat_widget = None
        self.init_ui()
        self.start_server()

    def init_ui(self):
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.setWidget(self.content)
        
        self.status_label = QLabel("Ожидание соединения с кем-то...")
        self.layout.addWidget(self.status_label)

    def start_server(self):
        self.server_thread = ServerThread(self.server)
        self.server_thread.client_connected.connect(self.on_client_connected)
        self.server_thread.chat_ready.connect(self.show_chat)
        self.server_thread.message_received.connect(self.on_message_received)
        self.server.message_box_signal.connect(self.show_message_box)
        self.server_thread.start()

    def on_client_connected(self):
        self.status_label.setText("Клиент подключен. Ожидание авторизации...")

    def show_chat(self, session_key):
        self.clear_layout()
        self.chat_widget = ServerChatWindow(rc4=self.server.rc4, 
                                          is_server=True, 
                                          socket=self.server.client_socket)
        self.layout.addWidget(self.chat_widget)

    def on_message_received(self, encrypted: str, decrypted: str):
        if self.chat_widget:
            self.chat_widget.message_received.emit(encrypted, decrypted)

    def show_message_box(self, title, message):
        QMessageBox.information(self, title, message)

    def clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Защищенный чат")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создание док-виджетов
        self.client_dock = ClientDockWidget(self)
        self.server_dock = ServerDockWidget(self)
        self.log_viewer = LogViewer(self)
        self.db_viewer = DatabaseViewer(self)
        
        # Добавление док-виджетов
        self.addDockWidget(Qt.LeftDockWidgetArea, self.client_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.server_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.db_viewer)
        
        # Разрешение перемещения док-виджетов
        self.setDockNestingEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
