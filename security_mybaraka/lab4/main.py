import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, 
                            QTextEdit, QVBoxLayout, QWidget, QTableWidget,
                            QTableWidgetItem, QHeaderView, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from c_client.main import Client, MainWindow as ClientMainWindow
from s_client.main import ClientO, MainWindow as ServerMainWindow
import sqlite3
from datetime import datetime

class LogViewer(QDockWidget):
    """Компонент для просмотра логов в реальном времени"""
    def __init__(self, parent=None):
        super().__init__("Логи", parent)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setWidget(self.text_edit)
        self.last_position = 0
        
        # Таймер для обновления логов
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logs)
        self.timer.start(1000)  # Обновление каждую секунду
        
    def update_logs(self):
        try:
            with open("app.log", "r", encoding='utf-8') as f:
                f.seek(self.last_position)
                new_data = f.read()
                if new_data:
                    self.text_edit.append(new_data)
                    self.last_position = f.tell()
                    # Прокрутка вниз
                    scrollbar = self.text_edit.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
        except FileNotFoundError:
            pass

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Защищенный чат")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Кнопки для запуска клиента и сервера
        client_btn = QPushButton("Запустить клиент")
        client_btn.clicked.connect(self.start_client)
        layout.addWidget(client_btn)
        
        server_btn = QPushButton("Запустить сервер")
        server_btn.clicked.connect(self.start_server)
        layout.addWidget(server_btn)
        
        # Добавление мониторов
        self.log_viewer = LogViewer(self)
        self.db_viewer = DatabaseViewer(self)
        
        # Добавление док-виджетов
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_viewer)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.db_viewer)
        
        # Разрешение перемещения док-виджетов
        self.setDockNestingEnabled(True)
        
    def start_client(self):
        """Запуск клиентского окна"""
        self.client = Client()
        self.client_window = ClientMainWindow(self.client)
        self.client_window.show()
        
    def start_server(self):
        """Запуск серверного окна"""
        self.server = ClientO()
        self.server_window = ServerMainWindow(self.server)
        self.server_window.show()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
