import sys
import sqlite3
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTableWidget, QTableWidgetItem, QPushButton)
from PyQt5.QtCore import QTimer
from logger import CustomLogger

class DatabaseViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = CustomLogger("db_viewer")
        self.init_ui()
        
        # Таймер для обновления данных
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_table)
        self.timer.start(1000)  # Обновление каждую секунду
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Просмотр базы данных (Реальное время)')
        self.setGeometry(300, 300, 800, 400)
        
        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Создание таблицы
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Логин', 'Пароль (Хэш)', 'Слово-вызов', 'Время действия'])
        layout.addWidget(self.table)
        
        # Кнопка обновления
        refresh_button = QPushButton('Обновить')
        refresh_button.clicked.connect(self.update_table)
        layout.addWidget(refresh_button)
        
        # Первоначальное заполнение таблицы
        self.update_table()
        
    def update_table(self):
        """Обновление данных таблицы"""
        try:
            # Подключение к базе данных
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # Получение данных
            cursor.execute('SELECT * FROM users')
            data = cursor.fetchall()
            
            # Обновление таблицы
            self.table.setRowCount(len(data))
            for row, record in enumerate(data):
                for col, value in enumerate(record):
                    item = QTableWidgetItem(str(value) if value else '')
                    self.table.setItem(row, col, item)
                    
            # Автоматическая подгонка размеров столбцов
            self.table.resizeColumnsToContents()
            
            conn.close()
            self.logger.info("Таблица обновлена")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении таблицы: {str(e)}")
            
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DatabaseViewer()
    viewer.show()
    sys.exit(app.exec_())
