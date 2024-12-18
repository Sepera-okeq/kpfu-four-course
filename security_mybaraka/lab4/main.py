import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QDockWidget, QMessageBox)
from PyQt5.QtCore import Qt
from client1 import Client1
from client2 import Client2
from db_viewer import DatabaseViewer
from logger import CustomLogger

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = CustomLogger("main")
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Защищенный чат - Все компоненты')
        self.setGeometry(100, 100, 1600, 900)
        
        try:
            # Создание док-виджета для Клиента 1 (Сервер)
            self.client1 = ModifiedClient1()
            dock1 = QDockWidget("Клиент 1 (Сервер)", self)
            dock1.setWidget(self.client1)
            dock1.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, dock1)
            self.logger.info("Клиент 1 инициализирован")
            
            # Создание док-виджета для Клиента 2
            self.client2 = ModifiedClient2()
            dock2 = QDockWidget("Клиент 2", self)
            dock2.setWidget(self.client2)
            dock2.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.RightDockWidgetArea, dock2)
            self.logger.info("Клиент 2 инициализирован")
            
            # Создание док-виджета для просмотра базы данных
            self.db_viewer = ModifiedDatabaseViewer()
            dock3 = QDockWidget("База данных", self)
            dock3.setWidget(self.db_viewer)
            dock3.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.BottomDockWidgetArea, dock3)
            self.logger.info("Просмотрщик БД инициализирован")
            
            # Разрешаем перемещение док-виджетов
            self.setDockOptions(
                QMainWindow.AnimatedDocks |
                QMainWindow.AllowNestedDocks |
                QMainWindow.AllowTabbedDocks
            )
            
            # Сохраняем ссылки на док-виджеты
            self.docks = [dock1, dock2, dock3]
            
            # Устанавливаем начальные размеры док-виджетов
            for dock in self.docks:
                dock.setMinimumWidth(400)
                dock.setMinimumHeight(300)
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации компонентов: {str(e)}")
            QMessageBox.critical(self, "Ошибка", 
                               f"Ошибка при инициализации компонентов: {str(e)}")
            
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        try:
            # Закрытие всех компонентов
            self.client1.close()
            self.client2.close()
            self.db_viewer.close()
            self.logger.info("Все компоненты успешно закрыты")
            event.accept()
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии компонентов: {str(e)}")
            event.accept()

class ModifiedClient1(Client1):
    """Модифицированная версия Client1 для работы в док-виджете"""
    def __init__(self):
        super().__init__()
        # Отключаем создание отдельного окна
        self.setWindowFlags(Qt.Widget)
        
    def closeEvent(self, event):
        """Переопределяем закрытие, чтобы не закрывалось всё приложение"""
        event.ignore()
        
class ModifiedClient2(Client2):
    """Модифицированная версия Client2 для работы в док-виджете"""
    def __init__(self):
        super().__init__()
        # Отключаем создание отдельного окна
        self.setWindowFlags(Qt.Widget)
        
    def closeEvent(self, event):
        """Переопределяем закрытие, чтобы не закрывалось всё приложение"""
        event.ignore()
        
class ModifiedDatabaseViewer(DatabaseViewer):
    """Модифицированная версия DatabaseViewer для работы в док-виджете"""
    def __init__(self):
        super().__init__()
        # Отключаем создание отдельного окна
        self.setWindowFlags(Qt.Widget)
        
    def closeEvent(self, event):
        """Переопределяем закрытие, чтобы не закрывалось всё приложение"""
        event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Устанавливаем стиль для лучшего отображения док-виджетов
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
