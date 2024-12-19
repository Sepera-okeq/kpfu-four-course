from datetime import datetime
from typing import TextIO

class Logger:
    """
    Универсальный логгер для клиента и сервера.
    Реализует паттерн Singleton для обеспечения единственного экземпляра логгера.
    """
    _instance = None
    
    def __new__(cls, name: str = "app"):
        """
        Создание или получение единственного экземпляра логгера.
        
        Args:
            name (str): Имя приложения для логирования ('client' или 'server')
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(name)
        return cls._instance
        
    def initialize(self, name: str):
        """
        Инициализация логгера.
        
        Args:
            name (str): Имя приложения для логирования
        """
        self.log_file: TextIO = open(f'{name}.log', 'a', encoding='utf-8')
        
    def _write_log(self, level: str, message: str):
        """
        Запись лога в файл и вывод в консоль.
        
        Args:
            level (str): Уровень логирования (INFO, ERROR, WARNING)
            message (str): Сообщение для логирования
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # Вывод в консоль
        print(log_message)
        
        # Запись в файл
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
        
    def info(self, message: str):
        """Информационное сообщение."""
        self._write_log('INFO', message)
        
    def error(self, message: str):
        """Сообщение об ошибке."""
        self._write_log('ERROR', message)
        
    def warning(self, message: str):
        """Предупреждение."""
        self._write_log('WARNING', message)
        
    def __del__(self):
        """Закрытие файла при удалении объекта."""
        if hasattr(self, 'log_file'):
            self.log_file.close()

# Пример использования:
# client_logger = Logger("client")
# client_logger.info("Клиент запущен")
# 
# server_logger = Logger("server")
# server_logger.info("Сервер запущен")
