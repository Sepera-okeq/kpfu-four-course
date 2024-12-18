import logging
import os
from datetime import datetime

class CustomLogger:
    def __init__(self, name, log_dir="logs"):
        self.name = name
        
        # Создаем директорию для логов если её нет
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Создаем форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Настраиваем файловый обработчик
        file_handler = logging.FileHandler(
            f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Настраиваем консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Создаем и настраиваем логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.error(message)
        
    def critical(self, message):
        self.logger.critical(message)
