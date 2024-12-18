import sqlite3
import hashlib
from datetime import datetime, timedelta
from logger import CustomLogger

class Database:
    def __init__(self):
        self.logger = CustomLogger("database")
        self.conn = sqlite3.connect('users.db')
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    def create_tables(self):
        """Создание таблицы пользователей если она не существует"""
        self.logger.info("Инициализация базы данных")
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    login TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    sw TEXT,
                    t TIMESTAMP
                )
            ''')
            self.conn.commit()
            self.logger.info("Таблица users успешно создана или уже существует")
        except Exception as e:
            self.logger.error(f"Ошибка при создании таблицы: {str(e)}")
            
    def add_user(self, login, password):
        """Добавление нового пользователя"""
        try:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute('INSERT INTO users (login, password) VALUES (?, ?)',
                              (login, hashed_password))
            self.conn.commit()
            self.logger.info(f"Пользователь {login} успешно добавлен")
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"Пользователь {login} уже существует")
            return False
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении пользователя: {str(e)}")
            return False
            
    def check_login(self, login):
        """Проверка существования логина"""
        try:
            self.cursor.execute('SELECT login FROM users WHERE login = ?', (login,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Ошибка при проверке логина: {str(e)}")
            return False
            
    def update_sw(self, login, sw):
        """Обновление слова-вызова и времени его действия"""
        try:
            t = datetime.now() + timedelta(hours=24)
            self.cursor.execute('UPDATE users SET sw = ?, t = ? WHERE login = ?',
                              (sw, t, login))
            self.conn.commit()
            self.logger.info(f"Обновлено слово-вызов для пользователя {login}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении слова-вызова: {str(e)}")
            return False
            
    def get_user_data(self, login):
        """Получение данных пользователя"""
        try:
            self.cursor.execute('SELECT * FROM users WHERE login = ?', (login,))
            return self.cursor.fetchone()
        except Exception as e:
            self.logger.error(f"Ошибка при получении данных пользователя: {str(e)}")
            return None
            
    def close(self):
        """Закрытие соединения с базой данных"""
        self.conn.close()
        self.logger.info("Соединение с базой данных закрыто")
