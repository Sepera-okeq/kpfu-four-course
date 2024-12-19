"""
Модуль для работы с базой данных пользователей.

Реализует функционал хранения и управления учетными записями:
1. Структура БД:
   - users таблица:
     * id: Уникальный идентификатор
     * login: Логин пользователя
     * password: Хеш пароля MD5
     * sw: Значение SW для аутентификации
     * time: Время последней аутентификации

2. Основные операции:
   - Создание и инициализация БД
   - Добавление новых пользователей
   - Поиск пользователей по логину
   - Обновление SW и времени аутентификации
   - Получение списка всех пользователей

3. Безопасность:
   - Автоматическое хеширование паролей
   - Контроль уникальности логинов
   - Безопасное хранение SW
   - Управление временем жизни сессий

Использование:
    db = UserDatabase('users.db')
    
    # Регистрация
    db.add_user('login', 'password')
    
    # Поиск
    user = db.find_user('login')
    
    # Обновление SW
    db.update_user_auth('login', 'new_sw', datetime.now())
"""

import sqlite3
from datetime import datetime
from typing import Optional, Tuple, List
from . import hash_md5, Logger

logger = Logger("server")

class UserDatabase:
    """
    Класс для работы с базой данных пользователей.
    
    Атрибуты:
        db_name (str): Путь к файлу базы данных
        conn: Соединение с базой данных
        cursor: Курсор для выполнения запросов
        
    Методы:
        add_user: Добавление нового пользователя
        find_user: Поиск пользователя по логину
        get_sw: Получение значения SW
        get_time: Получение времени регистрации
        update_user_auth: Обновление данных аутентификации
        get_all_users: Получение списка всех пользователей
    """
    
    def __init__(self, db_name: str):
        """
        Инициализация базы данных.
        
        Args:
            db_name: Путь к файлу базы данных
        """
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        with self:
            self._create_table()

    def __enter__(self):
        """Открытие соединения при входе в контекст"""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие соединения при выходе из контекста"""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            self.conn.close()
            self.conn = None
            self.cursor = None

    def _create_table(self):
        """Создание таблицы users если она не существует"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                sw TEXT,
                time TIMESTAMP
            )
        ''')
        self.conn.commit()

    def add_user(self, login: str, password: str) -> bool:
        """
        Добавление нового пользователя.
        
        Args:
            login: Логин пользователя
            password: Пароль пользователя
            
        Returns:
            bool: True если пользователь успешно добавлен
        """
        try:
            with self as db:
                db.cursor.execute('SELECT EXISTS(SELECT 1 FROM users WHERE login = ?)', (login,))
                exists = db.cursor.fetchone()[0]
                
                if exists:
                    return False
                
                hashed_password = hash_md5(password)
                db.cursor.execute('''
                    INSERT INTO users (login, password, sw, time)
                    VALUES (?, ?, ?, ?)
                ''', (login, hashed_password, "", ""))
                logger.info(f"Пользователь {login} зарегистрирован")
                logger.info(f"Хеш пароля: {hashed_password}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Ошибка добавления пользователя: {e}")
            return False

    def find_user(self, login: str) -> Optional[Tuple]:
        """
        Поиск пользователя по логину.
        
        Args:
            login: Логин пользователя
            
        Returns:
            Optional[Tuple]: Данные пользователя или None
        """
        with self as db:
            db.cursor.execute('SELECT * FROM users WHERE login = ?', (login,))
            return db.cursor.fetchone()

    def get_sw(self, login: str) -> Optional[str]:
        """
        Получение значения SW пользователя.
        
        Args:
            login: Логин пользователя
            
        Returns:
            Optional[str]: Значение SW или None
        """
        with self as db:
            db.cursor.execute('SELECT sw FROM users WHERE login = ?', (login,))
            result = db.cursor.fetchone()
            return result[0] if result else None

    def get_time(self, login: str) -> Optional[datetime]:
        """
        Получение времени регистрации пользователя.
        
        Args:
            login: Логин пользователя
            
        Returns:
            Optional[datetime]: Время регистрации или None
        """
        with self as db:
            db.cursor.execute('SELECT time FROM users WHERE login = ?', (login,))
            result = db.cursor.fetchone()
            return datetime.fromisoformat(result[0]) if result and result[0] else None
    
    def update_user_auth(self, login: str, sw: str, time: datetime) -> bool:
        """
        Обновление SW и времени для пользователя.
        
        Args:
            login: Логин пользователя
            sw: Новое значение SW
            time: Новое время
            
        Returns:
            bool: True если обновление успешно
        """
        try:
            with self:
                self.cursor.execute('''
                    UPDATE users 
                    SET sw = ?, time = ?
                    WHERE login = ?
                ''', (sw, time, login))
                logger.info(f"Обновлены данные пользователя {login}")
                logger.info(f"SW: {sw}")
                logger.info(f"Время: {time}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления данных пользователя: {e}")
            return False
            
    def get_all_users(self) -> List[Tuple]:
        """
        Получение списка всех пользователей.
        
        Returns:
            List[Tuple]: Список кортежей с данными пользователей
        """
        try:
            with self as db:
                db.cursor.execute('SELECT * FROM users')
                return db.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения списка пользователей: {e}")
            return []
