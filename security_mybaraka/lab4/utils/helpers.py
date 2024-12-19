"""
Вспомогательные функции для защищенного чата.

Модуль содержит набор утилит для:
1. Криптографических операций:
   - MD5 хеширование
   - Генерация случайных чисел
   - Модульная арифметика
   - Тесты на простоту

2. Валидации данных:
   - Проверка логинов (длина, символы)
   - Проверка паролей (сложность)
   - Генерация SW значений

3. Математические функции:
   - Символ Якоби
   - Тест Соловея-Штрассена
   - Генерация простых чисел
   - Числа Софи Жермен

Использование:
    # Хеширование
    hash = hash_md5("данные")
    
    # Валидация
    valid, msg = validate_credentials("login", "pass")
    
    # Генерация чисел
    sw = generate_sw()
    prime = generate_prime_512bit()
"""

import hashlib
import secrets
import random
import re
from . import Logger

logger = Logger("server")

def hash_md5(data: str) -> str:
    """
    MD5 хеширование строки.
    
    Args:
        data (str): Входная строка для хеширования
        
    Returns:
        str: MD5 хеш в шестнадцатеричном представлении
    """
    return hashlib.md5(data.encode()).hexdigest()

def validate_credentials(login: str, password: str) -> tuple[bool, str]:
    """
    Проверка валидности логина и пароля.
    
    Args:
        login (str): Логин для проверки
        password (str): Пароль для проверки
        
    Returns:
        tuple[bool, str]: (Валидны ли данные, Сообщение об ошибке)
    """
    if len(login) < 5:
        return False, "Логин должен содержать минимум 5 символов"
    if len(login) > 30:
        return False, "Логин не должен превышать 30 символов"
    if not re.match(r'^[a-zA-Z0-9]+$', login):
        return False, "Логин должен содержать только латинские буквы и цифры"
    
    if len(password) < 5:
        return False, "Пароль должен содержать минимум 5 символов"
    if len(password) > 30:
        return False, "Пароль не должен превышать 30 символов"
    if not re.match(r'^[a-zA-Z0-9?!&]+$', password):
        return False, "Пароль должен содержать только латинские буквы, цифры и символы ?!&"
    if not re.search(r'[A-Z]', password):
        return False, "Пароль должен содержать хотя бы одну заглавную букву"
    if not re.search(r'[0-9]', password):
        return False, "Пароль должен содержать хотя бы одну цифру"
    
    return True, ""

def generate_sw() -> str:
    """
    Генерация случайной 128-битной строки.
    
    Returns:
        str: Случайная строка из 32 шестнадцатеричных символов
    """
    return secrets.token_hex(16)

def generate_odd_64bit() -> int:
    """
    Генерация 64-битного нечетного числа.
    
    Returns:
        int: Случайное нечетное 64-битное число
    """
    num = random.getrandbits(64)
    if num % 2 == 0:
        num += 1
    return num

def mod_exp(base: int, exp: int, mod: int) -> int:
    """
    Быстрое возведение в степень по модулю.
    
    Args:
        base (int): Основание
        exp (int): Показатель степени 
        mod (int): Модуль
        
    Returns:
        int: Результат возведения в степень по модулю
    """
    if mod == 1:
        return 0
    
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
        
    return result

def jacobi_symbol(a: int, n: int) -> int:
    """
    Вычисление символа Якоби (a/n).
    
    Args:
        a (int): Верхнее число
        n (int): Нижнее число (должно быть нечетным положительным)
    
    Returns:
        int: Значение символа Якоби (-1, 0 или 1)
    """
    if a == 0:
        return 0
    if a == 1:
        return 1
    
    if a < 0:
        return (-1) ** ((n-1)//2) * jacobi_symbol(-a, n)
    
    if a % 2 == 0:
        return (-1) ** ((n*n-1)//8) * jacobi_symbol(a//2, n)
    
    if a >= n:
        return jacobi_symbol(a % n, n)
    
    return (-1) ** ((a-1)*(n-1)//4) * jacobi_symbol(n % a, a)

def solovay_strassen_test(n: int, k: int = 10) -> bool:
    """
    Вероятностный тест на простоту числа методом Соловея-Штрассена.
    
    Args:
        n (int): Тестируемое число
        k (int): Количество раундов тестирования
    
    Returns:
        bool: True если число вероятно простое, False если составное
    """
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    
    for _ in range(k):
        a = random.randrange(2, n)
        x = jacobi_symbol(a, n)
        
        if x == 0 or mod_exp(a, (n-1)//2, n) != (x % n):
            return False
            
    return True

def generate_prime_512bit() -> int:
    """
    Генерация 512-битного простого числа методом Соловея-Штрассена.
    
    Returns:
        int: 512-битное простое число
    """
    while True:
        num = random.getrandbits(512) | 1
        
        if num < 2**511:
            num |= (1 << 511)
            
        if solovay_strassen_test(num):
            return num

def is_sophie_germain_prime(p: int) -> bool:
    """
    Проверка является ли число простым числом Софи Жермен.
    
    Args:
        p (int): Проверяемое число
        
    Returns:
        bool: True если число является числом Софи Жермен, False иначе
    """
    if not solovay_strassen_test(p):
        return False
    return solovay_strassen_test(2 * p + 1)

def generate_sophie_germain_primes(count: int) -> list:
    """
    Генерация заданного количества чисел Софи Жермен.
    
    Args:
        count (int): Требуемое количество чисел
        
    Returns:
        list: Список чисел Софи Жермен
    """
    primes = []
    candidate = 3
    while len(primes) < count:
        if is_sophie_germain_prime(candidate):
            primes.append(candidate)
        candidate += 2
    return primes

def generate_generator() -> int:
    """
    Генерация генератора группы (числа Софи Жермен).
    
    Returns:
        int: Случайное число Софи Жермен
    """
    primes = generate_sophie_germain_primes(10)
    return random.choice(primes)
