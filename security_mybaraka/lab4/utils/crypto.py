"""
Криптографические алгоритмы для защищенного чата.

Модуль реализует основные криптографические операции:
1. RC4 - потоковое шифрование сообщений:
   - Инициализация S-блока на основе ключа
   - Генерация псевдослучайного потока
   - Шифрование/дешифрование XOR

2. RSA - асимметричное шифрование:
   - Генерация ключевой пары (открытый/закрытый)
   - Тест Миллера-Рабина на простоту
   - Расширенный алгоритм Евклида
   - Модульная арифметика

Использование:
1. RC4:
   rc4 = RC4("ключ")
   encrypted = rc4.encrypt("сообщение")
   decrypted = rc4.decrypt(encrypted)

2. RSA:
   (e, n), d = RSA.generate_keys()
   # e, n - открытый ключ
   # d - закрытый ключ
"""

import random
from .logger import Logger
from .helpers import mod_exp

logger = Logger()

class RC4:
    """
    Реализация потокового шифра RC4.
    RC4 генерирует псевдослучайный поток байтов и комбинирует его с открытым текстом операцией XOR.
    """
    def __init__(self, key: str):
        """
        Инициализация RC4 с ключом.
        
        Args:
            key (str): Ключ шифрования для инициализации S-блока
            
        Note:
            RC4 использует ключ для создания начальной перестановки в S-блоке,
            которая затем используется для генерации псевдослучайного потока.
            Этот поток XOR'ится с данными для шифрования/дешифрования.
        """
        logger.info(f"Инициализация RC4 с ключом: {key}")
        self.key = [ord(c) for c in key]
        self.S = list(range(256))
        self.init_sbox()
        
    def init_sbox(self):
        """
        Инициализация S-блока (Key Scheduling Algorithm - KSA).
        Перемешивает начальную перестановку на основе ключа.
        
        Note:
            1. Создает начальную перестановку 0-255 в S-блоке
            2. Использует ключ для перемешивания этой перестановки
            3. Результат используется в PRGA для генерации потока
            
        Безопасность:
            - Качество перемешивания влияет на криптостойкость
            - Длина ключа определяет размер пространства ключей
        """
        j = 0
        for i in range(256):
            j = (j + self.S[i] + self.key[i % len(self.key)]) % 256
            self.S[i], self.S[j] = self.S[j], self.S[i]
            
    def encrypt(self, data: str) -> str:
        """
        Шифрование/дешифрование данных (Pseudo-Random Generation Algorithm - PRGA).
        
        Args:
            data (str): Входные данные для шифрования/дешифрования
            
        Returns:
            str: Зашифрованный/расшифрованный текст
            
        Note:
            1. Генерирует псевдослучайный поток на основе состояния S-блока
            2. Каждый байт потока XOR'ится с байтом входных данных
            3. Процесс одинаков для шифрования и дешифрования (свойство XOR)
            
        Безопасность:
            - Каждый байт шифруется независимо
            - Требуется синхронизация состояний на обеих сторонах
            - Уязвим к атакам при повторном использовании ключа
        """
        S = self.S.copy()
        i = j = 0
        result = []
        
        bytes_data = [ord(c) for c in data]
        
        for byte in bytes_data:
            i = (i + 1) % 256
            j = (j + S[i]) % 256
            S[i], S[j] = S[j], S[i]
            k = S[(S[i] + S[j]) % 256]
            encrypted_byte = byte ^ k
            result.append(chr(encrypted_byte))
            
        return ''.join(result)
    
    # Дешифрование идентично шифрованию из-за свойств XOR
    decrypt = encrypt

class RSA:
    """Реализация алгоритма RSA для шифрования и генерации ключей."""
    
    @staticmethod
    def miller_rabin(n: int, k: int = 40) -> bool:
        """
        Тест Миллера-Рабина на простоту числа.
        
        Args:
            n (int): Тестируемое число
            k (int): Количество раундов тестирования
            
        Returns:
            bool: True если число вероятно простое, False если составное
            
        Note:
            1. Вероятностный тест на простоту
            2. Представляет n-1 как d*2^s
            3. Проверяет свойства простых чисел для k случайных оснований
            
        Безопасность:
            - Вероятность ошибки <= 4^(-k)
            - k=40 даёт очень высокую надежность
            - Используется для генерации простых чисел в RSA
        """
        if n == 2:
            return True
        if not n & 1 or n < 2:
            return False

        s = 0
        d = n - 1
        while not d & 1:
            s += 1
            d >>= 1

        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = mod_exp(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
                
            for _ in range(s - 1):
                x = mod_exp(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
                
        return True

    @staticmethod
    def generate_prime(bits: int) -> int:
        """
        Генерация простого числа заданной битовой длины.
        
        Args:
            bits (int): Требуемая длина числа в битах
            
        Returns:
            int: Простое число длиной bits бит
            
        Note:
            1. Генерирует случайное число нужной длины
            2. Устанавливает старший бит для гарантии длины
            3. Проверяет на простоту тестом Миллера-Рабина
            
        Безопасность:
            - Генерирует криптографически стойкие простые числа
            - Важно для безопасности RSA
            - Длина определяет сложность факторизации
        """
        while True:
            p = random.getrandbits(bits) | (1 << bits - 1) | 1
            if RSA.miller_rabin(p):
                return p

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Наибольший общий делитель.
        
        Args:
            a, b (int): Числа для поиска НОД
            
        Returns:
            int: НОД чисел a и b
        """
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        """
        Расширенный алгоритм Евклида.
        
        Args:
            a, b (int): Целые числа
            
        Returns:
            tuple[int, int, int]: (НОД, x, y), где ax + by = НОД(a,b)
        """
        if a == 0:
            return b, 0, 1
            
        gcd, x1, y1 = RSA.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y

    @staticmethod
    def modinv(a: int, m: int) -> int:
        """
        Нахождение мультипликативного обратного по модулю.
        
        Args:
            a (int): Число, для которого ищем обратное
            m (int): Модуль
            
        Returns:
            int: Мультипликативно обратное число
            
        Raises:
            ValueError: Если обратное не существует
        """
        gcd, x, _ = RSA.extended_gcd(a, m)
        
        if gcd != 1:
            raise ValueError("Мультипликативное обратное не существует")
            
        return x % m

    @staticmethod
    def generate_keys(bits: int = 512) -> tuple[tuple[int, int], int]:
        """
        Генерация пары ключей RSA.
        
        Args:
            bits (int): Длина ключа в битах
            
        Returns:
            tuple[tuple[int, int], int]: ((e, n), d) - открытый и закрытый ключи
            
        Note:
            1. Генерирует два простых числа p и q
            2. Вычисляет n = p*q и φ(n) = (p-1)(q-1)
            3. Выбирает открытую экспоненту e
            4. Вычисляет закрытую экспоненту d
            
        Безопасность:
            - Длина ключа определяет стойкость к факторизации
            - e выбирается случайно для доп. безопасности
            - Закрытый ключ d вычисляется через расш. алг. Евклида
        """
        p = RSA.generate_prime(bits)
        q = RSA.generate_prime(bits)
        
        n = p * q
        phi = (p - 1) * (q - 1)
        
        e_bits = bits // 3
        max_e = min(phi, 1 << e_bits)
        
        e = random.randint(1, max_e)
        while RSA.gcd(e, phi) != 1:
            e = random.randint(1, max_e)
        
        d = RSA.modinv(e, phi)
        
        return (e, n), d
