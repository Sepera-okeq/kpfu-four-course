import random
import hashlib
from logger import CustomLogger

class CryptoUtils:
    def __init__(self):
        self.logger = CustomLogger("crypto")
        
    def generate_sw(self):
        """Генерация 128-битного слова-вызова"""
        try:
            sw = ''.join(random.choice('0123456789abcdef') for _ in range(32))
            self.logger.info("Сгенерировано новое слово-вызов")
            return sw
        except Exception as e:
            self.logger.error(f"Ошибка при генерации слова-вызова: {str(e)}")
            return None

    def hash_password(self, password):
        """Хэширование пароля"""
        try:
            return hashlib.sha256(password.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Ошибка при хэшировании пароля: {str(e)}")
            return None

    def hash_sw(self, sw):
        """Хэширование слова-вызова"""
        try:
            return hashlib.sha256(sw.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Ошибка при хэшировании слова-вызова: {str(e)}")
            return None

    def jacobi(self, a, n):
        """Вычисление символа Якоби"""
        if n <= 0 or n % 2 == 0:
            return 0
        
        j = 1
        while a != 0:
            while a % 2 == 0:
                a //= 2
                if n % 8 in [3, 5]:
                    j = -j
            a, n = n, a
            if a % 4 == 3 and n % 4 == 3:
                j = -j
            a %= n
        return j if n == 1 else 0

    def solovay_strassen(self, n, k=10):
        """Тест простоты Соловея-Штрассена"""
        if n == 2:
            return True
        if n < 2 or n % 2 == 0:
            return False

        for _ in range(k):
            a = random.randrange(2, n)
            x = self.jacobi(a, n)
            if x == 0:
                return False
            y = pow(a, (n-1)//2, n)
            if y != x % n:
                return False
        return True

    def generate_prime(self, bits=512):
        """Генерация простого числа заданной длины"""
        self.logger.info(f"Генерация простого числа длиной {bits} бит")
        while True:
            n = random.getrandbits(bits) | (1 << bits - 1) | 1
            if self.solovay_strassen(n):
                self.logger.info("Простое число успешно сгенерировано")
                return n

    def rc4_init(self, key):
        """Инициализация состояния RC4"""
        S = list(range(256))
        j = 0
        for i in range(256):
            j = (j + S[i] + key[i % len(key)]) % 256
            S[i], S[j] = S[j], S[i]
        return S

    def rc4_encrypt(self, key, plaintext):
        """Шифрование/дешифрование RC4"""
        try:
            if isinstance(key, str):
                key = [ord(c) for c in key]
            if isinstance(plaintext, str):
                plaintext = [ord(c) for c in plaintext]
                
            S = self.rc4_init(key)
            i = j = 0
            ciphertext = []
            
            for byte in plaintext:
                i = (i + 1) % 256
                j = (j + S[i]) % 256
                S[i], S[j] = S[j], S[i]
                k = S[(S[i] + S[j]) % 256]
                ciphertext.append(byte ^ k)
                
            self.logger.info("RC4 шифрование/дешифрование выполнено успешно")
            return bytes(ciphertext)
        except Exception as e:
            self.logger.error(f"Ошибка в RC4: {str(e)}")
            return None

    def generate_rsa_keys(self, bits=512):
        """Генерация ключей RSA"""
        try:
            self.logger.info("Начало генерации ключей RSA")
            
            # Генерация p и q
            p = self.generate_prime(bits)
            q = self.generate_prime(bits)
            n = p * q
            phi = (p - 1) * (q - 1)
            
            # Выбор открытой экспоненты e
            e = 65537  # Стандартное значение
            while True:
                if self.gcd(e, phi) == 1:
                    break
                e += 2
                
            # Вычисление закрытой экспоненты d
            d = self.mod_inverse(e, phi)
            
            self.logger.info("Ключи RSA успешно сгенерированы")
            return ((e, n), (d, n))  # ((public), (private))
        except Exception as ex:
            self.logger.error(f"Ошибка при генерации ключей RSA: {str(ex)}")
            return None

    def gcd(self, a, b):
        """Наибольший общий делитель"""
        while b:
            a, b = b, a % b
        return a

    def mod_inverse(self, a, m):
        """Вычисление модульного обратного"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        _, x, _ = extended_gcd(a, m)
        return (x % m + m) % m

    def rsa_encrypt(self, message, public_key):
        """Шифрование RSA"""
        try:
            e, n = public_key
            if isinstance(message, str):
                message = int.from_bytes(message.encode(), 'big')
            cipher = pow(message, e, n)
            self.logger.info("RSA шифрование выполнено успешно")
            return cipher
        except Exception as e:
            self.logger.error(f"Ошибка при шифровании RSA: {str(e)}")
            return None

    def rsa_decrypt(self, cipher, private_key):
        """Дешифрование RSA"""
        try:
            d, n = private_key
            message = pow(cipher, d, n)
            self.logger.info("RSA дешифрование выполнено успешно")
            return message
        except Exception as e:
            self.logger.error(f"Ошибка при дешифровании RSA: {str(e)}")
            return None
