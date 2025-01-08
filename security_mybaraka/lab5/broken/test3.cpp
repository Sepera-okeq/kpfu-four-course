#include <iostream>
#include <cmath>
using namespace std;

// НОД
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Быстрое возведение в степень по модулю
long long mod_pow(long long base, long long exp, long long modulus) {
    long long result = 1;
    base = base % modulus;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % modulus;
        base = (base * base) % modulus;
        exp >>= 1;
    }
    return result;
}

// Мультипликативное обратное
long long mod_inverse(long long a, long long m) {
    long long m0 = m;
    long long y = 0, x = 1;
    
    if (m == 1) return 0;
    
    while (a > 1) {
        long long q = a / m;
        long long t = m;
        m = a % m;
        a = t;
        t = y;
        y = x - q * y;
        x = t;
    }
    
    if (x < 0)
        x += m0;
    
    return x;
}

// Оптимизированный алгоритм Полларда (p-1)
long long pollard_pm1(long long n) {
    long long base = 2;
    long long exponent = 2;
    long long max_iter = 100;  // Ограничиваем количество итераций для маленьких чисел
    
    cout << "Starting factorization..." << endl;
    
    for (long long i = 0; i < max_iter; i++) {
        base = mod_pow(base, exponent, n);
        long long d = gcd(base - 1, n);
        
        if (d > 1 && d < n) {
            cout << "Found factor after " << i << " iterations" << endl;
            return d;
        }
        
        exponent++;
    }
    
    return 1;
}

int main() {
    // Тестовые значения
    long long N = 493;  // 11 * 17
    long long e = 45;
    long long SW = 56;
    
    cout << "Testing with:" << endl;
    cout << "N = " << N << endl;
    cout << "e = " << e << endl;
    cout << "SW = " << SW << endl;
    
    // Факторизация
    long long p = pollard_pm1(N);
    if (p == 1) {
        cout << "Failed to factor N" << endl;
        return 1;
    }
    
    long long q = N / p;
    cout << "Factorization successful:" << endl;
    cout << "p = " << p << endl;
    cout << "q = " << q << endl;
    
    // Вычисление закрытого ключа
    long long phi = (p - 1) * (q - 1);
    long long d = mod_inverse(e, phi);
    
    // Расшифровка
    long long decrypted = mod_pow(SW, d, N);
    
    cout << "Private key d = " << d << endl;
    cout << "Decrypted message = " << decrypted << endl;
    
    return 0;
}
