#include <iostream>
#include <cmath>
using namespace std;

// Функция для нахождения НОД
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

// Реализация алгоритма Полларда p-1
long long pollard_pm1(long long n) {
    long long base = 2;
    int itr = 2;
    cout << "Starting factorization of " << n << endl;

    while (true) {
        base = mod_pow(base, itr, n);
        long long d = gcd(base - 1, n);
        
        if (d > 1 && d < n) {
            cout << "Found factor after " << itr << " iterations" << endl;
            return d;
        }

        if (itr % 100 == 0) {
            cout << "Iteration " << itr << ", base = " << base << endl;
        }

        itr++;
    }
}

int main() {
    // Тестовые данные
    long long N = 187; // 11 * 17
    cout << "Testing with N = " << N << endl;

    long long p = pollard_pm1(N);
    long long q = N / p;

    cout << "Factorization result:" << endl;
    cout << "p = " << p << endl;
    cout << "q = " << q << endl;

    return 0;
}
