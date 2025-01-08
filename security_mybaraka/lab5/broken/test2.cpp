#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
using namespace std;

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

// НОД
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Функция для одного потока
void pollard_thread(long long n, long long start_base, atomic<bool>& found, long long& result) {
    long long base = start_base;
    int itr = 2;
    
    while (!found) {
        base = mod_pow(base, itr, n);
        long long d = gcd(abs(base - 1), n);
        
        if (d > 1 && d < n) {
            if (!found) {
                found = true;
                result = d;
                cout << "Thread with base " << start_base << " found factor " << d << endl;
            }
            break;
        }
        
        if (itr % 1000 == 0) {
            cout << "Base " << start_base << ": iteration " << itr << endl;
        }
        
        itr++;
    }
}

// Параллельный алгоритм Полларда (p-1)
long long parallel_pollard_pm1(long long n) {
    atomic<bool> found(false);
    long long result = 1;
    vector<thread> threads;
    vector<long long> bases = {2, 3, 5, 7, 11, 13, 17, 19};
    
    cout << "Starting factorization with " << bases.size() << " threads..." << endl;
    
    // Запускаем потоки с разными базами
    for (long long base : bases) {
        threads.emplace_back(pollard_thread, n, base, ref(found), ref(result));
    }
    
    // Ждем завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
    
    return result;
}

int main() {
    // Тестовые данные
    long long N = 187; // 11 * 17
    cout << "Testing with N = " << N << endl;
    
    auto start = chrono::high_resolution_clock::now();
    long long p = parallel_pollard_pm1(N);
    auto end = chrono::high_resolution_clock::now();
    
    long long q = N / p;
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "\nFactorization results:" << endl;
    cout << "p = " << p << endl;
    cout << "q = " << q << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;
    
    return 0;
}
