#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <algorithm>
#include <omp.h>

using namespace std;

// Класс для работы с большими числами
class BigInt {
    string digits;
    bool negative;
public:
    BigInt(const string& num = "0") : negative(false) {
        if (num[0] == '-') {
            negative = true;
            digits = num.substr(1);
        } else {
            digits = num;
        }
        // Удаляем ведущие нули
        while (digits.size() > 1 && digits[0] == '0')
            digits.erase(digits.begin());
    }

    BigInt(long long num) {
        if (num < 0) {
            negative = true;
            num = -num;
        } else {
            negative = false;
        }
        digits = to_string(num);
    }

    // Конструктор копирования
    BigInt(const BigInt& other) : digits(other.digits), negative(other.negative) {}

    BigInt operator-() const {
        BigInt result = *this;
        if (result.digits != "0") {
            result.negative = !result.negative;
        }
        return result;
    }

    // Вспомогательная функция для сложения положительных чисел
    BigInt add_positive(const BigInt& other) const {
        string result;
        int carry = 0;
        int i = digits.length() - 1;
        int j = other.digits.length() - 1;

        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += digits[i--] - '0';
            if (j >= 0) sum += other.digits[j--] - '0';
            carry = sum / 10;
            result.push_back(sum % 10 + '0');
        }

        reverse(result.begin(), result.end());
        BigInt res(result);
        return res;
    }

    // Вспомогательная функция для вычитания положительных чисел (this >= other)
    BigInt subtract_positive(const BigInt& other) const {
        string result;
        int borrow = 0;
        int i = digits.length() - 1;
        int j = other.digits.length() - 1;

        while (i >= 0) {
            int digit = (digits[i] - '0') - borrow;
            if (j >= 0) digit -= (other.digits[j] - '0');
            
            if (digit < 0) {
                digit += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }
            
            result.push_back(digit + '0');
            i--;
            j--;
        }

        reverse(result.begin(), result.end());
        BigInt res(result);
        return res;
    }

    // Сложение
    BigInt operator+(const BigInt& other) const {
        if (negative == other.negative) {
            // Если знаки одинаковые, просто складываем и сохраняем знак
            BigInt result = add_positive(other);
            result.negative = negative;
            return result;
        }
        
        // Если знаки разные, вычитаем меньшее по модулю из большего
        if (digits.length() > other.digits.length() || 
            (digits.length() == other.digits.length() && digits > other.digits)) {
            BigInt result = subtract_positive(other);
            result.negative = negative;
            return result;
        } else {
            BigInt result = other.subtract_positive(*this);
            result.negative = other.negative;
            return result;
        }
    }

    // Вычитание
    BigInt operator-(const BigInt& other) const {
        // a - b = a + (-b)
        return *this + (-other);
    }

    // Умножение
    BigInt operator*(const BigInt& other) const {
        vector<int> result(digits.length() + other.digits.length(), 0);
        
        for (int i = digits.length() - 1; i >= 0; i--) {
            for (int j = other.digits.length() - 1; j >= 0; j--) {
                int product = (digits[i] - '0') * (other.digits[j] - '0');
                int pos1 = i + j;
                int pos2 = i + j + 1;
                
                product += result[pos2];
                result[pos2] = product % 10;
                result[pos1] += product / 10;
            }
        }
        
        string s;
        bool leading_zero = true;
        for (int digit : result) {
            if (digit != 0) leading_zero = false;
            if (!leading_zero) s += to_string(digit);
        }
        if (s.empty()) s = "0";
        
        BigInt res(s);
        // Результат отрицательный, если знаки множителей разные
        res.negative = negative != other.negative && s != "0";
        return res;
    }

    // Деление
    pair<BigInt, BigInt> divmod(const BigInt& other) const {
        if (other == BigInt("0")) throw runtime_error("Division by zero");
        
        // Работаем с положительными числами
        BigInt abs_this(*this);
        BigInt abs_other(other);
        abs_this.negative = false;
        abs_other.negative = false;

        if (abs_this < abs_other) {
            BigInt zero("0");
            return make_pair(zero, BigInt(*this));
        }

        BigInt quotient("0");
        BigInt remainder(abs_this);
        BigInt one("1");

        while (remainder >= abs_other) {
            BigInt temp = remainder - abs_other;
            remainder = temp;
            temp = quotient + one;
            quotient = temp;
        }

        // Устанавливаем правильный знак для частного
        if (negative != other.negative && quotient.digits != "0") {
            quotient.negative = true;
        }
        
        // Остаток имеет тот же знак, что и делимое
        if (negative && remainder.digits != "0") {
            remainder.negative = true;
        }

        return make_pair(quotient, remainder);
    }

    BigInt operator/(const BigInt& other) const {
        return divmod(other).first;
    }

    BigInt operator%(const BigInt& other) const {
        return divmod(other).second;
    }

    // Сравнение
    bool operator<(const BigInt& other) const {
        if (negative != other.negative)
            return negative;
        
        if (negative) {
            if (digits.length() != other.digits.length())
                return digits.length() > other.digits.length();
            return digits > other.digits;
        } else {
            if (digits.length() != other.digits.length())
                return digits.length() < other.digits.length();
            return digits < other.digits;
        }
    }

    bool operator>(const BigInt& other) const {
        return other < *this;
    }

    bool operator>=(const BigInt& other) const {
        return (*this > other) || (*this == other);
    }

    bool operator==(const BigInt& other) const {
        return negative == other.negative && digits == other.digits;
    }

    bool operator!=(const BigInt& other) const {
        return !(*this == other);
    }

    BigInt& operator=(const BigInt& other) {
        if (this != &other) {
            digits = other.digits;
            negative = other.negative;
        }
        return *this;
    }

    string toString() const {
        return digits;
    }

    friend ostream& operator<<(ostream& os, const BigInt& num) {
        if (num.negative && num.digits != "0") {
            os << "-";
        }
        os << num.digits;
        return os;
    }

    bool isNegative() const {
        return negative && digits != "0";
    }
};

// Функция для нахождения НОД
BigInt gcd(BigInt a, BigInt b) {
    while (!(b == BigInt("0"))) {
        BigInt temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Функция для быстрого возведения в степень по модулю
BigInt mod_pow(const BigInt& base, const BigInt& exp, const BigInt& modulus) {
    if (modulus == BigInt("1")) return BigInt("0");
    
    BigInt result("1");
    BigInt b(base % modulus);
    BigInt e(exp);
    BigInt two("2");
    
    while (!(e == BigInt("0"))) {
        BigInt remainder = e % two;
        if (remainder == BigInt("1")) {
            BigInt temp = result * b;
            result = temp % modulus;
        }
        BigInt temp = b * b;
        b = temp % modulus;
        e = e / two;
    }
    return result;
}

// Функция для нахождения мультипликативного обратного
BigInt mod_inverse(const BigInt& a, const BigInt& m) {
    BigInt m0(m);
    BigInt y("0"), x("1");
    
    if (m == BigInt("1")) return BigInt("0");
    
    BigInt a1(a);
    BigInt m1(m);
    
    while (a1 > BigInt("1")) {
        // Сохраняем старые значения
        BigInt old_a1(a1);
        BigInt old_m1(m1);
        BigInt old_y(y);
        
        // Вычисляем частное
        BigInt q = old_a1 / old_m1;
        
        // Обновляем значения
        a1 = old_m1;
        m1 = old_a1 % old_m1;
        
        // y = x - q * y
        BigInt temp = q * y;
        y = x - temp;
        x = old_y;
    }
    
    // Если x отрицательное, добавляем m0
    if (x < BigInt("0")) {
        BigInt temp = x + m0;
        x = temp;
    }
    
    return x;
}

// Функция для декодирования русского текста
string decode_russian_text(BigInt number) {
    vector<int> codes;
    BigInt hundred("100");
    BigInt temp = number;
    
    // Получаем коды символов
    while (!(temp == BigInt("0"))) {
        pair<BigInt, BigInt> division = temp.divmod(hundred);
        string remainder_str = division.second.toString();
        int remainder = 0;
        for (char c : remainder_str) {
            remainder = remainder * 10 + (c - '0');
        }
        codes.push_back(remainder);
        temp = division.first;
    }
    
    // Формируем строку из русских символов
    string result;
    for (auto it = codes.rbegin(); it != codes.rend(); ++it) {
        int code = *it;
        if (code >= 16 && code < 48) {  // Заглавные буквы (А-Я)
            result.push_back(static_cast<char>(code - 16 + 128));
        } else if (code >= 48 && code < 80) {  // Строчные буквы (а-я)
            result.push_back(static_cast<char>(code - 48 + 160));
        }
    }
    
    return result;
}

// Функция для поиска множителя методом Полларда
BigInt pollard_rho(const BigInt& N) {
    BigInt d("1");
    int max_threads = omp_get_max_threads();
    bool found = false;
    
    #pragma omp parallel for shared(found)
    for (int i = 0; i < max_threads; i++) {
        if (!found) {
            BigInt x(2 + i), y(2 + i);
            BigInt local_d("1");
            
            while (local_d == BigInt("1") && !found) {
                x = (x * x + BigInt("1")) % N;
                y = ((y * y + BigInt("1")) * (y * y + BigInt("1")) + BigInt("1")) % N;
                local_d = gcd((x - y >= BigInt("0") ? x - y : y - x), N);
                
                if (local_d != BigInt("1") && local_d != N) {
                    #pragma omp critical
                    {
                        if (!found) {
                            d = local_d;
                            found = true;
                        }
                    }
                }
            }
        }
    }
    
    return d;
}

int main() {
    // Установка кодировки консоли в OEM (DOS)
    system("chcp 866");
    
    // Вариант 69
    BigInt N("75332357154462380976079586039");
    BigInt e("24639182129584606917471570503");
    BigInt SW("26752566085475942776180898092");
    
    cout << "Starting RSA breaking..." << endl;
    cout << "N = " << N << endl;
    cout << "e = " << e << endl;
    cout << "SW = " << SW << endl;
    
    // Начало отсчета времени
    auto start = chrono::high_resolution_clock::now();
    
    cout << "Using " << omp_get_max_threads() << " threads..." << endl;
    
    // Поиск множителей с помощью ро-алгоритма Полларда
    BigInt p = pollard_rho(N);
    
    if (p == BigInt("1") || p == N) {
        cout << "Failed to factor N" << endl;
        return 1;
    }
    BigInt q = N / p;
    
    // Вычисление закрытого ключа
    BigInt phi = (p - BigInt("1")) * (q - BigInt("1"));
    BigInt d_val = mod_inverse(e, phi);
    
    // Расшифровка сообщения
    BigInt decrypted = mod_pow(SW, d_val, N);
    
    // Конец отсчета времени
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // Вывод результатов
    cout << "\nFactorization results:" << endl;
    cout << "p = " << p << endl;
    cout << "q = " << q << endl;
    cout << "Threads used: " << omp_get_max_threads() << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;
    
    cout << "\nPrivate key d = " << d_val << endl;
    cout << "Decrypted number: " << decrypted << endl;
    
    // Декодирование и вывод русского текста
    string decoded_text = decode_russian_text(decrypted);
    cout << "\nDecoded Russian text: " << decoded_text << endl;
    
    return 0;
}
