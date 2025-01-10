#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>
#include <vector>
#include <queue>
#include <stdlib.h>
#include <stdint.h>
#include <gmp.h>
#include <gmpxx.h>
#include "matrix.h"


#define ENABLE_TIMER

#ifdef ENABLE_TIMER
    /*
     * Если ENABLE_TIMER определен, мы создаем часы и два макроса START и
     * STOP. Используйте START() для запуска таймера и STOP("сообщение") для
     * его остановки и вывода времени, прошедшего с момента вызова START, в мс.
     */
    clock_t timer;
    #define START() timer = std::clock();
    #define STOP(msg) \
        std::cout << msg << " за ";\
        std::cout << (1000.0 * (std::clock() - timer)/CLOCKS_PER_SEC);\
        std::cout << " мс" << std::endl;
#else
    // Иначе макросы определяются как пустые операции.
    #define START(x)
    #define STOP(x)
#endif // ENABLE_TIMER

// Минимальная граница гладкости.
const static uint32_t MINIMAL_BOUND = 300;

// Длина интервала просеивания.
const static uint32_t INTERVAL_LENGTH = 65536;

/*
 * Генерирует факторную базу для факторизации N с границей гладкости B.
 *
 * Возвращает вектор простых чисел <= B таких, что N является квадратичным
 * вычетом по модулю p.
 */
std::vector<uint32_t> generateFactorBase(const mpz_class &N, uint32_t B)
{
    std::vector<uint32_t> factorBase;
    /*
     * По сути это решето Эратосфена с дополнительной проверкой
     * условия (N/p) = 1 (символ Лежандра).
     */
    std::vector<char> sieve(B + 1, 0);
    for (uint32_t p = 2; p <= B; ++p)
    {
        if (sieve[p])
            continue;

        // Добавляем p в факторную базу, если N является квадратичным вычетом по модулю p.
        if (mpz_legendre(N.get_mpz_t(), mpz_class(p).get_mpz_t()) == 1)
            factorBase.push_back(p);

        // Добавляем кратные p в решето.
        for (uint32_t i = p; i <= B; i += p)
            sieve[i] = 1;
    }

    return factorBase;
}

/*
 * Возвращает b^e (mod m), используя двоичный метод справа налево.
 */
uint64_t modularPow(uint64_t b, uint64_t e, uint64_t m)
{
    uint64_t result = 1;
    while (e > 0)
    {
            if (e & 1)                     // Для каждого установленного бита в показателе степени.
                result = (result * b) % m; // Умножаем результат на b^2^i.
            e >>= 1;
            b = (b * b) % m; // Возводим основание в квадрат.
    }
    return result;
}

/*
 * Вычисляет символ Лежандра (является ли целое число a квадратичным вычетом по модулю p).
 *
 *   1 если a является квадратичным вычетом по модулю p и a != 0 (mod p)
 *  -1 если a является квадратичным невычетом по модулю p
 *   0 если a = 0 (mod p)
 */
int32_t legendreSymbol(uint32_t a, uint32_t p)
{
    uint64_t result = modularPow(a, (p - 1) / 2, p);
    return result > 1 ? -1 : result;
}

/*
 * Возвращает решения сравнения x^2 = n (mod p).
 *
 * Использует алгоритм Тонелли-Шенкса. p должно быть нечетным простым числом,
 * а n - квадратичным вычетом по модулю p.
 *
 * Алгоритм Тонелли-Шенкса — это алгоритм для нахождения квадратных корней по модулю p, где p — простое число.
 * Т.Е, он решает сравнение x² ≡ n (mod p) для заданных n и p.
 * Он короче работает, только если n является квадратичным вычетом по модулю p (т.е., символ Лежандра (n/p) = 1).
 * 
 * Алгоритм реализован на основе псевдокода из Wikipedia, где он хорошо
 * (хотя и кратко) описан. Алгоритм находит квадратный корень по модулю
 * простого числа.
 */
std::pair<uint32_t, uint32_t> tonelliShanks(uint32_t n, uint32_t p)
{
    if (p == 2)
        return std::make_pair(n, n); // Двойной корень.

    // Определяем Q2^S = p - 1.
    uint64_t Q = p - 1;
    uint64_t S = 0;
    while (Q % 2 == 0)
    {
        Q /= 2;
        ++S;
    }

    // Определяем z как первый квадратичный невычет по модулю p.
    uint64_t z = 2;
    while (legendreSymbol(z, p) != -1)
        ++z;

    // Инициализируем c, R, t и M.
    uint64_t c = modularPow(z, Q, p);           // c = z^Q         (mod p)
    uint64_t R = modularPow(n, (Q + 1) / 2, p); // R = n^((Q+1)/2) (mod p)
    uint64_t t = modularPow(n, Q, p);           // t = n^Q         (mod p)
    uint64_t M = S;

    // Инвариант: R^2 = nt (mod p)
    while (t % p != 1)
    {
        // Находим наименьшее 0 < i < M такое, что t^2^i = 1 (mod p).
        int32_t i = 1;
        while (modularPow(t, std::pow(2, i), p) != 1)
            ++i;

        // Устанавливаем b = c^2^(M - i - 1)
        uint64_t b = modularPow(c, std::pow(2, M - i - 1), p);

        // Обновляем c, R, t и M.
        R = R * b % p;     // R = Rb (mod p)
        t = t * b * b % p; // t = tb^2
        c = b * b % p;     // c = b^2 (mod p)
        M = i;

        // Инвариант: R^2 = nt (mod p)
    }

    return std::make_pair(R, p - R);
}

/*
 * Базовая реализация алгоритма квадратичного решета.
 *
 * Принимает целое число N в качестве входных данных и возвращает его делитель.
 * Алгоритм основан на поиске чисел вида (x + sqrt(N))^2 - N, которые 
 * раскладываются на множители из факторной базы (B-гладкие числа).
 */
mpz_class quadraticSieve(const mpz_class &N)
{
    std::cout << "В данный момент факторизуем " << N << ".\n";

    // Некоторые полезные функции от N.
    const float logN = mpz_sizeinbase(N.get_mpz_t(), 2) * std::log(2); // Приближенное значение
    const float loglogN = std::log(logN);
    const mpz_class sqrtN = sqrt(N);

    // Граница гладкости B.
    const uint32_t B = MINIMAL_BOUND + std::ceil(std::exp(0.55 * std::sqrt(logN * loglogN)));
    std::cout << "Граница гладкости установлена как " << B << ".\n";

    /******************************
     *                            *
     * ЭТАП 1: Сбор данных       *
     *                            *
     ******************************/

    /*
     * Шаг 1
     *
     * Генерация факторной базы.
     */
    // START();
    // std::cout << "Факторная база сгенерирована.\n";
    const std::vector<uint32_t> factorBase = generateFactorBase(N, B);
    // STOP("Факторная база сгенерирована");

    /*
     * Шаг 2
     *
     * Вычисление начальных индексов для каждого числа в факторной базе.
     */
    // START();
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> startIndex(
        std::vector<uint32_t>(factorBase.size()), // Вектор первого начального индекса.
        std::vector<uint32_t>(factorBase.size())  // Вектор второго начального индекса.
    );

    for (uint32_t i = 0; i < factorBase.size(); ++i)
    {
        uint32_t p = factorBase[i];                   // Простое число из нашей факторной базы.
        uint32_t N_mod_p = mpz_class(N % p).get_ui(); // N по модулю p.

        /*
         * Нам нужны начальные значения a такие, что (a + sqrt(N))^2 - N 
         * делится на N. Для этого решаем сравнение x^2 = N (mod p), которое
         * даст нам искомые значения a в виде a = x - sqrt(N).
         */
        std::pair<uint32_t, uint32_t> x = tonelliShanks(N_mod_p, p);

        /* 
         * Искомое значение теперь a = x - sqrt(N) (mod p). Оно может быть
         * отрицательным, поэтому добавляем p, чтобы получить положительное число.
         */
        startIndex.first[i] = mpz_class((((x.first - sqrtN) % p) + p) % p).get_ui();
        startIndex.second[i] = mpz_class((((x.second - sqrtN) % p) + p) % p).get_ui();
    }
    // STOP("Индексы вычислены");

    // std::cout << "Вычисление индексов завершено.\n";
    /************************************
     *                                  *
     * ЭТАП 2: Этап просеивания        *
     *                                  *
     ***********************************/

    // В комментариях ниже, Q = (a + sqrt(N))^2 - N , a = 1, 2, ...

    /*
     * Шаг 2.1
     *
     * Просеиваем через логарифмические приближения в интервалах длины INTERVAL_LENGTH,
     * пока не найдем как минимум factorBase.size() + 20 B-гладких чисел.
     */
    uint32_t intervalStart = 0;
    uint32_t intervalEnd = INTERVAL_LENGTH;

    std::vector<uint32_t> smooth;                     // B-гладкие числа.
    std::vector<std::vector<uint32_t>> smoothFactors; // Разложение каждого B-гладкого числа.
    std::vector<float> logApprox(INTERVAL_LENGTH, 0); // Приближенные двоичные логарифмы a^2 - N.

    // Грубые логарифмические оценки вместо полных приближений.
    float prevLogEstimate = 0;
    uint32_t nextLogEstimate = 1;

    std::cout << "Факторизация.\n";

    while (smooth.size() < factorBase.size() + 20)
    {
        /*
         * Шаг 2.1.1
         *
         * Генерируем логарифмические приближения Q = (a + sqrt(N))^2 - N в текущем интервале.
         */
        for (uint32_t i = 1, a = intervalStart + 1; i < INTERVAL_LENGTH; ++i, ++a)
        {
            if (nextLogEstimate <= a)
            {
                const mpz_class Q = (a + sqrtN) * (a + sqrtN) - N;
                
                prevLogEstimate = mpz_sizeinbase(Q.get_mpz_t(), 2); // ~log_2(Q) - приближенное значение
                nextLogEstimate = nextLogEstimate * 1.8 + 1;
            }
            
            logApprox[i] = prevLogEstimate;
        }

        /*
         * Шаг 2.1.2
         *
         * Просеиваем числа в последовательности, которые полностью раскладываются по факторной базе.
         */
        for (uint32_t i = 0; i < factorBase.size(); ++i)
        {
            const uint32_t p = factorBase[i];
            const float logp = std::log(factorBase[i]) / std::log(2);

            // Просеиваем первую последовательность.
            while (startIndex.first[i] < intervalEnd)
            {
                if (startIndex.first[i] >= intervalStart) {
                    logApprox[startIndex.first[i] - intervalStart] -= logp;
                }
                startIndex.first[i] += p;
            }
            
            if (p == 2)
                continue; // a^2 = N (mod 2) имеет только один корень.

            // Просеиваем вторую последовательность.
            while (startIndex.second[i] < intervalEnd)
            {
                if (startIndex.second[i] >= intervalStart) {
                    logApprox[startIndex.second[i] - intervalStart] -= logp;
                }
                startIndex.second[i] += p;
            }
        }
        
        /*
         * Шаг 2.1.3
         *
         * Раскладываем на множители значения Q, чьи ~логарифмы были уменьшены до ~нуля во время просеивания.
         */
        const float threshold = std::log(factorBase.back()) / std::log(2);
        for (uint32_t i = 0, a = intervalStart; i < INTERVAL_LENGTH; ++i, ++a)
        {
            if (std::fabs(logApprox[i]) < threshold)
            {
                mpz_class Q = (a + sqrtN) * (a + sqrtN) - N;
                std::vector<uint32_t> factors;

                // Для каждого множителя p в факторной базе.
                for (uint32_t j = 0; j < factorBase.size(); ++j)
                {
                    // Последовательно делим Q на p, пока это возможно.
                    const uint32_t p = factorBase[j];
                    while (mpz_divisible_ui_p(Q.get_mpz_t(), p))
                    {
                        mpz_divexact_ui(Q.get_mpz_t(), Q.get_mpz_t(), p);
                        factors.push_back(j); // j-й элемент факторной базы был множителем.
                    }
                }
                if (Q == 1)
                {
                    // Q действительно является B-гладким, сохраняем его множители и соответствующее a.
                    smoothFactors.push_back(factors);
                    smooth.push_back(a);
                }
                if (smooth.size() >= factorBase.size() + 20)
                    break; // У нас достаточно гладких чисел, прекращаем факторизацию.
            }
        }

        // Переходим к следующему интервалу.
        intervalStart += INTERVAL_LENGTH;
        intervalEnd += INTERVAL_LENGTH;
    }

    /************************************
     *                                  *
     * ЭТАП 3: Этап линейной алгебры   *
     *                                  *
     ***********************************/

    std::cout << "Этап линейной алгебры.\n";

    /*
     * Шаг 3.1
     *
     * Построение бинарной матрицы M, где M_ij = четность i-го простого множителя
     * из факторной базы в разложении j-го B-гладкого числа.
     */
    Matrix M(factorBase.size(), smoothFactors.size() + 1);
    for (uint32_t i = 0; i < smoothFactors.size(); ++i)
    {
        for (uint32_t j = 0; j < smoothFactors[i].size(); ++j)
        {
            M(smoothFactors[i][j], i).flip();
        }
    }

    /*
     * Шаг 3.2
     *
     * Приведение матрицы к ступенчатому виду и её многократное решение,
     * пока не будет найден делитель.
     */
    M.reduce();
    mpz_class a;
    mpz_class b;

    do
    {
        std::vector<uint32_t> x = M.solve();

        a = 1;
        b = 1;

        /*
         * Вычисляем b = произведение(smooth[i] + sqrt(N)).
         *
         * Также вычисляем степень каждого простого числа в разложении a
         * по факторной базе.
         */
        std::vector<uint32_t> decomp(factorBase.size(), 0);
        for (uint32_t i = 0; i < smoothFactors.size(); ++i)
        {
            if (x[i] == 1)
            {
                for (uint32_t p = 0; p < smoothFactors[i].size(); ++p)
                    ++decomp[smoothFactors[i][p]];
                b *= (smooth[i] + sqrtN);
            }
        }

        /*
         * Вычисляем a = sqrt(произведение(factorBase[p])).
         */
        for (uint32_t p = 0; p < factorBase.size(); ++p)
        {
            for (uint32_t i = 0; i < (decomp[p] / 2); ++i)
                a *= factorBase[p];
        }

        // a = +/- b (mod N) означает, что мы нашли тривиальный делитель :(
    } while (a % N == b % N || a % N == (-b) % N + N);

    /************************************
     *                                  *
     * ЭТАП 4: Успех!                  *
     *                                  *
     ***********************************/

    mpz_class factor;
    mpz_gcd(factor.get_mpz_t(), mpz_class(b - a).get_mpz_t(), N.get_mpz_t());
    return factor;
}

int main()
{
    std::vector<uint32_t> primes;
    uint32_t max = 10001;
    std::vector<char> sieve(max, 0);
    std::queue<mpz_class> qu;
    for (uint32_t p = 2; p < max; ++p)
    {
        if (sieve[p])
            continue;
        primes.push_back(p);
        for (uint32_t i = p; i < max; i += p)
            sieve[i] = 1;
    }
    mpz_class N;
    std::string input;
    std::cin >> input;
    if (input.find_first_not_of("0123456789") != std::string::npos || input.empty()) {
        std::cerr << "Ошибка: введено не число." << std::endl;
        return 1;
    }
    N = mpz_class(input);
    START();
    if (mpz_probab_prime_p(N.get_mpz_t(), 10))
    {
        // N является простым числом.
        std::cout << N << std::endl
                  << std::endl;
    }
    std::stack<mpz_class> factors;
    factors.push(N);
    while (!factors.empty())
    {
        mpz_class factor = factors.top();
        factors.pop();
        if (mpz_probab_prime_p(factor.get_mpz_t(), 10))
        {
            // N является простым числом.
            qu.push(factor);
            std::cout << "Мы факторизовали " << factor << ".\n";
            continue;
        }
        // Выполняем пробное деление перед запуском решета.

        bool foundFactor = false;
        for (uint32_t i = 0; i < primes.size(); ++i)
        {
            if (mpz_divisible_ui_p(factor.get_mpz_t(), primes[i]))
            {
                factors.push(primes[i]);
                factors.push(factor / primes[i]);
                foundFactor = true;
                std::cout << "Мы факторизовали " << primes[i] << " методом пробного деления.\n";
                break;
            }
        }
        if (foundFactor)
        {
            // Пробное деление было успешным.
            continue;
        }

        // Обрабатываем точные степени отдельно (QS с ними плохо работает).
        if (mpz_perfect_power_p(factor.get_mpz_t()))
        {
            mpz_class root, r;
            uint32_t max = mpz_sizeinbase(factor.get_mpz_t(), 2) / 2;
            for (uint32_t n = 2; n < max; ++n)
            {
                mpz_rootrem(root.get_mpz_t(), r.get_mpz_t(), factor.get_mpz_t(), n);
                if (r == 0)
                {
                    for (uint32_t i = 0; i < n; ++i)
                        factors.push(root);
                }
            }
        }
        else
        {
            // Запускаем алгоритм квадратичного решета.
            mpz_class result = quadraticSieve(factor);
            factors.push(result);
            factors.push(factor / result);
            std::cout << "Мы факторизовали " << result << " из " << factor << ".\n";
        }
    }
    std::cout << "Результат факторизации: \n";
    while (!qu.empty())
    {
        std::cout << qu.front() << "\n";
        qu.pop();
    }

    STOP("Факторизация завершена");
}
