#include <iostream>
#include <gmp.h>
#include <omp.h>

using namespace std;

// Функция НОД (наибольший общий делитель)
void gcd(mpz_t result, const mpz_t a, const mpz_t b) {
    mpz_gcd(result, a, b);
}

// Быстрое возведение в степень по модулю
void mod_pow(mpz_t result, const mpz_t base, const mpz_t exp, const mpz_t modulus) {
    mpz_powm(result, base, exp, modulus);
}

// Мультипликативное обратное по модулю
void mod_inverse(mpz_t result, const mpz_t a, const mpz_t m) {
    if (!mpz_invert(result, a, m)) {
        mpz_set_ui(result, 0);  // Мультипликативного обратного не существует
    }
}

// Алгоритм Полларда ро с параллелизацией для факторизации числа N
void pollard_rho(mpz_t factor, const mpz_t N) {
    // Инициализация переменных
    mpz_t x, y, d, temp;
    mpz_inits(x, y, d, temp, NULL);
    int found = 0;
    unsigned long seed;

    cout << "Начинаем факторизацию алгоритмом Полларда ро..." << endl;

    // Параллельная секция
    #pragma omp parallel private(x, y, d, temp, seed) shared(found)
    {
        mpz_inits(x, y, d, temp, NULL);

        // Инициализация генератора случайных чисел для каждого потока
        gmp_randstate_t state;
        gmp_randinit_default(state);
        seed = omp_get_thread_num() + time(NULL);
        gmp_randseed_ui(state, seed);

        // Генерация случайного начального значения x
        mpz_urandomm(x, state, N);
        mpz_set(y, x);

        while (!found) {
            // x = (x^2 + 1) mod N
            mpz_mul(x, x, x);
            mpz_add_ui(x, x, 1);
            mpz_mod(x, x, N);

            // y = ((y^2 + 1)^2 + 1) mod N
            mpz_mul(y, y, y);
            mpz_add_ui(y, y, 1);
            mpz_mod(y, y, N);

            mpz_mul(y, y, y);
            mpz_add_ui(y, y, 1);
            mpz_mod(y, y, N);

            // d = НОД(|x - y|, N)
            mpz_sub(temp, x, y);
            mpz_abs(temp, temp);
            mpz_gcd(d, temp, N);

            // Если найден нетривиальный делитель
            if (mpz_cmp_ui(d, 1) > 0 && mpz_cmp(d, N) < 0) {
                // Критическая секция для записи найденного делителя
                #pragma omp critical
                {
                    if (!found) {
                        mpz_set(factor, d);
                        found = 1;
                        cout << "Делитель найден потоком " << omp_get_thread_num() << endl;
                    }
                }
                break;
            }
        }

        // Очистка переменных потока
        mpz_clears(x, y, d, temp, NULL);
        gmp_randclear(state);
    }

    // Если делитель не найден
    if (!found) {
        mpz_set_ui(factor, 1);
    }
}

int main() {
    // Инициализация переменных большого размера
    mpz_t N, e, SW, p, q, phi, d, decrypted;
    mpz_inits(N, e, SW, p, q, phi, d, decrypted, NULL);

    // Ввод данных с клавиатуры
    cout << "Введите N (модуль): ";
    gmp_scanf("%Zd", N);

    cout << "Введите e (открытая экспонента): ";
    gmp_scanf("%Zd", e);

    cout << "Введите SW (зашифрованное сообщение): ";
    gmp_scanf("%Zd", SW);

    cout << "Тестирование с параметрами:" << endl;
    gmp_printf("N = %Zd\n", N);
    gmp_printf("e = %Zd\n", e);
    gmp_printf("SW = %Zd\n", SW);

    // Факторизация модуля N
    pollard_rho(p, N);
    if (mpz_cmp_ui(p, 1) == 0) {
        cout << "Не удалось факторизовать N с помощью алгоритма Полларда ро." << endl;
        return 1;
    }

    // Вычисление q = N / p
    mpz_divexact(q, N, p);

    cout << "Факторизация успешна:" << endl;
    gmp_printf("p = %Zd\n", p);
    gmp_printf("q = %Zd\n", q);

    // Вычисление функции Эйлера phi = (p - 1)*(q - 1)
    mpz_sub_ui(p, p, 1);     // p = p - 1
    mpz_sub_ui(q, q, 1);     // q = q - 1
    mpz_mul(phi, p, q);      // phi = (p - 1)*(q - 1)

    // Вычисление мультипликативного обратного d = e^(-1) mod phi
    mod_inverse(d, e, phi);

    if (mpz_cmp_ui(d, 0) == 0) {
        cout << "Не найден мультипликативный обратный. Расшифрование невозможно." << endl;
        return 1;
    }

    // Расшифрование сообщения: M = SW^d mod N
    mod_pow(decrypted, SW, d, N);

    // Вывод результатов
    gmp_printf("Закрытый ключ d = %Zd\n", d);
    gmp_printf("Расшифрованное сообщение = %Zd\n", decrypted);

    // Очистка переменных
    mpz_clears(N, e, SW, p, q, phi, d, decrypted, NULL);

    return 0;
}