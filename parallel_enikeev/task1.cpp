#include <matplot/matplot.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdlib>

using namespace matplot;
using namespace std;

// =============================================
// Функции для вычисления частичных сумм (линейно)
// =============================================
void compute_linear_prefix_sum(vector<double>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        arr[i] += arr[i - 1];
    }
}

// =============================================
/* 
 * Задача 1:
 * Реализовать параллельное вычисление частичных сумм массива размера n с помощью OpenMP.
 * Для этого нужно написать функцию, которая принимает массив, считает частичную сумму, изменяя сам массив.
 * 
 * Принимаем массив по ссылке, чтобы изменения внутри функции были видны снаружи.
 */
// =============================================
void compute_parallel_prefix_sum(vector<double>& arr) {
    int n = arr.size();
    int num_threads = omp_get_max_threads();
    
    // Распараллеливаем первичное накопление сумм в блоках
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = n / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n : start + chunk_size;

        // Локальное накопление в блоке
        for (int i = start + 1; i < end; ++i) {
            arr[i] += arr[i-1];
        }
    }

    // Объединение блоков
    vector<double> block_sums(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        int chunk_size = n / num_threads;
        block_sums[i] = arr[(i + 1) * chunk_size - 1];
    }

    // Префиксная сумма для блоков
    for (int i = 1; i < num_threads; ++i) {
        block_sums[i] += block_sums[i-1];
    }

    // Распространение сумм блоков
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = n / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n : start + chunk_size;

        double block_offset = (thread_id > 0) ? block_sums[thread_id - 1] : 0;
        
        for (int i = start; i < end; ++i) {
            arr[i] += block_offset;
        }
    }
}

// =============================================
/* 
 * Задача 2:
 * Написать функцию для вычисления размера массива M , начиная с которого параллельное выполнение алгоритма эффективнее последовательного.
 * Полученное значение необходимо сохранить в файл.
 * 
 * Функция для вычисления M: минимальный размер массива,
 * при котором параллельный алгоритм быстрее линейного 
 * для вычисления частичных сумм.
 * Записывает значение M в файл output.txt.
 * 
 * Для запуска: task1.exe -task2
 */
// =============================================
int calculate_effectify_parralel_value_M() {
    ofstream output_file;
    output_file.open("output.txt", ofstream::trunc);

    if (!output_file.is_open()) {
        cerr << "Ошибка: невозможно открыть файл для записи значения M!\n";
        return -1;
    }

    int number = 10; 
    int iterations = 10; // Добавляем количество итераций для усреднения
    
    while (true) {
        double avg_lin_time = 0, avg_par_time = 0;

        // Цикл для усреднения времени
        for (int iter = 0; iter < iterations; ++iter) {
            vector<double> arr(number);
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = static_cast<double>(rand() % 100);
            }
            vector<double> arr_clone(arr);

            // Измеряем линейное время
            double start_lin = omp_get_wtime();
            compute_linear_prefix_sum(arr_clone);
            double end_lin = omp_get_wtime();
            double lin_time = end_lin - start_lin;
            avg_lin_time += lin_time;

            // Измеряем параллельное время
            arr_clone = vector<double>(arr);
            double start_par = omp_get_wtime();
            compute_parallel_prefix_sum(arr_clone);
            double end_par = omp_get_wtime();
            double par_time = end_par - start_par;
            avg_par_time += par_time;
        }

        // Усредняем результаты
        avg_lin_time /= iterations;
        avg_par_time /= iterations;

        cout << "Размер массива: " << number << endl;
        cout << "Среднее время (линейное): " << avg_lin_time << " сек.\n";
        cout << "Среднее время (параллельное): " << avg_par_time << " сек.\n";

        // Критерий эффективности: параллельное время должно быть меньше линейного
        if (avg_par_time < avg_lin_time) {
            break;
        }

        number += 10; // Увеличиваем размер более существенно
    }

    // Сохраняем значение M в файл
    output_file << number;
    output_file.close();

    cout << "Минимальный размер M записан в файл: " << number << "\n";
    return number;
}

// =============================================
/* 
 * Задача 3:
 * Реализуем гибридную функцию, которая выбирает 
 * последовательный или параллельный алгоритм на основе M.
 * 
 * Читает значение M из файла output.txt.
 */
// =============================================
void hybrid_prefix_sum(vector<double>& arr) {
    ifstream input_file("output.txt");
    int threshold_M = 0;

    if (input_file.is_open()) {
        input_file >> threshold_M;
        input_file.close();
        cout << "Прочитано значение M из файла: " << threshold_M << endl;
    } else {
        cerr << "Ошибка: файл output.txt не найден. Запустите программу с аргументом -task2 для вычисления M.\n";
        exit(1);
    }

    // Гибридный выбор: последовательный или параллельный алгоритм
    if (arr.size() < static_cast<size_t>(threshold_M)) {
        cout << "Используется последовательный алгоритм (n < M).\n";
        compute_linear_prefix_sum(arr);
    } else {
        cout << "Используется параллельный алгоритм (n >= M).\n";
        compute_parallel_prefix_sum(arr);
    }
}

// =============================================
/*
 * Задача 4:
 * Построить диаграмму ускорения для N=2*M.
 * Диаграмма отражает, насколько увеличивается производительность
 * в зависимости от количества потоков.
 */
// =============================================
void generate_plot() {
    // Чтение значения M из файла
    ifstream input_file("output.txt");
    int threshold_M = 0;

    if (input_file.is_open()) {
        input_file >> threshold_M;
        input_file.close();
        cout << "Прочитано значение M из файла: " << threshold_M << endl;
    } else {
        cerr << "Ошибка: файл output.txt не найден. Запустите программу с аргументом -task2 для вычисления M.\n";
        exit(1);
    }

    // Размер массива для тестирования - удвоенное значение M
    int N = static_cast<int>(threshold_M * 2); 

    // Максимальное количество потоков
    int max_threads = omp_get_max_threads();

    // Векторы для хранения данных
    vector<double> thread_counts;       // Количество потоков
    vector<double> execution_times;     // Времена выполнения
    vector<double> speedup_times;       // Коэффициенты ускорения

    // Количество итераций для усреднения результатов
    int iterations = 10;

    // Базовый массив для тестирования
    vector<double> base_arr(static_cast<size_t>(N));
    for (int i = 0; i < N; i++) {
        base_arr[i] = static_cast<double>(rand() % 100);
    }

    // Замер времени для базового (последовательного) варианта
    double base_time = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> arr = base_arr;
        double start = omp_get_wtime();
        compute_linear_prefix_sum(arr);  // Последовательный алгоритм
        double end = omp_get_wtime();
        base_time += (end - start);
    }
    base_time /= iterations;

    // Тестирование для разного количества потоков
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        // Установка количества потоков
        omp_set_num_threads(num_threads);
        
        // Усреднение времени выполнения
        double avg_parallel_time = 0.0;
        for (int iter = 0; iter < iterations; ++iter) {
            vector<double> arr = base_arr;
            
            // Замер времени параллельного алгоритма
            double start = omp_get_wtime();
            compute_parallel_prefix_sum(arr);
            double end = omp_get_wtime();
            
            avg_parallel_time += (end - start);
        }
        avg_parallel_time /= iterations;

        // Сохранение результатов
        thread_counts.push_back(static_cast<double>(num_threads));
        execution_times.push_back(avg_parallel_time);
        
        // Вычисление ускорения
        speedup_times.push_back(base_time / avg_parallel_time);
    }

    // Создание графиков с использованием современного синтаксиса Matplot++
    auto fig = matplot::figure();
    fig->width(1200);  // Увеличиваем ширину окна
    fig->height(800);  // Увеличиваем высоту окна

    // Первый subplot - времена выполнения
    matplot::subplot(2, 1, 0);
    auto p1 = matplot::plot(thread_counts, execution_times);
    p1->line_width(2);
    p1->marker("o");  // Круглые маркеры
    p1->color("blue");
    matplot::xlabel("Количество потоков");
    matplot::ylabel("Время выполнения (сек)");
    matplot::title("Время выполнения параллельного алгоритма");
    matplot::grid(true);

    // Второй subplot - ускорение
    matplot::subplot(2, 1, 1);
    auto p2 = matplot::plot(thread_counts, speedup_times);
    p2->line_width(2);
    p2->marker("o");  // Круглые маркеры
    p2->color("red");
    matplot::xlabel("Количество потоков");
    matplot::ylabel("Ускорение");
    matplot::title("Коэффициент ускорения");
    matplot::grid(true);

    // Сохранение графика
    matplot::save("performance_plot.png");

    // Вывод графиков
    matplot::show();

    // Дополнительный вывод результатов в консоль
    cout << "\nРезультаты тестирования:" << endl;
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        cout << "Потоки: " << thread_counts[i] 
             << ", Время: " << std::fixed << execution_times[i] 
             << ", Ускорение: " << speedup_times[i] << endl;
    }
}

// =============================================
// Главная функция
// =============================================
int main(int argc, char* argv[]) {
    if (argc > 1) {
        string arg = argv[1];

        if (arg == "-task2") {
            int M = calculate_effectify_parralel_value_M();
            if (M != -1) {
                cout << "Задача 2 выполнена: M = " << M << "\n";
            }
            return 0;
        } 
        if (arg == "-task3") {
            cout << "Выполнение задачи 3: гибридное вычисление...\n";
            cout << "Введите размер массива: ";
            size_t size;
            cin >> size;
            vector<double> arr(size);
            for (size_t i = 0; i < size; i++) {
                arr[i] = i + 1;
            }
            hybrid_prefix_sum(arr);

            cout << "Результат частичных сумм:\n";
            for (const auto& val : arr) {
                cout << val << " ";
            }
            cout << "\n";
            return 0;
        }
        if (arg == "-task4") {
            cout << "\n";
            cout << "Выполнение задачи 4: построение графика ускорения...\n";
            generate_plot();
            return 0;
        }

        cerr << "Неизвестный аргумент: " << arg << "\n";
        return 1;
    }

    cerr << "Не передан аргумент задачи. Доступные задачи:\n";
    cerr << "  -task2: Подсчитать минимальный размер M.\n";
    cerr << "  -task3: Выполнить гибридное вычисление частичных сумм.\n";
    cerr << "  -task4: Построить график ускорения.\n";
    return 1;
}