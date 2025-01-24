#include <matplot/matplot.h> // Библиотека для построения графиков
#include <omp.h> // Библиотека OpenMP для параллельного программирования
#include <iostream> // Ввод/вывод
#include <fstream> // Работа с файлами
#include <vector> // Динамические массивы
#include <cmath> // Математические функции
#include <algorithm> // Алгоритмы STL
#include <string> // Строки
#include <cstdlib> // Стандартные функции C (rand, srand)

using namespace matplot;
using namespace std;

// =============================================
// Функции для вычисления частичных сумм (линейно)
// =============================================
void compute_linear_prefix_sum(vector<double>& arr) {
    // Последовательное вычисление префиксных сумм
    for (size_t i = 1; i < arr.size(); i++) {
        // Каждый элемент становится суммой себя и предыдущих элементов
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
    int n = arr.size(); // Размер массива
    int num_threads = omp_get_max_threads(); // Получаем максимальное количество доступных потоков

    // Распараллеливаем первичное накопление сумм в блоках
    // Директива OpenMP для создания параллельной области
    #pragma omp parallel num_threads(num_threads) 
    {
        // Получаем ID текущего потока
        int thread_id = omp_get_thread_num();
        // Вычисляем размер блока для каждого потока
        int chunk_size = n / num_threads;
        // Начало блока для текущего потока
        int start = thread_id * chunk_size;
        // Конец блока для текущего потока (обработка последнего блока, если n не делится на num_threads)
        int end = (thread_id == num_threads - 1) ? n : start + chunk_size;

        // Локальное накопление в блоке. Каждый поток считает префиксные суммы в своем блоке
        for (int i = start + 1; i < end; ++i) {
            arr[i] += arr[i-1];
        }
    }

    // Объединение блоков. Создаем массив для хранения сумм каждого блока
    vector<double> block_sums(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        int chunk_size = n / num_threads;
        // Записываем последний элемент каждого блока (который содержит сумму блока) в block_sums
        block_sums[i] = arr[(i + 1) * chunk_size - 1]; 
    }

    // Префиксная сумма для блоков. Считаем префиксные суммы для block_sums
    for (int i = 1; i < num_threads; ++i) {
        block_sums[i] += block_sums[i-1];
    }

    // Распространение сумм блоков. Добавляем сумму предыдущих блоков к каждому элементу текущего блока
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = n / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n : start + chunk_size;

        // Смещение для текущего блока (сумма всех предыдущих блоков)
        double block_offset = (thread_id > 0) ? block_sums[thread_id - 1] : 0; 
        
        // Добавляем смещение к каждому элементу в блоке
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
    ofstream output_file; // Создаем объект для записи в файл
    output_file.open("output.txt", ofstream::trunc); // Открываем файл output.txt для записи,  ofstream::trunc очищает файл перед записью

    if (!output_file.is_open()) { // Проверяем, успешно ли открыт файл
        cerr << "Ошибка: невозможно открыть файл для записи значения M!\n";
        return -1; // Возвращаем -1 в случае ошибки
    }

    int number = 10; // Начальный размер массива
    int iterations = 10; // Количество итераций для усреднения времени

    while (true) {
        double avg_lin_time = 0, avg_par_time = 0; // Переменные для среднего времени

        // Цикл для усреднения времени
        for (int iter = 0; iter < iterations; ++iter) {
            vector<double> arr(number); // Создаем массив заданного размера
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = static_cast<double>(rand() % 100); // Заполняем массив случайными числами
            }
            vector<double> arr_clone(arr); // Копируем массив для корректного сравнения

            // Измеряем линейное время
            double start_lin = omp_get_wtime(); // Получаем текущее время
            compute_linear_prefix_sum(arr_clone); // Вызываем линейную функцию
            double end_lin = omp_get_wtime();  // Получаем текущее время
            double lin_time = end_lin - start_lin; // Вычисляем время выполнения
            avg_lin_time += lin_time; //  Добавляем к суммарному времени

            // Измеряем параллельное время
            arr_clone = vector<double>(arr);  // Восстанавливаем исходный массив
            double start_par = omp_get_wtime();  // Получаем текущее время
            compute_parallel_prefix_sum(arr_clone); // Вызываем параллельную функцию
            double end_par = omp_get_wtime(); // Получаем текущее время
            double par_time = end_par - start_par; // Вычисляем время выполнения
            avg_par_time += par_time; //  Добавляем к суммарному времени
        }

        // Усредняем результаты
        avg_lin_time /= iterations;
        avg_par_time /= iterations;

        cout << "Размер массива: " << number << endl;
        cout << "Среднее время (линейное): " << avg_lin_time << " сек.\n";
        cout << "Среднее время (параллельное): " << avg_par_time << " сек.\n";

        // Критерий эффективности: параллельное время должно быть меньше линейного
        if (avg_par_time < avg_lin_time) {
            break; // Выходим из цикла, если параллельный алгоритм быстрее
        }

        number += 10; // Увеличиваем размер массива для следующей итерации
    }

    // Сохраняем значение M в файл
    output_file << number;
    output_file.close(); // Закрываем файл

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
    ifstream input_file("output.txt"); // Открываем файл output.txt для чтения
    int threshold_M = 0;

    if (input_file.is_open()) { // Проверяем, успешно ли открыт файл
        input_file >> threshold_M; // Читаем значение M из файла
        input_file.close(); // Закрываем файл
        cout << "Прочитано значение M из файла: " << threshold_M << endl;
    } else {
        cerr << "Ошибка: файл output.txt не найден. Запустите программу с аргументом -task2 для вычисления M.\n";
        exit(1); // Завершаем программу с кодом ошибки
    }

    // Гибридный выбор: последовательный или параллельный алгоритм
    if (arr.size() < static_cast<size_t>(threshold_M)) {
        cout << "Используется последовательный алгоритм (n < M).\n";
        compute_linear_prefix_sum(arr); // Вызываем линейный алгоритм
    } else {
        cout << "Используется параллельный алгоритм (n >= M).\n";
        compute_parallel_prefix_sum(arr); // Вызываем параллельный алгоритм
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

    // Создание графиков
    auto fig = matplot::figure();
    fig->width(1200);  // Устанавливаем ширину окна графика
    fig->height(800); // Устанавливаем высоту окна графика

    // Первый subplot - времена выполнения. subplot(rows, cols, index) создает сетку графиков rows x cols и выбирает график с индексом index
    matplot::subplot(2, 1, 0); // Создаем график в первой строке (из двух)
    auto p1 = matplot::plot(thread_counts, execution_times); // Строим график времени выполнения от количества потоков
    p1->line_width(2); // Устанавливаем толщину линии
    p1->marker("o");  // Устанавливаем маркер точки как кружок
    p1->color("blue"); // Устанавливаем цвет линии
    matplot::xlabel("Количество потоков"); // Подпись оси X
    matplot::ylabel("Время выполнения (сек)"); // Подпись оси Y
    matplot::title("Время выполнения параллельного алгоритма"); // Заголовок графика
    matplot::grid(true); // Включаем сетку

    // Второй subplot - ускорение
    matplot::subplot(2, 1, 1); // Создаем график во второй строке (из двух)
    auto p2 = matplot::plot(thread_counts, speedup_times);  // Строим график ускорения от количества потоков
    p2->line_width(2);  // Устанавливаем толщину линии
    p2->marker("o"); // Устанавливаем маркер точки как кружок
    p2->color("red"); // Устанавливаем цвет линии
    matplot::xlabel("Количество потоков"); // Подпись оси X
    matplot::ylabel("Ускорение"); // Подпись оси Y
    matplot::title("Коэффициент ускорения"); // Заголовок графика
    matplot::grid(true); // Включаем сетку

    // Сохранение графика в файл
    matplot::save("performance_plot.png");

    // Вывод графиков на экран
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