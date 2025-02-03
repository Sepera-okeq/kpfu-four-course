#define _CRT_SECURE_NO_WARNINGS

/*
 * Параллельный конвейер с использованием MPI
 * 
 * Программа реализует параллельный конвейер обработки данных с использованием MPI.
 * Архитектура конвейера состоит из следующих компонентов:
 * 1. Генератор (ранг 0) - создает последовательность чисел для обработки
 * 2. Работники (ранги 1..N-1) - выполняют функции обработки данных (Ф1, Ф2, Ф3)
 * 3. Коллектор (последний ранг) - собирает результаты и вычисляет итоговую сумму
 * 
 * Особенности реализации:
 * - Использует MPI для межпроцессного взаимодействия
 * - Каждая функция конвейера может выполняться на нескольких процессах
 * - Измерение времени выполнения с помощью MPI_Wtime
 * - Логирование действий с временными метками
 * - Обработка сигналов завершения для корректного завершения работы
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <functional>
#include <memory>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>

// Мьютекс для синхронизации вывода в консоль
std::mutex cout_mutex;

// Функция для получения форматированного времени с использованием chrono
std::string get_formatted_time() {
    auto now = std::chrono::system_clock::now();
    auto time_point = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm local_tm;
    localtime_s(&local_tm, &time_point);
    
    std::stringstream ss;
    ss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Безопасный вывод в консоль с временной меткой используя mutex для синхронизации
template<typename T>
void safe_cout(T message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[" << get_formatted_time() << "] " << message << std::endl;
}

// Форматирование времени выполнения в микросекундах
std::string format_duration(double duration) {
    return std::to_string(static_cast<long long>(duration * 1e6)) + "мкс";
}

// Теги для MPI сообщений
enum Tags {
    DATA_TAG = 1,        // Тег для передачи данных
    TERMINATION_TAG = 2  // Тег для сигнала завершения
};

// Структура сообщения для передачи данных между процессами
struct DataMessage {
    int value;              // Значение для обработки
    int source;             // Исходный процесс
    bool is_termination;    // Флаг завершения
    double timestamp;       // Временная метка (используется MPI_Wtime)

    DataMessage() : value(0), source(0), is_termination(false),
        timestamp(MPI_Wtime()) {}
};

// Базовый абстрактный класс для функций конвейера
class PipelineFunction {
public:
    virtual int process(int input) = 0;              // Метод обработки входных данных
    virtual std::string getName() const = 0;         // Получение имени функции
    virtual ~PipelineFunction() = default;
};

// Первая функция конвейера: увеличивает значение на 1
class F1 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Имитация сложных вычислений
        return input + 1;
    }
    std::string getName() const override { return "Ф1 (инкремент)"; }
};

// Вторая функция конвейера: возводит число в квадрат
class F2 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Имитация сложных вычислений
        return input * input;
    }
    std::string getName() const override { return "Ф2 (квадрат)"; }
};

// Третья функция конвейера: возвращает значение без изменений
class F3 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Имитация сложных вычислений
        return input;
    }
    std::string getName() const override { return "Ф3 (идентичность)"; }
};

// Класс для генерации последовательности чисел
class Generator {
private:
    int count;          // Количество генерируемых чисел
    int start_value;    // Начальное значение
    int step;           // Шаг между числами

public:
    Generator(int count = 3, int start = 0, int step = 1)
        : count(count), start_value(start), step(step) {}

    // Генерация последовательности чисел
    std::vector<int> generate() {
        std::vector<int> result;
        for (int i = 0; i < count; ++i) {
            result.push_back(start_value + i * step);
        }
        return result;
    }
};

// Структура конфигурации конвейера
struct PipelineConfig {
    std::vector<std::shared_ptr<PipelineFunction>> functions;  // Список функций конвейера
    std::vector<int> proc_per_func;                           // Количество процессов на каждую функцию
    bool use_reduction;                                       // Флаг использования редукции
    int generator_count;                                      // Количество генерируемых чисел
    int generator_start;                                      // Начальное значение для генератора
    int generator_step;                                       // Шаг генератора

    PipelineConfig() : generator_count(3), generator_start(0), generator_step(1) {}
};

// Основной класс конвейера
class Pipeline {
private:
    int rank;                                           // Ранг текущего процесса
    int total_procs;                                    // Общее количество процессов
    PipelineConfig config;                             // Конфигурация конвейера
    std::vector<std::pair<int, int>> rank_ranges;      // Диапазоны рангов для каждой функции
    double start_time;                                  // Время начала работы (MPI_Wtime)

    // Получение прошедшего времени в миллисекундах
    std::string getElapsedTime() {
        double current_time = MPI_Wtime();
        double elapsed = (current_time - start_time) * 1000;  // Конвертация в миллисекунды
        return std::to_string(static_cast<int>(elapsed)) + "мс";
    }

    // Логирование сообщений с информацией о процессе и времени
    void log(const std::string& message) {
        std::stringstream ss;
        ss << "[Процесс " << std::setw(2) << rank << " | " << getElapsedTime() << "] " << message;
        safe_cout(ss.str());
    }

    // Настройка диапазонов рангов для каждой функции конвейера
    void setupRankRanges() {
        int current_rank = 1;  // Начинаем с 1, так как 0 - генератор
        for (size_t i = 0; i < config.proc_per_func.size(); ++i) {
            rank_ranges.push_back({ current_rank, current_rank + config.proc_per_func[i] - 1 });
            current_rank += config.proc_per_func[i];
        }
    }

    void processMessage(const DataMessage& msg, int stage, int next_rank) {
        auto start_process = MPI_Wtime();

        if (!msg.is_termination) {
            int result = config.functions[stage]->process(msg.value);

            auto end_process = MPI_Wtime();
            auto duration = end_process - start_process;

            log("Функция " + config.functions[stage]->getName() +
                " обработала значение " + std::to_string(msg.value) +
                " -> " + std::to_string(result) +
                " (заняло " + format_duration(duration) + ")");

            // Проверяем, что процесс-получатель все еще активен
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, TERMINATION_TAG, MPI_COMM_WORLD, &flag, &status);
            if (!flag) {
                DataMessage output{};
                output.value = result;
                output.source = rank;
                output.is_termination = false;
                MPI_Send(&output, sizeof(DataMessage), MPI_BYTE, next_rank, DATA_TAG, MPI_COMM_WORLD);
            }
        }
        else {
            // Проверяем, что процесс-получатель все еще активен
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, TERMINATION_TAG, MPI_COMM_WORLD, &flag, &status);
            if (!flag) {
                MPI_Send(&msg, sizeof(DataMessage), MPI_BYTE, next_rank, DATA_TAG, MPI_COMM_WORLD);
            }
        }
    }

public:
    // Конструктор конвейера
    Pipeline(int rank, int total_procs, const PipelineConfig& config)
        : rank(rank), total_procs(total_procs), config(config) {
        start_time = MPI_Wtime();  // Засекаем время начала работы
        setupRankRanges();
    }

    void run() {
        if (rank == 0) {
            runGenerator();
        }
        else if (rank == total_procs - 1) {
            runCollector();
        }
        else {
            runWorker();
        }
    }

private:
    void runWorker() {
        int stage = -1;
        for (size_t i = 0; i < rank_ranges.size(); ++i) {
            if (rank >= rank_ranges[i].first && rank <= rank_ranges[i].second) {
                stage = i;
                break;
            }
        }

        if (stage == -1) {
            log("Работник инициализирован с неверным этапом. Завершение.");
            return;
        }

        log("Запущен работник для этапа " + std::to_string(stage) +
            " (" + config.functions[stage]->getName() + ")");

        bool continue_working = true;
        int processed_count = 0;
        std::vector<DataMessage> pending_messages;

        while (continue_working) {
            MPI_Status status;
            DataMessage msg;

            MPI_Recv(&msg, sizeof(DataMessage), MPI_BYTE, MPI_ANY_SOURCE,
                MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (msg.is_termination) {
                // Обрабатываем все оставшиеся сообщения перед завершением
                for (const auto& pending : pending_messages) {
                    int result = config.functions[stage]->process(pending.value);

                    DataMessage output{};
                    output.value = result;
                    output.source = rank;
                    output.is_termination = false;

                    int next_rank;
                    if (stage == static_cast<int>(rank_ranges.size()) - 1) {
                        next_rank = total_procs - 1;
                    }
                    else {
                        next_rank = rank_ranges[stage + 1].first;
                    }

                    MPI_Send(&output, sizeof(DataMessage), MPI_BYTE,
                        next_rank, DATA_TAG, MPI_COMM_WORLD);
                }

                // Передаем сигнал завершения дальше
                if (stage < static_cast<int>(rank_ranges.size()) - 1) {
                    for (int next = rank_ranges[stage + 1].first;
                        next <= rank_ranges[stage + 1].second; ++next) {
                        MPI_Send(&msg, sizeof(DataMessage), MPI_BYTE,
                            next, DATA_TAG, MPI_COMM_WORLD);
                    }
                }
                else {
                    MPI_Send(&msg, sizeof(DataMessage), MPI_BYTE,
                        total_procs - 1, DATA_TAG, MPI_COMM_WORLD);
                }

                continue_working = false;
                break;
            }

            // Обработка обычного сообщения
            auto start_process = MPI_Wtime();
            int result = config.functions[stage]->process(msg.value);
            auto end_process = MPI_Wtime();
            auto duration = end_process - start_process;

            log("Функция " + config.functions[stage]->getName() +
                " обработала значение " + std::to_string(msg.value) +
                " -> " + std::to_string(result) +
                " (заняло " + format_duration(duration) + ")");

            DataMessage output{};
            output.value = result;
            output.source = rank;
            output.is_termination = false;

            int next_rank;
            if (stage == static_cast<int>(rank_ranges.size()) - 1) {
                next_rank = total_procs - 1;
            }
            else {
                next_rank = rank_ranges[stage + 1].first +
                    (processed_count % (rank_ranges[stage + 1].second -
                        rank_ranges[stage + 1].first + 1));
            }

            MPI_Send(&output, sizeof(DataMessage), MPI_BYTE,
                next_rank, DATA_TAG, MPI_COMM_WORLD);
            processed_count++;
        }

        log("Работник завершил работу");
    }

    void runGenerator() {
        log("Генератор запускается...");

        Generator gen(config.generator_count, config.generator_start, config.generator_step);
        auto data = gen.generate();

        log("Сгенерировано " + std::to_string(data.size()) + " значений");

        int current_worker = rank_ranges[0].first;
        for (int value : data) {
            DataMessage msg{};
            msg.value = value;
            msg.source = rank;
            msg.is_termination = false;

            log("Отправка значения " + std::to_string(value) +
                " работнику " + std::to_string(current_worker));

            MPI_Send(&msg, sizeof(DataMessage), MPI_BYTE,
                current_worker, DATA_TAG, MPI_COMM_WORLD);

            // Ждем небольшую паузу между отправками
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            current_worker++;
            if (current_worker > rank_ranges[0].second) {
                current_worker = rank_ranges[0].first;
            }
        }

        // Отправляем сигналы завершения
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        DataMessage termination{};
        termination.is_termination = true;

        for (int i = rank_ranges[0].first; i <= rank_ranges[0].second; ++i) {
            log("Отправка сигнала завершения работнику " + std::to_string(i));
            MPI_Send(&termination, sizeof(DataMessage), MPI_BYTE,
                i, DATA_TAG, MPI_COMM_WORLD);
        }

        log("Генератор завершил работу");
    }

    void runCollector() {
        log("Коллектор запускается");

        int received = 0;
        int sum = 0;
        int termination_signals = 0;
        int expected_termination = rank_ranges.back().second -
            rank_ranges.back().first + 1;

        while (termination_signals < expected_termination) {
            DataMessage msg;
            MPI_Status status;

            MPI_Recv(&msg, sizeof(DataMessage), MPI_BYTE, MPI_ANY_SOURCE,
                DATA_TAG, MPI_COMM_WORLD, &status);

            if (msg.is_termination) {
                termination_signals++;
                log("Получен сигнал завершения (" +
                    std::to_string(termination_signals) + "/" +
                    std::to_string(expected_termination) + ")");
            }
            else {
                sum += msg.value;
                received++;
                log("Получено значение: " + std::to_string(msg.value) +
                    ", Текущая сумма: " + std::to_string(sum));
            }
        }

        log("Финальный результат: " + std::to_string(sum));
    }
};

// Основная функция программы
int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Настройка конфигурации конвейера
    PipelineConfig config;
    config.proc_per_func = { 1, 1, 1 };  // По одному процессу на каждую функцию
    config.use_reduction = true;
    config.generator_count = 3;           // Генерировать 3 числа
    config.generator_start = 0;           // Начать с 0
    config.generator_step = 1;            // Шаг 1

    // Добавление функций в конвейер
    config.functions.push_back(std::make_shared<F1>());
    config.functions.push_back(std::make_shared<F2>());
    config.functions.push_back(std::make_shared<F3>());

    try {
        // Создание и запуск конвейера
        Pipeline pipeline(rank, size, config);
        pipeline.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка в процессе " << rank << ": " << e.what() << std::endl;
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}
