#define _CRT_SECURE_NO_WARNINGS

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

std::mutex cout_mutex;

std::string get_formatted_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S:");
    ss << "-" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Безопасный вывод в консоль с временной меткой
template<typename T>
void safe_cout(T message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[" << get_formatted_time() << "] " << message << std::endl;
}

// Форматирование времени выполнения
std::string format_duration(std::chrono::microseconds duration) {
    auto microseconds = duration.count();
    return std::to_string(microseconds) + "μs";
}

enum Tags {
    DATA_TAG = 1,
    TERMINATION_TAG = 2
};

struct DataMessage {
    int value;
    int source;
    bool is_termination;
    long long timestamp;  // Добавляем временную метку

    DataMessage() : value(0), source(0), is_termination(false),
        timestamp(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) {}
};

class PipelineFunction {
public:
    virtual int process(int input) = 0;
    virtual std::string getName() const = 0;
    virtual ~PipelineFunction() = default;
};

class F1 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Имитация работы
        return input + 1;
    }
    std::string getName() const override { return "F1 (increment)"; }
};

class F2 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Имитация работы
        return input * input;
    }
    std::string getName() const override { return "F2 (square)"; }
};

class F3 : public PipelineFunction {
public:
    int process(int input) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Имитация работы
        return input;
    }
    std::string getName() const override { return "F3 (identity)"; }
};

class Generator {
private:
    int count;
    int start_value;
    int step;

public:
    Generator(int count = 3, int start = 0, int step = 1)
        : count(count), start_value(start), step(step) {}

    std::vector<int> generate() {
        std::vector<int> result;
        for (int i = 0; i < count; ++i) {
            result.push_back(start_value + i * step);
        }
        return result;
    }
};

struct PipelineConfig {
    std::vector<std::shared_ptr<PipelineFunction>> functions;
    std::vector<int> proc_per_func;
    bool use_reduction;
    int generator_count;
    int generator_start;
    int generator_step;

    PipelineConfig() : generator_count(3), generator_start(0), generator_step(1) {}
};

class Pipeline {
private:
    int rank;
    int total_procs;
    PipelineConfig config;
    std::vector<std::pair<int, int>> rank_ranges;
    std::chrono::system_clock::time_point start_time;

    std::string getElapsedTime() {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        return std::to_string(duration.count()) + "ms";
    }

    void log(const std::string& message) {
        std::stringstream ss;
        ss << "[Process " << std::setw(2) << rank << " | " << getElapsedTime() << "] " << message;
        safe_cout(ss.str());
    }

    void setupRankRanges() {
        int current_rank = 1;
        for (size_t i = 0; i < config.proc_per_func.size(); ++i) {
            rank_ranges.push_back({ current_rank, current_rank + config.proc_per_func[i] - 1 });
            current_rank += config.proc_per_func[i];
        }
    }

    void processMessage(const DataMessage& msg, int stage, int next_rank) {
        auto start_process = std::chrono::high_resolution_clock::now();

        if (!msg.is_termination) {
            int result = config.functions[stage]->process(msg.value);

            auto end_process = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_process - start_process);

            log("Function " + config.functions[stage]->getName() +
                " processed value " + std::to_string(msg.value) +
                " -> " + std::to_string(result) +
                " (took " + format_duration(duration) + ")");

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
    Pipeline(int rank, int total_procs, const PipelineConfig& config)
        : rank(rank), total_procs(total_procs), config(config) {
        start_time = std::chrono::system_clock::now();
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
            log("Worker initialized with invalid stage. Exiting.");
            return;
        }

        log("Started worker for stage " + std::to_string(stage) +
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
            auto start_process = std::chrono::high_resolution_clock::now();
            int result = config.functions[stage]->process(msg.value);
            auto end_process = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_process - start_process);

            log("Function " + config.functions[stage]->getName() +
                " processed value " + std::to_string(msg.value) +
                " -> " + std::to_string(result) +
                " (took " + format_duration(duration) + ")");

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

        log("Worker finished");
    }

    void runGenerator() {
        log("Generator starting...");

        Generator gen(config.generator_count, config.generator_start, config.generator_step);
        auto data = gen.generate();

        log("Generated " + std::to_string(data.size()) + " values");

        int current_worker = rank_ranges[0].first;
        for (int value : data) {
            DataMessage msg{};
            msg.value = value;
            msg.source = rank;
            msg.is_termination = false;

            log("Sending value " + std::to_string(value) +
                " to worker " + std::to_string(current_worker));

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
            log("Sending termination signal to worker " + std::to_string(i));
            MPI_Send(&termination, sizeof(DataMessage), MPI_BYTE,
                i, DATA_TAG, MPI_COMM_WORLD);
        }

        log("Generator finished");
    }

    void runCollector() {
        log("Collector starting");

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
                log("Received termination signal (" +
                    std::to_string(termination_signals) + "/" +
                    std::to_string(expected_termination) + ")");
            }
            else {
                sum += msg.value;
                received++;
                log("Received value: " + std::to_string(msg.value) +
                    ", Current sum: " + std::to_string(sum));
            }
        }

        log("Final result: " + std::to_string(sum));
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PipelineConfig config;
    config.proc_per_func = { 1, 1, 1 };
    config.use_reduction = true;
    config.generator_count = 3;
    config.generator_start = 0;
    config.generator_step = 1;

    config.functions.push_back(std::make_shared<F1>());
    config.functions.push_back(std::make_shared<F2>());
    config.functions.push_back(std::make_shared<F3>());

    try {
        Pipeline pipeline(rank, size, config);
        pipeline.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in process " << rank << ": " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
