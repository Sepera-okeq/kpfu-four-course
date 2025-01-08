#ifndef MATRIX_H
#define MATRIX_H

#include <limits>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <vector>

#include <stdint.h>

/**
 * Матрица битов.
 *
 * Биты эффективно хранятся в блоках целочисленного типа (в данном случае uint64_t).
 *
 * Пример использования:
 *
 * Matrix M(4, 4);
 * M(1, 1) = true;
 * std::cout << M << std::endl;
 *
 * Вывод:
 *
 * 0 0 0 0
 * 0 1 0 0
 * 0 0 0 0
 * 0 0 0 0
 */
class Matrix {
public:
    // Целочисленный тип, используемый для блоков битов.
    typedef uint64_t Block;

    // Количество битов в блоке.
    const static uint32_t BitsPerBlock = std::numeric_limits<Block>::digits;

    // Вспомогательный класс для ссылки на отдельный бит.
    //
    // Используется как возвращаемый тип для оператора() (int, int), позволяя, например:
    // M(i, j) = true и if (!M(i, j)) { ... }.
    class Value {
    public:
        friend class Matrix;

        // Присваивание булева значения биту.
        inline Value& operator=(bool rhs) {
            m_block = rhs ? m_block | m_mask : m_block & ~m_mask;
            return *this;
        }

        // Неявное преобразование бита в булево значение.
        inline operator bool() const {
            return (m_block & m_mask) != 0;
        }

        // Инвертирование бита.
        inline void flip() {
            m_block ^= m_mask;
        }

    private:
        Value();
        Value(const Matrix &matrix, uint32_t row, uint32_t col) :
            m_block(matrix.m_matrix[row][col / BitsPerBlock]),
            m_mask(Block(1) << col % BitsPerBlock) { }

        Block& m_block;  // Блок, в котором находится этот бит.
        Block m_mask;    // Маска для получения бита в блоке.
    };

    // Создание новой матрицы с заданными размерами, инициализированной нулями.
    Matrix(uint32_t rows, uint32_t cols) :
        m_blocksPerRow(cols / BitsPerBlock + 1),
        m_rows(rows),
        m_cols(cols)
    {
        m_matrix = new Block*[m_rows];
        for (uint32_t i = 0; i < m_rows; ++i) {
            m_matrix[i] = new Block[m_blocksPerRow];
            std::fill(m_matrix[i], m_matrix[i] + m_blocksPerRow, 0);
        }
    }

    // Создание новой матрицы на основе другой матрицы (конструктор копирования).
    Matrix(const Matrix& other) {
        m_rows = other.rows();
        m_cols = other.cols();
        m_blocksPerRow = m_cols / BitsPerBlock + 1;

        m_matrix = new Block*[m_rows];
        for (uint32_t i = 0; i < m_rows; ++i) {
            m_matrix[i] = new Block[m_blocksPerRow];
            for (uint32_t j = 0; j < m_cols; ++j)
                (*this)(i, j) = (bool)other(i, j);
        }
    }

    // Загрузка данных матрицы из строки.
    void load(const std::string &in) {
        std::stringstream ss(in);
        for (uint32_t i = 0; i < m_rows; ++i) {
            for (uint32_t j = 0; j < m_cols; ++j) {
                bool bit; ss >> bit;
                (*this)(i, j) = bit;
            }
        }
    }

    // Деструктор.
    ~Matrix() {
        for (uint32_t i = 0; i < m_rows; ++i)
            delete[] m_matrix[i];
        delete[] m_matrix;
    }

    // Количество строк / столбцов.
    inline uint32_t rows() const { return m_rows; }
    inline uint32_t cols() const { return m_cols; }

    // Возвращает ссылку на отдельный бит (константную).
    inline Value operator() (uint32_t row, uint32_t col) const {
        return Value(*this, row, col);
    }

    // Возвращает ссылку на отдельный бит (неконстантную).
    inline Value operator() (uint32_t row, uint32_t col) {
        return Value(*this, row, col);
    }

    // Добавляет строку i к строке j (по модулю 2), сохраняя результат в строке j.
    inline void addRows(uint32_t i, uint32_t j) {
        for (uint32_t k = 0; k < m_blocksPerRow; ++k)
            m_matrix[j][k] ^= m_matrix[i][k];
    }

    // Меняет местами строки i и j.
    inline void swapRows(uint32_t i, uint32_t j) {
        std::swap(m_matrix[i], m_matrix[j]);
    }

    // Очищает строку i, устанавливая все элементы в 0.
    inline void clearRow(uint32_t i) {
        std::fill(m_matrix[i], m_matrix[i] + m_blocksPerRow, 0);
    }

    // Приводит матрицу к ступенчатому виду, используя метод Гаусса.
    inline void reduce() {
        uint32_t i = 0, j = 0;
        while (i < rows() && j < cols()) {
            uint32_t maxi = i;
            // Находим ведущий элемент.
            for (uint32_t k = i + 1; k < rows(); ++k) {
                if ((*this)(k, j)) {
                    maxi = k;
                    break;
                }
            }
            if ((*this)(maxi, j)) {
                // Выполняем перестановку.
                swapRows(i, maxi);
                for (uint32_t l = i + 1; l < rows(); ++l) {
                    if ((*this)(l, j)) {
                        addRows(i, l);
                    }
                }
                ++i;
            }
            ++j;
        }
    }

    /*
     * Выполняет обратную подстановку на копии матрицы и возвращает вектор решения
     * x для Ax = b. Предполагается, что матрица уже приведена к ступенчатому виду
     * и система является недоопределенной.
     */
    std::vector<uint32_t> solve() const {
        Matrix M(*this); // Работаем с копией.

        std::vector<uint32_t> x(cols() - 1, 0);
        int32_t i = rows() - 1;
        while (i >= 0) {
            // Подсчитываем количество единиц в текущей строке.
            int32_t count = 0;
            int32_t current = -1;
            for (uint32_t j = 0; j < cols() - 1; ++j) {
                count += M(i, j);
                current = M(i, j) ? j : current;
            }
            if (count == 0) {
                --i;
                continue; // Строка пустая, переходим выше.
            }

            // Добавляем случайность, чтобы избежать тривиального решения.
            uint32_t x_current = count > 1 ? rand() % 2 : M(i, cols() - 1);
            x[current] = x_current;

            for (int32_t k = 0; k <= i; ++k) {
                if (M(k, current)) {
                    if (x_current == 1)
                        M(k, cols() - 1).flip(); // Добавляем к правой части.
                    M(k, current) = false;       // Удаляем из левой части.
                }
            }
            if (count == 1)
                --i; // Закончили с этой строкой, переходим выше.
        }

        return x;
    }

private:
    Matrix();     // Закрытый конструктор по умолчанию.

    Block **m_matrix;        // Блоки матрицы (порядок по строкам).
    uint32_t m_blocksPerRow; // Количество блоков в строке.
    uint32_t m_rows;         // Количество строк (в битах).
    uint32_t m_cols;         // Количество столбцов (в битах).
};

// Оператор вывода в поток.
std::ostream& operator<<(std::ostream& os, const Matrix &matrix)
{
    for (uint32_t i = 0; i < matrix.rows(); ++i) {
        for (uint32_t j = 0; j < matrix.cols(); ++j) {
            os << std::left << std::setw(2) << matrix(i, j);
        }
        if (i < matrix.rows() - 1)
            os << std::endl;
    }
    return os;
}

#endif // MATRIX_H
