"""
Модуль предоставляет комплексную реализацию математических структур и операций над полями Галуа,
матрицами и многочленами, необходимых для эффективного кодирования по методу Рида-Соломона,
включая полную поддержку конечных полей GF(2^m), операции сложения, умножения, деления и поиска обратных элементов в поле,
матричные преобразования (приведение к ступенчатому виду, поиск ядра, решение систем линейных уравнений),
а также широкий набор операций над многочленами (сложение, вычитание, деление с остатком, поиск НОД) -
все эти компоненты реализованы с учетом специфики работы в конечных полях и оптимизированы для использования в задачах
помехоустойчивого кодирования.
"""

class Field:
    """
    Абстрактный базовый класс для реализации полей.
    
    Поле - это математическая структура, удовлетворяющая определенным аксиомам:
    1. Замкнутость относительно операций сложения и умножения
    2. Ассоциативность сложения и умножения
    3. Коммутативность сложения и умножения
    4. Существование нейтральных элементов (0 для сложения, 1 для умножения)
    5. Существование обратных элементов
    6. Дистрибутивность умножения относительно сложения
    
    Подробнее: https://mathworld.wolfram.com/FieldAxioms.html
    """

    def zero(self):
        """
        Возвращает аддитивный нейтральный элемент поля (0).
        Для любого элемента x в поле: x + 0 = x
        """
        raise NotImplementedError("Метод zero() должен быть реализован в подклассе")

    def one(self):
        """
        Возвращает мультипликативный нейтральный элемент поля (1).
        Для любого элемента x в поле: x * 1 = x
        """
        raise NotImplementedError("Метод one() должен быть реализован в подклассе")

    def equals(self, x, y):
        """
        Проверяет равенство двух элементов поля.
        
        Args:
            x: Первый элемент поля
            y: Второй элемент поля
            
        Returns:
            bool: True если элементы равны, False в противном случае
        """
        raise NotImplementedError("Метод equals() должен быть реализован в подклассе")

    def add(self, x, y):
        """
        Выполняет операцию сложения двух элементов поля.
        
        Args:
            x: Первый элемент поля
            y: Второй элемент поля
            
        Returns:
            Результат сложения x + y
        """
        raise NotImplementedError("Метод add() должен быть реализован в подклассе")

    def negate(self, x):
        """
        Возвращает аддитивно обратный элемент для x.
        Для элемента x возвращает такой -x, что x + (-x) = 0
        
        Args:
            x: Элемент поля
            
        Returns:
            Аддитивно обратный элемент для x
        """
        raise NotImplementedError("Метод negate() должен быть реализован в подклассе")

    def subtract(self, x, y):
        """
        Выполняет операцию вычитания y из x.
        
        Args:
            x: Уменьшаемое
            y: Вычитаемое
            
        Returns:
            Результат вычитания x - y
        """
        raise NotImplementedError("Метод subtract() должен быть реализован в подклассе")

    def multiply(self, x, y):
        """
        Выполняет операцию умножения двух элементов поля.
        
        Args:
            x: Первый множитель
            y: Второй множитель
            
        Returns:
            Результат умножения x * y
        """
        raise NotImplementedError("Метод multiply() должен быть реализован в подклассе")

    def reciprocal(self, x):
        """
        Возвращает мультипликативно обратный элемент для x.
        Для ненулевого элемента x возвращает такой x^(-1), что x * x^(-1) = 1
        
        Args:
            x: Ненулевой элемент поля
            
        Returns:
            Мультипликативно обратный элемент для x
        """
        raise NotImplementedError("Метод reciprocal() должен быть реализован в подклассе")

    def divide(self, x, y):
        """
        Выполняет операцию деления x на y.
        
        Args:
            x: Делимое
            y: Делитель (не равный нулю)
            
        Returns:
            Результат деления x / y
        """
        raise NotImplementedError("Метод divide() должен быть реализован в подклассе")

    def __eq__(self, other):
        """
        Проверяет равенство двух полей.
        
        Args:
            other: Другое поле для сравнения
            
        Returns:
            bool: True если поля равны, False в противном случае
        """
        raise NotImplementedError("Метод __eq__() должен быть реализован в подклассе")

    def __ne__(self, other):
        """
        Проверяет неравенство двух полей.
        
        Args:
            other: Другое поле для сравнения
            
        Returns:
            bool: True если поля не равны, False в противном случае
        """
        return not self.__eq__(other)


class BinaryField(Field):
    """
    Реализация двоичного поля Галуа GF(2^m).
    
    В этом поле каждый элемент представляется как многочлен над GF(2),
    где коэффициенты могут быть только 0 или 1. Операции выполняются по модулю
    неприводимого многочлена.
    
    Подробнее:
    - https://sites.math.washington.edu/~morrow/336_12/papers/juan.pdf
    - https://en.wikipedia.org/wiki/Finite_field_arithmetic
    """

    def __init__(self, mod):
        """
        Инициализирует двоичное поле Галуа GF(2^m).
        
        Args:
            mod: Неприводимый многочлен (в двоичном представлении).
                 Рекомендуется использовать 0x11d для GF(2^8)
        """
        self.mod = mod  # Неприводимый многочлен
        self.size = 1 << (mod.bit_length() - 1)  # Размер поля (2^m)
        
        # Инициализация таблиц для оптимизации вычислений
        self.mults = [[None] * self.size for _ in range(self.size)]  # Таблица умножения
        self.recips = [None] * self.size  # Таблица обратных элементов
        self.pows = [[None] * self.size for _ in range(self.size)]  # Таблица степеней
        self._init_tables()

    def _init_tables(self):
        """
        Инициализирует таблицы умножения, обратных элементов и степеней
        для оптимизации вычислений.
        """
        # Заполнение таблицы умножения
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.mults[i][j] = self._multiply_no_lookup(i, j)

        # Заполнение таблицы степеней
        for i in range(self.size):
            for j in range(self.size):
                self.pows[i][j] = pow_over_field(i, j, self)

        # Заполнение таблицы обратных элементов (кроме нуля)
        for i in range(1, self.size):
            self.recips[i] = self._reciprocal_no_lookup(i)

    def _is_valid(self, x):
        """
        Проверяет, является ли элемент допустимым для данного поля.
        
        Args:
            x: Проверяемый элемент
            
        Returns:
            int: Проверенный элемент
            
        Raises:
            TypeError: Если x не является целым числом
            ValueError: Если x не принадлежит полю
        """
        if not isinstance(x, int):
            raise TypeError("Элемент поля должен быть целым числом")
        if x < 0 or x >= self.size:
            raise ValueError(f"Элемент {x} не принадлежит полю")
        return x

    def zero(self):
        """Возвращает нулевой элемент поля (0)"""
        return 0

    def one(self):
        """Возвращает единичный элемент поля (1)"""
        return 1

    def equals(self, x, y):
        """
        Проверяет равенство двух элементов поля.
        
        Args:
            x, y: Сравниваемые элементы
            
        Returns:
            bool: True если элементы равны
        """
        return self._is_valid(x) == self._is_valid(y)

    def add(self, x, y):
        """
        Складывает два элемента поля.
        В двоичном поле сложение выполняется как XOR коэффициентов.
        
        Args:
            x, y: Слагаемые
            
        Returns:
            Сумма x + y
        """
        return self._is_valid(x) ^ self._is_valid(y)

    def negate(self, x):
        """
        Возвращает аддитивно обратный элемент.
        В двоичном поле элемент является обратным сам к себе: -x = x
        
        Args:
            x: Элемент поля
            
        Returns:
            Аддитивно обратный элемент
        """
        return self._is_valid(x)

    def subtract(self, x, y):
        """
        Вычитает y из x.
        В двоичном поле вычитание эквивалентно сложению: x - y = x + y
        
        Args:
            x: Уменьшаемое
            y: Вычитаемое
            
        Returns:
            Разность x - y
        """
        return self._is_valid(x) ^ self._is_valid(y)

    def multiply(self, x, y):
        """
        Умножает два элемента поля.
        Использует предварительно вычисленную таблицу умножения.
        
        Args:
            x, y: Множители
            
        Returns:
            Произведение x * y
        """
        self._is_valid(x)
        self._is_valid(y)
        if self.mults[x][y] is not None:
            return self.mults[x][y]
        return self._multiply_no_lookup(x, y)

    def _multiply_no_lookup(self, x, y):
        """
        Выполняет умножение без использования таблицы умножения.
        Реализует классический алгоритм умножения многочленов по модулю.
        
        Args:
            x, y: Множители
            
        Returns:
            Произведение x * y
        """
        p = 0
        while y != 0:
            if y & 1 != 0:
                p ^= x
            x <<= 1
            if x >= self.size:
                x ^= self.mod
            y >>= 1
        return p

    def reciprocal(self, x):
        """
        Возвращает мультипликативно обратный элемент.
        Использует предварительно вычисленную таблицу обратных элементов.
        
        Args:
            x: Ненулевой элемент поля
            
        Returns:
            Мультипликативно обратный элемент
        """
        self._is_valid(x)
        if self.recips[x] is not None:
            return self.recips[x]
        return self._reciprocal_no_lookup(x)

    def _reciprocal_no_lookup(self, x):
        """
        Вычисляет обратный элемент без использования таблицы.
        Использует расширенный алгоритм Ито-Цудзии.
        
        Args:
            x: Ненулевой элемент поля
            
        Returns:
            Мультипликативно обратный элемент
        """
        return pow_over_field(x, self.size - 2, self)

    def divide(self, x, y):
        """
        Делит x на y.
        Реализуется как умножение x на обратный элемент y.
        
        Args:
            x: Делимое
            y: Делитель (не равный нулю)
            
        Returns:
            Частное x / y
        """
        return self.multiply(x, self.reciprocal(y))

    def __eq__(self, other):
        """
        Проверяет равенство с другим полем.
        Поля равны, если они имеют одинаковый неприводимый многочлен.
        
        Args:
            other: Другое поле
            
        Returns:
            bool: True если поля равны
        """
        if isinstance(other, BinaryField):
            return self.mod == other.mod
        return False


def pow_over_field(base, exp, field):
    """
    Возводит элемент поля в степень.
    
    Args:
        base: Основание (элемент поля)
        exp: Показатель степени (неотрицательное целое число)
        field: Поле, в котором выполняется операция
        
    Returns:
        Результат возведения в степень
        
    Raises:
        ValueError: Если exp отрицательный
        Exception: Если field не является полем
    """
    if exp < 0:
        raise ValueError("Показатель степени не может быть отрицательным")
    if not isinstance(field, Field):
        raise Exception("Аргумент field должен быть экземпляром класса Field")
        
    # Проверяем наличие предварительно вычисленного значения
    if isinstance(field, BinaryField):
        if field.pows[base][exp] is not None:
            return field.pows[base][exp]
        if field.pows[base][exp - 1] is not None:
            return field.multiply(base, field.pows[base][exp - 1])

    # Вычисляем степень последовательным умножением
    res = field.one()
    for _ in range(exp):
        res = field.multiply(base, res)
    return res


class Matrix:
    """
    Реализация матрицы над произвольным полем.
    Поддерживает основные матричные операции, необходимые для
    кодирования Рида-Соломона.
    """

    def __init__(self, rows, columns, field, init_val=None):
        """
        Создает матрицу заданного размера над указанным полем.
        
        Args:
            rows: Количество строк
            columns: Количество столбцов
            field: Поле, над которым определена матрица
            init_val: Начальное значение для всех элементов (по умолчанию None)
            
        Raises:
            ValueError: Если размеры неположительны
            TypeError: Если аргументы имеют неверный тип
        """
        if rows <= 0 or columns <= 0:
            raise ValueError("Размеры матрицы должны быть положительными")
        if not isinstance(field, Field) or not isinstance(rows, int) or not isinstance(columns, int):
            raise TypeError("Неверные типы аргументов")

        self.f = field
        self.values = [[init_val] * columns for _ in range(rows)]
        self.rows = rows
        self.columns = columns

    def get(self, r, c):
        """
        Возвращает элемент матрицы.
        
        Args:
            r: Индекс строки
            c: Индекс столбца
            
        Returns:
            Элемент матрицы в позиции (r,c)
            
        Raises:
            IndexError: Если индексы выходят за границы матрицы
        """
        if r < 0 or c < 0 or r >= self.rows or c >= self.columns:
            raise IndexError("Индексы выходят за границы матрицы")
        return self.values[r][c]

    def set(self, r, c, val):
        """
        Устанавливает значение элемента матрицы.
        
        Args:
            r: Индекс строки
            c: Индекс столбца
            val: Новое значение
            
        Raises:
            IndexError: Если индексы выходят за границы матрицы
        """
        if r < 0 or c < 0 or r >= self.rows or c >= self.columns:
            raise IndexError("Индексы выходят за границы матрицы")
        self.values[r][c] = val

    def get_sub_matrix(self, row_i, row_t, col_i, col_t):
        """
        Возвращает подматрицу.
        
        Args:
            row_i: Начальный индекс строк (включительно)
            row_t: Конечный индекс строк (исключительно)
            col_i: Начальный индекс столбцов (включительно)
            col_t: Конечный индекс столбцов (исключительно)
            
        Returns:
            Matrix: Подматрица
            
        Raises:
            Exception: Если границы подматрицы некорректны
        """
        if row_i is None:
            row_i = 0
        if row_t is None:
            row_t = self.rows
        if col_i is None:
            col_i = 0
        if col_t is None:
            col_t = self.columns

        if row_t <= row_i or col_t <= col_i:
            raise Exception("Некорректные границы подматрицы")
            
        result = self.__class__(row_t - row_i, col_t - col_i, self.f)
        for r in range(row_i, row_t):
            for c in range(col_i, col_t):
                result.set(r - row_i, c - col_i, self.get(r, c))
        return result

    def to_list(self, single=False):
        """
        Преобразует матрицу в список.
        
        Args:
            single: Если True, возвращает одномерный список
                   Если False, возвращает список списков
            
        Returns:
            list: Матрица в виде списка
        """
        lst = []
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                if single:
                    lst.append(self.get(r, c))
                else:
                    row.append(self.get(r, c))
            if not single:
                lst.append(row)
        return lst

    def __str__(self):
        """
        Возвращает строковое представление матрицы.
        
        Returns:
            str: Форматированная строка с элементами матрицы
        """
        matrix_str = "    "
        for (i, row) in enumerate(self.values):
            if i > 0:
                matrix_str += " \n    "
            matrix_str += " ".join(str(val) + " "*(5-len(str(val))) for val in row)
        return matrix_str + ""

    def __mul__(self, other):
        """
        Умножает матрицу на другую матрицу.
        
        Args:
            other: Вторая матрица
            
        Returns:
            Matrix: Результат умножения
            
        Raises:
            TypeError: Если other не является матрицей
            Exception: Если поля не совпадают или размеры несовместимы
        """
        if not isinstance(other, Matrix):
            raise TypeError("Умножение возможно только с другой матрицей")
        if self.f != other.f:
            raise Exception("Матрицы должны быть над одним полем")
        if self.columns != other.rows:
            raise Exception("Размеры матриц несовместимы для умножения")

        result = self.__class__(self.rows, other.columns, self.f)
        for r in range(result.rows):
            for c in range(result.columns):
                val = self.f.zero()
                for i in range(self.columns):
                    val = self.f.add(val, self.f.multiply(self.get(r, i), other.get(i, c)))
                result.set(r, c, val)
        return result

    def transpose(self):
        """
        Возвращает транспонированную матрицу.
        
        Returns:
            Matrix: Транспонированная матрица
        """
        result = self.__class__(self.columns, self.rows, self.f)
        for r in range(result.rows):
            for c in range(result.columns):
                result.set(r, c, self.get(c, r))
        return result

    def any(self):
        """
        Проверяет, есть ли в матрице ненулевые элементы.
        
        Returns:
            bool: True если есть хотя бы один ненулевой элемент
        """
        for r in range(self.rows):
            for c in range(self.columns):
                if not self.f.equals(self.get(r, c), self.f.zero()):
                    return True
        return False

    def copy(self):
        """
        Создает копию матрицы.
        
        Returns:
            Matrix: Копия матрицы
        """
        result = self.__class__(self.rows, self.columns, self.f)
        result.values = self.values
        return result

    def swap_rows(self, r1, r2):
        """
        Меняет местами две строки матрицы.
        
        Args:
            r1, r2: Индексы строк
            
        Raises:
            IndexError: Если индексы выходят за границы матрицы
        """
        if r1 < 0 or r2 < 0 or r1 >= self.rows or r2 >= self.rows:
            raise IndexError("Индексы выходят за границы матрицы")
        self.values[r1], self.values[r2] = self.values[r2], self.values[r1]

    def multiply_row(self, r, multiplier):
        """
        Умножает строку на скаляр.
        
        Args:
            r: Индекс строки
            multiplier: Множитель
            
        Raises:
            IndexError: Если индекс выходит за границы матрицы
        """
        if r < 0 or r >= self.rows:
            raise IndexError("Индекс выходит за границы матрицы")
        self.values[r] = [self.f.multiply(multiplier, val) for val in self.values[r]]

    def add_rows(self, r1, r2, multiplier):
        """
        Прибавляет к строке r2 строку r1, умноженную на multiplier.
        
        Args:
            r1: Индекс первой строки
            r2: Индекс второй строки
            multiplier: Множитель для первой строки
            
        Raises:
            IndexError: Если индексы выходят за границы матрицы
        """
        if r1 < 0 or r2 < 0 or r1 >= self.rows or r2 >= self.rows:
            raise IndexError("Индексы выходят за границы матрицы")
        self.values[r2] = [self.f.add(val_r2, self.f.multiply(multiplier, val_r1)) 
                          for val_r1, val_r2 in zip(self.values[r1], self.values[r2])]

    def row_echelon_form(self):
        """
        Приводит матрицу к ступенчатому виду (форме Гаусса).
        Модифицирует матрицу на месте.
        """
        lead = 0
        for r in range(self.rows):
            if lead >= self.columns:
                return
            i = r
            while self.values[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.columns == lead:
                        return
            self.swap_rows(i, r)
            lv_recip = self.f.reciprocal(self.values[r][lead])

            for i in range(r, self.rows):
                if i != r:
                    lv = self.values[i][lead]
                    self.add_rows(r, i, self.f.negate(self.f.multiply(lv, lv_recip)))
            lead += 1

    def reduced_row_echelon_form(self):
        """
        Приводит матрицу к приведенной ступенчатой форме (форме Жордана-Гаусса).
        Модифицирует матрицу на месте.
        
        Основано на алгоритме:
        https://rosettacode.org/wiki/Reduced_row_echelon_form#Python
        """
        lead = 0
        for r in range(self.rows):
            if lead >= self.columns:
                return
            i = r
            while self.values[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.columns == lead:
                        return
            self.swap_rows(i, r)
            lv_recip = self.f.reciprocal(self.values[r][lead])
            self.multiply_row(r, lv_recip)

            for i in range(self.rows):
                if i != r:
                    lv = self.values[i][lead]
                    self.add_rows(r, i, self.f.negate(lv))
            lead += 1

    def kernel_space(self):
        """
        Вычисляет ядро матрицы (null space).
        
        Использует метод Гаусса, как описано в:
        https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
        
        Returns:
            Matrix или int: Матрица, представляющая базис ядра, или 0 если ядро тривиально
        """
        ai_matrix = augmented_a_b_matrix(self.transpose(), identity_n(self.columns, self.f))
        ai_matrix.reduced_row_echelon_form()
        res = []
        
        for r in range(ai_matrix.rows):
            valid = True
            for c in range(self.rows):
                if not self.f.equals(ai_matrix.get(r, c), self.f.zero()):
                    valid = False
                    break
            if valid:
                res.append([ai_matrix.get(r, c) for c in range(self.rows, ai_matrix.columns)])

        if len(res) == 0 or len(res[0]) == 0:
            return 0
            
        result = self.__class__(len(res), len(res[0]), self.f)
        for r in range(len(res)):
            for c in range(len(res[0])):
                result.set(r, c, res[r][c])
        return result.transpose()


def identity_n(n, field):
    """
    Создает единичную матрицу размера n×n над заданным полем.
    
    Args:
        n: Размер матрицы
        field: Поле
        
    Returns:
        Matrix: Единичная матрица
        
    Raises:
        TypeError: Если field не является полем
    """
    if not isinstance(field, Field):
        raise TypeError("Аргумент field должен быть экземпляром класса Field")

    result = Matrix(n, n, field)
    for r in range(result.rows):
        for c in range(result.columns):
            if r == c:
                result.set(r, c, field.one())
            else:
                result.set(r, c, field.zero())
    return result


def augmented_a_b_matrix(a, b):
    """
    Создает расширенную матрицу [A|B] из двух матриц A и B.
    
    Args:
        a: Первая матрица
        b: Вторая матрица
        
    Returns:
        Matrix: Расширенная матрица
        
    Raises:
        TypeError: Если a или b не являются матрицами
        Exception: Если поля не совпадают или размеры несовместимы
    """
    if not isinstance(a, Matrix) or not isinstance(b, Matrix):
        raise TypeError("Аргументы должны быть матрицами")
    if a.f != b.f:
        raise Exception("Матрицы должны быть над одним полем")
    if a.rows != b.rows:
        raise Exception(f"Несовместимые размеры матриц: a: {a.rows}x{a.columns} и b: {b.rows}x{b.columns}")

    axb = Matrix(a.rows, a.columns + b.columns, a.f)
    for r in range(axb.rows):
        for c in range(axb.columns):
            if c >= a.columns:
                val = b.get(r, c - a.columns)
            else:
                val = a.get(r, c)
            axb.set(r, c, val)
    return axb


def solve_ax_b(a, b):
    """
    Решает систему линейных уравнений Ax = b методом Гаусса.
    
    Args:
        a: Матрица коэффициентов A
        b: Вектор свободных членов b (матрица-столбец)
        
    Returns:
        Matrix: Решение системы x
        
    Raises:
        TypeError: Если a или b не являются матрицами 
        Exception: Если размеры матриц несовместимы или система не имеет решения
    """

    if not isinstance(a, Matrix) or not isinstance(b, Matrix):
        raise TypeError
    if b.columns != 1 or b.rows != a.rows:
        raise Exception("Матрица b должна быть размера nx1.")

    axb = augmented_a_b_matrix(a, b)
    axb.row_echelon_form()

    # Поиск решения в ступенчатой форме
    c = 0
    result = Matrix(a.columns, 1, a.f, init_val=a.f.zero())
    if axb.rows < axb.columns - 1:
        raise Exception("Невозможно решить, слишком мало строк")
    for r in range(axb.rows-1, -1, -1):
        c = (min(axb.columns - 2, r))
        if c != r:
            if axb.get(r, c) != 0 or axb.get(r, c + 1) != 0:
                raise Exception("Несовместная система или вырожденная матрица A")
        else:
            to_sub = axb.f.zero()
            for c2 in range(c + 1, axb.columns - 1):
                to_sub = axb.f.add(to_sub, axb.f.multiply(axb.get(r, c2), result.get(c2, 0)))
            result.set(c, 0, axb.f.multiply(axb.f.reciprocal(axb.get(r, c)),
                                            axb.f.subtract(axb.get(r, axb.columns - 1), to_sub)))
    return result


def solve_lstsq(a, b):
    """
    Решает систему линейных уравнений методом наименьших квадратов.
    Находит решение нормального уравнения A^T*A*x = A^T*b.
    
    Args:
        a: Матрица коэффициентов A
        b: Вектор свободных членов b
        
    Returns:
        Matrix: Решение системы x
        
    Raises:
        TypeError: Если a или b не являются матрицами
        
    Alert:
        Не работает для двоичных полей и конечных полей в целом.
    """
    if not isinstance(a, Matrix) or not isinstance(a, Matrix):
        raise TypeError

    ata = a.transpose() * a
    atb = a.transpose() * b

    x = solve_ax_b(ata, atb)
    return x


def create_matrix(lst, field):
    """
    Создает матрицу из списка значений над заданным полем.
    
    Args:
        lst: Список значений (может быть списком списков)
        field: Поле, над которым создается матрица
        
    Returns:
        Matrix: Созданная матрица
        
    Raises:
        Exception: Если входные данные некорректны
    """
    rows = len(lst)
    if rows == 0:
        raise Exception("Некорректные входные данные")
    if isinstance(lst[0], list):
        columns = len(lst[0])
    else:
        columns = 1

    result = Matrix(rows, columns, field)
    for r in range(rows):
        for c in range(columns):
            result.set(r, c, lst[r][c])
    return result

class PolynomialOperations:
    """
    Класс для выполнения операций с многочленами над произвольным полем.
    
    Многочлен представляется в виде списка коэффициентов, где индекс
    соответствует степени. Например, многочлен 3 + 3x + x^4 представляется
    как [3, 3, 0, 0, 1].
    """
    
    def __init__(self, field):
        """
        Инициализирует операции над многочленами для заданного поля.
        
        Args:
            field: Поле, над которым определены многочлены
            
        Raises:
            Exception: Если field не является экземпляром класса Field
        """
        if isinstance(field, Field):
            self.f = field
        else:
            raise Exception("Аргумент field должен быть экземпляром класса Field")

    def poly_call(self, poly, x):
        """
        Вычисляет значение многочлена в точке x.
        Использует схему Горнера для эффективного вычисления.
        
        Args:
            poly: Список коэффициентов многочлена
            x: Точка, в которой вычисляется значение
            
        Returns:
            Значение многочлена в точке x
        """
        res = self.f.zero()
        for c in range(len(poly)-1, -1, -1):
            res = self.f.add(self.f.multiply(res, x), poly[c])
        return res

    def poly_degree(self, poly):
        """
        Возвращает степень многочлена.
        
        Args:
            poly: Список коэффициентов многочлена
            
        Returns:
            int: Степень многочлена
            
        Example:
            Для многочлена 3 + 3x + x^4 ([3, 3, 0, 0, 1]) вернёт 4
        """
        self.poly_trim(poly)
        if poly:
            return len(poly) - 1
        else:
            return 0

    def poly_trim(self, poly):
        """
        Удаляет ведущие нулевые коэффициенты многочлена.
        Модифицирует список коэффициентов на месте.
        
        Args:
            poly: Список коэффициентов многочлена
            
        Example:
            [1, 2, 0, 0] -> [1, 2]
        """
        while poly and poly[-1] == self.f.zero():
            poly.pop()
        if len(poly) == 0:
            poly.append(self.f.zero())

    def poly_lead(self, poly):
        """
        Возвращает старший коэффициент многочлена.
        
        Args:
            poly: Список коэффициентов многочлена
            
        Returns:
            Старший коэффициент
            
        Example:
            Для многочлена 3 + 3x + x^4 ([3, 3, 0, 0, 1]) вернёт 1
        """
        self.poly_trim(poly)
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                return poly[i]
        return self.f.zero()
    
    def set(self, poly, degree_term, coeff_val):
        """
        Устанавливает коэффициент при заданной степени.
        
        Args:
            poly: Список коэффициентов многочлена
            degree_term: Степень члена
            coeff_val: Новое значение коэффициента
            
        Example:
            Для многочлена x + 2, set(2, 2) даст 2x^2 + x + 2
        """
        if len(poly) <= degree_term:
            poly += [self.f.zero()] * (degree_term - len(poly) + 1)
        poly[degree_term] = coeff_val

    def poly_divmod(self, poly1, poly2):
        """
        Выполняет деление многочленов с остатком.
        Реализует алгоритм длинного деления многочленов.
        
        Основано на алгоритме:
        https://rosettacode.org/wiki/Polynomial_long_division
        
        Args:
            poly1: Делимое (список коэффициентов)
            poly2: Делитель (список коэффициентов)
            
        Returns:
            tuple: (частное, остаток)
            
        Raises:
            ZeroDivisionError: Если делитель нулевой
        """
        deg_top = self.poly_degree(poly1)
        deg_bot = self.poly_degree(poly2)

        if deg_bot < 0:
            raise ZeroDivisionError("Деление на нулевой многочлен")

        if deg_top >= deg_bot:
            q = [self.f.zero()] * deg_top
            while deg_top >= deg_bot and self.poly_lead(poly1) != self.f.zero():
                # Создаем временный многочлен для вычитания
                d = [self.f.zero()] * (deg_top - deg_bot) + poly2
                # Вычисляем коэффициент для текущего шага деления
                inv = self.f.reciprocal(self.poly_lead(d))
                multiplier = self.f.multiply(self.poly_lead(poly1), inv)
                # Записываем коэффициент в частное
                self.set(q, deg_top - deg_bot, multiplier)
                # Вычитаем произведение делителя на текущий член частного
                d = [self.f.multiply(multiplier, c) for c in d]
                poly1 = self.poly_subtract(poly1, d)
                deg_top = self.poly_degree(poly1)
        else:
            q = [self.f.zero()]
        return q, poly1

    def poly_add(self, poly1, poly2):
        """
        Складывает два многочлена.
        
        Args:
            poly1, poly2: Списки коэффициентов многочленов
            
        Returns:
            list: Сумма многочленов
            
        Example:
            (3 + 2x) + (1 + x) = 4 + 3x
            [3, 2] + [1, 1] = [4, 3]
        """
        return [self.f.add(t1, t2) for t1, t2 in zip_longest(poly1, poly2, fillvalue=self.f.zero())]

    def poly_subtract(self, poly1, poly2):
        """
        Вычитает второй многочлен из первого.
        
        Args:
            poly1: Уменьшаемое (список коэффициентов)
            poly2: Вычитаемое (список коэффициентов)
            
        Returns:
            list: Разность многочленов
            
        Example:
            (3 + 2x) - (1 + x) = 2 + x
            [3, 2] - [1, 1] = [2, 1]
        """
        return [self.f.subtract(t1, t2) for t1, t2 in zip_longest(poly1, poly2, fillvalue=self.f.zero())]

    def poly_equal(self, poly1, poly2):
        """
        Проверяет равенство двух многочленов.
        
        Args:
            poly1, poly2: Списки коэффициентов многочленов
            
        Returns:
            bool: True если многочлены равны
        """
        self.poly_trim(poly1)
        self.poly_trim(poly2)
        return poly1 == poly2

    def poly_not_equal(self, poly1, poly2):
        """
        Проверяет неравенство двух многочленов.
        
        Args:
            poly1, poly2: Списки коэффициентов многочленов
            
        Returns:
            bool: True если многочлены не равны
        """
        return not self.poly_equal(poly1, poly2)

    def poly_gcd(self, poly1, poly2):
        """
        Вычисляет наибольший общий делитель двух многочленов.
        Использует алгоритм Евклида.
        
        Args:
            poly1, poly2: Списки коэффициентов многочленов
            
        Returns:
            list: НОД многочленов
        """
        if self.poly_degree(poly1) > self.poly_degree(poly2):
            return self.poly_gcd(poly2, poly1)

        if self.poly_equal(poly1, [self.f.zero()]):
            return poly2

        _, r = self.poly_divmod(poly2, poly1)
        self.poly_trim(r)

        return self.poly_gcd(r, poly1)


def zip_longest(iter1, iter2, fillvalue=None):
    """
    Вспомогательная функция для объединения двух итерируемых объектов.
    Аналог itertools.zip_longest для работы со списками разной длины.
    
    Args:
        iter1, iter2: Итерируемые объекты
        fillvalue: Значение для заполнения более короткого списка
        
    Yields:
        tuple: Пары элементов из обоих списков
    """
    for i in range(max(len(iter1), len(iter2))):
        if i >= len(iter1):
            yield (fillvalue, iter2[i])
        elif i >= len(iter2):
            yield (iter1[i], fillvalue)
        else:
            yield (iter1[i], iter2[i])
        i += 1
