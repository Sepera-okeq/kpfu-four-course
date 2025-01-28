"""
Модуль реализации кодов Рида-Соломона.

Коды Рида-Соломона - это линейные [n,k,n-k+1]-коды, способные исправлять
до floor((n-k)/2) ошибок, где:
- n: длина закодированного сообщения
- k: длина исходного сообщения
- n-k+1: минимальное расстояние Хэмминга
"""

from extra_math import Field, BinaryField, PolynomialOperations, Matrix, solve_ax_b, create_matrix, pow_over_field

class GeneralizedReedSolomon:
    """
    Реализация обобщенного кода Рида-Соломона.
    
    Основные характеристики:
    - Линейный блочный код
    - Способен исправлять до floor((n-k)/2) ошибок
    - Использует конечные поля Галуа для вычислений
    - Поддерживает как стандартное, так и обобщенное кодирование
    """

    def __init__(self, f, k, n=None, conventional_creation=False, alpha=None, v_arr=1, print_matrices=False):
        """
        Инициализирует кодер/декодер Рида-Соломона.
        
        Args:
            f: Поле Галуа для вычислений (рекомендуется BinaryField(0x11d))
            k: Размер исходного сообщения
            n: Размер закодированного сообщения (опционально)
            conventional_creation: Использовать ли классическое построение
            alpha: Локатор кода или список локаторов
            v_arr: Множители столбцов для матрицы H (1 для нормализованного GRS)
            print_matrices: Выводить ли матрицы G и H после создания
            
        Raises:
            Exception: При некорректных параметрах или несовместимых размерах
        """
        # Проверка корректности поля
        if isinstance(f, Field):
            self.f = f
        else:
            raise Exception("Необходимо предоставить корректное поле")

        self.p = PolynomialOperations(self.f)

        # Непустой массив элементов поля Галуа, определяющий точки оценки для кодирования. 
        # В классическом варианте это степени примитивного элемента, в обобщенном – произвольные ненулевые элементы.
        
        # Инициализация массива локаторов кода
        self.alpha_arr = []
        if alpha and isinstance(alpha, list):
            self.alpha_arr = alpha
        elif n is not None:
            if conventional_creation:
                if alpha and not isinstance(alpha, list):
                    # Классическое построение: alpha^1, alpha^2, ..., alpha^n
                    for i in range(1, n+1):
                        self.alpha_arr.append(pow_over_field(alpha, i, self.f))
                else:
                    raise Exception("alpha должно быть одним элементом поля при conventional_creation=True")
            else:
                try:
                    # Последовательное построение: 2, 3, ..., n+1
                    self.alpha_arr = [self.f.multiply(self.f.one(), i) for i in range(2, n + 2)]
                except:
                    raise Exception("Ошибка создания массива alpha. Попробуйте conventional_creation=True")
        else:
            raise Exception("Необходимо указать либо alpha_arr, либо n")

        if print_matrices:
            print("Массив локаторов кода:", self.alpha_arr)

        # Установка основных параметров кода
        self.n = len(self.alpha_arr)  # Длина закодированного сообщения
        self.k = k                    # Длина исходного сообщения
        self.d = self.n - self.k + 1  # Минимальное расстояние Хэмминга

        # Инициализация множителей столбцов
        if isinstance(v_arr, int):
            self.v_arr = [self.f.multiply(v_arr, self.f.one())]*self.n
        elif isinstance(v_arr, list):
            self.v_arr = v_arr
        else:
            raise TypeError("v_arr должен быть целым числом или списком")

        # Проверка корректности размеров
        if not self.k <= self.n:
            raise Exception(f"Должно выполняться k <= n: k={self.k}, n={self.n}")

        # Создание матриц кодирования
        print(f"Инициализация линейного GRS [{self.n}, {self.k}, {self.d}]-кода")
        print(f"Данный код может исправить до {int((self.d-1)/2)} ошибок\n")
        
        self.generator_matrix = None      # Порождающая матрица
        self.parity_check_matrix = None   # Проверочная матрица
        self.vp_arr = None               # Множители для порождающей матрицы
        
        self.create_matrices()
        
        if print_matrices:
            print(f"Порождающая матрица:\n{self.generator_matrix}")
            print(f"Проверочная матрица:\n{self.parity_check_matrix}\n")

    def encode(self, msg, use_poly=True):
        """
        Кодирует сообщение с помощью кода Рида-Соломона.
        
        Поддерживает два метода кодирования:
        1. Через порождающий многочлен (по умолчанию)
        2. Через умножение на порождающую матрицу
        
        Args:
            msg: Исходное сообщение длины k
            use_poly: Использовать ли метод порождающего многочлена
            
        Returns:
            list: Закодированное сообщение длины n
            
        Raises:
            Exception: Если длина входного сообщения не равна k
        """
        if len(msg) != self.k:
            raise Exception(f"Длина входного сообщения должна быть k={self.k}")

        if use_poly:
            # Кодирование через порождающий многочлен
            encoded_msg = []
            for i in range(0, self.n):
                encoded_msg.append(self.f.multiply(self.vp_arr[i], 
                                                 self.p.poly_call(msg, self.alpha_arr[i])))
            return encoded_msg
        else:
            # Кодирование через умножение матриц: E(m) = m*G
            msg_matrix = Matrix(1, self.k, self.f)
            for i in range(self.k):
                msg_matrix.set(0, i, msg[i])
            return (msg_matrix * self.generator_matrix).to_list(single=True)

    def decode(self, msg):
        """
        Декодирует сообщение и исправляет ошибки.
        
        Алгоритм Петерсона-Горенштейна-Цирера - используется для декодирования и исправления ошибок.
        Он включает в себя нахождение многочлена локаторов ошибок, определение позиций ошибок и вычисление значений ошибок.
        
        Этапы алгоритма декодирования Петерсона-Горенштейна-Цирера:
        1. Вычисляет синдром принятого сообщения
        2. Находит многочлен локаторов ошибок
        3. Находит позиции ошибок
        4. Вычисляет значения ошибок
        5. Исправляет ошибки
        
        Args:
            msg: Принятое сообщение длины n
            
        Returns:
            list: Декодированное сообщение длины k
        """
        # Вычисление синдрома
        msg_synd = self.syndrome(msg)
        msg_matrix = create_matrix([msg], self.f)

        # Если синдром ненулевой, есть ошибки
        if any([syn != self.f.zero() for syn in msg_synd]):
            msg_syndrome = create_matrix([msg_synd], self.f)
            tau = (self.d - 1) // 2  # Максимальное число исправляемых ошибок
            
            # Создание матрицы синдромов для системы уравнений
            syndrome_matrix = Matrix(self.d - 1, tau + 1, self.f, init_val=self.f.zero())
            for i in range(self.d - 1):
                for j in range(i, max(-1, i - tau - 1), -1):
                    syndrome_matrix.set(i, i - j, msg_syndrome.get(0, j))

            # Решение системы для нахождения многочлена локаторов ошибок
            lam_eqs = syndrome_matrix.get_sub_matrix(tau, None, None, None)
            lam_kernel_space = lam_eqs.kernel_space()
            
            if lam_kernel_space != 0:
                # Нахождение коэффициентов многочлена локаторов ошибок
                lam_coeff_matrix = lam_kernel_space * create_matrix(
                    [[1]] * lam_kernel_space.columns, self.f)
                lam_coeff = lam_coeff_matrix.to_list(single=True)

                # Нахождение многочлена значений ошибок
                gamma_coeff_matrix = syndrome_matrix.get_sub_matrix(None, tau, None, None) * lam_coeff_matrix
                gamma_coeff = gamma_coeff_matrix.to_list(single=True)

                # Вычисление НОД для получения истинного многочлена локаторов
                gcd_lg = self.p.poly_gcd(lam_coeff, gamma_coeff)
                error_locator_poly, _ = self.p.poly_divmod(lam_coeff, gcd_lg)

                # Поиск позиций ошибок
                error_locations = []
                for j in range(len(self.alpha_arr)):
                    alpha_inv = self.f.reciprocal(self.alpha_arr[j])
                    if self.p.poly_call(error_locator_poly, alpha_inv) == 0:
                        error_locations.append(j)

                if len(error_locations) != 0:
                    # Решение системы уравнений для нахождения значений ошибок
                    err_matrix = Matrix(self.d - 1, len(error_locations), self.f)
                    for r in range(err_matrix.rows):
                        for c in range(err_matrix.columns):
                            val = self.f.multiply(
                                self.v_arr[error_locations[c]],
                                pow_over_field(
                                    self.alpha_arr[error_locations[c]], r, self.f))
                            err_matrix.set(r, c, val)

                    try:
                        errors = solve_ax_b(err_matrix, msg_syndrome.transpose())
                    except Exception as e:
                        print(f"Ошибка при решении системы для нахождения ошибок: {e}")
                        errors = create_matrix([[0]]*err_matrix.columns, self.f)

                    # Исправление ошибок
                    for i in range(len(error_locations)):
                        msg_matrix.set(0, error_locations[i],
                                     self.f.subtract(msg_matrix.get(0, error_locations[i]), 
                                                   errors.get(i, 0)))

        # Если синдром исправленного сообщения нулевой, восстанавливаем исходное сообщение
        if not self.syndrome(msg_matrix).any():
            return solve_ax_b(self.generator_matrix.transpose(), 
                                      msg_matrix.transpose()).to_list(single=True)
        else:
            # Если не удалось исправить все ошибки, возвращаем нулевое сообщение
            return [0]*self.k

    def syndrome(self, msg):
        """
        Вычисляет синдром принятого сообщения.
        
        Args:
            msg: Принятое сообщение (список или матрица)
            
        Returns:
            list или Matrix: Синдром S_H(msg) = (S_0, ..., S_{d-2})
            
        Raises:
            TypeError: Если msg неверного типа
        """
        if isinstance(msg, list):
            syndrome = []
            for l in range(self.d-1):
                syn_l = self.f.zero()
                for j in range(self.n):
                    syn_l = self.f.add(syn_l, 
                                     self.f.multiply(
                                         self.f.multiply(msg[j], self.v_arr[j]),
                                         pow_over_field(self.alpha_arr[j], l, self.f)))
                syndrome.append(syn_l)
            return syndrome
        elif isinstance(msg, Matrix):
            if msg.columns != self.n:
                raise Exception("Длина входного сообщения должна быть n")
            # Вычисление через умножение матриц: y*H^T
            return msg * self.parity_check_matrix.transpose()
        else:
            raise TypeError("Неподдерживаемый тип входного сообщения")

    def create_matrices(self):
        """
        Создает порождающую и проверочную матрицы кода.
        
        Матрицы не приводятся к стандартной форме G = (I|X), H = (X^T|I),
        так как это усложнило бы декодирование.
        """
        self.generator_matrix = Matrix(self.k, self.n, self.f)
        self.parity_check_matrix = Matrix(self.n - self.k, self.n, self.f)

        # Создание проверочной матрицы
        self.parity_check_matrix = self.create_parity_check_matrix(self.k)

        # Нахождение множителей столбцов для порождающей матрицы
        k_one_t = self.create_parity_check_matrix(1)
        k_one_kernel_space = k_one_t.kernel_space()
        if isinstance(k_one_kernel_space, int):
            raise Exception("Не удалось найти матрицу с k=1. Попробуйте другое значение alpha")

        self.vp_arr = (k_one_kernel_space * create_matrix(
            [[1]] * k_one_kernel_space.columns, self.f)).to_list(single=True)

        # Создание порождающей матрицы
        # Используемая для кодирования сообщения
        for i in range(self.generator_matrix.rows):
            for j in range(self.generator_matrix.columns):
                val = self.f.multiply(
                    self.vp_arr[j], 
                    pow_over_field(self.alpha_arr[j], i, self.f))
                self.generator_matrix.set(i, j, val)

        # Проверка корректности матриц
        if (self.parity_check_matrix * self.generator_matrix.transpose()).any():
            raise Exception("Порождающая и проверочная матрицы несовместимы")
        if not isinstance(self.generator_matrix.transpose().kernel_space(), int):
            raise Exception("Строки порождающей матрицы линейно зависимы")
        if not isinstance(self.parity_check_matrix.transpose().kernel_space(), int):
            raise Exception("Строки проверочной матрицы линейно зависимы")

    def create_parity_check_matrix(self, k):
        """
        Создает проверочную матрицу для заданного k.
        Используемая для вычисления синдрома и проверки корректности кода.
        
        Args:
            k: Размер исходного сообщения
            
        Returns:
            Matrix: Проверочная матрица
        """
        pc_matrix = Matrix(self.n - k, self.n, self.f)
        for i in range(pc_matrix.rows):
            for j in range(pc_matrix.columns):
                val = self.f.multiply(
                    self.v_arr[j],
                    pow_over_field(self.alpha_arr[j], i, self.f))
                pc_matrix.set(i, j, val)
        return pc_matrix