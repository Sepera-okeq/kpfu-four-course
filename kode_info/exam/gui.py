"""
Графический интерфейс для демонстрации работы кодов Рида-Соломона.
Позволяет кодировать сообщения, вносить ошибки и декодировать их обратно.
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit,
                           QSpinBox, QMessageBox, QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont
from reed_solomon import GeneralizedReedSolomon
from extra_math import BinaryField

class ReedSolomonGUI(QMainWindow):
    """
    Главное окно приложения для работы с кодами Рида-Соломона.
    
    Предоставляет интерфейс для:
    - Ввода исходного сообщения
    - Настройки параметров кодирования
    - Установки вероятностей ошибок
    - Кодирования и декодирования сообщений
    - Визуализации результатов
    """
    
    def __init__(self):
        """
        Инициализация главного окна и всех его компонентов.
        Настраивает внешний вид, создает элементы управления и устанавливает обработчики событий.
        """
        super().__init__()
        
        # Основные параметры окна
        self.setWindowTitle("Кодирование Рида-Соломона")
        self.setGeometry(100, 100, 1000, 800)
        
        # Установка темной темы
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QLineEdit, QSpinBox {
                background-color: #3b3b3b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QTextEdit {
                background-color: #3b3b3b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        
        # Создание основного виджета и компоновки
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        
        # Заголовок приложения
        title_label = QLabel("Система кодирования Рида-Соломона")
        title_label.setStyleSheet("font-size: 24px; color: #ffffff; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Секция ввода сообщения
        self._create_message_section(layout)
        
        # Секция параметров кодирования
        self._create_parameters_section(layout)
        
        # Секция вероятностей ошибок
        self._create_error_probabilities_section(layout)
        
        # Секция кнопок управления
        self._create_control_buttons_section(layout)
        
        # Индикатор прогресса
        self._create_progress_bar(layout)
        
        # Область вывода результатов
        self._create_output_section(layout)
        
        # Инициализация поля Галуа
        self.field = BinaryField(0x11d)
        
    def _create_message_section(self, layout):
        """
        Создает секцию для ввода исходного сообщения.
        
        Args:
            layout: Основной компоновщик окна
        """
        msg_frame = QFrame()
        msg_frame.setStyleSheet("QFrame { background-color: #333333; border-radius: 5px; padding: 10px; }")
        msg_layout = QVBoxLayout(msg_frame)
        
        msg_label = QLabel("Введите сообщение для кодирования:")
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("Введите текст...")
        
        msg_layout.addWidget(msg_label)
        msg_layout.addWidget(self.msg_input)
        layout.addWidget(msg_frame)

    def _create_parameters_section(self, layout):
        """
        Создает секцию для настройки параметров кодирования.
        
        Args:
            layout: Основной компоновщик окна
        """
        params_frame = QFrame()
        params_frame.setStyleSheet("QFrame { background-color: #333333; border-radius: 5px; padding: 10px; }")
        params_layout = QHBoxLayout(params_frame)
        
        # Параметр k - размер исходного сообщения
        k_layout = QVBoxLayout()
        k_label = QLabel("k (размер сообщения):")
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 255)
        self.k_input.setValue(24)
        k_layout.addWidget(k_label)
        k_layout.addWidget(self.k_input)
        
        # Параметр n - размер закодированного сообщения
        n_layout = QVBoxLayout()
        n_label = QLabel("n (размер кода):")
        self.n_input = QSpinBox()
        self.n_input.setRange(1, 255)
        self.n_input.setValue(30)
        n_layout.addWidget(n_label)
        n_layout.addWidget(self.n_input)
        
        params_layout.addLayout(k_layout)
        params_layout.addLayout(n_layout)
        layout.addWidget(params_frame)

    def _create_error_probabilities_section(self, layout):
        """
        Создает секцию для установки вероятностей ошибок.
        
        Args:
            layout: Основной компоновщик окна
        """
        error_frame = QFrame()
        error_frame.setStyleSheet("QFrame { background-color: #333333; border-radius: 5px; padding: 10px; }")
        error_layout = QHBoxLayout(error_frame)
        
        # Вероятность одиночной ошибки
        pe_layout = QVBoxLayout()
        pe_label = QLabel("P(ошибка в байте):")
        self.pe_input = QLineEdit()
        self.pe_input.setText("0.033")
        pe_layout.addWidget(pe_label)
        pe_layout.addWidget(self.pe_input)
        
        # Вероятность последовательной ошибки
        pc_layout = QVBoxLayout()
        pc_label = QLabel("P(послед. ошибка):")
        self.pc_input = QLineEdit()
        self.pc_input.setText("0.055")
        pc_layout.addWidget(pc_label)
        pc_layout.addWidget(self.pc_input)
        
        error_layout.addLayout(pe_layout)
        error_layout.addLayout(pc_layout)
        layout.addWidget(error_frame)

    def _create_control_buttons_section(self, layout):
        """
        Создает секцию с кнопками управления.
        
        Args:
            layout: Основной компоновщик окна
        """
        button_frame = QFrame()
        button_frame.setStyleSheet("QFrame { background-color: #333333; border-radius: 5px; padding: 10px; }")
        button_layout = QHBoxLayout(button_frame)
        
        # Кнопка кодирования
        self.encode_btn = QPushButton("Закодировать")
        self.encode_btn.clicked.connect(self.encode_message)
        self.encode_btn.setMinimumWidth(200)
        
        # Кнопка декодирования
        self.decode_btn = QPushButton("Декодировать с ошибками")
        self.decode_btn.clicked.connect(self.decode_message)
        self.decode_btn.setMinimumWidth(200)
        
        button_layout.addWidget(self.encode_btn)
        button_layout.addWidget(self.decode_btn)
        layout.addWidget(button_frame)

    def _create_progress_bar(self, layout):
        """
        Создает индикатор прогресса операций.
        
        Args:
            layout: Основной компоновщик окна
        """
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #3b3b3b;
            }
            QProgressBar::chunk {
                background-color: #1565c0;
                border-radius: 3px;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

    def _create_output_section(self, layout):
        """
        Создает область для вывода результатов.
        
        Args:
            layout: Основной компоновщик окна
        """
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(300)
        layout.addWidget(self.output_text)

    def start_progress(self, duration=2000):
        """
        Запускает анимацию индикатора прогресса.
        
        Args:
            duration: Длительность анимации в миллисекундах
        """
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_step = 0
        self.progress_timer.start(20)

    def update_progress(self):
        """
        Обновляет состояние индикатора прогресса.
        Вызывается по таймеру каждые 20мс.
        """
        self.progress_step += 1
        progress = int((self.progress_step * 20 / 2000) * 100)
        if progress >= 100:
            self.progress_timer.stop()
            self.progress_bar.hide()
        else:
            self.progress_bar.setValue(progress)

    def encode_message(self):
        """
        Обработчик нажатия кнопки "Закодировать".
        Кодирует введенное сообщение с помощью кода Рида-Соломона.
        """
        try:
            self.encode_btn.setEnabled(False)
            self.decode_btn.setEnabled(False)
            self.start_progress()
            
            # Получение параметров кодирования
            k = self.k_input.value()
            n = self.n_input.value()
            message = self.msg_input.text()
            
            # Создание кодера Рида-Соломона
            grs = GeneralizedReedSolomon(
                f=self.field,
                k=k,
                n=n,
                alpha=0x2,
                v_arr=1,
                conventional_creation=True
            )
            
            # Разбиение сообщения на блоки размера k
            msgs = [message[i:i+k] for i in range(0, len(message), k)]
            if len(msgs[-1]) < k:
                msgs[-1] = msgs[-1] + ' ' * (k - len(msgs[-1]))
            
            # Кодирование каждого блока
            encoded_msg = []
            for msg_block in msgs:
                field_msg = [ord(c) for c in msg_block]
                encoded_msg.extend(grs.encode(field_msg))
            
            # Формирование вывода результатов
            self.output_text.clear()
            html_text = f"""
            <p style="color: #4CAF50; font-size: 14px; margin: 10px 0;">
                <b>Исходное сообщение:</b> {message}
            </p>
            
            <p style="color: #2196F3; font-size: 14px; margin: 10px 0;">
                <b>Закодированное сообщение:</b><br>
            """
            
            # Добавление закодированного сообщения с форматированием
            for i, val in enumerate(encoded_msg):
                html_text += f'<span style="color: #64B5F6;">{val}</span> '
                if (i + 1) % n == 0:
                    html_text += "<br>"
            
            html_text += '<hr style="border: 1px solid #555555; margin: 15px 0;">'
            self.output_text.setHtml(html_text)
            
            # Сохранение для последующего декодирования
            self.last_encoded = encoded_msg
            self.last_grs = grs
            
            QTimer.singleShot(2000, self.enable_buttons)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при кодировании: {str(e)}")
            self.enable_buttons()
    
    def enable_buttons(self):
        """
        Включает кнопки после завершения операции.
        """
        self.encode_btn.setEnabled(True)
        self.decode_btn.setEnabled(True)

    def decode_message(self):
        """
        Обработчик нажатия кнопки "Декодировать с ошибками".
        Вносит случайные ошибки в закодированное сообщение и пытается его декодировать.
        """
        try:
            if not hasattr(self, 'last_encoded'):
                raise Exception("Сначала закодируйте сообщение")
            
            self.encode_btn.setEnabled(False)
            self.decode_btn.setEnabled(False)
            self.start_progress()
            
            # Получение вероятностей ошибок
            p_e = float(self.pe_input.text())  # Вероятность одиночной ошибки
            p_c = float(self.pc_input.text())  # Вероятность последовательной ошибки
            
            # Внесение ошибок в сообщение
            import random
            msg_with_errors = []
            error_positions = []
            
            for i in range(len(self.last_encoded)):
                if i == 0 or msg_with_errors[i-1] == self.last_encoded[i-1]:
                    # Проверка на одиночную ошибку
                    if random.random() < p_e:
                        error = random.randint(0, self.field.size-1)
                        msg_with_errors.append(self.field.add(self.last_encoded[i], error))
                        error_positions.append(i)
                    else:
                        msg_with_errors.append(self.last_encoded[i])
                else:
                    # Проверка на последовательную ошибку
                    if random.random() < (p_e + p_c):
                        error = random.randint(0, self.field.size-1)
                        msg_with_errors.append(self.field.add(self.last_encoded[i], error))
                        error_positions.append(i)
                    else:
                        msg_with_errors.append(self.last_encoded[i])
            
            # Разбиение на блоки для декодирования
            n = self.n_input.value()
            msg_blocks = [msg_with_errors[i:i+n] for i in range(0, len(msg_with_errors), n)]
            
            # Декодирование каждого блока
            decoded_msg = []
            for block in msg_blocks:
                decoded_block = self.last_grs.decode(block)
                decoded_msg.extend(decoded_block)
            
            # Преобразование в текст
            text_msg = ''.join(chr(x) for x in decoded_msg).rstrip()
            
            # Формирование вывода результатов
            self.output_text.clear()
            html_text = """
            <p style="color: #F44336; font-size: 14px; margin: 10px 0;">
                <b>Сообщение с ошибками:</b><br>
            """
            
            # Вывод сообщения с ошибками
            for i, val in enumerate(msg_with_errors):
                if i in error_positions:
                    html_text += f'<span style="color: #FF5252; font-weight: bold;">{val}</span> '
                else:
                    html_text += f'<span style="color: #64B5F6;">{val}</span> '
                if (i + 1) % n == 0:
                    html_text += "<br>"
            
            # Вывод декодированного сообщения и статистики
            error_rate = len(error_positions) / len(msg_with_errors) * 100
            html_text += f"""
            </p>
            <p style="color: #4CAF50; font-size: 14px; margin: 10px 0;">
                <b>Декодированное сообщение:</b> {text_msg}
            </p>
            <p style="color: #FFC107; font-size: 14px; margin: 10px 0;">
                <b>Статистика ошибок:</b><br>
                Количество ошибок: {len(error_positions)}<br>
                Процент ошибок: {error_rate:.2f}%
            </p>
            <hr style="border: 1px solid #555555; margin: 15px 0;">
            """
            
            self.output_text.setHtml(html_text)
            QTimer.singleShot(2000, self.enable_buttons)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при декодировании: {str(e)}")
            self.enable_buttons()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReedSolomonGUI()
    window.show()
    sys.exit(app.exec_())
