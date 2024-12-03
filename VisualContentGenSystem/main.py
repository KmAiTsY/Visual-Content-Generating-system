import sys
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QLineEdit, QStyle, QFileDialog, QMessageBox, QComboBox, QGroupBox,
    QFormLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image, ImageEnhance
import numpy as np


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.generated_video_path = None

    def initUI(self):
        self.setWindowTitle('Генерація та відтворення відео')
        self.resize(1000, 700)  # Змінити розміри основного вікна
        self.setStyleSheet("QWidget { background-color: #333; color: #EEE; }")

        button_style = """
        QPushButton {
            background-color: #5C85AD;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #41688A;
        }
        QPushButton:pressed {
            background-color: #315673;
        }
        """
        
        slider_style = """
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1, stop: 0 #5D9CEC, stop: 1 #B4E1FA);
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #eee, stop: 1 #ccc);
            border: 1px solid #777;
            width: 18px;
            margin-top: -2px;
            margin-bottom: -2px;
            border-radius: 4px;
        }
        QSlider::add-page:horizontal {
            background: #fff;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        """

        # Розміщення вікна
        self.layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Лівий бік: введення, кнопки, ползунки
        self.entry_group = QGroupBox("Параметри генерації")
        self.entry_layout = QFormLayout()
        self.entry = QLineEdit(self)
        self.entry.setStyleSheet(
            "QLineEdit { background-color: #555; color: #EEE; border: none; padding: 8px; font-size: 16px; }"
        )
        self.entry_layout.addRow("Текст для генерації:", self.entry)
        self.entry_group.setLayout(self.entry_layout)
        self.left_layout.addWidget(self.entry_group)

        self.style_group = QGroupBox("Стиль відео")
        self.style_layout = QFormLayout()
        self.style_combo = QComboBox(self)
        self.style_combo.addItems(["Оригінальний", "Високий контраст", "Монохромний", "Насичений"])
        self.style_layout.addRow("Виберіть стиль:", self.style_combo)
        self.style_group.setLayout(self.style_layout)
        self.left_layout.addWidget(self.style_group)

        self.frames_slider = QSlider(Qt.Horizontal)
        self.frames_slider.setMinimum(1)
        self.frames_slider.setMaximum(30)
        self.frames_slider.setValue(16)
        self.frames_slider.valueChanged.connect(self.update_frames_label)
        self.frames_slider.setStyleSheet(slider_style)
        self.frames_label = QLabel('Вибрано: 16 кадрів')
        self.left_layout.addWidget(self.frames_label)
        self.left_layout.addWidget(self.frames_slider)

        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(5000)
        self.steps_slider.setValue(50)
        self.steps_slider.valueChanged.connect(self.update_steps_label)
        self.steps_slider.setStyleSheet(slider_style)
        self.steps_label = QLabel('Вибрано: 50 кроків')
        self.left_layout.addWidget(self.steps_label)
        self.left_layout.addWidget(self.steps_slider)

        buttons = [
            ("Згенерувати відео 1", self.on_button_click_1),
            ("Згенерувати відео 2", self.on_button_click_2),
            ("Зберегти відео", self.save_video),
            ("Профіль", self.show_profile),
            ("Інформація про продукт", self.show_product_info),
            ("Порівняти моделі", self.compare_models),
        ]
        for text, callback in buttons:
            button = QPushButton(text, self)
            button.clicked.connect(callback)
            button.setStyleSheet(button_style)
            self.left_layout.addWidget(button)

        # Правий бік: відеовіджети
        self.video_widget1 = QVideoWidget(self)
        self.right_layout.addWidget(self.video_widget1)
        self.player1 = QMediaPlayer(self)
        self.player1.setVideoOutput(self.video_widget1)

        self.video_widget2 = QVideoWidget(self)
        self.right_layout.addWidget(self.video_widget2)
        self.player2 = QMediaPlayer(self)
        self.player2.setVideoOutput(self.video_widget2)

        self.player1.mediaStatusChanged.connect(self.loop_video1)
        self.player2.mediaStatusChanged.connect(self.loop_video2)

        # Додавання компонентів до основного вікна
        self.layout.addLayout(self.left_layout, 1)
        self.layout.addLayout(self.right_layout, 2)
        self.setLayout(self.layout)
        self.show()

    def loop_video1(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player1.setPosition(0)
            self.player1.play()

    def loop_video2(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player2.setPosition(0)
            self.player2.play()

    def update_frames_label(self, value):
        self.frames_label.setText(f'Вибрано: {value} кадрів')

    def update_steps_label(self, value):
        self.steps_label.setText(f'Вибрано: {value} кроків')

    def auto_select_parameters(self, prompt):
        if "fast" in prompt:
            return 10, 100  # Швидка генерація
        elif "detailed" in prompt:
            return 30, 1000  # Детальна генерація
        return 16, 500  # Стандартні параметри

    def on_button_click_1(self):
        self.generate_video(self.player1)

    def on_button_click_2(self):
        self.generate_video(self.player2)

    def generate_video(self, player):
        prompt = self.entry.text()
        num_frames = self.frames_slider.value()
        num_steps = self.steps_slider.value()
        filter_type = self.style_combo.currentText()
        self.generated_video_path = self.return_video_with_filters(prompt, num_frames, num_steps, filter_type)
        video_path = self.generated_video_path
        player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        player.play()
        
    def compare_models(self):
        prompt = self.entry.text()
        num_frames, num_steps = self.auto_select_parameters(prompt)
        video_path_1 = self.return_video_with_filters(prompt, num_frames, num_steps)
        video_path_2 = self.return_video_with_filters(prompt, 
        num_frames, num_steps, filter_type="Високий контраст")
        # Завантаження відео для порівняння
 
        self.player1.setMedia(QMediaContent(QUrl.fromLocalFile(video_path_1)))
        self.player1.play()
 
        self.player2.setMedia(QMediaContent(QUrl.fromLocalFile(video_path_2)))
        self.player2.play()


    

    def return_video_with_filters(self, prompt, num_frames, num_inference_steps, filter_type=None):
        # Завантажуємо модель для генерації відео
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Генеруємо кадри
        video_frames = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames
        ).frames

        # Переводимо кадри з numpy.ndarray в PIL.Image
        # Масштабуємо пікселі від [0, 1] до [0, 255] і конвертуємо до uint8
        video_frames = [
            Image.fromarray(np.squeeze(np.clip(frame, 0, 1) * 255).astype(np.uint8)) for frame in video_frames
        ]

        # Застосування фільтрів
        if filter_type == "Високий контраст":
            video_frames = [ImageEnhance.Contrast(frame).enhance(2) for frame in video_frames]
        elif filter_type == "Монохромний":
            video_frames = [frame.convert("L").convert("RGB") for frame in video_frames]
        elif filter_type == "Насичений":
            video_frames = [ImageEnhance.Color(frame).enhance(2) for frame in video_frames]

        # Переконатися, що всі кадри у форматі RGB
        video_frames = [frame.convert("RGB") for frame in video_frames]

        # Експортуємо відео
        video_path = export_to_video(video_frames)
        torch.cuda.empty_cache()

        return video_path



    def apply_filter(self, video_frames, filter_type):
        if filter_type == "Високий контраст":
            return [ImageEnhance.Contrast(frame).enhance(2) for frame in video_frames]
        elif filter_type == "Монохромний":
            return [frame.convert("L") for frame in video_frames]
        elif filter_type == "Насичений":
            return [ImageEnhance.Color(frame).enhance(2) for frame in video_frames]
        return video_frames

    def save_video(self):
        if not self.generated_video_path:
            QMessageBox.warning(self, "Помилка", "Немає згенерованого відео для збереження.")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Зберегти відео", "", "Video Files (*.mp4);;All Files (*)", options=options)
        if file_path:
            try:
                shutil.copy(self.generated_video_path, file_path)
                QMessageBox.information(self, "Збереження", "Відео збережено успішно!")
            except Exception as e:
                QMessageBox.warning(self, "Помилка", f"Не вдалося зберегти відео: {e}")

    def show_profile(self):
        QMessageBox.information(self, "Профіль", "Інформація про користувача")

    def show_product_info(self):
        product_info = """
        Продукт: Генератор Відео на Основі Тексту
        Версія: 1.0.0
        """
        QMessageBox.information(self, "Про продукт", product_info)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    sys.exit(app.exec_())
