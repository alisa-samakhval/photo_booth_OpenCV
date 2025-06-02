import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from filters import to_grayscale, to_negative


class PhotoBoothUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Photo Booth')
        self.setGeometry(100, 100, 1280, 800)
        self.cap = cv2.VideoCapture(0)
        self.current_filter = 'Normal'
        self.face_detection_active = False
        self.filters = ['Normal', 'Grayscale', 'Negative']

        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load stickers directly here
        self.stickers = [
            cv2.imread('/Users/alisasamohval/Desktop/photo_booth_openCV/stickers/proxy-image.jpeg', cv2.IMREAD_UNCHANGED),
            cv2.imread('/Users/alisasamohval/Desktop/photo_booth_openCV/stickers/proxy-image.png', cv2.IMREAD_UNCHANGED)
        ]
        for i, s in enumerate(self.stickers):
            if s is None:
                print(f"Sticker {i+1} failed to load!")
            else:
                print(f"Sticker {i+1} loaded: shape {s.shape}")
        self.selected_sticker_index = 0  # Default to first sticker

        # Webcam display
        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 720)

        # Buttons
        self.filter_button = QPushButton('Filters')
        self.filter_button.setStyleSheet('background-color: #191919; color: white; border-radius: 8px; font-size: 16px; padding: 8px 24px;')
        self.filter_button.clicked.connect(self.toggle_filter_menu)

        self.face_button = QPushButton('Face Detection')
        self.face_button.setStyleSheet('background-color: #191919; color: white; border-radius: 8px; font-size: 16px; padding: 8px 24px;')
        self.face_button.setCheckable(True)
        self.face_button.clicked.connect(self.toggle_face_detection)

        self.stickers_button = QPushButton('Stickers')
        self.stickers_button.setStyleSheet('background-color: #191919; color: white; border-radius: 8px; font-size: 16px; padding: 8px 24px;')
        self.stickers_button.clicked.connect(self.toggle_stickers_menu)

        # Stickers menu (as buttons)
        self.stickers_menu = QWidget()
        self.stickers_menu_layout = QVBoxLayout()
        self.stickers_menu.setLayout(self.stickers_menu_layout)
        self.stickers_menu.setVisible(False)
        self.sticker1_btn = QPushButton('1')
        self.sticker1_btn.setStyleSheet('background-color: #232323; color: white; border-radius: 4px; font-size: 14px; padding: 4px 16px;')
        self.sticker1_btn.clicked.connect(lambda: self.set_sticker(0))
        self.sticker2_btn = QPushButton('2')
        self.sticker2_btn.setStyleSheet('background-color: #232323; color: white; border-radius: 4px; font-size: 14px; padding: 4px 16px;')
        self.sticker2_btn.clicked.connect(lambda: self.set_sticker(1))
        self.stickers_menu_layout.addWidget(self.sticker1_btn)
        self.stickers_menu_layout.addWidget(self.sticker2_btn)

        # Filter menu (as buttons)
        self.filter_buttons = []
        self.filter_menu = QWidget()
        self.filter_menu_layout = QVBoxLayout()
        self.filter_menu.setLayout(self.filter_menu_layout)
        self.filter_menu.setVisible(False)
        for f in self.filters:
            btn = QPushButton(f)
            btn.setStyleSheet('background-color: #232323; color: white; border-radius: 4px; font-size: 14px; padding: 4px 16px;')
            btn.clicked.connect(lambda checked, name=f: self.set_filter(name))
            self.filter_buttons.append(btn)
            self.filter_menu_layout.addWidget(btn)

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.filter_button)
        top_layout.addWidget(self.face_button)
        top_layout.addWidget(self.stickers_button)
        top_layout.addWidget(self.stickers_menu)
        top_layout.addStretch()
        top_layout.addWidget(self.filter_menu)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)
        self.setLayout(main_layout)

        # Timer for webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def toggle_filter_menu(self):
        self.filter_menu.setVisible(not self.filter_menu.isVisible())

    def set_filter(self, name):
        self.current_filter = name
        self.filter_menu.setVisible(False)

    def toggle_face_detection(self):
        self.face_detection_active = not self.face_detection_active

    def toggle_stickers_menu(self):
        self.stickers_menu.setVisible(not self.stickers_menu.isVisible())

    def set_sticker(self, idx):
        self.selected_sticker_index = idx
        self.stickers_menu.setVisible(False)
        print(f"Sticker {idx+1} selected.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from webcam!")
            return
        frame = cv2.flip(frame, 1)
        frame = self.apply_filter(frame)
        if self.face_detection_active:
            frame = self.detect_and_draw_faces(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def overlay_sticker(self, frame, sticker, x, y, w, h):
        if sticker is None or sticker.shape[2] not in [3, 4]:
            print("Sticker not loaded or wrong format!")
            return frame
        # Resize sticker to face width
        sticker_width = w
        sticker_height = int(sticker.shape[0] * (w / sticker.shape[1]))
        sticker_resized = cv2.resize(sticker, (sticker_width, sticker_height), interpolation=cv2.INTER_AREA)
        sh, sw = sticker_resized.shape[:2]
        # Place the hat above the face
        y1 = max(0, y - sh)
        y2 = y1 + sh
        x1 = x
        x2 = x + sw
        # Check bounds
        if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            print("Sticker out of bounds!")
            return frame
        if sticker_resized.shape[2] == 4:
            # Split sticker into BGR and alpha
            sticker_bgr = sticker_resized[..., :3]
            mask = sticker_resized[..., 3:] / 255.0
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = (roi * (1 - mask) + sticker_bgr * mask).astype(np.uint8)
            print(f"Overlayed sticker with alpha at ({x1},{y1}) size ({sw},{sh})")
        else:
            # No alpha channel, just overlay
            frame[y1:y2, x1:x2] = sticker_resized
            print(f"Overlayed sticker without alpha at ({x1},{y1}) size ({sw},{sh})")
        return frame

    def detect_and_draw_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))
        print(f"Detected {len(faces)} faces.")
        sticker = self.stickers[self.selected_sticker_index]
        print(f"Sticker to overlay: type={type(sticker)}, shape={getattr(sticker, 'shape', None)}")
        for (x, y, w, h) in faces:
            # Draw yellow rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Overlay selected sticker
            frame = self.overlay_sticker(frame, sticker, x, y, w, h)
        return frame

    def apply_filter(self, frame):
        if self.current_filter == 'Grayscale':
            gray = to_grayscale(frame)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == 'Negative':
            return to_negative(frame)
        return frame

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhotoBoothUI()
    window.show()
    sys.exit(app.exec_())
