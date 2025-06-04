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
            cv2.imread('stickers/proxy-image.jpeg', cv2.IMREAD_UNCHANGED),
            cv2.imread('stickers/proxy-image.png', cv2.IMREAD_UNCHANGED),
            cv2.imread('stickers/glassesb.png', cv2.IMREAD_UNCHANGED)
        ]
        for i, s in enumerate(self.stickers):
            if s is None:
                print(f"Sticker {i+1} failed to load!")
            else:
                print(f"Sticker {i+1} loaded: shape {s.shape}")
        self.selected_sticker_index = None  # No sticker selected by default

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
        self.sticker3_btn = QPushButton('3')
        self.sticker3_btn.setStyleSheet('background-color: #232323; color: white; border-radius: 4px; font-size: 14px; padding: 4px 16px;')
        self.sticker3_btn.clicked.connect(lambda: self.set_sticker(2))
        self.stickers_menu_layout.addWidget(self.sticker1_btn)
        self.stickers_menu_layout.addWidget(self.sticker2_btn)
        self.stickers_menu_layout.addWidget(self.sticker3_btn)

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
        # Turn off face detection if it was on
        self.face_detection_active = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from webcam!")
            return
        frame = cv2.flip(frame, 1)
        frame = self.apply_filter(frame)
        # Only face detection (yellow rectangles)
        if self.face_detection_active:
            frame = self.draw_face_rects(frame)
        # Only stickers (no rectangles)
        elif self.selected_sticker_index is not None:
            frame = self.apply_stickers_only(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def draw_face_rects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Detected {len(faces)} faces.")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame

    def apply_stickers_only(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Detected {len(faces)} faces for stickers.")
        sticker = self.stickers[self.selected_sticker_index]
        print(f"Sticker to overlay: type={type(sticker)}, shape={getattr(sticker, 'shape', None)}")
        for (x, y, w, h) in faces:
            frame = self.overlay_sticker(frame, sticker, x, y, w, h)
        return frame

    def overlay_sticker(self, frame, sticker, x, y, w, h):
        if sticker is None or sticker.shape[2] not in [3, 4]:
            print("Sticker not loaded or wrong format!")
            return frame
        # Resize sticker to face width
        sticker_width = w
        sticker_height = int(sticker.shape[0] * (w / sticker.shape[1]))
        sticker_resized = cv2.resize(sticker, (sticker_width, sticker_height), interpolation=cv2.INTER_AREA)
        sh, sw = sticker_resized.shape[:2]
        # For glasses, place in the middle of the face rectangle (fine-tuned higher)
        if self.selected_sticker_index == 2:  # Glasses (index 2)
            y1 = y - h // 12
        else:  # For hats, place above the face
            y1 = max(0, y - sh)
        y2 = y1 + sh
        x1 = x
        x2 = x + sw

        # Clip to image bounds
        img_h, img_w = frame.shape[:2]
        x1_clip, x2_clip = max(0, x1), min(img_w, x2)
        y1_clip, y2_clip = max(0, y1), min(img_h, y2)

        # Calculate the region of the sticker to use
        sticker_x1 = x1_clip - x1
        sticker_x2 = sticker_x1 + (x2_clip - x1_clip)
        sticker_y1 = y1_clip - y1
        sticker_y2 = sticker_y1 + (y2_clip - y1_clip)

        if x2_clip > x1_clip and y2_clip > y1_clip:
            sticker_part = sticker_resized[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
            roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]
            if sticker_part.shape[2] == 4:
                sticker_bgr = sticker_part[..., :3]
                mask = sticker_part[..., 3:] / 255.0
                frame[y1_clip:y2_clip, x1_clip:x2_clip] = (roi * (1 - mask) + sticker_bgr * mask).astype(np.uint8)
            else:
                frame[y1_clip:y2_clip, x1_clip:x2_clip] = sticker_part
        else:
            print("Sticker completely out of bounds!")
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
