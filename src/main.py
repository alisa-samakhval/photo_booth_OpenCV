import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from filters import to_grayscale, to_negative


class PhotoBoothUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Photo Booth')
        self.setGeometry(100, 100, 1280, 800)

        # Ensure save directories exist
        # Photos are saved to: ./captures/ (relative to your project folder)
        # Videos are saved to: ./records/ (relative to your project folder)
        self.capture_dir = 'captures'
        self.record_dir = 'records'
        os.makedirs(self.capture_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.current_filter = 'Normal'
        self.face_detection_active = False
        self.filters = ['Normal', 'Grayscale', 'Negative']

        # Recording state
        self.recording = False
        self.video_writer = None

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load stickers
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
        self.selected_sticker_index = None

        # Display label
        self.image_label = QLabel()
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Top controls
        self.filter_button = QPushButton('Filters')
        self.filter_button.setCheckable(True)
        self.filter_button.clicked.connect(self.toggle_filter_menu)
        self.face_button = QPushButton('Face Detection')
        self.face_button.setCheckable(True)
        self.face_button.clicked.connect(self.toggle_face_detection)
        self.stickers_button = QPushButton('Stickers')
        self.stickers_button.setCheckable(True)
        self.stickers_button.clicked.connect(self.toggle_stickers_menu)

        # Bottom controls (round red buttons)
        self.capture_button = QPushButton('📸')
        self.capture_button.setFixedSize(60, 60)
        self.capture_button.setStyleSheet(
            'background-color: red; border-radius: 30px; color: white; font-size: 24px;'
        )
        self.capture_button.clicked.connect(self.capture_image)

        self.record_button = QPushButton('●')
        self.record_button.setCheckable(True)
        self.record_button.setFixedSize(60, 60)
        self.record_button.setStyleSheet(
            'background-color: red; border-radius: 30px; color: white; font-size: 24px;'
        )
        self.record_button.clicked.connect(self.toggle_recording)

        # Stickers menu
        self.stickers_menu = QWidget()
        stickers_layout = QVBoxLayout(self.stickers_menu)
        self.stickers_menu.setVisible(False)
        for idx in range(len(self.stickers)):
            btn = QPushButton(str(idx+1))
            btn.clicked.connect(lambda _, i=idx: self.set_sticker(i))
            stickers_layout.addWidget(btn)

        # Filter menu
        self.filter_menu = QWidget()
        filters_layout = QVBoxLayout(self.filter_menu)
        self.filter_menu.setVisible(False)
        for f in self.filters:
            btn = QPushButton(f)
            btn.clicked.connect(lambda _, name=f: self.set_filter(name))
            filters_layout.addWidget(btn)

        # Layout assembly
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.filter_button)
        top_layout.addWidget(self.face_button)
        top_layout.addWidget(self.stickers_button)
        top_layout.addWidget(self.stickers_menu)
        top_layout.addStretch()
        top_layout.addWidget(self.filter_menu)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.capture_button)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(self.record_button)
        bottom_layout.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def toggle_filter_menu(self):
        if self.filter_button.isChecked():
            self.filter_menu.setVisible(True)
            # Close other menus
            self.stickers_menu.setVisible(False)
            self.stickers_button.setChecked(False)
        else:
            self.filter_menu.setVisible(False)

    def set_filter(self, name):
        if self.current_filter == name:
            # If clicking the same filter, turn it off
            self.current_filter = 'Normal'
            self.filter_button.setChecked(False)
            self.filter_menu.setVisible(False)
        else:
            self.current_filter = name
            # Keep menu open for further selections

    def toggle_face_detection(self):
        self.face_detection_active = self.face_button.isChecked()
        
        # Automatically enable grayscale filter when face detection is on
        if self.face_detection_active:
            self.current_filter = 'Grayscale'
            # Update filter button state to show grayscale is active
            self.filter_button.setChecked(True)
        else:
            # Reset to normal filter when face detection is turned off
            self.current_filter = 'Normal'
            self.filter_button.setChecked(False)

    def toggle_stickers_menu(self):
        if self.stickers_button.isChecked():
            self.stickers_menu.setVisible(True)
            # Close other menus
            self.filter_menu.setVisible(False)
            self.filter_button.setChecked(False)
        else:
            self.stickers_menu.setVisible(False)

    def set_sticker(self, idx):
        if self.selected_sticker_index == idx:
            # If clicking the same sticker, remove it
            self.selected_sticker_index = None
            self.stickers_button.setChecked(False)
            self.stickers_menu.setVisible(False)
        else:
            self.selected_sticker_index = idx
            # Keep menu open for further selections
        self.face_detection_active = False
        print(f"Sticker {idx+1} {'removed' if self.selected_sticker_index is None else 'selected'}.")

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Capture failed: no frame")
            return
        frame = cv2.flip(frame, 1)
        frame = self.process_frame(frame)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.capture_dir, f'capture_{timestamp}.jpg')
        if cv2.imwrite(path, frame):
            print(f"Saved snapshot {path}")
            # Show confirmation message
            self.show_capture_confirmation()
        else:
            print(f"Failed to save snapshot to {path}")

    def show_capture_confirmation(self):
        # Create a temporary confirmation label
        confirmation = QLabel("📸 Photo Captured!")
        confirmation.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 255, 0, 0.8);
                color: white;
                font-size: 24px;
                padding: 20px;
                border-radius: 10px;
                font-weight: bold;
            }
        """)
        confirmation.setAlignment(Qt.AlignCenter)
        
        # Position it over the image
        confirmation.setFixedSize(300, 80)
        confirmation.move(
            self.image_label.x() + (self.image_label.width() - 300) // 2,
            self.image_label.y() + (self.image_label.height() - 80) // 2
        )
        confirmation.show()
        
        # Hide after 2 seconds
        QTimer.singleShot(2000, confirmation.deleteLater)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.record_dir, f'record_{timestamp}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            print(f"Started recording to {path}")
            # Change button appearance to show recording
            self.record_button.setText('⏹')
            self.record_button.setStyleSheet(
                'background-color: #8B0000; border-radius: 30px; color: white; font-size: 24px;'
            )
        else:
            if self.video_writer:
                self.video_writer.release()
                print("Stopped recording")
                # Show completion message
                self.show_recording_confirmation()
            self.video_writer = None
            # Reset button appearance
            self.record_button.setText('●')
            self.record_button.setStyleSheet(
                'background-color: red; border-radius: 30px; color: white; font-size: 24px;'
            )

    def show_recording_confirmation(self):
        # Create a temporary confirmation label
        confirmation = QLabel("🎬 Recording Saved!")
        confirmation.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 255, 0.8);
                color: white;
                font-size: 24px;
                padding: 20px;
                border-radius: 10px;
                font-weight: bold;
            }
        """)
        confirmation.setAlignment(Qt.AlignCenter)
        
        # Position it over the image
        confirmation.setFixedSize(300, 80)
        confirmation.move(
            self.image_label.x() + (self.image_label.width() - 300) // 2,
            self.image_label.y() + (self.image_label.height() - 80) // 2
        )
        confirmation.show()
        
        # Hide after 2 seconds
        QTimer.singleShot(2000, confirmation.deleteLater)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return
        frame = cv2.flip(frame, 1)
        frame = self.process_frame(frame)

        if self.recording and self.video_writer:
            self.video_writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def process_frame(self, frame):
        frame = self.apply_filter(frame)
        if self.face_detection_active:
            frame = self.draw_face_rects(frame)
        elif self.selected_sticker_index is not None:
            frame = self.apply_stickers_only(frame)
        return frame

    def draw_face_rects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame

    def apply_stickers_only(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        sticker = self.stickers[self.selected_sticker_index]
        for x, y, w, h in faces:
            frame = self.overlay_sticker(frame, sticker, x, y, w, h)
        return frame

    def overlay_sticker(self, frame, sticker, x, y, w, h):
        if sticker is None or sticker.shape[2] not in (3, 4):
            return frame
        scale = w / sticker.shape[1]
        sw, sh = int(sticker.shape[1] * scale), int(sticker.shape[0] * scale)
        sticker_resized = cv2.resize(sticker, (sw, sh), interpolation=cv2.INTER_AREA)
        y1 = y - sh if self.selected_sticker_index != 2 else y - h // 12
        y1, x1 = max(0, y1), x
        y2, x2 = y1 + sh, x1 + sw
        y1c, x1c = y1, x1
        y2c, x2c = min(frame.shape[0], y2), min(frame.shape[1], x2)
        sy1, sx1 = 0, 0
        sy2, sx2 = y2c - y1c, x2c - x1c
        part = sticker_resized[sy1:sy2, sx1:sx2]
        roi = frame[y1c:y2c, x1c:x2c]
        if part.shape[2] == 4:
            bgr, mask = part[..., :3], part[..., 3:] / 255.0
            frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + bgr * mask).astype(np.uint8)
        else:
            frame[y1c:y2c, x1c:x2c] = part
        return frame

    def apply_filter(self, frame):
        if self.current_filter == 'Grayscale':
            gray = to_grayscale(frame)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self.current_filter == 'Negative':
            return to_negative(frame)
        return frame

    def closeEvent(self, event):
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhotoBoothUI()
    window.show()
    sys.exit(app.exec_())
