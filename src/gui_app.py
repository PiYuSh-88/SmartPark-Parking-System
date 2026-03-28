import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from src.utils import load_parking_model
from src.detect_slots import detect_parking_slots

class SmartParkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SmartPark Parking Detection")
        self.setGeometry(100, 100, 800, 600)
        
        self.model_path = 'model/parking_model.h5'
        self.model = None
        self.current_image_path = None
        self.loaded_image = None
        
        self.init_ui()
        self.load_model()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title Label
        title_label = QLabel("SmartPark - Slot Detection System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Image Display Area
        self.image_label = QLabel("Upload an image to get started.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #eee; font-size: 16px;")
        self.image_label.setMinimumSize(600, 400)
        main_layout.addWidget(self.image_label, stretch=1)
        
        # Results Info Label
        self.info_label = QLabel("Status: Waiting for input...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        main_layout.addWidget(self.info_label)
        
        # Buttons Layout
        btn_layout = QHBoxLayout()
        
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_upload.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.btn_upload)
        
        self.btn_detect = QPushButton("Detect Parking")
        self.btn_detect.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_detect.clicked.connect(self.run_detection)
        self.btn_detect.setEnabled(False)
        btn_layout.addWidget(self.btn_detect)
        
        main_layout.addLayout(btn_layout)
        
    def load_model(self):
        """Loads the CNN model on startup."""
        if os.path.exists(self.model_path):
            self.info_label.setText("Status: Loading model...")
            self.model = load_parking_model(self.model_path)
            if self.model:
                self.info_label.setText("Status: Model loaded. Ready.")
            else:
                self.info_label.setText("Status: Failed to load model. Check console.")
                QMessageBox.warning(self, "Model Error", "Failed to load model from " + self.model_path)
        else:
            self.info_label.setText("Status: Model not found. Please train the model first.")
            QMessageBox.information(self, "Model Not Found", "Please run train_model.py first to generate " + self.model_path)

    def upload_image(self):
        """Opens a file dialog to select an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Parking Lot Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
            
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_detect.setEnabled(True)
            self.info_label.setText("Status: Image loaded. Ready for detection.")
            
    def display_image(self, file_path_or_img_array):
        """Displays an image on the QLabel. Handles both path and cv2 image array."""
        if isinstance(file_path_or_img_array, str):
            # Load from path
            pixmap = QPixmap(file_path_or_img_array)
        else:
            # OpenCV image (BGR) format -> QPixmap
            height, width, channel = file_path_or_img_array.shape
            bytes_per_line = 3 * width
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(file_path_or_img_array, cv2.COLOR_BGR2RGB)
            
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit the label while keeping aspect ratio
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def run_detection(self):
        """Runs the slot detection on the loaded image."""
        if not self.model:
            QMessageBox.critical(self, "Error", "Model not loaded. Please train first.")
            return
            
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
            
        try:
            self.info_label.setText("Status: Detecting slots... Please wait.")
            # Process UI events so label updates immediately
            QApplication.processEvents() 
            
            annotated_img, empty_count, occupied_count = detect_parking_slots(
                self.current_image_path, self.model
            )
            
            # Show results
            self.display_image(annotated_img)
            
            total = empty_count + occupied_count
            self.info_label.setText(
                f"Status: Detection Complete | Total Slots: {total} | "
                f"Empty: {empty_count} (GREEN) | Occupied: {occupied_count} (RED)"
            )
            
        except Exception as e:
            self.info_label.setText("Status: Detection failed.")
            QMessageBox.critical(self, "Error", f"An error occurred during detection: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SmartParkApp()
    window.show()
    sys.exit(app.exec_())
