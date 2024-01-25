import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import tensorflow as tf
import tensorflow_hub as hub

class ImageClassifierApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()

        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont('Arial', 14, QFont.Bold))

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")

        layout = QVBoxLayout()
        layout.addWidget(self.result_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)

        if file_name:
            image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (128, 128))
            image = image[:, :, ::-1]  # Convert BGR to RGB
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Convert image data to bytes
            image_data = image.tobytes(order='C')

            # Create QImage with the converted image data
            q_image = QImage(image_data, 128, 128, QImage.Format_RGB888)
            pixmap = QPixmap(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

            # Inference
            prediction = self.model.predict(image)
            predicted_class = np.argmax(prediction)
            class_names = ['clean', 'dirty']
            result_text = f"<span style='color: {'green' if predicted_class == 0 else 'red'};'>Prediction: {class_names[predicted_class]} (Confidence: {prediction[0][predicted_class]:.2f})</span>"
            self.result_label.setText(result_text)

def main():
    app = QApplication(sys.argv)

    model_path = "D:/Downloads_DataDrive/mobilenetv2_garbage_classifier_retrained_model.h5"  # Update with your actual model path
    image_classifier_app = ImageClassifierApp(model_path)
    image_classifier_app.setGeometry(100, 100, 800, 600)
    image_classifier_app.setWindowTitle('Image Classifier App')
    image_classifier_app.setStyleSheet("background-color: #f0f0f0;")

    image_classifier_app.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
