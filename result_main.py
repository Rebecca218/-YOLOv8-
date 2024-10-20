import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QScrollArea, QFrame, QSplitter, QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import cv2
import torch
from ultralytics import YOLO

class VideoSegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_path = ""
        self.segments = []
        self.yolo_models = [
            YOLO('C:/Users/Administrator/Desktop/result_demo/football_model.pt'),
            YOLO('C:/Users/Administrator/Desktop/result_demo/full_best.pt'),
            YOLO('C:/Users/Administrator/Desktop/result_demo/person_best.pt'),
            YOLO('C:/Users/Administrator/Desktop/result_demo/rules_best.pt')
        ]
        self.output_folder = 'detected_images'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def initUI(self):
        self.setWindowTitle('Video Segmentation and YOLO Detection')
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: #333;
            }
        """)

        main_layout = QHBoxLayout()

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)

        upload_button = QPushButton('Upload Video', self)
        upload_button.clicked.connect(self.upload_video)
        left_layout.addWidget(upload_button)

        self.segment_list = QListWidget(self)
        left_layout.addWidget(self.segment_list)

        self.selected_segment_label = QLabel('Selected segment: None', self)
        self.selected_segment_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.selected_segment_label)

        detect_button = QPushButton('Detect Objects', self)
        detect_button.clicked.connect(self.detect_objects)
        left_layout.addWidget(detect_button)

        self.segment_list.itemClicked.connect(self.on_segment_selected)

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        right_layout.addWidget(self.scroll_area)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def upload_video(self):
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if self.video_path:
            self.segment_video()

    def segment_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        segment_duration = 10  # seconds
        num_segments = int(duration // segment_duration)

        self.segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            self.segments.append((start_time, end_time))
            self.segment_list.addItem(f"Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s")

        if duration % segment_duration > 0:
            start_time = num_segments * segment_duration
            end_time = duration
            self.segments.append((start_time, end_time))
            self.segment_list.addItem(f"Segment {num_segments+1}: {start_time:.2f}s - {end_time:.2f}s")

        cap.release()

    def on_segment_selected(self, item):
        index = self.segment_list.row(item)
        start_time, end_time = self.segments[index]
        self.selected_segment_label.setText(f"Selected segment: {start_time:.2f}s - {end_time:.2f}s")

    def detect_objects(self):
        if not self.video_path or not self.segments:
            return

        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            return

        index = self.segment_list.row(selected_items[0])
        start_time, end_time = self.segments[index]

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Clear previous results
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > end_time * 1000:
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Process every 30th frame
                fall_detected = False
                rules_detected = False
                for model in self.yolo_models:
                    result = model(frame)
                    
                    # Draw bounding boxes on the frame
                    for det in result[0].boxes.data:
                        x1, y1, x2, y2, conf, cls = det
                        if model == self.yolo_models[0]:  # football_model
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red color
                            cv2.putText(frame, 'Football', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        elif model == self.yolo_models[1]:  # full_best model
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color
                            cv2.putText(frame, 'Fall', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                            fall_detected = True
                        elif model == self.yolo_models[2]:  # person_best model
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color
                            cv2.putText(frame, 'Person', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)    
                        elif model == self.yolo_models[3]:  # rules_best model
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Yellow color
                            cv2.putText(frame, 'Rules', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)   
                            rules_detected = True
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Save the image for every detection
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                filename = f'detected_{timestamp:.2f}.jpg'
                cv2.imwrite(os.path.join(self.output_folder, filename), frame)
                
                # Create HTML file for every detection
                html_filename = f'detected_{timestamp:.2f}.html'
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Detection Result</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        h1 {{ color: #333; }}
                        img {{ max-width: 100%; height: auto; }}
                        p {{ font-size: 18px; }}
                    </style>
                </head>
                <body>
                    <h1>Detection at {timestamp:.2f} seconds</h1>
                    <img src="{filename}" alt="Detection Result">
                    <p>Fall detected: {'Yes' if fall_detected else 'No'}</p>
                    <p>Rules detected: {'Yes' if rules_detected else 'No'}</p>
                </body>
                </html>
                """
                with open(os.path.join(self.output_folder, html_filename), 'w') as f:
                    f.write(html_content)
                
                # Convert frame to QImage
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Create QLabel to display the image
                label = QLabel()
                pixmap = QPixmap.fromImage(q_image)
                label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
                
                self.scroll_layout.addWidget(label)
                QApplication.processEvents()  # Update the UI

        cap.release()

        # Save the results for the current segment
        segment_folder = os.path.join(self.output_folder, f"segment_{index + 1}")
        if not os.path.exists(segment_folder):
            os.makedirs(segment_folder)

        for item in os.listdir(self.output_folder):
            if item.startswith("detected_"):
                os.rename(os.path.join(self.output_folder, item), os.path.join(segment_folder, item))

        QMessageBox.information(self, "Detection Completed", f"Detection completed for segment {index + 1}. Results saved in {segment_folder}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoSegmentationApp()
    ex.show()
    sys.exit(app.exec_())
