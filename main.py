import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox, QHBoxLayout, QStackedWidget, QFrame, QSizePolicy, QListWidget, QListWidgetItem, QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QLinearGradient, QIcon
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
import os
import torch
from datetime import datetime

class VideoProcessThread(QThread):
    update_frame = pyqtSignal(QImage)
    update_event = pyqtSignal(bool, bool, bool)  # 足球检测、摔倒检测和犯规检测
    foul_detected = pyqtSignal(float)  # 发送犯规时间的信号
    segment_finished = pyqtSignal()

    def __init__(self, video_path, models, start_time, end_time):
        super().__init__()
        self.video_path = video_path
        self.models = models
        self.skip_frames = 15  # 增加到15，即每15帧处理一帧
        self.start_time = start_time
        self.end_time = end_time
        self.is_running = True
        self.fall_detection_threshold = 3
        self.fall_detection_counter = 0
        self.foul_detection_window = 60  # 加检测窗口以适应更高的跳帧率
        self.fall_detected_frames = []
        self.last_foul_time = None
        self.foul_detected_flag = False  # 新增：用于标记是否检测到犯规

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(self.start_time * fps)
        end_frame = int(self.end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        while cap.isOpened() and frame_count < end_frame and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % (self.skip_frames + 1) != 0:
                continue

            annotated_frame = frame.copy()
            football_detected = False
            fall_detected = False
            foul_detected = False

            # 对每一帧使用所有模型
            for model_name, model in self.models.items():
                results = model(frame, conf=0.7, iou=0.5)
                
                if model_name == 'football':
                    football_detected = self.process_football_detection(results)
                elif model_name == 'full':
                    fall_detected = self.process_fall_detection(results)
                    foul_detected = self.process_foul_detection(results)
                
                # 处理 'person' 和 'rules' 模型的结果
                if model_name in ['person', 'rules']:
                    # 这里可以添加特定的处理逻辑
                    pass

                # 在annotated_frame上绘制所有模型的结果
                for r in results:
                    annotated_frame = r.plot(img=annotated_frame)
            
            current_time = (frame_count - start_frame) / fps + self.start_time

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.update_frame.emit(qt_image)
            self.update_event.emit(football_detected, fall_detected, foul_detected)

            if foul_detected and fall_detected:
                self.foul_detected.emit(current_time)

        cap.release()
        self.segment_finished.emit()

    def stop(self):
        self.is_running = False

    def process_fall_detection(self, results):
        fall_detected = False
        for r in results:
            for c in r.boxes.cls:
                if self.models['full'].names[int(c)] == 'fall':
                    self.fall_detection_counter += 1
                    if self.fall_detection_counter >= self.fall_detection_threshold:
                        fall_detected = True
                    break
            if fall_detected:
                break
        
        if not fall_detected:
            self.fall_detection_counter = max(0, self.fall_detection_counter - 1)
        
        return fall_detected

    def process_football_detection(self, results):
        for r in results:
            for c in r.boxes.cls:
                if self.models['football'].names[int(c)] == 'football':
                    return True
        return False

    def process_foul_detection(self, results):
        for r in results:
            for c in r.boxes.cls:
                if self.models['full'].names[int(c)] == 'foul':
                    return True
        return False

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.segment_finished_label = None
        self.initUI()
        self.load_models()
        self.thread = None
        self.video_path = None
        self.video_segments = []
        self.football_detected = False
        self.fall_detected = False
        self.foul_detected = False
        self.foul_recorded = False
        self.foul_list = QListWidget()

    def load_models(self):
        model_paths = {
            'football': 'C:/Users/Administrator/Desktop/result_demo/football_model.pt',
            'full': 'C:/Users/Administrator/Desktop/result_demo/full_best.pt',
            'person':'C:/Users/Administrator/Desktop/result_demo/person_best.pt',
            'rules' : 'C:/Users/Administrator/Desktop/result_demo/rules_best.pt'
        }
        self.models = {}
        for name, path in model_paths.items():
            if not os.path.exists(path):
                QMessageBox.critical(self, "错误", f"模型文件不存在: {path}")
                sys.exit(1)
            try:
                self.models[name] = YOLO(path)
                self.models[name].to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU加速
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型时出错: {str(e)}")
                sys.exit(1)

    def initUI(self):
        self.setWindowTitle('基于YOLOv8的足球赛场识别检测')
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        self.stacked_widget = QStackedWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)

        # 主页面
        home_page = QWidget()
        home_layout = QVBoxLayout(home_page)
        home_layout.setSpacing(40)
        home_layout.setContentsMargins(60, 60, 60, 60)

        title_label = QLabel('基于YOLOv8的足球赛场识别检测')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Segoe UI', 36, QFont.Bold))
        title_label.setStyleSheet("""
            color: #00ffff;
            margin-bottom: 50px;
            padding: 30px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        """)
        home_layout.addWidget(title_label)

        self.upload_btn = QPushButton('上传视频')
        self.upload_btn.setFont(QFont('Segoe UI', 20))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aaff;
                color: white;
                border: none;
                padding: 25px 50px;
                border-radius: 15px;
                box-shadow: 0 0 15px rgba(0, 170, 255, 0.5);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #0088cc;
                transform: translateY(-3px);
                box-shadow: 0 0 25px rgba(0, 170, 255, 0.7);
            }
            QPushButton:pressed {
                background-color: #006699;
                transform: translateY(1px);
                box-shadow: 0 0 10px rgba(0, 170, 255, 0.3);
            }
        """)
        self.upload_btn.clicked.connect(self.upload_video)
        home_layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)

        # 视频段落选择页面
        segment_page = QWidget()
        segment_layout = QVBoxLayout(segment_page)
        segment_layout.setContentsMargins(20, 20, 20, 20)
        segment_layout.setSpacing(20)

        self.segment_list = QListWidget()
        self.segment_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border-radius: 15px;
                padding: 10px;
            }
            QListWidget::item {
                background-color: #3a3a3a;
                color: #ffffff;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: #4a4a4a;
            }
        """)
        segment_layout.addWidget(self.segment_list)

        process_btn = QPushButton('处理选中段落')
        process_btn.setFont(QFont('Segoe UI', 16))
        process_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aaff;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 170, 255, 0.5);
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:pressed {
                background-color: #006699;
            }
        """)
        process_btn.clicked.connect(self.process_selected_segment)
        segment_layout.addWidget(process_btn, alignment=Qt.AlignCenter)

        # 视频处理页面
        video_page = QWidget()
        video_layout = QHBoxLayout(video_page)
        video_layout.setContentsMargins(20, 20, 20, 20)
        video_layout.setSpacing(20)

        # 左侧事件检测区域
        event_frame = QFrame()
        event_frame.setFixedWidth(200)
        event_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 15px;
                padding: 10px;
            }
        """)
        event_layout = QVBoxLayout(event_frame)
        
        self.football_icon = QLabel()
        self.football_icon.setPixmap(QIcon("path_to_football_icon.png").pixmap(64, 64))
        self.football_icon.setAlignment(Qt.AlignCenter)
        event_layout.addWidget(self.football_icon)
        
        self.fall_icon = QLabel()
        self.fall_icon.setPixmap(QIcon("path_to_fall_icon.png").pixmap(64, 64))
        self.fall_icon.setAlignment(Qt.AlignCenter)
        self.fall_icon.hide()  # 初始时隐藏
        event_layout.addWidget(self.fall_icon)
        
        self.foul_icon = QLabel()
        self.foul_icon.setPixmap(QIcon("path_to_foul_icon.png").pixmap(64, 64))
        self.foul_icon.setAlignment(Qt.AlignCenter)
        self.foul_icon.hide()  # 初始时隐藏
        event_layout.addWidget(self.foul_icon)
        
        self.segment_finished_label = QLabel("段落播放完成")
        self.segment_finished_label.setAlignment(Qt.AlignCenter)
        self.segment_finished_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 16px;
                font-weight: bold;
                background-color: #1a1a1a;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.segment_finished_label.hide()  # 初始时隐藏
        event_layout.addWidget(self.segment_finished_label)
        
        # 在左侧事件检测区域添加犯规列表
        self.foul_list = QListWidget()
        self.foul_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border-radius: 10px;
                padding: 5px;
                color: #ffffff;
            }
            QListWidget::item {
                padding: 5px;
            }
        """)
        event_layout.addWidget(self.foul_list)
        
        event_layout.addStretch()
        
        video_layout.addWidget(event_frame)

        # 中央视频区域
        central_layout = QVBoxLayout()
        video_frame = QFrame()
        video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 25px;
                padding: 20px;
                box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
            }
        """)
        video_frame_layout = QVBoxLayout(video_frame)
        video_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 3px solid #00ffff;
                border-radius: 15px;
            }
        """)
        video_frame_layout.addWidget(self.image_label)

        central_layout.addWidget(video_frame)

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00aaff;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        central_layout.addWidget(self.progress_slider)

        video_layout.addLayout(central_layout)

        button_layout = QHBoxLayout()
        back_btn = QPushButton('返回主页')
        back_btn.setFont(QFont('Segoe UI', 16))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff3b30;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(255, 59, 48, 0.5);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #d63030;
                transform: translateY(-2px);
                box-shadow: 0 0 25px rgba(255, 59, 48, 0.7);
            }
            QPushButton:pressed {
                background-color: #b02a2a;
                transform: translateY(1px);
                box-shadow: 0 0 10px rgba(255, 59, 48, 0.3);
            }
        """)
        back_btn.clicked.connect(self.show_home_page)
        button_layout.addWidget(back_btn)

        switch_segment_btn = QPushButton('切换段落')
        switch_segment_btn.setFont(QFont('Segoe UI', 16))
        switch_segment_btn.setStyleSheet("""
            QPushButton {
                background-color: #4cd964;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(76, 217, 100, 0.5);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #3cb371;
                transform: translateY(-2px);
                box-shadow: 0 0 25px rgba(76, 217, 100, 0.7);
            }
            QPushButton:pressed {
                background-color: #2e8b57;
                transform: translateY(1px);
                box-shadow: 0 0 10px rgba(76, 217, 100, 0.3);
            }
        """)
        switch_segment_btn.clicked.connect(self.switch_segment)
        button_layout.addWidget(switch_segment_btn)

        # Add a new button to save results
        save_results_btn = QPushButton('保存结果')
        save_results_btn.setFont(QFont('Segoe UI', 16))
        save_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 122, 255, 0.5);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #0056b3;
                transform: translateY(-2px);
                box-shadow: 0 0 25px rgba(0, 122, 255, 0.7);
            }
            QPushButton:pressed {
                background-color: #003d80;
                transform: translateY(1px);
                box-shadow: 0 0 10px rgba(0, 122, 255, 0.3);
            }
        """)
        save_results_btn.clicked.connect(self.save_results_as_html)
        button_layout.addWidget(save_results_btn)

        central_layout.addLayout(button_layout)

        self.stacked_widget.addWidget(home_page)
        self.stacked_widget.addWidget(segment_page)
        self.stacked_widget.addWidget(video_page)

        self.setLayout(main_layout)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.segment_video()
            self.show_segment_page()

    def segment_video(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps
        segment_duration = 40  # 每段40秒

        self.video_segments = []
        for i in range(0, int(total_duration), segment_duration):
            start_time = i
            end_time = min(i + segment_duration, total_duration)
            self.video_segments.append((start_time, end_time))

        cap.release()

        self.segment_list.clear()
        for i, (start, end) in enumerate(self.video_segments):
            item = QListWidgetItem(f"段落 {i+1}: {start:.1f}s - {end:.1f}s")
            self.segment_list.addItem(item)

    def show_segment_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def process_selected_segment(self):
        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请选择一个视频段落")
            return

        selected_index = self.segment_list.row(selected_items[0])
        start_time, end_time = self.video_segments[selected_index]
        self.process_video(self.video_path, start_time, end_time)
        self.stacked_widget.setCurrentIndex(2)  # 切换到视频处理页面

    def process_video(self, video_path, start_time, end_time):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        self.thread = VideoProcessThread(video_path, self.models, start_time, end_time)
        self.thread.update_frame.connect(self.update_frame)
        self.thread.update_event.connect(self.update_event)
        self.thread.foul_detected.connect(self.record_foul)
        self.thread.segment_finished.connect(self.on_segment_finished)
        self.thread.start()
        if self.segment_finished_label:
            self.segment_finished_label.hide()
        self.football_detected = False
        self.fall_detected = False
        self.foul_detected = False
        self.foul_recorded = False

    def update_event(self, football_detected, fall_detected, foul_detected):
        self.football_detected = football_detected
        self.fall_detected = fall_detected
        self.foul_detected = foul_detected
        
        if football_detected:
            self.football_icon.show()
        else:
            self.football_icon.hide()
        
        if fall_detected:
            self.fall_icon.show()
        else:
            self.fall_icon.hide()
        
        if foul_detected:
            self.foul_icon.show()
        else:
            self.foul_icon.hide()

    def record_foul(self, foul_time):
        if not self.foul_recorded:
            foul_time_str = f"{int(foul_time // 60):02d}:{int(foul_time % 60):02d}"
            foul_item = QListWidgetItem(f"犯规发生时间: {foul_time_str}")
            self.foul_list.addItem(foul_item)
            self.foul_recorded = True
            
            # 记录到txt文件
            with open("foul_record.txt", "a") as f:
                f.write(f"犯规发生时间: {foul_time_str}, 记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def update_frame(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))
        # 更新进度条（如果需要的话）
        # self.progress_slider.setValue(...)

    def show_home_page(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        self.stacked_widget.setCurrentIndex(0)  # 切换回主页面

    def switch_segment(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        self.show_segment_page()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_label.pixmap():
            self.update_frame(self.image_label.pixmap().toImage())

    def on_segment_finished(self):
        if self.segment_finished_label:
            self.segment_finished_label.show()  # 显示段落完成标签
        else:
            print("Warning: segment_finished_label is not initialized")

    def save_results_as_html(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先上传并处理视频")
            return

        video_name = os.path.basename(self.video_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"results_{current_time}.html"

        with open(html_filename, "w", encoding="utf-8") as f:
            f.write("<html><head><title>视频处理结果</title></head><body>")
            f.write(f"<h1>视频处理结果 - {video_name}</h1>")
            f.write("<h2>检测到的犯规:</h2>")
            f.write("<ul>")
            for i in range(self.foul_list.count()):
                f.write(f"<li>{self.foul_list.item(i).text()}</li>")
            f.write("</ul>")
            f.write("</body></html>")

        QMessageBox.information(self, "保存成功", f"结果已保存为 {html_filename}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gradient = QLinearGradient(0, 0, 0, 400)
    gradient.setColorAt(0.0, QColor(0, 0, 0))
    gradient.setColorAt(1.0, QColor(25, 25, 25))
    palette = QPalette()
    palette.setBrush(QPalette.Window, gradient)
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())