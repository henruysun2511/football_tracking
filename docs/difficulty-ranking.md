# Bảng Xếp Hạng Độ Khó Của Các File Trong Dự Án

## Rất phức tạp (Very Complex)

| Hạng | File | Dòng | Pha | Lý do |
|------|------|------|-----|-------|
| 1 | `main.py` | 266 | Module chính | Orchestrator toàn bộ pipeline: YOLO tracking, optical flow, homography, speed/distance, KMeans, formation, heatmap, minimap. Tích hợp mọi subsystem. |
| 2 | `app/gradio_app.py` | 336 | Triển khai | Web UI Gradio đầy đủ: upload video, progress tracking, caching, pipeline hoàn chỉnh, heatmap, thống kê cầu thủ, xử lý lỗi. |

## Phức tạp (Complex)

| Hạng | File | Dòng | Pha | Lý do |
|------|------|------|-----|-------|
| 3 | `trackers/tracker.py` | 231 | Module chính | Wrapper YOLO + ByteTrack, batch detection, interpolation ball & player tracking, 4 utility vẽ annotations. |
| 4 | `formations/formation_analyzer.py` | 255 | Module chính | Phát hiện đội hình bóng đá: KMeans clustering, quantile-based, direction normalization, multi-frame voting, 27 đội hình. |
| 5 | `pitch_keypoint_detector/pitch_keypoint_detector.py` | 139 | Module chính | YOLO-pose 32 keypoints, smoothing, RANSAC homography, perspective transform, SoccerPitchConfig. |
| 6 | `app/streamlit_app.py` | 318 | Triển khai | Web UI Streamlit: caching, session state, pipeline split, collapsible sections. |
| 7 | `estimators/camera_movement_estimator.py` | 99 | Module chính | Lucas-Kanade optical flow + Shi-Tomasi features, dominant motion vector, position adjustment. |
| 8 | `estimators/view_transformer.py` | 75 | Module chính | Homography từ pitch keypoints, perspective transformation, exponential smoothing, boundary validation. |

## Trung bình (Medium)

| Hạng | File | Dòng | Pha | Lý do |
|------|------|------|-----|-------|
| 9 | `estimators/speed_distance_estimator.py` | 104 | Module chính | Window-based speed/distance, missing frame handling, clamping. |
| 10 | `minimap/minimap_renderer.py` | 124 | Module chính | Vẽ sân bóng tỷ lệ, mapping tọa độ thế giới → minimap, alpha blending. |
| 11 | `cleanings/pitch_keypoint_cleaning.py` | 379 | Tiền xử lý | Cleaning dataset YOLO Pose (32 keypoints + visibility), statistics, visualization. |
| 12 | `cleanings/player_cleaning.py` | 247 | Tiền xử lý | Cleaning dataset phát hiện cầu thủ: validation, fix bbox, visualization. |
| 13 | `analysis/visualize_perspective.py` | 87 | Đánh giá | Homography + perspectiveTransform, custom crop matrix cho bird's-eye view. |
| 14 | `analysis/visualize_mosaic_augmentation.py` | 119 | Tiền xử lý | Mô phỏng mosaic augmentation, coordinate transformation. |
| 15 | `analysis/visualize_optical_flow.py` | 79 | Đánh giá | Lucas-Kanade optical flow, feature detection, flow visualization. |

## Đơn giản (Simple)

| Hạng | File | Dòng | Pha | Lý do |
|------|------|------|-----|-------|
| 16 | `asigners/team_assigner.py` | 48 | Module chính | KMeans clustering màu áo, background subtraction. |
| 17 | `asigners/player_ball_assigner.py` | 22 | Module chính | Gán bóng cho cầu thủ gần nhất (threshold 70px). |
| 18 | `utils/video_util.py` | 45 | Module chính | Đọc/ghi video với ffmpeg fallback. |
| 19 | `utils/bbox_util.py` | 16 | Module chính | 5 hàm toán học một dòng: center, width, foot position, distance. |
| 20 | `heatmap_generator/heatmap_generator.py` | 52 | Module chính | Accumulate positions, Gaussian blur, OpenCV colormap. |
| 21 | `analysis/capture_result_frame.py` | 61 | Đánh giá | Capture frame từ output video, matplotlib subplot. |
| 22 | `analysis/visualize_player_stats.py` | 67 | Đánh giá | Bar charts speed/distance từ tracking stubs. |
| 23 | `analysis/visualize_team_assignment.py` | 73 | Đánh giá | KMeans color segmentation visualization. |
| 24 | `analysis/check_dataset_quality.py` | 109 | Tiền xử lý | Scan dataset quality issues, text report. |
| 25 | `analysis/class_imbalance_analysis.py` | 114 | Tiền xử lý | Class distribution, bbox area, small-object analysis. |
| 26 | `analysis/eda_dataset.py` | 123 | Tiền xử lý | EDA cơ bản: image counts, class distribution, pie chart. |
| 27 | `analysis/verify_data_split.py` | 79 | Tiền xử lý | Verify train/val/test split distribution. |
| 28 | `analysis/visualize_label_format.py` | 110 | Tiền xử lý | Giải thích YOLO label format với annotated image. |
| 29 | `analysis/visualize_preprocessing.py` | 85 | Tiền xử lý | Augmentation: flip, HSV jitter, noise, random crop. |
| 30 | `analysis/visualize_field_labels.py` | 193 | Tiền xử lý | Keypoint field visualization với YOLO Pose format. |
| 31 | `analysis/visualize_training_results.py` | 80 | Đánh giá | Plot YOLO training metrics từ results.csv. |
| 32 | `analysis/visualize_ball_interpolation.py` | 61 | Đánh giá | So sánh raw vs interpolated ball positions. |
| 33 | `analysis/evaluate_player_model.py` | 120 | Đánh giá | Validation YOLOv8x player detector, metrics + plots. |
| 34 | `analysis/evaluate_pitch_model.py` | 155 | Đánh giá | Validation YOLOv8x-pose pitch keypoint, metrics + plots. |
| 35 | `trainings/football_training.py` | 56 | Huấn luyện | Fine-tune YOLOv8x, download dataset từ Roboflow. |
| 36 | `trainings/pitch_keypoint_training.py` | 67 | Huấn luyện | Fine-tune YOLOv8x-pose với custom hyperparameters. |
| 37 | `trainings/player_dataset.py` | 19 | Tiền xử lý | Download dataset Roboflow. |
| 38 | `trainings/pitch_keypoint_dataset.py` | 19 | Tiền xử lý | Download dataset Roboflow. |


## Cơ bản (Trivial)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 37+ | Tất cả `__init__.py` | 0-4 | Export package, không có logic. |
