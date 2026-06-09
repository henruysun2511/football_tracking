# Bảng Xếp Hạng Độ Khó Của Các File Trong Dự Án

## Rất phức tạp (Very Complex)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 1 | `main.py` | 266 | Orchestrator toàn bộ pipeline: YOLO tracking, optical flow, homography, speed/distance, KMeans, formation, heatmap, minimap. Tích hợp mọi subsystem. |
| 2 | `app/gradio_app.py` | 336 | Web UI Gradio đầy đủ: upload video, progress tracking, caching, pipeline hoàn chỉnh, heatmap, thống kê cầu thủ, xử lý lỗi. |

## Phức tạp (Complex)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 3 | `trackers/tracker.py` | 231 | Wrapper YOLO + ByteTrack, batch detection, interpolation ball & player tracking, 4 utility vẽ annotations. |
| 4 | `formations/formation_analyzer.py` | 255 | Phát hiện đội hình bóng đá: KMeans clustering, quantile-based, direction normalization, multi-frame voting, 27 đội hình. |
| 5 | `pitch_keypoint_detector/pitch_keypoint_detector.py` | 139 | YOLO-pose 32 keypoints, smoothing, RANSAC homography, perspective transform, SoccerPitchConfig. |
| 6 | `app/streamlit_app.py` | 318 | Web UI Streamlit: caching, session state, pipeline split, collapsible sections. |
| 7 | `estimators/camera_movement_estimator.py` | 99 | Lucas-Kanade optical flow + Shi-Tomasi features, dominant motion vector, position adjustment. |
| 8 | `estimators/view_transformer.py` | 75 | Homography từ pitch keypoints, perspective transformation, exponential smoothing, boundary validation. |

## Trung bình (Medium)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 9 | `estimators/speed_distance_estimator.py` | 104 | Window-based speed/distance, missing frame handling, clamping. |
| 10 | `minimap/minimap_renderer.py` | 124 | Vẽ sân bóng tỷ lệ, mapping tọa độ thế giới → minimap, alpha blending. |
| 11 | `cleanings/pitch_keypoint_cleaning.py` | 379 | Cleaning dataset YOLO Pose (32 keypoints + visibility), statistics, visualization. |
| 12 | `cleanings/player_cleaning.py` | 247 | Cleaning dataset phát hiện cầu thủ: validation, fix bbox, visualization. |
| 13 | `analysis/visualize_perspective.py` | 87 | Homography + perspectiveTransform, custom crop matrix cho bird's-eye view. |
| 14 | `analysis/visualize_mosaic_augmentation.py` | 119 | Mô phỏng mosaic augmentation, coordinate transformation. |
| 15 | `analysis/visualize_optical_flow.py` | 79 | Lucas-Kanade optical flow, feature detection, flow visualization. |

## Đơn giản (Simple)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 16 | `asigners/team_assigner.py` | 48 | KMeans clustering màu áo, background subtraction. |
| 17 | `asigners/player_ball_assigner.py` | 22 | Gán bóng cho cầu thủ gần nhất (threshold 70px). |
| 18 | `utils/video_util.py` | 45 | Đọc/ghi video với ffmpeg fallback. |
| 19 | `utils/bbox_util.py` | 16 | 5 hàm toán học một dòng: center, width, foot position, distance. |
| 20 | `heatmap_generator/heatmap_generator.py` | 52 | Accumulate positions, Gaussian blur, OpenCV colormap. |
| 21 | `analysis/capture_result_frame.py` | 61 | Capture frame từ output video, matplotlib subplot. |
| 22 | `analysis/visualize_player_stats.py` | 67 | Bar charts speed/distance từ tracking stubs. |
| 23 | `analysis/visualize_team_assignment.py` | 73 | KMeans color segmentation visualization. |
| 24 | `analysis/check_dataset_quality.py` | 109 | Scan dataset quality issues, text report. |
| 25 | `analysis/class_imbalance_analysis.py` | 114 | Class distribution, bbox area, small-object analysis. |
| 26 | `analysis/eda_dataset.py` | 123 | EDA cơ bản: image counts, class distribution, pie chart. |
| 27 | `analysis/verify_data_split.py` | 79 | Verify train/val/test split distribution. |
| 28 | `analysis/visualize_label_format.py` | 110 | Giải thích YOLO label format với annotated image. |
| 29 | `analysis/visualize_preprocessing.py` | 85 | Augmentation: flip, HSV jitter, noise, random crop. |
| 30 | `analysis/visualize_training_results.py` | 80 | Plot YOLO training metrics từ results.csv. |
| 31 | `analysis/visualize_ball_interpolation.py` | 61 | So sánh raw vs interpolated ball positions. |
| 32 | `trainings/football_training.py` | 56 | Fine-tune YOLOv8x, download dataset từ Roboflow. |
| 33 | `trainings/pitch_keypoint_training.py` | 67 | Fine-tune YOLOv8x-pose với custom hyperparameters. |
| 34 | `trainings/player_dataset.py` | 19 | Download dataset Roboflow. |
| 35 | `trainings/pitch_keypoint_dataset.py` | 19 | Download dataset Roboflow. |
| 36 | `test_model.py` | 32 | Inference đơn giản, predict với stream=True. |

## Cơ bản (Trivial)

| Hạng | File | Dòng | Lý do |
|------|------|------|-------|
| 37+ | Tất cả `__init__.py` | 0-4 | Export package, không có logic. |
