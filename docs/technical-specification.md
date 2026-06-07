# BÁO CÁO ĐẶC TẢ KỸ THUẬT
## Hệ thống Phân tích Bóng đá bằng AI (Football AI Tracking & Analysis)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu

Xây dựng hệ thống thị giác máy tính ứng dụng trí tuệ nhân tạo để phân tích các trận đấu bóng đá từ dữ liệu video. Hệ thống có khả năng phát hiện, theo dõi cầu thủ, bóng và trọng tài; chiếu tọa độ lên bản đồ sân 2D; tính toán các chỉ số thống kê như tốc độ, quãng đường di chuyển, tỉ lệ kiểm soát bóng và phát hiện đội hình chiến thuật.

### 1.2 Phạm vi

- **Đầu vào**: Video trận đấu bóng đá góc quay cố định (broadcast camera)
- **Đầu ra**: Video đã chú thích với các lớp overlay (ellipse cầu thủ, minimap, thông số tốc độ, camera movement, tỉ lệ kiểm soát bóng, đội hình), ảnh heatmap, file dữ liệu thống kê dạng pickle
- **Xử lý**: Tracking real-time, phát hiện keypoint sân, biến đổi phối cảnh, phân cụm màu áo, tính toán chuyển động camera

### 1.3 Đối tượng sử dụng

- Huấn luyện viên và ban huấn luyện: phân tích chiến thuật và thể lực cầu thủ
- Bình luận viên thể thao: cung cấp số liệu trực quan
- Nhà phân tích dữ liệu thể thao: nghiên cứu và phát triển
- Người hâm mộ: trải nghiệm xem bóng đá tương tác

### 1.4 Kiến trúc tổng thể

Hệ thống được tổ chức theo kiến trúc pipeline gồm hai pha chính:

```
[Input Video]
     ↓
PHA 1 — TRACKING
     ├── 1. Đọc video → frames
     ├── 2. Phát hiện + Tracking (YOLOv8 + ByteTrack)
     ├── 3. Nội suy vị trí cầu thủ & bóng
     ├── 4. Ước lượng chuyển động camera (Optical Flow)
     ├── 5. Phát hiện keypoint sân → Homography
     ├── 6. Chiếu tọa độ lên mặt sân thực (Perspective Transform)
     ├── 7. Tính tốc độ & quãng đường
     ├── 8. Phân cụm màu áo → gán đội
     └── 9. Gán bóng → tính kiểm soát bóng
     ↓
PHA 2 — RENDERING
     ├── 10. Phát hiện đội hình chiến thuật
     ├── 11. Vẽ annotation lên từng frame
     ├── 12. Vẽ minimap, heatmap
     └── 13. Xuất video + ảnh heatmap
     ↓
[Output Video] [Heatmaps] [Thống kê]
```

### 1.5 Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | ≥ 3.9 | Ngôn ngữ lập trình |
| Ultralytics YOLOv8 | ≥ 8.1.0 | Phát hiện đối tượng & keypoint |
| Supervision | 0.18 – 0.21 | ByteTrack, công cụ hậu xử lý |
| OpenCV | ≥ 4.9.0 | Xử lý ảnh, video, homography |
| NumPy | — | Tính toán số học |
| Pandas | ≥ 2.2.0 | Nội suy dữ liệu tracking |
| Scikit-learn | ≥ 1.4.0 | K-Means clustering |
| PyTorch | — | Framework deep learning |
| Gradio | ≥ 4.0.0 | Giao diện web (tùy chọn) |
| Streamlit | ≥ 1.32.0 | Giao diện web (tùy chọn) |
| FFmpeg | — | Mã hóa video H.264 |

---

## 2. YÊU CẦU HỆ THỐNG

### 2.1 Functional Requirements

| ID | Chức năng | Mô tả |
|----|-----------|-------|
| FR-01 | Phát hiện cầu thủ | Phát hiện bounding box của cầu thủ, thủ môn, trọng tài và bóng trong từng frame |
| FR-02 | Theo dõi đối tượng | Gán ID duy nhất cho mỗi cầu thủ/trọng tài xuyên suốt video bằng ByteTrack |
| FR-03 | Nội suy tracking | Lấp đầy frame bị mất tracking bằng nội suy tuyến tính |
| FR-04 | Ước lượng camera | Tính toán độ dịch chuyển (dx, dy) của camera giữa các frame bằng optical flow |
| FR-05 | Phát hiện keypoint sân | Nhận diện 32 keypoint trên sân bóng (đường biên, vòng cấm, chấm phạt đền...) |
| FR-06 | Biến đổi phối cảnh | Tính homography để chiếu tọa độ 2D từ góc camera xuống mặt sân top-down |
| FR-07 | Tính tốc độ | Tính tốc độ tức thời (km/h) của từng cầu thủ dựa trên tọa độ thực tế |
| FR-08 | Tính quãng đường | Tính tổng quãng đường di chuyển (m) của từng cầu thủ |
| FR-09 | Gán đội | Phân cụm màu áo cầu thủ bằng K-Means để gán vào đội 1 hoặc đội 2 |
| FR-10 | Kiểm soát bóng | Xác định cầu thủ nào đang kiểm soát bóng dựa trên khoảng cách |
| FR-11 | Phát hiện đội hình | Phân tích vị trí cầu thủ trên sân để nhận diện đội hình (4-3-3, 4-4-2...) |
| FR-12 | Vẽ minimap | Hiển thị sơ đồ tactical 2D với vị trí các cầu thủ |
| FR-13 | Tạo heatmap | Tạo ảnh nhiệt mật độ di chuyển cho từng đội |
| FR-14 | Xuất video | Ghi video đầu ra với đầy đủ annotation |
| FR-15 | Giao diện web | Cho phép upload video và xem kết quả qua trình duyệt |

### 2.2 Non-functional Requirements

| ID | Yêu cầu | Mô tả |
|----|---------|-------|
| NFR-01 | Tốc độ xử lý | Pipeline tracking đạt ≥ 5 FPS trên GPU (NVIDIA T4+); rendering đạt ≥ 30 FPS |
| NFR-02 | Độ chính xác phát hiện | mAP@0.5 ≥ 85% cho lớp player, ball, referee |
| NFR-03 | Độ chính xác keypoint | mAP@0.5 ≥ 75% cho 32 keypoint sân |
| NFR-04 | Độ chính xác gán đội | ≥ 90% trên điều kiện ánh sáng tốt, màu áo rõ ràng |
| NFR-05 | Yêu cầu phần cứng | GPU NVIDIA với ≥ 4GB VRAM khuyến nghị; CPU 8+ cores tối thiểu |
| NFR-06 | Độ trễ | Toàn bộ pipeline xử lý video 5 phút trong ≤ 15 phút (GPU T4) |
| NFR-07 | Mở rộng | Hỗ trợ video đầu vào độ phân giải tối đa 1920×1080, FPS bất kỳ |
| NFR-08 | Tương thích | Chạy được trên Windows, Linux (Ubuntu 20.04+), Kaggle, Google Colab |

---

## 3. ĐẶC TẢ TỪNG MODULE

### 3.1 Module: Trackers

**Tập tin**: `trackers/tracker.py`  
**Class chính**: `Tracker`

#### Mục đích

Module trung tâm chịu trách nhiệm phát hiện và theo dõi đối tượng (cầu thủ, bóng, trọng tài) xuyên suốt video, đồng thời cung cấp các phương thức nội suy và vẽ annotation.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `__init__` | `model_path: str | None` | `Tracker` instance |
| `detect_frames` | `frames: list[np.array], batch_size=20` | `list[ultralytics.Results]` |
| `get_object_tracks` | `frames, read_from_stub, stub_path` | `tracks: dict` (players, referees, ball) |
| `interpolate_ball_positions` | `ball_positions: list` | `list` — bbox đã nội suy |
| `interpolate_player_positions` | `player_positions: list` | `list` — bbox đã nội suy |
| `add_position_to_tracks` | `tracks: dict` | Cập nhật key `position` |
| `draw_ellipse` | `frame, bbox, color, track_id` | `frame: np.array` |
| `draw_triangle` | `frame, bbox, color` | `frame: np.array` |
| `draw_team_ball_control` | `frame, frame_num, team_ball_control` | `frame: np.array` |
| `draw_annotations` | `video_frames, tracks, team_ball_control` | `list[np.array]` |

#### Thuật toán / Model

- **Model phát hiện**: YOLOv8x fine-tuned (`models/player_detector.pt`)
- **Tracker**: ByteTrack từ thư viện Supervision, với các tham số:
  - `track_activation_threshold = 0.25`
  - `lost_track_buffer = 30` frames
  - `minimum_matching_threshold = 0.8`
  - `frame_rate = 25`
- **Nội suy**: Pandas DataFrame interpolate(method='linear') + bfill()

#### Tham số cấu hình

```python
batch_size = 20              # Số frame xử lý cùng lúc
conf = 0.1                   # Ngưỡng confidence YOLO
# ByteTrack:
track_activation_threshold = 0.25
lost_track_buffer = 30
minimum_matching_threshold = 0.8
frame_rate = 25
```

#### Xử lý lỗi

- Tự động chọn CPU/GPU dựa trên VRAM khả dụng (nếu VRAM < 1GB → CPU)
- Gộp class "goalkeeper" thành "player" để đơn giản hóa
- Nếu `read_from_stub=True` và file stub tồn tại → load thẳng, bỏ qua detection
- Nếu stub_path không có filename → không lưu stub

---

### 3.2 Module: Camera Movement Estimator

**Tập tin**: `estimators/camera_movement_estimator.py`  
**Class chính**: `CameraMovementEstimator`

#### Mục đích

Tính toán độ dịch chuyển (pan) của camera giữa các frame liên tiếp để bù trừ tọa độ cầu thủ.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `__init__` | `first_frame: np.array` | Khởi tạo mask, tham số optical flow |
| `get_camera_movement` | `frames: list, read_from_stub, stub_path` | `list[[dx, dy]]` — mỗi frame |
| `add_adjust_positions_to_tracks` | `tracks, movement` | Cập nhật key `position_adjusted` |
| `draw_camera_movement` | `frames, movement` | `list[np.array]` — frame đã vẽ |

#### Thuật toán

- Sử dụng **Lucas-Kanade Optical Flow** (`cv2.calcOpticalFlowPyrLK`)
- Chỉ tracking feature ở biên trái (20px) và biên phải (150px) của frame → vùng ít bị ảnh hưởng bởi chuyển động của cầu thủ
- Với mỗi frame, tìm feature có độ dịch chuyển lớn nhất → ghi nhận (dx, dy)
- Nếu max_distance ≤ 5px → coi như không có chuyển động (set [0, 0])

#### Tham số cấu hình

```python
minimum_distance = 5                         # Ngưỡng phát hiện chuyển động (px)
# Lucas-Kanade params:
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Feature detection:
feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=3, blockSize=7, mask=mask)
```

---

### 3.3 Module: View Transformer

**Tập tin**: `estimators/view_transformer.py`  
**Class chính**: `ViewTransformer`

#### Mục đích

Thực hiện phép biến đổi phối cảnh (perspective transformation) để chuyển tọa độ cầu thủ từ không gian ảnh 2D của camera sang tọa độ mặt sân thực tế (top-down).

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `__init__` | `kp_detector, smooth_alpha=0.15` | `ViewTransformer` instance |
| `add_transformed_position_to_tracks` | `tracks, video_frames` | Cập nhật key `position_transformed` |

#### Thuật toán

1. Với mỗi frame, phát hiện 32 keypoint sân qua `PitchKeypointDetector`
2. Tính ma trận homography H (3×3) bằng `cv2.findHomography(src, dst, RANSAC, 5.0)`:
   - `src`: keypoint phát hiện từ ảnh
   - `dst`: keypoint chuẩn từ `SoccerPitchConfig.vertices` (tọa độ thực tế)
3. Với mỗi đối tượng (player, referee, ball), áp dụng:
   - `cv2.perspectiveTransform(position_adjusted, H)` → tọa độ thực tế (cm)
4. Làm mịn bằng **Exponential Moving Average** (EMA):
   - `smoothed = alpha × raw + (1 - alpha) × prev`
   - Loại bỏ outlier: vị trí ngoài biên sân ±25%, khoảng cách nhảy > 600cm/frame

#### Tham số cấu hình

```python
smooth_alpha = 0.15          # Hệ số EMA (0 = giữ nguyên, 1 = không smooth)
max_jump = {
    'players':  600,         # cm/frame (~14 m/s @ 24fps)
    'referees': 600,
    'ball':     1500,        # bóng có thể bay nhanh hơn
}
margin = 0.10                # 10% margin cho in-bounds check
```

---

### 3.4 Module: Speed & Distance Estimator

**Tập tin**: `estimators/speed_distance_estimator.py`  
**Class chính**: `SpeedDistanceEstimator`

#### Mục đích

Tính toán tốc độ tức thời (km/h) và quãng đường tích lũy (m) của từng cầu thủ dựa trên tọa độ thực tế.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `add_speed_and_distance_to_tracks` | `tracks, fps=25` | Cập nhật key `speed`, `distance` |
| `draw_speed_and_distance` | `frames, tracks` | `list[np.array]` |

#### Thuật toán

1. Duyệt tracks theo cửa sổ `WINDOW = 5` frame
2. Với mỗi cầu thủ trong window:
   - Tính `dist_cm = euclidean(p_start, p_end)` từ `position_transformed`
   - Loại bỏ window nếu `dist_cm > 300cm` (nhiễu homography)
   - `dist_m = dist_cm × 0.01`
   - `speed_ms = dist_m / elapsed_time`
   - `speed_kmh = min(speed_ms × 3.6, 38.0)` — giới hạn sinh lý học
3. Tích lũy distance: `total_dist[tid] += dist_m`
4. Ghi `speed` và `distance` vào mỗi frame trong window

#### Tham số cấu hình

```python
WINDOW = 5                    # Số frame trong cửa sổ tính toán
_UNIT_TO_METER = 0.01         # cm → m
_MAX_SPEED_KMH = 38.0         # Giới hạn tốc độ tối đa (Mbappé ~38 km/h)
_MAX_DIST_CM = 300.0          # Giới hạn khoảng cách tối đa mỗi window
```

---

### 3.5 Module: Team Assigner

**Tập tin**: `asigners/team_assigner.py`  
**Class chính**: `TeamAssigner`

#### Mục đích

Phân cụm cầu thủ vào 2 đội dựa trên màu áo (màu sắc vùng jersey phía trên).

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `assign_team_color` | `frame, player_detections` | Xác định `team_colors[1]`, `team_colors[2]` |
| `get_player_team` | `frame, bbox, player_id` | `1` hoặc `2` |
| `get_player_color` | `frame, bbox` | `tuple(B, G, R)` — màu áo |

#### Thuật toán

1. Crop vùng cầu thủ từ bounding box, lấy nửa trên (vùng áo)
2. Áp dụng K-Means với `n_clusters=2` để tách màu áo khỏi nền:
   - Cluster 1: nền (background) — dựa trên góc ảnh
   - Cluster 2: màu áo — `cluster_center` là màu đại diện
3. Gán đội: K-Means lần 2 trên toàn bộ màu áo các cầu thủ với `n_clusters=2` (2 đội)
4. Lưu kết quả vào `player_team_dict` để cache

---

### 3.6 Module: Player-Ball Assigner

**Tập tin**: `asigners/player_ball_assigner.py`  
**Class chính**: `PlayerBallAssigner`

#### Mục đích

Xác định cầu thủ nào đang kiểm soát bóng tại mỗi frame.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `assign_ball_to_player` | `players: dict, ball_bbox: list` | `player_id: int` hoặc `-1` |

#### Thuật toán

1. Tính tâm bóng: `((x1+x2)/2, (y1+y2)/2)`
2. Với mỗi cầu thủ, tính khoảng cách từ tâm bóng đến `(foot_x ± bbox_width/2, foot_y)` (vị trí chân trái/phải)
3. Chọn cầu thủ có khoảng cách nhỏ nhất và ≤ `max_distance` (mặc định 70px)

---

### 3.7 Module: Pitch Keypoint Detector

**Tập tin**: `pitch_keypoint_detector/pitch_keypoint_detector.py`  
**Class chính**: `PitchKeypointDetector`, `SoccerPitchConfig`

#### Mục đích

Phát hiện 32 điểm mốc trên sân bóng (góc sân, góc vòng cấm, chấm phạt đền, vòng tròn trung tâm...) để tính ma trận homography.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `detect` | `frame` | `(xy_array, conf_array)` hoặc `None` |
| `detect_smoothed` | `frame, alpha=0.6` | `(smoothed_xy, confs)` |
| `get_homography` | `frame_keypoints` | `3×3 matrix` hoặc `None` |
| `transform_point` | `point, M` | `(x', y')` |

#### Tham số cấu hình

```python
# PitchKeypointDetector
conf_threshold = 0.3           # Ngưỡng confidence keypoint

# SoccerPitchConfig (kích thước sân)
width  = 7000  cm   (70m)
length = 12000 cm   (120m)
penalty_box_width     = 4100 cm
penalty_box_length    = 2015 cm
goal_box_width        = 1832 cm
goal_box_length       = 550  cm
centre_circle_radius  = 915  cm
penalty_spot_distance = 1100 cm
```

---

### 3.8 Module: Minimap Renderer

**Tập tin**: `minimap/minimap_renderer.py`  
**Class chính**: `MinimapRenderer`

#### Mục đích

Tạo bản đồ tactical 2D (góc nhìn từ trên xuống) hiển thị vị trí các cầu thủ và bóng.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `render` | `tracks, frame_num, team_colors` | `np.array` (350×230) |
| `overlay` | `frame, minimap, pos='bottom_right'` | `np.array` |

#### Xử lý

1. Vẽ sân 2D với các đường kẻ, vòng tròn trung tâm, vòng cấm
2. Chiếu `position_transformed` từ cm → pixel trên minimap
3. Vẽ cầu thủ dạng circle với màu theo đội, trọng tài màu vàng, bóng màu xanh cyan
4. Overlay lên frame gốc ở góc dưới phải với background opacity 30%

---

### 3.9 Module: Heatmap Generator

**Tập tin**: `heatmap_generator/heatmap_generator.py`  
**Class chính**: `HeatmapGenerator`

#### Mục đích

Tạo ảnh nhiệt mật độ di chuyển của cầu thủ cho từng đội và tổng hợp.

#### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `update_from_tracks` | `tracks` | Cập nhật internal heatmap arrays |
| `render_both` | — | `np.array` (630×420) — heatmap tổng hợp (JET colormap) |
| `render_team` | `team, colormap` | `np.array` — heatmap từng đội |

---

### 3.10 Module: Formation Analyzer

**Tập tin**: `formations/formation_analyzer.py`  
**Hàm chính**: `detect_team_formation`, `detect_formation`

#### Mục đích

Phát hiện đội hình chiến thuật của mỗi đội (ví dụ: 4-3-3, 4-4-2) dựa trên vị trí cầu thủ trên sân.

#### Thuật toán

1. Với mỗi frame mẫu (cách 30 frame), lấy tọa độ `position_transformed` của 10 cầu thủ (loại thủ môn)
2. Chuẩn hóa hướng tấn công (defenders bên trái)
3. Sắp xếp theo trục x (từ hậu vệ → tiền đạo)
4. Thử K-Means với `n_clusters=3` (3 tuyến) và `n_clusters=4` (4 tuyến)
5. Chọn kết quả phù hợp nhất dựa trên tổng số cầu thủ mỗi tuyến
6. Bỏ phiếu trên toàn bộ frame → đội hình có số phiếu cao nhất

---

### 3.11 Module: Web UI (Gradio)

**Tập tin**: `app/gradio_app.py`

#### Mục đích

Cung cấp giao diện web cho phép người dùng upload video và nhận kết quả phân tích.

#### Input / Output

| Element | Type | Mô tả |
|---------|------|-------|
| Video input | `gr.Video` | Upload file .mp4 |
| Pitch Keypoints | `gr.Checkbox` | Bật/tắt vẽ keypoint |
| Minimap | `gr.Checkbox` | Bật/tắt minimap |
| Heatmap | `gr.Checkbox` | Bật/tắt heatmap |
| Analyze button | `gr.Button` | Kích hoạt pipeline |
| Video output | `gr.Video` | Kết quả video đã annotation |
| Heatmap gallery | `gr.Gallery` | 3 ảnh heatmap |
| Status log | `gr.HTML` | Log thời gian thực |

#### Tính năng

- Cache dùng MD5 hash nội dung file → upload lại file trùng sẽ dùng stub cũ
- Fallback ffmpeg → OpenCV nếu không có ffmpeg
- Hỗ trợ share link trên Kaggle/Colab

---

### 3.12 Module: Cleaning Dataset

**Tập tin**: `cleanings/player_cleaning.py` , `cleanings/pitch_keypoint_cleaning.py`

#### Mục đích

Làm sạch dữ liệu gốc từ Roboflow trước khi huấn luyện, bao gồm phát hiện và loại bỏ ảnh/ label bị lỗi, đồng thời chuẩn hóa tọa độ bounding box và keypoint. Module này đảm bảo chất lượng đầu vào cho quá trình huấn luyện, tránh gây nhiễu mô hình.

#### Cleaning player detection dataset (`player_cleaning.py`)

##### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `scan_split` | `split: str` (train/valid/test) | `(stats, bad_stems, cls_counter)` |
| `clean_split` | `split, bad_stems` | Copy ảnh/label sạch sang `CLEANED_ROOT` |
| `fix_bbox_labels` | `split` | Clip tọa độ bbox outlier về [0,1] |
| `visualize_before_after` | `split, n_samples=6` | Ảnh PNG so sánh trước/sau lưu tại `analysis/figures/` |
| `print_cleaning_report` | `before_stats, after_counts` | Bảng tổng kết số lượng trước/sau cleaning |

##### Pipeline xử lý

```
BƯỚC 1: SCAN
  ├── Đọc từng ảnh → kiểm tra corrupt (cv2.imread == None)
  ├── Kiểm tra tồn tại file label (.txt)
  ├── Kiểm tra label rỗng
  ├── Parse từng dòng: class_id, cx, cy, bw, bh
  │   ├── class_id ∉ [0, NUM_CLASSES=4] → bad
  │   ├── bw, bh ∉ (0, 1] hoặc cx, cy ∉ [0, 1] → bad
  └── Ghi nhận bad_stems (stem ảnh lỗi)

BƯỚC 2: CLEAN (COPY)
  └── Copy ảnh & label không nằm trong bad_stems
      sang thư mục CLEANED_ROOT / split / {images, labels}

BƯỚC 3: FIX BBOX
  └── Clip cx, cy, bw, bh vào [0,1] bằng np.clip()
      Đảm bảo bw, bh > 0.001 sau clip, nếu không thì xóa dòng

BƯỚC 4: VISUALIZE
  └── Vẽ bounding box trước/sau lên cùng ảnh, lưu so sánh

BƯỚC 5: BÁO CÁO
  └── In bảng tổng kết: Split, Trước, Sau, Xóa, %giữ
```

##### Tham số cấu hình

```python
DATASET_ROOT  = "datasets/football-players-detection-2"      # Thư mục gốc Roboflow
CLEANED_ROOT  = "datasets/football-players-detection-cleaned" # Thư mục sau cleaning
SPLITS        = ["train", "valid", "test"]
CLASS_NAMES   = ["ball", "goalkeeper", "player", "referee"]
NUM_CLASSES   = 4
```

##### Các tiêu chí lỗi phát hiện

| Loại lỗi | Mô tả | Hành động |
|----------|-------|-----------|
| Corrupt | Ảnh không đọc được bởi OpenCV | Xóa ảnh + label |
| Missing label | Không có file .txt tương ứng | Xóa ảnh |
| Empty label | File label tồn tại nhưng không có dòng dữ liệu | Xóa ảnh |
| Invalid bbox | Tọa độ ngoài [0,1] hoặc kích thước ≤ 0 | Fix/clip về [0,1] |
| Bad class | class_id nằm ngoài [0, 3] | Xóa dòng đó |

#### Cleaning pitch keypoint dataset (`pitch_keypoint_cleaning.py`)

##### Input / Output

| Phương thức | Input | Output |
|-------------|-------|--------|
| `scan_keypoint_split` | `split: str` | `(stats, bad_stems, kp_vis_count, kp_per_image)` |
| `fix_and_copy_keypoint_split` | `split, bad_stems` | Copy ảnh sạch + fix label vào `CLEANED_ROOT` |
| `visualize_keypoint_stats` | `kp_vis_count, kp_per_image, split` | Biểu đồ bar + histogram PNG |
| `visualize_keypoints_on_image` | `split, n=4` | Ảnh so sánh keypoint trước/sau PNG |

##### Pipeline xử lý

```
BƯỚC 1: SCAN
  ├── Đọc ảnh → kiểm tra corrupt
  ├── Kiểm tra file label .txt
  ├── Parse format YOLO Pose: class cx cy w h [x1 y1 v1 x2 y2 v2 ... xN yN vN]
  │   ├── Số phần tử < 5 + NUM_KEYPOINTS*3 → malformed
  │   ├── bbox không hợp lệ → invalid_bbox
  │   ├── visibility (v) ∉ {0,1,2} → invalid_visibility
  │   └── n_visible < MIN_VISIBLE_KPS → too_few_keypoints
  └── Ghi nhận bad_stems

BƯỚC 2: FIX + COPY
  ├── Copy ảnh sạch
  ├── Clip bbox về [0,1]
  ├── Với mỗi keypoint:
  │   ├── Clip visibility về {0,1,2}
  │   ├── Nếu vis > 0: clip tọa độ (x,y) về [0,1]
  │   └── Nếu x,y bị đẩy > 0.01 so với gốc → set vis = 0
  └── Ghi label đã fix

BƯỚC 3: THỐNG KÊ
  ├── Bar chart: tần suất visible của 32 keypoint
  └── Histogram: phân bố số keypoint visible mỗi ảnh

BƯỚC 4: VISUALIZE TRỰC QUAN
  └── Vẽ keypoint lên ảnh (xanh=visible, vàng=occluded) trước/sau
```

##### Tham số cấu hình

```python
DATASET_ROOT   = "datasets/pitch-keypoint-detection-2"          # Roboflow gốc
CLEANED_ROOT   = "datasets/pitch-keypoint-detection-cleaned"    # Sau cleaning
SPLITS         = ["train", "valid", "test"]
NUM_CLASSES    = 1           # Chỉ 1 class "pitch"
NUM_KEYPOINTS  = 32          # 32 keypoint sân
MIN_VISIBLE_KPS = 4          # Tối thiểu 4 keypoint visible mới giữ ảnh
```

##### Ý nghĩa tham số MIN_VISIBLE_KPS

- Homography cần tối thiểu 4 cặp điểm tương ứng để tính ma trận H (3×3) với 8 bậc tự do
- Ảnh có < 4 keypoint visible không thể dùng cho perspective transformation → loại bỏ
- Giúp giảm nhiễu trong quá trình huấn luyện keypoint detector

---

### 3.13 Module: Training

**Tập tin**: `trainings/player_dataset.py` , `trainings/pitch_keypoint_dataset.py` , `trainings/football_training.py` , `trainings/pitch_keypoint_training.py`

#### Mục đích

Cung cấp các script tải dữ liệu và huấn luyện hai mô hình chính của hệ thống: YOLOv8x cho phát hiện cầu thủ/bóng/trọng tài và YOLOv8x-pose cho phát hiện keypoint sân (32 keypoints). Cả hai đều sử dụng dữ liệu từ Roboflow và pre-trained weights từ Ultralytics.

#### Input / Output tổng quan

| Script | Input | Output |
|--------|-------|--------|
| `player_dataset.py` | ROBOFLOW_API_KEY (env) | Dataset tại `datasets/football-players-detection-2/` |
| `pitch_keypoint_dataset.py` | ROBOFLOW_API_KEY (env) | Dataset tại `datasets/pitch-keypoint-detection-2/` |
| `football_training.py` | Dataset cleaned + `yolov8x.pt` | `models/player_detector/player_detector/weights/best.pt` |
| `pitch_keypoint_training.py` | Dataset cleaned + `yolov8x-pose.pt` | `models/pitch_keypoint/pitch_keypoint/weights/best.pt` |

#### Download Dataset

##### player_dataset.py

```python
# Class: ball (0), goalkeeper (1), player (2), referee (3)
# Nguồn: Roboflow workspace "roboflow-jvuqo"
# Project: "football-players-detection-3zvbc", version 2
# Format: YOLOv8
rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(2).download("yolov8")
```

##### pitch_keypoint_dataset.py

```python
# Class: pitch (1 class), 32 keypoints
# Nguồn: Roboflow workspace "roboflow-jvuqo"
# Project: "football-field-detection-f07vi", version 14
# Format: YOLOv8
project_pitch = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
dataset = project_pitch.version(14).download("yolov8")
```

#### Huấn luyện

##### football_training.py — Player Detection

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `model` | `yolov8x.pt` | Pre-trained YOLOv8x (COCO) |
| `data` | Dataset cleaned `data.yaml` | Đường dẫn file YAML định nghĩa dataset |
| `epochs` | 100 | Số epoch huấn luyện |
| `imgsz` | 1280 | Kích thước ảnh đầu vào (px) |
| `batch` | 8 | Batch size (phù hợp GPU T4 16GB) |
| `device` | 0 | GPU ID (CPU nếu không có GPU) |
| `patience` | 20 | Early stopping nếu mAP không cải thiện sau 20 epochs |
| `workers` | 2 | Số worker load dữ liệu |
| `project` | `models/player_detector` | Thư mục lưu kết quả |
| `name` | `player_detector` | Tên run experiment |
| `save` | True | Lưu checkpoint |
| `plots` | True | Vẽ biểu đồ training |

**Quy trình**:
1. Kiểm tra dataset cleaned tại `datasets/football-players-detection-cleaned/`
2. Nếu chưa có, tự động tải từ Roboflow (version 2)
3. Load `yolov8x.pt` (COCO pre-trained)
4. Fine-tune trên dataset bóng đá
5. Lưu best weights tại `models/player_detector/player_detector/weights/best.pt`
6. Nếu chạy trên Google Colab (có `/content/drive`), lưu thêm vào Google Drive

##### pitch_keypoint_training.py — Pitch Keypoint Detection

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `model` | `yolov8x-pose.pt` | Pre-trained YOLOv8x-pose (COCO keypoints) |
| `data` | Dataset cleaned `data.yaml` | Đường dẫn dataset keypoint sân |
| `epochs` | 100 | Số epoch |
| `imgsz` | 640 | Kích thước ảnh đầu vào (px) |
| `batch` | 8 | Batch size |
| `device` | 0 | GPU ID |
| `patience` | 50 | Early stopping sau 50 epochs (cao hơn player vì pose khó học hơn) |
| `lr0` | 0.01 | Learning rate ban đầu |
| `lrf` | 0.1 | Learning rate final = lr0 * lrf = 0.001 |
| `warmup_epochs` | 5 | Số epoch warmup tăng dần lr |
| `mosaic` | 1.0 | Tỉ lệ mosaic augmentation |
| `close_mosaic` | 10 | Tắt mosaic ở 10 epoch cuối |
| `degrees` | 10.0 | Augmentation xoay ảnh (độ) |
| `scale` | 0.2 | Augmentation scale |
| `fliplr` | 0.5 | Augmentation lật ngang |
| `flipud` | 0.0 | Không lật dọc |
| `project` | `models/pitch_keypoint` | Thư mục lưu kết quả |
| `name` | `pitch_keypoint` | Tên run experiment |
| `save` | True | Lưu checkpoint |
| `plots` | True | Vẽ biểu đồ |

**Quy trình**:
1. Kiểm tra dataset cleaned tại `datasets/pitch-keypoint-detection-cleaned/`
2. Nếu chưa có, tải từ Roboflow (version 14)
3. Load `yolov8x-pose.pt` (COCO keypoints pre-trained)
4. Fine-tune với augmentation nhẹ (scale=0.2, degrees=10, không flip dọc)
5. Mosaic augmentation được tắt ở 10 epoch cuối để ổn định hội tụ
6. Lưu best weights tại `models/pitch_keypoint/pitch_keypoint/weights/best.pt`

##### Chiến lược augmentation cho Pitch Keypoint

| Augmentation | Giá trị | Lý do |
|-------------|---------|-------|
| `mosaic=1.0` | Bật hoàn toàn | Tăng đa dạng ngữ cảnh sân, giúp model học mối quan hệ giữa các keypoint |
| `close_mosaic=10` | Tắt 10 epoch cuối | Tránh nhiễu mosaic làm giảm độ chính xác localization ở giai đoạn hội tụ |
| `degrees=10.0` | Xoay ±10° | Mô phỏng các góc camera nghiêng nhẹ |
| `scale=0.2` | Scale ±20% | Sân có thể ở các tỉ lệ xa/gần khác nhau |
| `fliplr=0.5` | Lật ngang 50% | Sân đối xứng trái-phải |
| `flipud=0.0` | Không lật dọc | Sân bóng không bao giờ lộn ngược trong thực tế |

#### Kết quả đầu ra

Sau khi huấn luyện, mỗi run tạo ra thư mục `project/name/` chứa:

| File | Mô tả |
|------|-------|
| `weights/best.pt` | Best model weights (dùng cho inference) |
| `weights/last.pt` | Checkpoint epoch cuối cùng |
| `confusion_matrix.png` | Ma trận nhầm lẫn |
| `results.png` | Biểu đồ loss, mAP, precision, recall |
| `labels.jpg` | Label visualization trên ảnh mẫu |
| `val_batch*.jpg` | Validation predictions |
| `hyp.yaml` | Hyperparameters đã dùng |
| `opt.yaml` | Tất cả tham số / cấu hình |

#### Lưu ý triển khai

- Cả hai training script đều kiểm tra sự tồn tại của thư mục datasets cục bộ trước khi tải từ Roboflow → tiết kiệm thời gian nếu đã có dữ liệu
- `player_dataset.py` và `pitch_keypoint_dataset.py` có thể chạy độc lập để tải dữ liệu mà không cần training
- Pipeline chuẩn: **Download** → **Clean** (Section 3.12) → **Train** → **Inference** (Section 3.1)
- Mô hình player detection input 1280px (độ phân giải cao → phát hiện cầu thủ nhỏ tốt hơn)
- Mô hình keypoint input 640px (cân bằng giữa độ chính xác và tốc độ, keypoint sân ít chi tiết hơn)

---

### 3.14 Module: Analysis & Visualization

**Tập tin**: `analysis/` (14 scripts + thư mục `figures/`, `reports/`)

#### Mục đích

Cung cấp các script phân tích và trực quan hóa dữ liệu, kết quả huấn luyện và đầu ra pipeline. Module này phục vụ nghiên cứu, debug và báo cáo — không nằm trong pipeline chính mà chạy độc lập sau khi có dữ liệu đầu vào (dataset, stub, video kết quả).

#### Phân loại

14 script được chia thành 4 nhóm chức năng:

| Nhóm | Script | Đầu vào | Đầu ra |
|------|--------|---------|--------|
| **Dataset EDA** | `eda_dataset.py`, `class_imbalance_analysis.py`, `check_dataset_quality.py`, `verify_data_split.py` | Dataset Roboflow gốc | PNG biểu đồ + TXT report |
| **Labeling & Augmentation** | `visualize_label_format.py`, `visualize_preprocessing.py`, `visualize_mosaic_augmentation.py` | Dataset Roboflow gốc | PNG trực quan hóa |
| **Pipeline Component** | `visualize_perspective.py`, `visualize_optical_flow.py`, `visualize_ball_interpolation.py`, `visualize_team_assignment.py` | Dataset + stub + model | PNG minh họa |
| **Kết quả & Stats** | `visualize_player_stats.py`, `visualize_training_results.py`, `capture_result_frame.py` | Stub tracking + video | PNG biểu đồ + ảnh frame |

---

#### 3.14.1 Dataset EDA

##### `eda_dataset.py` — Thống kê tổng quan dataset

| Chức năng | Mô tả |
|-----------|-------|
| `count_images(split)` | Đếm số ảnh mỗi split (train/valid/test) |
| `parse_labels(label_dir)` | Đọc tất cả label YOLO, trả về list (class_id, w, h) |
| Phân bố class | Bar chart số bbox theo 4 class |
| Split distribution | Pie chart tỉ lệ train/val/test |
| Bbox size distribution | Histogram diện tích bbox (normalized), đánh dấu ngưỡng small object 1% |
| Bảng thống kê | In tỉ lệ phần trăm từng class, số small objects, imbalance ratio |

**Output**: `analysis/figures/dataset_eda.png`

##### `class_imbalance_analysis.py` — Phân tích mất cân bằng dữ liệu

| Biểu đồ | Mô tả |
|---------|-------|
| 1. Số lượng bbox theo class | Bar chart + % tỉ lệ |
| 2. Box plot diện tích bbox | Phân bố w×h theo từng class (log scale) |
| 3. Scatter width × height | 300 mẫu/class, trực quan kích thước tương đối |
| 4. Small object analysis | % bbox dưới ngưỡng diện tích [0.1%, 0.2%, 0.5%, 1%] |

**Output**: `analysis/figures/class_imbalance.png` + report in console

| Class | Đặc điểm |
|-------|----------|
| **ball** | Rất ít (~3%), diện tích rất nhỏ → small object, dễ bị miss |
| **goalkeeper** | Ít (~5%), kích thước trung bình |
| **player** | Nhiều nhất (~75%), kích thước đa dạng |
| **referee** | Trung bình (~17%), dễ nhầm với player |

##### `check_dataset_quality.py` — Kiểm tra chất lượng dataset

Kiểm tra 6 loại vấn đề trên cả 3 split:

| Vấn đề | Kiểm tra | Hành động |
|--------|----------|-----------|
| `corrupt_images` | `cv2.imread()` trả về None | Ghi nhận file ảnh lỗi |
| `missing_labels` | File .txt không tồn tại | Ghi nhận ảnh thiếu label |
| `empty_labels` | File label tồn tại nhưng không có dòng dữ liệu | Ghi nhận label rỗng |
| `malformed_lines` | Dòng label không đúng 5 phần tử | Ghi nhận và in dòng lỗi |
| `invalid_class` | class_id ngoài [0, NUM_CLASSES) | Ghi nhận |
| `bbox_out_of_bounds` | cx, cy, bw, bh ngoài [0,1] | Ghi nhận tọa độ |

**Output**: `analysis/reports/dataset_quality_report.txt` (dạng văn bản, in 3 ví dụ đầu mỗi lỗi)

##### `verify_data_split.py` — Xác minh phân chia dữ liệu

| Chức năng | Mô tả |
|-----------|-------|
| Bảng tóm tắt | Số ảnh + tỉ lệ + số bbox mỗi class theo split |
| Pie chart | Phân chia ảnh train/val/test |
| Grouped bar | So sánh phân bố class giữa các split (kiểm tra stratified split) |

**Output**: `analysis/figures/data_split.png`

Ghi chú: Nếu tỉ lệ class giữa các split tương đồng → dataset được stratified split đúng cách, model sẽ không bị bias do phân phối lệch giữa train và val.

---

#### 3.14.2 Labeling & Augmentation

##### `visualize_label_format.py` — Giải thích format YOLO

Chọn ngẫu nhiên ảnh có nhiều class nhất từ train set, vẽ 2 biểu đồ:

| Biểu đồ | Mô tả |
|---------|-------|
| 1. Ảnh annotated đầy đủ | Bbox 4 màu khác nhau cho 4 class, kèm legend |
| 2. Giải thích 1 bbox chi tiết | Crop vùng bbox, đánh dấu center point, hiển thị: `class cx cy w h`, pixel coordinates, tỉ lệ % ảnh |

**Công thức chuyển đổi YOLO → Pixel**:
```python
x1 = int((cx - bw / 2) * img_w)
y1 = int((cy - bh / 2) * img_h)
x2 = int((cx + bw / 2) * img_w)
y2 = int((cy + bh / 2) * img_h)
```

**Output**: `analysis/figures/label_format_explanation.png`

##### `visualize_preprocessing.py` — Minh họa augmentation

| Ảnh | Mô tả |
|-----|-------|
| 1. Original (640×640) | Resize ảnh gốc |
| 2. Horizontal Flip | Lật ngang 100% |
| 3. HSV Jitter | Tăng saturation ×1.4, giảm brightness ×0.7 |
| 4. Gaussian Noise | Nhiễu Gaussian σ=15 |
| 5. Random Crop + Resize | Crop 80px mỗi cạnh rồi resize về 640 |

Kèm ảnh so sánh raw vs labeled:
- `analysis/figures/augmentation_comparison.png` — 5 augmentation
- `analysis/figures/labeled_sample.png` — Raw ↔ YOLO labels

##### `visualize_mosaic_augmentation.py` — Mosaic augmentation

Tái tạo (simulate) Mosaic Augmentation: ghép 4 ảnh ngẫu nhiên thành 1 ảnh 640×640.

| Thuật toán | Chi tiết |
|------------|----------|
| `create_mosaic()` | Chọn 4 ảnh, xác định cut point (cx, cy) ngẫu nhiên trong [¼, ¾] |
| | Resize mỗi ảnh vào ¼ vùng tương ứng |
| | Chuyển đổi tọa độ label về hệ tọa độ mosaic |
| | Clip bbox vào biên ảnh |
| Vẽ đường giao | Đường dashed vàng tại cut point |

**Output**: `analysis/figures/mosaic_augmentation.png`
- 4 ảnh gốc đã annotation (hàng trên)
- 1 ảnh mosaic kết quả (hàng dưới) với cut point + tất cả bbox

---

#### 3.14.3 Pipeline Component Visualization

##### `visualize_perspective.py` — Perspective Transform (Bird's eye view)

| Bước | Mô tả |
|------|-------|
| 1 | Load YOLO-pose model `pitch_keypoint_detector.pt` |
| 2 | Detect 32 keypoints trên frame đầu tiên của video |
| 3 | Tính homography matrix với keypoint confidence > 0.3 |
| 4 | Warp perspective: `cv2.warpPerspective(frame, M, (1200, 700))` |

**Output**: `analysis/figures/perspective_transform.png`
- Trái: frame gốc với keypoints (vàng)
- Phải: bird's eye view (sân 120m × 70m, tỉ lệ 1:10)

##### `visualize_optical_flow.py` — Optical Flow camera estimation

Tái tạo pipeline `CameraMovementEstimator` trên 2 frame đầu:

| Bước | Mô tả |
|------|-------|
| 1 | Tạo mask chỉ track feature ở dải trên (40px) và dải dưới (40px) của frame |
| 2 | `cv2.goodFeaturesToTrack` với `maxCorners=100, qualityLevel=0.3` |
| 3 | `cv2.calcOpticalFlowPyrLK` với Lucas-Kanade params (winSize=15, maxLevel=2) |
| 4 | Lọc điểm track thành công, tính median (dx, dy) |

**Output**: `analysis/figures/optical_flow.png`
- Trái: frame t với feature points (xanh)
- Giữa: frame t+1 với motion vectors (mũi tên đỏ) + giá trị Δx, Δy
- Phải: histogram độ lớn chuyển động, median (đường đỏ dashed)

##### `visualize_ball_interpolation.py` — Nội suy vị trí bóng

Minh họa hiệu quả của nội suy Pandas trên bóng:

| Bước | Mô tả |
|------|-------|
| 1 | Load `stubs/track_stubs.pkl`, trích xuất tọa độ trung tâm bóng mỗi frame |
| 2 | Tạo DataFrame, `interpolate(method='linear').bfill()` |
| 3 | Lấy 150 frame đầu, so sánh trước/sau |

**Output**: `analysis/figures/ball_interpolation.png`
- Trên: Trước interpolation — các frame thiếu đánh dấu đỏ '×'
- Dưới: Sau interpolation — các điểm nội suy hình kim cương cam, detection giữ nguyên xanh

##### `visualize_team_assignment.py` — K-Means gán đội

Minh họa chi tiết thuật toán phân tách màu áo:

| Cột | Hiển thị |
|-----|----------|
| 1 | Crop vùng cầu thủ (nửa trên = vùng áo) |
| 2 | Mask K-Means (đen = nền, trắng = áo) |
| 3 | Màu áo trích xuất (RGB swatch) |
| 4 | Bar chart 2 cluster centers (nền vs áo) |

**Output**: `analysis/figures/kmeans_team_assignment.png`

---

#### 3.14.4 Kết quả & Thống kê

##### `visualize_player_stats.py` — Thống kê cầu thủ

Đọc `stubs/tracks_full.pkl`, vẽ 3 biểu đồ cho 14 cầu thủ đầu tiên:

| Biểu đồ | Mô tả |
|---------|-------|
| 1. Tốc độ tối đa (km/h) | Max speed mỗi cầu thủ, màu theo đội |
| 2. Tốc độ trung bình (km/h) | Mean speed |
| 3. Quãng đường di chuyển (m) | Max distance tích lũy |

- Yêu cầu: mỗi cầu thủ ≥ 10 frames có dữ liệu speed
- Màu: team 1 = đỏ (`#e74c3c`), team 2 = xanh (`#3498db`)
- Grid alpha 0.3

**Output**: `analysis/figures/player_stats.png`

##### `visualize_training_results.py` — Biểu đồ huấn luyện

Vẽ kết quả training từ file `runs/detect/train/results.csv` (hoặc tạo dữ liệu demo nếu chưa có):

| Biểu đồ | Mô tả |
|---------|-------|
| 1. Loss curves | Train loss + Val loss (box regression) |
| 2. mAP curves | mAP@0.5 + mAP@0.5:0.95 + best mAP |
| 3. Precision & Recall | Precision + Recall |
| 4. mAP@0.5 per class | Horizontal bar: ball, goalkeeper, player, referee |

**Output**: `analysis/figures/training_results.png`

Nếu chưa train, script tự tạo dữ liệu synthetic (exponential saturation + noise) để demo cấu trúc biểu đồ.

##### `capture_result_frame.py` — Chụp frame kết quả

| Chức năng | Mô tả |
|-----------|-------|
| `capture_best_frames()` | Lấy 6 frame cách đều từ video kết quả, ghép vào lưới 2×3 |
| High-quality frame | Lưu 1 frame ở midpoint video với chất lượng gốc |

- Đầu vào: `output_videos/output-5.mp4`
- Đầu ra: `analysis/figures/result_showcase.png` (lưới 2×3) + `analysis/figures/best_frame.png` (1 frame)

---

#### 3.14.5 Thư mục đầu ra

| Thư mục | Định dạng | Mô tả |
|---------|-----------|-------|
| `analysis/figures/` | 15 file .png | Biểu đồ và ảnh minh họa cho báo cáo |
| `analysis/reports/` | 1 file .txt | Báo cáo chất lượng dataset |

Danh sách file trong `analysis/figures/`:

| File | Nguồn script | Mô tả |
|------|-------------|-------|
| `dataset_eda.png` | `eda_dataset.py` | EDA tổng quan dataset |
| `class_imbalance.png` | `class_imbalance_analysis.py` | Phân tích mất cân bằng class |
| `data_split.png` | `verify_data_split.py` | Phân chia train/val/test |
| `label_format_explanation.png` | `visualize_label_format.py` | Giải thích YOLO format |
| `augmentation_comparison.png` | `visualize_preprocessing.py` | 5 augmentation so sánh |
| `labeled_sample.png` | `visualize_preprocessing.py` | Ảnh raw vs labeled |
| `mosaic_augmentation.png` | `visualize_mosaic_augmentation.py` | Mosaic augmentation |
| `perspective_transform.png` | `visualize_perspective.py` | Bird's eye view |
| `optical_flow.png` | `visualize_optical_flow.py` | Optical flow camera |
| `ball_interpolation.png` | `visualize_ball_interpolation.py` | Nội suy vị trí bóng |
| `kmeans_team_assignment.png` | `visualize_team_assignment.py` | K-Means gán đội |
| `player_stats.png` | `visualize_player_stats.py` | Thống kê tốc độ/quãng đường |
| `training_results.png` | `visualize_training_results.py` | Biểu đồ training |
| `result_showcase.png` | `capture_result_frame.py` | 6 frame kết quả |
| `best_frame.png` | `capture_result_frame.py` | 1 frame chất lượng cao |

---

### 4.1 Schema của Tracks Dictionary

```python
tracks = {
    "players": [
        {
            <track_id: int>: {
                "bbox": [x1, y1, x2, y2],          # float — tọa độ pixel
                "position": (cx, foot_y),           # (center_x, y2)
                "position_adjusted": (cx, foot_y),  # đã bù camera movement
                "position_transformed": (x, y),     # tọa độ sân (cm)
                "team": 1 | 2,                      # ID đội
                "team_color": (B, G, R),            # BGR 0-255
                "has_ball": True | False,           # đang có bóng?
                "speed": float,                     # km/h
                "distance": float,                  # m (tích lũy)
            },
        },
        ...
    ],
    "referees": [{<track_id>: {"bbox": [...], "position": ..., ...}}, ...],
    "ball": [{1: {"bbox": [...], "position": (cx, cy), ...}}, ...],
}
```

### 4.2 Camera Movement Array

```python
cam_move = [
    [dx_0, dy_0],   # frame 0 — luôn [0, 0]
    [dx_1, dy_1],   # frame 1
    ...
]  # length = num_frames
```

### 4.3 Team Ball Control Array

```python
team_ball_control = np.array([
    0,   # chưa xác định / frame đầu
    1,   # đội 1
    1,   # đội 1
    2,   # đội 2
    ...
], dtype=int)  # shape: (num_frames,)
```

### 4.4 Format Video Input

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | .mp4, .avi, .mov |
| Codec khuyến nghị | H.264 |
| Độ phân giải | ≤ 1920×1080 |
| FPS | Bất kỳ (khuyến nghị 25-30) |
| Góc quay | Fixed broadcast camera |

### 4.5 Format Video Output

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | .mp4 (H.264) — ưu tiên, fallback .avi (MJPG) |
| Kích thước | Giữ nguyên input resolution |
| Annotation | Ellipse + ID cầu thủ, triangle bóng, camera movement overlay, speed/distance text, ball control, formation, minimap, keypoints |

### 4.6 Format Heatmap Output

| Thuộc tính | Giá trị |
|------------|---------|
| Định dạng | .png |
| Kích thước | 630×420 px |
| Số lượng | 3 ảnh: both (JET), team1 (WINTER), team2 (AUTUMN) |

### 4.7 Cấu trúc Dataset

#### Dataset phát hiện cầu thủ

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | Roboflow — `football-players-detection-3zvbc` |
| Classes | ball (0), goalkeeper (1), player (2), referee (3) |
| Format | YOLO (.txt) |
| Độ phân giải ảnh | 1280×720 — 1920×1080 |
| Augmentations | Horizontal flip, HSV jitter, mosaic, noise |

#### Dataset keypoint sân

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | Roboflow — `football-field-detection-f07vi` |
| Classes | pitch (1 class) |
| Keypoints | 32 keypoints |
| Format | YOLO Pose (.txt: class cx cy w h x1 y1 v1 x2 y2 v2 ...) |
| Visibility | 0=invisible, 1=occluded, 2=visible |

---

## 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY

### 5.1 Môi trường

- **Python**: ≥ 3.9 (khuyến nghị 3.10 hoặc 3.11)
- **GPU**: NVIDIA CUDA-compatible (≥ 4GB VRAM khuyến nghị)
- **Hệ điều hành**: Windows 10+, Ubuntu 20.04+, Kaggle/Colab

### 5.2 Cài đặt

```bash
# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux

# Cài dependencies
pip install -r requirements.txt

# Cài PyTorch riêng theo CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5.3 Download Models

Đặt các file model vào `models/`:

| File | Nguồn |
|------|-------|
| `models/player_detector.pt` | Train từ `trainings/football_training.py` hoặc download |
| `models/pitch_keypoint_detector.pt` | Train từ `trainings/pitch_keypoint_training.py` hoặc download |

### 5.4 Chạy Pipeline

```bash
# Chạy toàn bộ pipeline (tracking + render)
python main.py --mode all --video input_videos/sample.mp4

# Chỉ chạy tracking (lưu stub)
python main.py --mode tracking --video input_videos/sample.mp4

# Chỉ render từ stub đã có
python main.py --mode render --video input_videos/sample.mp4

# Giao diện Gradio
python app/gradio_app.py

# Giao diện Streamlit
streamlit run app/streamlit_app.py
```

### 5.5 Cấu hình Team

Trong trường hợp muốn chỉ định màu đội thủ công, sửa trong `phase2_render_from_stubs` của `main.py`:

```python
team_assigner.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}  # BGR
```

---

## 6. CẤU TRÚC THƯ MỤC

```
football_tracking/
├── main.py                          # Pipeline chính (entry point)
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore rules
│
├── trackers/
│   └── tracker.py                   # YOLO detection + ByteTrack + interpolation + drawing
│
├── estimators/
│   ├── camera_movement_estimator.py # Optical flow → camera pan
│   ├── view_transformer.py          # Homography → tọa độ sân thực
│   └── speed_distance_estimator.py  # Tốc độ & quãng đường
│
├── asigners/
│   ├── team_assigner.py             # K-Means phân cụm màu áo
│   └── player_ball_assigner.py      # Gán bóng cho cầu thủ gần nhất
│
├── pitch_keypoint_detector/
│   └── pitch_keypoint_detector.py   # YOLO-pose keypoint + SoccerPitchConfig
│
├── minimap/
│   └── minimap_renderer.py          # Vẽ tactical 2D minimap
│
├── heatmap_generator/
│   └── heatmap_generator.py         # Tạo ảnh nhiệt mật độ
│
├── formations/
│   └── formation_analyzer.py        # Phát hiện đội hình (K-Means)
│
├── app/
│   ├── gradio_app.py                # Giao diện Gradio
│   └── streamlit_app.py             # Giao diện Streamlit
│
├── utils/
│   ├── video_util.py                # Đọc/ghi video
│   └── bbox_util.py                 # Hàm tiện ích bounding box
│
├── cleanings/
│   ├── player_cleaning.py           # Làm sạch dataset phát hiện cầu thủ
│   └── pitch_keypoint_cleaning.py   # Làm sạch dataset keypoint sân
│
├── trainings/
│   ├── football_training.py         # Huấn luyện YOLOv8 phát hiện cầu thủ
│   ├── pitch_keypoint_training.py   # Huấn luyện YOLOv8-pose keypoint sân
│   ├── player_dataset.py            # Download dataset từ Roboflow
│   └── pitch_keypoint_dataset.py    # Download dataset từ Roboflow
│
├── analysis/                        # Scripts phân tích & trực quan hóa
│   ├── eda_dataset.py               # EDA tổng quan dataset
│   ├── class_imbalance_analysis.py  # Phân tích mất cân bằng class
│   ├── check_dataset_quality.py     # Kiểm tra chất lượng dataset
│   ├── verify_data_split.py         # Xác minh phân chia train/val/test
│   ├── visualize_label_format.py    # Giải thích YOLO format
│   ├── visualize_preprocessing.py   # Minh họa augmentation
│   ├── visualize_mosaic_augmentation.py # Mosaic augmentation
│   ├── visualize_perspective.py     # Minh họa perspective transform
│   ├── visualize_optical_flow.py    # Minh họa optical flow
│   ├── visualize_ball_interpolation.py # Minh họa nội suy bóng
│   ├── visualize_team_assignment.py # Minh họa K-Means gán đội
│   ├── visualize_player_stats.py    # Thống kê tốc độ/quãng đường
│   ├── visualize_training_results.py# Biểu đồ huấn luyện
│   ├── capture_result_frame.py      # Chụp frame từ video kết quả
│   ├── figures/                     # 15 file PNG ảnh kết quả
│   └── reports/                     # Báo cáo dạng text (.txt)
│
├── models/                          # Model weights
│   ├── player_detector.pt           # YOLOv8x — phát hiện cầu thủ
│   └── pitch_keypoint_detector.pt   # YOLOv8x-pose — keypoint sân
│
├── input_videos/                    # Video đầu vào (git-ignored)
├── output_videos/                   # Video đầu ra + heatmap (git-ignored)
├── stubs/                           # File pickle cache (git-ignored)
└── cache_gradio/                    # Gradio cache (git-ignored)
```

---

## 7. SƠ ĐỒ LUỒNG DỮ LIỆU

### 7.1 Pipeline Tracking (Pha 1)

```
[Video File]
    │
    ▼
read_video() ───────────────────────────────── frames: list[np.array]
    │
    ▼
Tracker.get_object_tracks()
    ├── YOLOv8 detect_frames() ──► raw detections
    ├── ByteTrack.update_with_detections() ──► track IDs
    └── return ──► tracks (raw, chỉ có bbox)
    │
    ▼
Tracker.add_position_to_tracks()
    └── thêm key "position" (foot center / ball center)
    │
    ▼
CameraMovementEstimator.get_camera_movement()
    ├── calcOpticalFlowPyrLK()
    └── return ──► cam_move: list[[dx, dy]]
    │
    ▼
CameraMovementEstimator.add_adjust_positions_to_tracks()
    └── thêm key "position_adjusted"
    │
    ▼
PitchKeypointDetector → ViewTransformer
    ├── detect_smoothed() ──► keypoints
    ├── get_homography() ──► ma trận H
    └── perspectiveTransform() ──► thêm key "position_transformed"
    │
    ▼
Tracker.interpolate_ball_positions()
    └── nội suy bbox bóng bằng Pandas
    │
    ▼
Tracker.interpolate_player_positions()
    └── nội suy bbox cầu thủ theo từng track_id
    │
    ▼
SpeedDistanceEstimator.add_speed_and_distance_to_tracks()
    └── thêm key "speed", "distance"
    │
    ▼
TeamAssigner
    ├── assign_team_color(frame[0], players[0])
    └── get_player_team() per frame ──► thêm key "team", "team_color"
    │
    ▼
PlayerBallAssigner.assign_ball_to_player()
    └── thêm key "has_ball" ──► team_ball_control array
    │
    ▼
Save stubs (tracks_full.pkl, cam_move.pkl, team_ball_control.npy)
```

### 7.2 Pipeline Rendering (Pha 2)

```
[Stubs: tracks_full.pkl, cam_move.pkl, team_ball_control.npy]
    │
    ▼
detect_team_formation()
    └── (f1, n1, c1), (f2, n2, c2) ──► formation names
    │
    ▼
For each frame:
    ├── Draw player ellipses (team-colored) + ID
    ├── Draw ball carrier triangle (green) if has_ball
    ├── Draw referee ellipses (yellow)
    ├── Draw ball triangle (cyan)
    ├── Draw camera movement overlay (top-left)
    ├── Draw speed & distance text (below foot)
    ├── Draw team ball control % (top-left)
    ├── Draw formation labels (bottom-left)
    ├── Draw pitch keypoints (optional)
    ├── Draw minimap overlay (bottom-right, optional)
    └── Append to output_frames list
    │
    ▼
save_video() ──► output_videos/output_enhanced.mp4
    │
    ▼
HeatmapGenerator
    ├── render_both() ──► heatmap_both.png
    ├── render_team(1, WINTER) ──► heatmap_team1.png
    └── render_team(2, AUTUMN) ──► heatmap_team2.png
```

### 7.3 Dependency Graph giữa các Module

```
read_video()
    └── phụ thuộc: OpenCV

Tracker.get_object_tracks()
    ├── phụ thuộc: Ultralytics YOLO, Supervision ByteTrack
    └── cần: read_video()

CameraMovementEstimator
    ├── phụ thuộc: OpenCV (optical flow, feature detection)
    └── cần: frame đầu tiên

PitchKeypointDetector
    ├── phụ thuộc: Ultralytics YOLO-pose, OpenCV (homography)
    └── cần: frame

ViewTransformer
    ├── phụ thuộc: PitchKeypointDetector
    └── cần: tracks (position_adjusted)

SpeedDistanceEstimator
    └── cần: tracks (position_transformed), fps

TeamAssigner
    ├── phụ thuộc: scikit-learn (K-Means)
    └── cần: frame, tracks (bbox)

PlayerBallAssigner
    └── cần: tracks (players + ball bbox)

FormationAnalyzer
    ├── phụ thuộc: scikit-learn (K-Means)
    └── cần: tracks (position_transformed, team)

MinimapRenderer
    └── cần: tracks (position_transformed, team_color)

HeatmapGenerator
    └── cần: tracks (position_transformed, team)

GradioApp
    └── cần: tất cả module trên

### 7.4 Pipeline Training & Cleaning

```
[Roboflow API Key]
    │
    ▼
player_dataset.py / pitch_keypoint_dataset.py
    └── download ──► datasets/ (raw)
        │
        ▼
    player_cleaning.py / pitch_keypoint_cleaning.py
        ├── scan_split() ──► phát hiện lỗi
        ├── clean_split() ──► copy ảnh sạch
        ├── fix_bbox_labels() / fix_and_copy_keypoint_split() ──► chuẩn hóa tọa độ
        └── visualize_before_after() ──► ảnh so sánh PNG
        │
        ▼
    datasets/*-cleaned/
        │
        ▼
    football_training.py / pitch_keypoint_training.py
        ├── YOLO pre-trained weights (yolov8x.pt / yolov8x-pose.pt)
        ├── model.train(epochs=100, imgsz=1280|640, ...)
        └── return ──► models/*/weights/best.pt
            │
            ▼
        Tracker (Section 3.1) / PitchKeypointDetector (Section 3.7)
```

### 7.5 Dependency Graph Cleaning & Training

```
player_dataset.py
    └── phụ thuộc: Roboflow API, dotenv

pitch_keypoint_dataset.py
    └── phụ thuộc: Roboflow API, dotenv

player_cleaning.py
    ├── phụ thuộc: OpenCV, NumPy, Matplotlib
    ├── cần: dataset từ player_dataset.py
    └── output: dataset cleaned

pitch_keypoint_cleaning.py
    ├── phụ thuộc: OpenCV, NumPy, Matplotlib
    ├── cần: dataset từ pitch_keypoint_dataset.py
    └── output: dataset cleaned

football_training.py
    ├── phụ thuộc: Ultralytics YOLO, Roboflow API, dotenv
    ├── cần: dataset cleaned từ player_cleaning.py
    └── output: models/player_detector.pt

pitch_keypoint_training.py
    ├── phụ thuộc: Ultralytics YOLO, Roboflow API, dotenv
    ├── cần: dataset cleaned từ pitch_keypoint_cleaning.py
    └── output: models/pitch_keypoint_detector.pt

### 7.6 Dependency Graph Analysis

```
eda_dataset.py
    ├── phụ thuộc: Seaborn, Matplotlib, PyYAML
    ├── cần: Roboflow dataset
    └── output: figures/dataset_eda.png

class_imbalance_analysis.py
    ├── phụ thuộc: NumPy, Matplotlib
    ├── cần: Roboflow dataset
    └── output: figures/class_imbalance.png

check_dataset_quality.py
    ├── phụ thuộc: OpenCV
    ├── cần: Roboflow dataset
    └── output: reports/dataset_quality_report.txt

verify_data_split.py
    ├── phụ thuộc: NumPy, Matplotlib
    ├── cần: Roboflow dataset
    └── output: figures/data_split.png

visualize_label_format.py
    ├── phụ thuộc: OpenCV, Matplotlib
    ├── cần: Roboflow dataset
    └── output: figures/label_format_explanation.png

visualize_preprocessing.py
    ├── phụ thuộc: OpenCV, Matplotlib
    ├── cần: Roboflow dataset
    └── output: figures/augmentation_comparison.png + labeled_sample.png

visualize_mosaic_augmentation.py
    ├── phụ thuộc: OpenCV, Matplotlib
    ├── cần: Roboflow dataset
    └── output: figures/mosaic_augmentation.png

visualize_perspective.py
    ├── phụ thuộc: OpenCV, Matplotlib, PitchKeypointDetector
    ├── cần: video mẫu + models/pitch_keypoint_detector.pt
    └── output: figures/perspective_transform.png

visualize_optical_flow.py
    ├── phụ thuộc: OpenCV, Matplotlib
    ├── cần: video mẫu
    └── output: figures/optical_flow.png

visualize_ball_interpolation.py
    ├── phụ thuộc: Pandas, Matplotlib
    ├── cần: stubs/track_stubs.pkl
    └── output: figures/ball_interpolation.png

visualize_team_assignment.py
    ├── phụ thuộc: OpenCV, Matplotlib, scikit-learn (KMeans)
    ├── cần: video mẫu + stubs/track_stubs.pkl
    └── output: figures/kmeans_team_assignment.png

visualize_player_stats.py
    ├── phụ thuộc: NumPy, Matplotlib
    ├── cần: stubs/tracks_full.pkl
    └── output: figures/player_stats.png

visualize_training_results.py
    ├── phụ thuộc: Pandas, Matplotlib
    ├── cần: runs/detect/train/results.csv (hoặc tự tạo demo)
    └── output: figures/training_results.png

capture_result_frame.py
    ├── phụ thuộc: OpenCV, Matplotlib
    ├── cần: output_videos/output-5.mp4
    └── output: figures/result_showcase.png + best_frame.png
```

---

*Tài liệu này được tạo ngày 07/06/2026, phiên bản tương ứng commit `139139c`.*
