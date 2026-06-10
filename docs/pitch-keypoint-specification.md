# Đặc tả PitchKeypointDetector

## 1. Tổng quan

`pitch_keypoint_detector/pitch_keypoint_detector.py` chịu trách nhiệm phát hiện 32 keypoint trên sân bóng đá từ ảnh broadcast và tính ma trận homography để chiếu tọa độ pixel lên mặt sẳng (top-down view). Đây là module nền tảng cho ViewTransformer, SpeedDistanceEstimator, và FormationAnalyzer.

```
Frame gốc (ảnh broadcast)
    │
    ├── PitchKeypointDetector.detect() → 32 keypoint (x, y, conf)
    │       │
    │       └── PitchKeypointDetector.get_homography() → ma trận H (3×3)
    │               │
    │               └── perspectiveTransform(pos, H) → tọa độ sân (cm)
    │
    └── SoccerPitchConfig: kích thước sân thực tế (12000×7000 cm)
```

## 2. Import (dòng 1–4)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import cv2` | OpenCV: `cv2.findHomography` (RANSAC), `cv2.perspectiveTransform`, `cv2.circle/putText` (draw). |
| 2 | `import numpy as np` | NumPy: mảng tọa độ, mask, vertices. |
| 3 | `from ultralytics import YOLO` | Ultralytics YOLOv8: load model pose-pretrained, inference. |
| 4 | `from ultralytics.nn.modules.head import Detect` | Fix version mismatch: gán `Detect.forward` cho head nếu thiếu attribute `detect`. |

## 3. Class SoccerPitchConfig (dòng 7–57)

**Mục đích:** Định nghĩa kích thước chuẩn của sân bóng đá (theo FIFA) và 32 keypoint vertices tương ứng.

**Kích thước sân (cm):**

| Thuộc tính | Giá trị (cm) | Giải thích |
|------------|-------------|------------|
| `width` | 7000 | Chiều rộng sân (70m) |
| `length` | 12000 | Chiều dài sân (120m) |
| `penalty_box_width` | 4100 | Rộng vùng cấm địa (41m) |
| `penalty_box_length` | 2015 | Dài vùng cấm địa (20.15m) |
| `goal_box_width` | 1832 | Rộng vùng 5m50 (18.32m) |
| `goal_box_length` | 550 | Dài vùng 5m50 (5.5m) |
| `centre_circle_radius` | 915 | Bán kính vòng tròn trung tâm (9.15m) |
| `penalty_spot_distance` | 1100 | Khoảng cách từ chấm phạt đền đến vạch vôi (11m) |

### `vertices` property (dòng 17–57)

**Mục đích:** Tạo mảng 32 điểm `(x, y)` tương ứng 32 keypoint trên sân thực tế.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 18–23 | Lấy biến config | `w, l, pbw, pbl, gbw, gbl, ccr, psd` — tên viết tắt cho dễ đọc. |
| 24–57 | `return np.array([...], dtype=np.float32)` | Mảng 32 điểm, mỗi điểm là (x, y) trên mặt phẳng sân (gốc (0,0) ở góc trái trên). Thứ tự khớp với YOLO keypoint index. |

**32 keypoint vertices:**
```
 0: (0, 0)                     — góc trái trên
 1: (0, (w-pbw)/2)            — biên trái, đầu vùng cấm
 2: (0, (w-gbw)/2)            — biên trái, đầu vùng 5m
 3: (0, (w+gbw)/2)            — biên trái, cuối vùng 5m
 4: (0, (w+pbw)/2)            — biên trái, cuối vùng cấm
 5: (0, w)                     — góc trái dưới
 6: (gbl, (w-gbw)/2)          — cột dọc trái (trên)
 7: (gbl, (w+gbw)/2)          — cột dọc trái (dưới)
 8: (psd, w/2)                 — chấm phạt đền trái
 9: (pbl, (w-pbw)/2)          — vạch 16m50 trái (trên)
10: (pbl, (w-gbw)/2)          — vạch 5m50 trái (trên)
11: (pbl, (w+gbw)/2)          — vạch 5m50 trái (dưới)
12: (pbl, (w+pbw)/2)          — vạch 16m50 trái (dưới)
13: (l/2, 0)                   — biên dọc giữa sân (trên)
14: (l/2, w/2 - ccr)          — vòng tròn trung tâm (trên)
15: (l/2, w/2 + ccr)          — vòng tròn trung tâm (dưới)
16: (l/2, w)                   — biên dọc giữa sân (dưới)
17: (l-pbl, (w-pbw)/2)        — vạch 16m50 phải (trên)
18: (l-pbl, (w-gbw)/2)        — vạch 5m50 phải (trên)
19: (l-pbl, (w+gbw)/2)        — vạch 5m50 phải (dưới)
20: (l-pbl, (w+pbw)/2)        — vạch 16m50 phải (dưới)
21: (l-psd, w/2)              — chấm phạt đền phải
22: (l-gbl, (w-gbw)/2)        — cột dọc phải (trên)
23: (l-gbl, (w+gbw)/2)        — cột dọc phải (dưới)
24: (l, 0)                     — góc phải trên
25: (l, (w-pbw)/2)            — biên phải, đầu vùng cấm
26: (l, (w-gbw)/2)            — biên phải, đầu vùng 5m
27: (l, (w+gbw)/2)            — biên phải, cuối vùng 5m
28: (l, (w+pbw)/2)            — biên phải, cuối vùng cấm
29: (l, w)                     — góc phải dưới
30: (l/2 - ccr, w/2)          — vòng tròn trung tâm (trái)
31: (l/2 + ccr, w/2)          — vòng tròn trung tâm (phải)
```

## 4. Class PitchKeypointDetector

### 4.1. `__init__(self, model_path='models/pitch_keypoint.pt', conf_threshold=0.3)` (dòng 61–77)

**Mục đích:** Load YOLO pose model, fix version mismatch, thiết lập device và config.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 62–63 | `model_path='models/pitch_keypoint.pt'; conf_threshold=0.3` | Default model path. `conf_threshold=0.3`: chỉ giữ keypoint có confidence ≥ 0.3. |
| 64 | `self.model = YOLO(model_path)` | Load Ultralytics YOLO model (YOLOv8x-pose, 70M params, 32 keypoints). |
| 65–67 | Fix version mismatch | Một số phiên bản training cũ không lưu attribute `detect` trong head. Gán `head.detect = Detect.forward` để tương thích. |
| 68–74 | Device selection | `device = 0 if torch.cuda.is_available() else 'cpu'`. Nếu có CUDA, chuyển model lên GPU. |
| 75 | `self.conf = conf_threshold` | Ngưỡng confidence. |
| 76 | `self.config = SoccerPitchConfig()` | Config sân — chứa vertices, kích thước. |
| 77 | `self.prev_keypoints = None` | Lưu keypoints frame trước cho smoothing temporal. |

### 4.2. `detect(self, frame)` (dòng 79–86)

**Mục đích:** Chạy YOLO inference, trả về keypoint coordinates + confidences.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 80 | `results = self.model(frame, conf=self.conf, verbose=False, device=self.device)` | Inference YOLO. `conf=0.3`: low threshold (sẽ lọc lại sau). `verbose=False`: tắt log. |
| 81 | `kps = results[0].keypoints` | Lấy keypoints result. `results[0]` là Frame 1 (batch 1). |
| 82–83 | Check None | Nếu không detect keypoint nào, trả về None. |
| 84 | `xy = kps.data[0][:, :2].cpu().numpy()` | `kps.data.shape = (1, 32, 3)`: [batch, num_kp, (x, y, visible)]. Lấy (x, y) và chuyển về numpy. |
| 85 | `confs = kps.conf[0].cpu().numpy()` | `kps.conf.shape = (1, 32)`: confidence của từng keypoint. Nếu None (model cũ), gán mặc định 1.0. |
| 86 | `return xy, confs` | `xy.shape=(32,2)`, `confs.shape=(32,)`. |

**Ví dụ trả về:**
```python
>>> kps = detector.detect(frame)
>>> kps is not None
True
>>> xy, confs = kps
>>> xy.shape, confs.shape
((32, 2), (32,))
>>> xy[:3]
array([[ 450.2, 320.5],
       [ 455.1, 350.2],
       [ 460.0, 400.8]], dtype=float32)
>>> confs[:3]
array([0.921, 0.883, 0.456])
```

### 4.3. `detect_smoothed(self, frame, alpha=0.6)` (dòng 88–102)

**Mục đích:** Temporal smoothing giữa frame trước và frame hiện tại để giảm jitter.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 89 | `detected = self.detect(frame)` | Raw detection. |
| 90–91 | Fallback | Nếu raw detection None, giữ nguyên keypoints frame trước (`self.prev_keypoints`). |
| 92 | `xy, confs = detected` | Tách coordinates và confidences. |
| 93 | `mask = confs > self.conf` | Chỉ smooth keypoint có confidence > 0.3. Keypoint low-conf không được smooth (giữ raw). |
| 94–98 | EMA smoothing | Nếu đã có `prev_keypoints`, áp dụng EMA: `smoothed[mask] = 0.6 * xy[mask] + 0.4 * prev_xy[mask]`. Chỉ smooth keypoint có conf cao. |
| 100–101 | First frame | Nếu chưa có prev, lưu raw làm reference. |
| 102 | `return self.prev_keypoints` | Trả về `(smoothed_xy, confs)`. |

**Ví dụ trả về:**
```python
>>> kps = detector.detect_smoothed(frame1)  # frame đầu
>>> kps[0][:3]  # giống raw (chưa có prev)
array([[ 450.2, 320.5], [ 455.1, 350.2], [ 460.0, 400.8]])

>>> kps2 = detector.detect_smoothed(frame2)  # frame 2
>>> kps2[0][:3]  # smoothed: 0.6*frame2 + 0.4*frame1
array([[ 451.0, 321.0], [ 456.0, 351.0], [ 461.0, 401.0]])
```

### 4.4. `get_homography(self, frame_keypoints)` (dòng 104–114)

**Mục đích:** Tính ma trận homography 3×3 từ keypoints detected → vertices sân thực.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 105–106 | None check | Nếu frame_keypoints None → return None. |
| 107 | `xy, confs = frame_keypoints` | Tách (32,2) và (32,). |
| 108 | `mask = confs > self.conf` | Chỉ giữ keypoint có confidence ≥ 0.3. |
| 109–110 | Cần ≥ 4 keypoints | Homography cần tối thiểu 4 cặp điểm để tính 8 degrees of freedom. `mask.sum() < 4` → None. |
| 111 | `src = xy[mask][:, :2].astype(np.float32)` | Source points: keypoint coordinates trên ảnh (pixel). |
| 112 | `dst = self.config.vertices[mask]` | Destination points: vertices trên sân thực (cm). Cùng mask — ghép cặp (keypoint_i → vertex_i). |
| 113 | `M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)` | RANSAC với threshold 5.0 pixel. Loại bỏ outlier (keypoint detect sai) và tính H từ inliers. |
| 114 | `return M` | Ma trận 3×3 hoặc None (nếu RANSAC thất bại). |

**Ví dụ trả về:**
```python
>>> M = detector.get_homography(kps)
>>> M.shape
(3, 3)
>>> M
array([[ 1.23e-01,  5.68e-02,  2.35e+02],
       [-3.21e-02,  1.99e-01,  1.23e+02],
       [ 2.35e-05,  1.23e-05,  1.00e+00]])
```

### 4.5. `transform_point(self, point, M)` (dòng 116–121)

**Mục đích:** Áp dụng homography lên một điểm pixel → tọa độ sân.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 117–118 | M is None | Return None. |
| 119 | `p = np.array([[point]], dtype=np.float32)` | Reshape `(2,)` → `(1, 1, 2)` cho perspectiveTransform. |
| 120 | `tp = cv2.perspectiveTransform(p, M)` | `tp.shape = (1, 1, 2)` — tọa độ transformed. |
| 121 | `return tp[0][0]` | Trả về `[x_san, y_san]` (cm). |

**Ví dụ trả về:**
```python
>>> pt = detector.transform_point([500, 300], M)
>>> pt
array([6120.3, 3450.7])   # tọa độ trên sân (cm)
```

### 4.6. `draw_keypoints(self, frame, frame_keypoints)` (dòng 123–139)

**Mục đích:** Vẽ 32 keypoint lên frame (debug/visualization).

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 124–125 | None check | Nếu không có keypoints → return nguyên frame. |
| 126 | `xy, confs = frame_keypoints` | Tách (32,2) và (32,). |
| 127 | `for i, ((x, y), c) in enumerate(zip(xy, confs)):` | Duyệt từng keypoint. |
| 128–129 | Skip low-conf | Bỏ qua keypoint có confidence < ngưỡng. |
| 130–131 | Color theo confidence | `intensity = min(1.0, c / 0.7)`. `color = (0, 255*intensity, 255*(1-intensity))`: conf cao → xanh, conf thấp → đỏ. |
| 133–134 | Vẽ circle | Circle filled với `color`, outline trắng 1px. |
| 135–138 | Vẽ index | Số thứ tự keypoint (0-31), outline trắng + fill đen để dễ đọc. |
| 139 | `return frame` | Frame đã vẽ (in-place + return). |

## 5. Sơ đồ luồng xử lý

```
PitchKeypointDetector.__init__(model_path, conf_threshold)
  ├── YOLO(model_path) → self.model (32 keypoints)
  ├── Fix version mismatch (head.detect)
  ├── SoccerPitchConfig() → self.config
  └── prev_keypoints = None

detect(frame)
  └── model(frame, conf=0.3) → [xy(32,2), confs(32,)]

detect_smoothed(frame, alpha=0.6)
  ├── detect(frame) → (xy, confs)
  ├── [None] → return prev_keypoints
  ├── EMA: smoothed = alpha * xy + (1-alpha) * prev_xy (only high-conf)
  └── return (smoothed, confs)

get_homography(frame_keypoints)
  ├── mask = confs > self.conf
  ├── [< 4 keypoints] → None
  ├── src = xy[mask]; dst = vertices[mask]
  ├── cv2.findHomography(src, dst, RANSAC, 5.0)
  └── return M (3×3) hoặc None

transform_point(point, M)
  └── cv2.perspectiveTransform → [x_san, y_san]

draw_keypoints(frame, frame_keypoints)
  └── Vẽ circle + index cho mỗi keypoint conf > threshold
```

## 6. Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Version mismatch Ultralytics** | Gán `head.detect = Detect.forward` nếu thiếu attribute (dòng 65–67) |
| **Keypoint jitter giữa frame** | `detect_smoothed()` EMA alpha=0.6 — chỉ smooth keypoint có conf cao |
| **Keypoint low confidence** | `conf_threshold=0.3`. Chỉ dùng keypoint conf > 0.3 cho homography |
| **Homography cần ≥ 4 keypoints** | Check `mask.sum() < 4` → return None; ViewTransformer fallback về `last_good_M` |
| **32 keypoints nhưng chỉ 4 cần thiết** | RANSAC tự động chọn inliers, outlier keypoint bị loại bỏ |
