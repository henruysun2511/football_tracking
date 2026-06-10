# Đặc tả ViewTransformer

## 1. Tổng quan

`estimators/view_transformer.py` chịu trách nhiệm biến đổi tọa độ pixel (2D ảnh) thành tọa độ thực tế trên sân (top-down view) bằng **homography mapping**. Đây là bước trung gian bắt buộc để tính toán tốc độ, khoảng cách, đội hình chiến thuật.

```
Tọa độ pixel (x, y)      Tọa độ sân (cm)
  ┌──────────┐    Homography     ┌──────────┐
  │  (500,300)│ ──────────────→  │(6000,3500)│
  └──────────┘    H (3×3)        └──────────┘
```

## 2. Import (dòng 1–2)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import cv2` | OpenCV: `cv2.findHomography` (tìm ma trận H), `cv2.perspectiveTransform` (áp dụng H lên điểm). |
| 2 | `import numpy as np` | NumPy: tạo mảng tọa độ, xử lý float32 cho perspectiveTransform. |

## 3. Class ViewTransformer

### 3.1. `__init__(self, kp_detector, smooth_alpha=0.05)` (dòng 6–15)

**Mục đích:** Khởi tạo transformer với pitch keypoint detector và tham số smoothing.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 7–8 | Comment `smooth_alpha` | `smooth_alpha` nhỏ (0.05) = smoothing mạnh: 95% giữ giá trị cũ, 5% cập nhật mới. Giảm nhiễu do keypoint detection sai lệch giữa các frame. |
| 9 | `self.kp_detector = kp_detector` | `PitchKeypointDetector` instance — dùng để detect 32 keypoint sân và tính homography. |
| 10–11 | `self.length = kp_detector.config.length; self.width = kp_detector.config.width` | Kích thước sân thực tế: length=12000cm (120m), width=7000cm (70m). Dùng để kiểm tra `_is_valid`. |
| 12 | `self.M_cache = {}` | Cache dictionary `{frame_num: homography_matrix}`. Tránh tính lại homography cho frame đã xử lý. |
| 13 | `self.last_good_M = None` | Homography của frame tốt nhất gần nhất. Dùng fallback nếu frame hiện tại detect keypoint kém. |
| 14 | `self.smooth_alpha = smooth_alpha` | Hệ số smoothing (EMA) cho tọa độ transformed. |
| 15 | `self._pos_smooth = {}` | Dictionary `{(obj, tid): [x, y]}` — lưu vị trí smoothed hiện tại. Key là tuple (object_type, track_id). |

### 3.2. `_get_homography_for_frame(self, video_frames, frame_num)` (dòng 17–27)

**Mục đích:** Lấy ma trận homography cho một frame cụ thể, có cache + fallback.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 18–19 | Cache hit | Nếu frame_num đã có trong M_cache, trả về ngay. |
| 20 | `kps = self.kp_detector.detect_smoothed(video_frames[frame_num])` | Detect keypoints với smoothing temporal (EMA giữa frame trước và frame hiện tại). Trả về `(xy, confs)` hoặc None. |
| 21 | `M = self.kp_detector.get_homography(kps)` | Tính homography từ keypoints detected → vertices sân thực. Dùng RANSAC để loại bỏ outlier. Trả về ma trận 3x3 hoặc None nếu < 4 keypoints hợp lệ. |
| 22–23 | Update last_good | Nếu homography hợp lệ, cập nhật `last_good_M`. |
| 24–25 | Fallback | Nếu homography None (detect kém), dùng `last_good_M` — frame trước đó có chất lượng tốt. |
| 26 | `self.M_cache[frame_num] = M` | Cache lại kết quả. |
| 27 | `return M` | Trả về ma trận 3x3 hoặc None. |

**Ví dụ trả về:**
```python
>>> M = transformer._get_homography_for_frame(frames, 0)
>>> type(M)
<class 'numpy.ndarray'>
>>> M.shape
(3, 3)
>>> M
array([[ 1.234e-01,  5.678e-02,  2.345e+02],
       [-3.210e-02,  1.987e-01,  1.234e+02],
       [ 2.345e-05,  1.234e-05,  1.000e+00]])
>>> transformer._get_homography_for_frame(frames, 0) is M  # cached
True
```

### 3.3. `_is_valid(self, pos)` (dòng 29–33)

**Mục đích:** Kiểm tra tọa độ transformed có nằm trong vùng hợp lệ không (có margin 25%).

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 30 | `margin_x = self.length * 0.25` | 25% chiều dài sân = 3000cm. |
| 31 | `margin_y = self.width * 0.25` | 25% chiều rộng sân = 1750cm. |
| 32–33 | Range check | `-margin_x <= x <= length + margin_x` và `-margin_y <= y <= width + margin_y`. Cho phép cầu thủ ở ngoài sân tối đa 25% kích thước sân. |

**Ví dụ trả về:**
```python
>>> transformer._is_valid([6000, 3500])   # tâm sân
True
>>> transformer._is_valid([-100, 3500])   # hơi ngoài lề
True   # trong margin 25% (3000cm)
>>> transformer._is_valid([20000, 3500])  # quá xa
False
```

### 3.4. `_smooth(self, key, raw)` (dòng 35–48)

**Mục đích:** Exponential Moving Average (EMA) để làm mượt tọa độ transformed.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 36–38 | Out-of-bounds | Nếu raw out-of-bounds (`_is_valid` == False), giữ nguyên prev (nếu có), không update state. Tránh nhiễu lớn làm sai lệch toàn bộ. |
| 40–42 | First frame | Nếu key chưa tồn tại, khởi tạo và trả về raw ngay. |
| 44–46 | EMA update | `smoothed = alpha * raw + (1 - alpha) * prev`. Với alpha=0.05, mỗi frame chỉ thay đổi 5% về phía giá trị mới. |
| 47 | `self._pos_smooth[key] = smoothed` | Lưu lại state cho frame sau. |
| 48 | `return smoothed` | Trả về list `[x, y]` đã smoothing. |

**Ví dụ trả về:**
```python
>>> transformer._smooth(('players', 1), [6000, 3500])
[6000.0, 3500.0]     # lần đầu: trả về raw
>>> transformer._smooth(('players', 1), [6010, 3510])
[6000.5, 3500.5]     # alpha=0.05: 0.05*6010 + 0.95*6000 = 6000.5
>>> transformer._smooth(('players', 1), [7000, 4000])  # nhảy xa
[6000.5, 3500.5]     # nếu is_valid=False: giữ prev
```

### 3.5. `add_transformed_position_to_tracks(self, tracks, video_frames)` (dòng 50–75)

**Mục đích:** Áp dụng perspective transform cho tất cả object (player, referee, ball) và lưu `position_transformed` vào tracks.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 51 | `for obj in ['players', 'referees', 'ball']:` | Duyệt 3 loại object. |
| 52 | `for frame_num, frame_track in enumerate(tracks[obj]):` | Duyệt từng frame. |
| 53 | `M = self._get_homography_for_frame(video_frames, frame_num)` | Lấy ma trận homography (có cache). |
| 54–60 | M is None | Nếu không có homography (keypoint detection thất bại hoàn toàn), giữ vị trí smoothed từ frame trước (key đã có trong `_pos_smooth`). |
| 63 | `for tid, data in frame_track.items():` | Duyệt từng object trong frame. |
| 64 | `pos = data.get('position')` | Lấy `position` (foot point / center) — đã được tính bởi `add_position_to_tracks`. |
| 65–66 | skip nếu None | Frame không có object này → bỏ qua. |
| 68–69 | `p = np.array([[pos]], dtype=np.float32); tp = cv2.perspectiveTransform(p, M)` | Perspective transform: nhân ma trận homography 3x3 với tọa độ đồng nhất `[x, y, 1]^T`. |
| 70–71 | Smooth và lưu | `raw = tp.tolist()` → `smoothed = self._smooth((obj, tid), raw)` → lưu `position_transformed`. |
| 74–75 | Exception handler | Bỏ qua lỗi (ví dụ pos None, M lỗi). |

**Ví dụ trả về:**
```python
>>> tracks["players"][10][1]
{
    'bbox': [500, 300, 550, 450],
    'position': (525.0, 450.0),                    # foot point (pixel)
    'position_transformed': [6120.3, 3450.7]        # tọa độ sân (cm)
}
>>> tracks["ball"][10][1]
{
    'bbox': [600, 320, 620, 340],
    'position': (610.0, 330.0),                     # center (pixel)
    'position_transformed': [6010.5, 3505.2]        # tọa độ sân (cm)
}
```

## 4. Sơ đồ luồng xử lý

```
ViewTransformer.__init__(kp_detector, smooth_alpha)
  ├── self.kp_detector = PitchKeypointDetector
  ├── self.M_cache = {}
  ├── self.last_good_M = None
  └── self._pos_smooth = {}

add_transformed_position_to_tracks(tracks, video_frames)
  │
  ├── For obj in [players, referees, ball]:
  │   ├── For frame_num, frame_track in enumerate(tracks[obj]):
  │   │   ├── _get_homography_for_frame(video_frames, frame_num)
  │   │   │   ├── [cache hit] → return M
  │   │   │   ├── kp_detector.detect_smoothed(frame)
  │   │   │   ├── kp_detector.get_homography(kps)
  │   │   │   │   ├── [M valid] → update last_good_M
  │   │   │   │   └── [M None]   → M = last_good_M
  │   │   │   └── cache + return M
  │   │   │
  │   │   ├── [M is None] → giữ vị trí smoothed cũ
  │   │   │
  │   │   └── For tid, data in frame_track:
  │   │       ├── perspectiveTransform(pos, M) → raw
  │   │       ├── _smooth(key, raw) → (EMA)
  │   │       └── tracks[obj][fn][tid]['position_transformed'] = smoothed
  │   │
  └── tracks đã được cập nhật
```

## 5. Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Keypoint detection không ổn định giữa các frame** | `detect_smoothed()` với EMA alpha=0.6 trên keypoint coordinates |
| **Homography thất bại ở một số frame** | Fallback về `last_good_M` — dùng homography frame tốt nhất gần nhất |
| **Tọa độ transformed bị nhiễu giật** | `_smooth()` với EMA alpha=0.05 — rất chậm, chỉ thay đổi 5% mỗi frame |
| **Cầu thủ chạy ra biên** | `_is_valid()` cho phép margin 25% → pos ngoài sân vẫn được giữ, không bị reset |
| **Tính toán lại homography liên tục** | `M_cache` lưu ma trận cho từng frame, tránh detect lại keypoint |
