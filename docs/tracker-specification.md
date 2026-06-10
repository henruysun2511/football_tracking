# Đặc tả File trackers/tracker.py

## 1. Tổng quan

`trackers/tracker.py` là module trung tâm chịu trách nhiệm phát hiện đối tượng (detection) và theo dõi (tracking) trong pipeline. File này định nghĩa class `Tracker` — một wrapper tích hợp YOLOv8 và ByteTrack, cung cấp các phương thức phát hiện theo batch, gán ID theo dõi, nội suy vị trí bị thiếu, tính tọa độ chân, và vẽ annotation.

```
YOLOv8 ──→ detections ──→ ByteTrack ──→ tracks (với ID)
                    ↓
          interpolation → add_position → tracks hoàn chỉnh
```

---

## 2. Import thư viện (dòng 1–7)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `from ultralytics import YOLO` | Thư viện Ultralytics YOLOv8. Cung cấp class `YOLO` để load model, predict, val, train. |
| 2 | `import supervision as sv` | Thư viện Supervision của Roboflow. Cung cấp `sv.Detections` — cấu trúc dữ liệu chuẩn hóa cho detection results, và `sv.ByteTrack` — thuật toán tracking by detection. |
| 3 | `import pickle` | Pickle serialize: lưu/đọc tracking results ra file .pkl (cơ chế stub). |
| 4 | `import os` | Thư viện hệ điều hành: kiểm tra file stub tồn tại, tạo thư mục. |
| 5 | `import numpy as np` | NumPy: xử lý mảng tọa độ, contour cho `drawContours`. |
| 6 | `import pandas as pd` | Pandas: nội suy tuyến tính (`interpolate()`, `bfill()`) cho ball và player positions. |
| 7 | `import cv2` | OpenCV: vẽ hình khối (ellipse, rectangle, contour, text) lên frame. |

---

## 3. Class Tracker

### 3.1. `__init__(self, model_path=None)` (dòng 11–18)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 12 | `self.model = YOLO(model_path) if model_path else None` | Load YOLO model từ file weights `.pt`. Nếu `model_path=None`, model không được load (chế độ render-only). |
| 13–18 | `self.tracker = sv.ByteTrack(...)` | Khởi tạo ByteTrack với 4 tham số: |

**Tham số ByteTrack:**

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `track_activation_threshold` | 0.25 | Ngưỡng confidence để kích hoạt track mới. Detection có confidence < 0.25 bị bỏ qua khi tạo track mới. |
| `lost_track_buffer` | 30 | Số frame tối đa một track có thể "mất" (không match được detection nào) trước khi bị xóa. Giá trị 30 frame ~ 1.2 giây ở 25 FPS. |
| `minimum_matching_threshold` | 0.8 | Ngưỡng IoU tối thiểu để match detection với track hiện có. IoU < 0.8 được coi là detection mới, không phải track cũ. |
| `frame_rate` | 25 | FPS của video đầu vào. Dùng để tính thời gian giữa các frame trong Kalman filter. |

ByteTrack sử dụng Kalman filter để dự đoán vị trí track ở frame tiếp theo, sau đó dùng Hungarian algorithm để match detection với track dựa trên IoU.

---

## 4. Detection (dòng 21–35)

### 4.1. `detect_frames(self, frames, batch_size=20)`

**Mục đích:** Chạy YOLO inference trên một list frame, xử lý theo batch để tối ưu GPU memory.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 22–23 | Kiểm tra model | Nếu `self.model is None`, raise `RuntimeError`. Trường hợp này xảy ra khi Tracker được khởi tạo không có `model_path` (ví dụ ở pha render). |
| 24–25 | `import torch; device = 0 if torch.cuda.is_available() else 'cpu'` | Kiểm tra GPU. `device=0` nếu có CUDA, `'cpu'` nếu không. Lưu ý: số 0 là chỉ số GPU đầu tiên, không phải tên device. |
| 26–27 | Kiểm tra VRAM | `torch.cuda.mem_get_info()[0]`: lấy dung lượng VRAM còn trống (bytes). Nếu < 1 GB, fallback về CPU để tránh lỗi OOM. |
| 28–29 | `self.model.to('cuda')` | Nếu đủ VRAM, chuyển model lên GPU. `YOLO()` load model mặc định trên CPU, cần `.to('cuda')` để inference trên GPU. |
| 30–34 | Vòng lặp batch | Chia frames thành các batch nhỏ (mặc định 20 frame/batch). `self.model.predict()` inference từng batch với `conf=0.1` (ngưỡng confidence tối thiểu). Kết quả được append vào list `detections`. |
| 35 | `return detections` | Trả về list `results` (Ultralytics Results objects) — mỗi phần tử tương ứng một frame. |

**Ví dụ trả về:**
```python
>>> detections = tracker.detect_frames(frames[:3])
>>> len(detections)
3
>>> type(detections[0])
<class 'ultralytics.engine.results.Results'>
>>> detections[0].boxes.xyxy
tensor([[8.1200e+02, 4.5600e+02, 9.6300e+02, 8.6400e+02],
        [2.3400e+02, 3.8900e+02, 3.6700e+02, 7.2100e+02],
        [4.5000e+02, 4.1200e+02, 5.2300e+02, 8.0500e+02]])
>>> detections[0].boxes.conf
tensor([0.8843, 0.7621, 0.6533])
>>> detections[0].boxes.cls
tensor([2., 2., 3.])  # 2=player, 3=referee
```

**`model.predict()` vs `model()`:** `predict()` cho phép truyền tham số như `conf`, `device`; `model()` chỉ inference với tham số mặc định.

---

## 5. Tracking (dòng 38–96)

### 5.1. `get_object_tracks(self, frames, read_from_stub=False, stub_path=None)`

**Mục đích:** Phát hiện và theo dõi đối tượng trên tất cả frame, trả về dictionary chứa tracks của players, referees, ball.

**Tham số:**

| Tham số | Mô tả |
|---------|-------|
| `frames` | List frame (NumPy array H×W×3) |
| `read_from_stub` | Nếu True và stub tồn tại, đọc từ file thay vì chạy detection |
| `stub_path` | Đường dẫn file .pkl để đọc/ghi cache |

**Kiến trúc dữ liệu đầu ra:**

```python
tracks = {
    "players": [
        {track_id: {"bbox": [x1,y1,x2,y2]}, ...},   # frame 0
        {track_id: {"bbox": [x1,y1,x2,y2]}, ...},   # frame 1
        ...
    ],
    "referees": [  # cấu trúc tương tự players
        {track_id: {"bbox": ...}, ...},
        ...
    ],
    "ball": [      # ball luôn có ID = 1
        {1: {"bbox": ...}},   # mỗi frame chỉ có 1 bóng hoặc rỗng
        {1: {"bbox": ...}},
        ...
    ]
}
```

**Ví dụ trả về:**
```python
>>> tracks = tracker.get_object_tracks(frames[:3])
>>> tracks.keys()
dict_keys(['players', 'referees', 'ball'])
>>> tracks["players"]  # 3 frames
[{1: {'bbox': [812, 456, 963, 864]}, 2: {'bbox': [234, 389, 367, 721]}},
 {1: {'bbox': [810, 452, 965, 868]}, 2: {'bbox': [236, 385, 370, 725]}},
 {1: {'bbox': [815, 460, 960, 862]}}]  # frame 3: mất player ID=2
>>> tracks["referees"]
[{3: {'bbox': [450, 412, 523, 805]}},
 {3: {'bbox': [448, 410, 525, 808]}},
 {3: {'bbox': [446, 408, 527, 812]}}]
>>> tracks["ball"]
[{1: {'bbox': [600, 320, 620, 340]}},
 {},
 {1: {'bbox': [605, 315, 625, 335]}}]  # frame giữa không detect được bóng
```

#### 5.1.1. Cache check (dòng 40–44)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 40–41 | Kiểm tra model | Raise error nếu model None. |
| 42 | `if read_from_stub and stub_path and os.path.exists(stub_path):` | Điều kiện đọc cache: tham số `read_from_stub=True`, có `stub_path`, và file tồn tại trên disk. |
| 43–44 | `pickle.load(f)` → `return` | Load và trả về ngay tracks từ file pickle, bỏ qua toàn bộ detection + tracking. |

#### 5.1.2. Detection (dòng 46)

```python
detections = self.detect_frames(frames)
```

Chạy YOLO batch inference trên tất cả frame. `detections` là list `Results` objects.

#### 5.1.3. Khởi tạo tracks dict (dòng 48–52)

```python
tracks = {"players": [], "referees": [], "ball": []}
```

Dictionary với 3 key, mỗi key tương ứng một list rỗng sẽ được populate cho từng frame.

#### 5.1.4. Vòng lặp frame (dòng 54–89)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 55–56 | `cls_names = detection.names; cls_names_inv = {v: k for k, v in cls_names.items()}` | `detection.names` là dict `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`. `cls_names_inv` đảo ngược: `{'ball': 0, 'goalkeeper': 1, ...}`. |
| 58 | `det_sv = sv.Detections.from_ultralytics(detection)` | Chuyển đổi kết quả YOLO (Ultralytics Results) sang định dạng `sv.Detections` của Supervision. Cấu trúc: `.xyxy` (bbox), `.confidence`, `.class_id`, `.tracker_id`. |
| 60–62 | Gộp goalkeeper → player | YOLO detect goalkeeper là class riêng, nhưng tracking chỉ cần phân biệt player (gồm cả thủ môn) vs referee. Gán lại `class_id` của goalkeeper thành player để đơn giản hóa. |
| 64 | `det_with_tracks = self.tracker.update_with_detections(det_sv)` | ByteTrack: update tracks với detection hiện tại. So sánh IoU giữa detection với track dự đoán (Kalman filter) → Hungarian algorithm → gán ID. `det_with_tracks` là `sv.Detections` đã có thêm cột `.tracker_id`. |
| 66–68 | Append dict rỗng | Mỗi frame bắt đầu với dict rỗng cho players, referees, ball. Các detection sẽ được thêm vào các dict này. |
| 71–72 | Kiểm tra tracker_id | Một số phiên bản Supervision cũ trả về `tracker_id = None`. `has_tid` kiểm tra điều này để xử lý tương thích. |
| 74–82 | Vòng lặp detection có track ID | Với mỗi detection đã được ByteTrack gán ID: lấy `bbox` (list 4 số), `class_id`, `track_id`. Nếu player → thêm vào `tracks["players"]`; nếu referee → `tracks["referees"]`. `track_id` là số nguyên do ByteTrack sinh ra. |
| 84–89 | Vòng lặp detection ban đầu (ball) | Bóng được xử lý riêng từ `det_sv` (không qua ByteTrack) vì bóng di chuyển nhanh, tracking riêng không hiệu quả. Ball luôn được gán ID = 1. |

#### 5.1.5. Lưu stub (dòng 91–94)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 91 | `if stub_path:` | Chỉ lưu nếu có đường dẫn stub. |
| 92 | `os.makedirs(os.path.dirname(stub_path), exist_ok=True)` | Tạo thư mục chứa stub nếu chưa tồn tại (ví dụ: `stubs/`). |
| 93–94 | `pickle.dump(tracks, f)` | Serialize toàn bộ tracks dict vào file pickle. Lần chạy sau sẽ đọc từ file này thay vì chạy detection lại. |

---

## 6. Interpolation (dòng 99–134)

### 6.1. `interpolate_ball_positions(self, ball_positions)`

**Mục đích:** Nội suy tuyến tính vị trí bóng ở các frame bị missing do YOLO không detect được.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 100 | `ball_list = [x.get(1, {}).get('bbox', []) for x in ball_positions]` | Duyệt từng frame: lấy `bbox` của ball (ID=1). Nếu frame không có bóng → `x.get(1, {})` trả về `{}`, `.get('bbox', [])` trả về `[]`. |
| 101–102 | `df = pd.DataFrame(ball_list, columns=['x1','y1','x2','y2'])` | Chuyển list bbox thành DataFrame 4 cột. Các frame thiếu sẽ có giá trị NaN. |
| 103 | `df = df.interpolate().bfill()` | `interpolate()`: nội suy tuyến tính giữa các giá trị không NaN. `bfill()`: các frame đầu còn NaN được lấp bằng giá trị không NaN đầu tiên (backward fill). |
| 104 | `return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]` | Chuyển DataFrame về list dict, giữ nguyên cấu trúc tracks ball. |

**Ví dụ trả về:**
```python
>>> ball_pos = [{1: {'bbox': [600, 320, 620, 340]}}, {}, {1: {'bbox': [605, 315, 625, 335]}}]
>>> interp = tracker.interpolate_ball_positions(ball_pos)
>>> interp  # frame giữa đã được nội suy
[{1: {'bbox': [600.0, 320.0, 620.0, 340.0]}},
 {1: {'bbox': [602.5, 317.5, 622.5, 337.5]}},  # linear interpolate
 {1: {'bbox': [605.0, 315.0, 625.0, 335.0]}}]
```

### 6.2. `interpolate_player_positions(self, player_positions)`

**Mục đích:** Nội suy bounding box cho từng player ID riêng biệt.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 107–109 | Thu thập tất cả track ID | Duyệt toàn bộ frame, thu thập tất cả `track_id` xuất hiện ít nhất một lần. `all_tids = set()` — mỗi ID chỉ được xử lý một lần. |
| 111–116 | Xác định khoảng frame | Với mỗi track ID: tìm `first_frame` (frame đầu tiên ID xuất hiện) và `last_frame` (frame cuối cùng). Chỉ nội suy trong khoảng [first, last]. |
| 118–123 | Xây dựng list bbox | Duyệt từ `first_frame` đến `last_frame`: nếu ID có trong frame → lấy bbox thật; nếu không → `[NaN, NaN, NaN, NaN]`. |
| 125–126 | DataFrame + interpolate | `pd.DataFrame(bboxes)` → `interpolate(method='linear')`. Không có `bfill()` vì khoảng đã được giới hạn trong [first, last] nên không có leading NaN. |
| 128–132 | Ghi đè bbox đã nội suy | `bboxes_interpolated[i]` là bbox đã được nội suy (hoặc giá trị gốc nếu không NaN). Ghi vào `player_positions[f][tid]['bbox']`. Nếu frame đó chưa có dict cho tid, tạo mới. |
| 134 | `return player_positions` | Trả về player_positions đã được cập nhật. |

**Ví dụ trả về:**
```python
>>> player_pos = [
...     {1: {'bbox': [100, 200, 150, 400]}, 2: {'bbox': [500, 300, 560, 500]}},   # frame 0
...     {1: {'bbox': [102, 202, 152, 402]}},                                        # frame 1: mất ID=2
...     {1: {'bbox': [106, 206, 156, 406]}, 2: {'bbox': [505, 305, 565, 505]}},   # frame 2: ID=2 xuất hiện lại
... ]
>>> interp = tracker.interpolate_player_positions(player_pos)
>>> interp[1][2]['bbox']  # frame 1, ID=2 đã được nội suy
[502.5, 302.5, 562.5, 502.5]  # (500+505)/2=502.5, (300+305)/2=302.5, ...
```

**So sánh interpolation ball vs player:**

| Đặc điểm | Ball | Player |
|-----------|------|--------|
| Phạm vi | Toàn bộ frame | [first_frame, last_frame] của từng ID |
| Xử lý đầu | `bfill()` — fill leading NaN | Không có leading NaN (khoảng đã giới hạn) |
| Traversal | Một DataFrame cho tất cả | Một DataFrame riêng cho mỗi ID |

---

## 7. Position Helpers (dòng 137–147)

### 7.1. `add_position_to_tracks(self, tracks)`

**Mục đích:** Tính tọa độ "chân" (foot position) cho player và tọa độ "tâm" cho bóng.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 138 | `for obj, obj_tracks in tracks.items():` | Duyệt 3 object type: `players`, `referees`, `ball`. |
| 139–141 | Vòng lặp frame + track | Với mỗi frame, với mỗi track trong frame, lấy `bbox = [x1, y1, x2, y2]`. |
| 142–144 | Ball: tâm | `pos = ((x1+x2)/2, (y1+y2)/2)` — tâm của bounding box. Bóng là vật thể nhỏ, tâm bbox xấp xỉ tâm bóng. |
| 145–146 | Player/Referee: chân | `pos = ((x1+x2)/2, y2)` — trung điểm cạnh đáy của bounding box. Giả định chân cầu thủ luôn ở dưới cùng bbox. |
| 147 | `tracks[obj][frame_num][tid]['position'] = pos` | Lưu position vào tracks. Key `position` được các module sau sử dụng (camera adjustment, homography, speed/distance). |

**Ví dụ trả về:**
```python
>>> tracks = {
...     "players": [{1: {'bbox': [100, 200, 150, 400]}}],
...     "ball":    [{1: {'bbox': [600, 320, 620, 340]}}],
...     "referees": [{3: {'bbox': [450, 412, 523, 805]}}]
... }
>>> tracker.add_position_to_tracks(tracks)
>>> tracks["players"][0][1]['position']
(125.0, 400.0)    # foot: ((100+150)/2, 400)
>>> tracks["ball"][0][1]['position']
(610.0, 330.0)    # center: ((600+620)/2, (320+340)/2)
>>> tracks["referees"][0][3]['position']
(486.5, 805.0)    # foot: ((450+523)/2, 805)
```

---

## 8. Draw Annotations (dòng 149–231)

### 8.1. `draw_ellipse(self, frame, bbox, color, track_id=None)`

**Mục đích:** Vẽ ellipse dưới chân cầu thủ/trọng tài + track ID.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 151–153 | Tính tham số | `y2` = tọa độ y đáy bbox. `x_c` = tâm x. `w` = chiều rộng bbox. |
| 154–158 | `cv2.ellipse(...)` | Vẽ ellipse: center tại `(x_c, y2)` (chân cầu thủ). `axes=(w, 0.35*w)` — rộng bằng bbox, cao = 35% bbox (tỷ lệ đẹp). `angle=0` (không xoay). `startAngle=-45, endAngle=235` — chỉ vẽ nửa dưới ellipse (hình vòng cung dưới chân). `color`, `thickness=2`. `lineType=cv2.LINE_4` — kiểu nét 4-connected. |
| 159 | `if track_id is not None:` | ID chỉ được vẽ nếu được cung cấp. |
| 160–164 | Vẽ background rect | Rectangle trắng 40×20px, bo tròn, đặt dưới chân để làm nền cho text ID. `y1r = y2 - rect_h//2 + 15`: căn giữa rect dưới ellipse. `color, -1`: fill đầy rect với màu đội. |
| 165–168 | Vẽ text ID | `cv2.putText` với số ID, font Hershey Simplex 0.6, màu đen, thickness 2. |

**Ví dụ trả về:** Hàm không return (modify frame in-place). Frame đầu vào được vẽ thêm ellipse + ID text.

**Đặc điểm ellipse:** Góc -45° đến 235° (tổng 280°) thay vì 360° để tạo khoảng hở phía trên — tạo cảm giác ellipse "nằm" trên mặt sân.

### 8.2. `draw_triangle(self, frame, bbox, color)`

**Mục đích:** Vẽ tam giác chỉ hướng phía trên bounding box — dùng cho bóng và cầu thủ có bóng.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 172–173 | Tính đỉnh tam giác | `y1` = tọa độ đỉnh bbox. `xc` = tâm x. |
| 174–176 | `np.array([[xc, y1], [xc-10, y1-20], [xc+10, y1-20]])` | Ba đỉnh tạo tam giác cân hướng lên: đỉnh dưới tại `(xc, y1)`, hai đỉnh trên cách nhau 20px. |
| 177 | `cv2.drawContours(frame, [pts], 0, color, -1)` | Vẽ tam giác filled với màu `color`. `-1` = fill. |
| 178 | `cv2.drawContours(frame, [pts], 0, (0,0,0), 2)` | Vẽ viền đen cho tam giác (outline), độ dày 2px. |

**Ví dụ trả về:** Hàm không return (modify frame in-place). Frame đầu vào được vẽ thêm tam giác filled + outline đen.

**Kỹ thuật outline 2 lớp:** Fill trước → viền đen sau. Tạo hiệu ứng viền đen bao quanh tam giác màu, giúp tam giác nổi bật trên mọi nền.

### 8.3. `draw_team_ball_control(self, frame, frame_num, team_ball_control)`

**Mục đích:** Vẽ overlay tỉ lệ kiểm soát bóng ở góc phải dưới frame.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 183 | `h, w = frame.shape[:2]` | Lấy kích thước frame. |
| 184–187 | Tạo overlay | Copy frame → rect trắng (góc phải dưới) → `addWeighted(0.4 overlay + 0.6 frame)` tạo nền mờ. |
| 189–191 | Tính % | `team1` / `team2`: đếm số frame đội 1/2 có bóng từ đầu đến frame hiện tại. `total += 1e-6` tránh chia 0. |
| 194–201 | Vẽ text | "Team 1: xx%" màu xanh dương `(255,0,0)` và "Team 2: xx%" màu đỏ `(0,0,255)` ở góc phải dưới. Font size 1, thickness 3. |

**Ví dụ trả về:** Hàm không return (modify frame in-place). Góc phải dưới frame hiển thị `Team 1: 63%` (xanh) và `Team 2: 37%` (đỏ). |

### 8.4. `draw_annotations(self, video_frames, tracks, team_ball_control)`

**Mục đích:** Vẽ toàn bộ annotation lên tất cả frame — hàm tổng hợp cho pipeline đơn giản.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 206–207 | Khởi tạo + copy frame | `frame = frame.copy()` tránh mutate frame gốc. |
| 210–217 | Vẽ player | Ellipse màu đội + ID. Nếu `has_ball=True`, vẽ thêm triangle xanh lá. `color = data.get("team_color", (0,255,0))` — màu xanh lá mặc định nếu chưa gán đội. |
| 219–221 | Vẽ referee | Ellipse màu vàng `(255,255,0)`. |
| 223–226 | Vẽ bóng | Triangle màu cyan `(0,255,255)`. |
| 228–229 | Vẽ ball control | Gọi `draw_team_ball_control()` để vẽ overlay %. |
| 230–231 | Append + return | Append frame đã annotate vào list output. |

**Ví dụ trả về:**
```python
>>> annotated = tracker.draw_annotations(frames[:3], tracks, team_ball_control)
>>> len(annotated)
3
>>> annotated[0].shape  # giữ nguyên kích thước frame gốc
(1080, 1920, 3)
>>> type(annotated[0])
<class 'numpy.ndarray'>  # mỗi frame là ảnh BGR đã vẽ annotation
```

---

## 9. Sơ đồ luồng xử lý

```
Tracker.__init__(model_path)
    │
    ├── YOLO(model_path) ──→ self.model
    └── sv.ByteTrack(...) ──→ self.tracker

get_object_tracks(frames, read_from_stub, stub_path)
    │
    ├── [stub exists] ──→ pickle.load → tracks
    │
    └── [no stub] ──→ detect_frames(frames)
                        │
                        └── predict(batch, conf=0.1, device)
                            │
                            ↓
                        sv.Detections.from_ultralytics()
                            │
                            ├── goalkeeper → player (remap)
                            │
                            └── ByteTrack.update_with_detections()
                                │
                                ├── players → tracks["players"]
                                ├── referees → tracks["referees"]
                                └── ball (ID=1) → tracks["ball"]
                                    │
                                    └── pickle.dump(stub)

interpolate_ball_positions(ball_positions)
    └── DataFrame.interpolate().bfill()

interpolate_player_positions(player_positions)
    └── per-track-id: DataFrame.interpolate()

add_position_to_tracks(tracks)
    ├── ball → center point
    └── player/ref → foot point (x_center, y_bottom)
```

---

## 10. Các lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Goalkeeper là class riêng** | Gộp goalkeeper → player (dòng 61–62) để đơn giản hóa. Thông tin thủ môn không cần thiết cho team assignment và formation detection. |
| **VRAM không đủ** | Tự động fallback CPU nếu VRAM < 1 GB (dòng 26–27). Kiểm tra trước khi inference để tránh OOM crash. |
| **ByteTrack không gán ID trên supervision cũ** | Kiểm tra `tracker_id is not None` (dòng 71–72); fallback dùng index i làm ID. |
| **Bóng không tracking bằng ByteTrack** | Ball được lấy trực tiếp từ detection (không qua ByteTrack) vì kích thước nhỏ, di chuyển nhanh, dễ bị mất track. Luôn gán ID=1. |
| **Frame thiếu detection** | Nội suy tuyến tính bằng Pandas — fill gap bbox dựa trên các frame lân cận. |

---

## 11. Phụ thuộc module

```mermaid
graph TD
    main.py --> trackers.tracker
    trackers.tracker --> YOLO[ultralytics.YOLO]
    trackers.tracker --> ByteTrack[supervision.ByteTrack]
    trackers.tracker --> pandas
    trackers.tracker --> numpy
    trackers.tracker --> cv2
    
    estimators.view_transformer --> tracks.position
    estimators.camera_movement_estimator --> tracks.position
    estimators.speed_distance_estimator --> tracks.position_transformed
    asigners.team_assigner --> tracks.players[].bbox
    asigners.player_ball_assigner --> tracks.players[].position + tracks.ball
    formations.formation_analyzer --> tracks.players[].position_transformed
    minimap.minimap_renderer --> tracks (all)
    heatmap_generator --> tracks.players[].position_transformed
```

`Tracker` sản xuất dữ liệu `tracks` — đây là đầu vào cho **tất cả** module phía sau trong pipeline (estimators, asigners, formations, minimap, heatmap). Chất lượng của tracking quyết định trực tiếp chất lượng toàn bộ hệ thống.
