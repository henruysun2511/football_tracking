# Đặc tả SpeedDistanceEstimator

## 1. Tổng quan

`estimators/speed_distance_estimator.py` tính toán tốc độ (km/h) và quãng đường đã di chuyển (mét) cho mỗi cầu thủ, dựa trên tọa độ `position_transformed` (top-down view, đơn vị cm) đã được tính bởi `ViewTransformer`.

## 2. Import (dòng 1–3)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import cv2` | OpenCV: `cv2.putText`, `cv2.FONT_HERSHEY_SIMPLEX` để vẽ speed/distance lên frame. |
| 2 | `import numpy as np` | NumPy: xử lý mảng tọa độ (được dùng gián tiếp qua `measure_distance`). |
| 3 | `from utils import measure_distance` | Hàm Euclidean distance: `sqrt((x1-x2)² + (y1-y2)²)`. |

## 3. Hằng số (dòng 5–7)

| Dòng | Tên | Giá trị | Giải thích |
|------|-----|---------|------------|
| 5 | `_UNIT_TO_METER` | 0.01 | Chuyển cm → m. Vị trí transformed từ ViewTransformer tính bằng cm (sân 12000×7000 cm). |
| 6 | `_MAX_SPEED_KMH` | 38.0 | Tốc độ tối đa của cầu thủ (km/h). Giới hạn trên để loại bỏ nhiễu. Tốc độ thực tế cao nhất ~37 km/h (Kylian Mbappé). |
| 7 | `_MAX_DIST_CM` | 300.0 | Khoảng cách tối đa cho phép trong 1 window (cm). Nếu cầu thủ "dịch chuyển" > 3m trong 5 frame (~0.2s), đó là nhiễu → bỏ qua. |

## 4. Class SpeedDistanceEstimator

### 4.1. `add_speed_and_distance_to_tracks(self, tracks, fps=25)` (dòng 13–82)

**Mục đích:** Tính tốc độ và quãng đường cho mỗi player dựa trên sliding window.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 14 | `total_dist = {}` | Dictionary `{obj: {tid: tổng_mét}}` — lưu tổng quãng đường tích lũy. |
| 15–17 | Skip ball & referees | Chỉ tính speed/distance cho players. Bóng di chuyển không đều, trọng tài không cần metric. |
| 19 | `n_frames = len(obj_tracks)` | Tổng số frame video. |
| 22–24 | Thu thập tất cả track ID | Duyệt toàn bộ frame, lấy tất cả player ID từng xuất hiện. `all_tids = set()`. |
| 26–30 | Khởi tạo `total_dist` | Với mỗi tid, khởi tạo `total_dist[obj][tid] = 0.0`. `last_speed = None` dùng để fill frame không có data. |
| 32–33 | Sliding window size = 5 | Duyệt từ `start=0` đến `n_frames` với bước nhảy `WINDOW=5`. Mỗi window 5 frame, không overlap. `end = min(start+4, n_frames-1)`. |
| 35–36 | Bỏ qua window 1 frame | Nếu `end == start`, window chỉ có 1 frame → không tính được speed. |
| 38–46 | Tìm first/last frame có data | Duyệt `start..end` để tìm frame đầu tiên và cuối cùng mà track ID này có `position_transformed`. Cần ít nhất 2 frame khác nhau. |
| 48–54 | Không đủ 2 frame | Nếu `first_fn == last_fn` (chỉ 1 frame có data), dùng `last_speed` (nếu có) để fill tất cả frame trong window. Giữ khoảng cách không đổi. |
| 56–57 | Lấy tọa độ đầu/cuối | `p_start = obj_tracks[first_fn][tid]['position_transformed']`; tương tự `p_end`. Đây là tọa độ (x, y) trên mặt sân (cm). |
| 59 | `dist_cm = measure_distance(p_start, p_end)` | Khoảng cách Euclidean giữa 2 điểm (cm). |
| 60–66 | Kiểm tra max distance | Nếu `dist_cm > 300cm`, coi là nhiễu (homography sai, keypoint nhảy). Dùng `last_speed` fill và skip. |
| 68 | `elapsed = (last_fn - first_fn) / fps` | Thời gian thực tế giữa 2 frame (giây). Với fps=25, 5 frame = 0.2s. |
| 69–70 | elapsed <= 0 | Tránh chia 0. |
| 72–73 | Tính tốc độ | `speed_ms = (dist_cm * 0.01) / elapsed` → `speed_kh = min(speed_ms * 3.6, 38.0)`. Công thức: km/h = (m/s) × 3.6. |
| 75 | `total_dist[obj][tid] += dist_cm * 0.01` | Cộng dồn quãng đường (mét). |
| 76 | `last_speed = speed_kh` | Lưu tốc độ hiện tại để fill frame không có data. |
| 79–82 | Ghi vào tất cả frame trong window | Gán `speed` và `distance` cho tất cả frame trong window có chứa tid. Nếu một frame không có bbox cho tid đó, nó sẽ không được ghi (vì `tid not in obj_tracks[fn]`). |

**Ví dụ trả về:**
```python
>>> # tracks["players"][5] (frame 5, player ID=1)
>>> tracks["players"][5][1]
{
    'bbox': [500, 300, 550, 450],
    'position': (525.0, 450.0),
    'position_transformed': [6120.3, 3450.7],
    'speed': 24.5,          # km/h
    'distance': 125.3        # mét (tổng tích lũy đến frame 5)
}
>>> tracks["players"][6][1]['speed']   # frame 6 cùng window
24.5                                     # speed giống nhau
>>> tracks["ball"][5].get(1, {}).get('speed')  # ball không tính
None
```

### 4.2. `draw_speed_and_distance(self, frames, tracks)` (dòng 84–104)

**Mục đích:** Vẽ speed và distance lên frame (dưới chân cầu thủ).

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 85–88 | Loop frames | Duyệt từng frame, bỏ qua ball và referees. |
| 89–94 | Lấy data | `speed`, `distance`, `bbox` từ tracks. Nếu thiếu speed hoặc bbox → skip. |
| 95–97 | Tính vị trí text | `foot_x = (x1 + x2) // 2` (tâm x), `foot_y = y2` (dưới cùng bbox). Text hiển thị bên dưới chân. |
| 98–100 | Vẽ speed | `f"{speed:.1f} km/h"` — 1 số lẻ, font 0.5, màu đen, độ dày 2. Offset Y: +40px (dưới foot). |
| 101–104 | Vẽ distance | `f"{dist:.1f} m"` — bên dưới speed, offset Y: +58px. |

**Ví dụ đầu ra (trên frame):**
```
                          ← ellipse của player
                   24.5 km/h   ← speed (y2+40)
                  125.3 m      ← distance (y2+58)
```

## 5. Sơ đồ luồng xử lý

```
SpeedDistanceEstimator.add_speed_and_distance_to_tracks(tracks, fps=25)
  │
  ├── For obj in [players] (skip ball, referees):
  │   ├── Collect all_tids từ toàn bộ video
  │   │
  │   └── For tid in all_tids:
  │       ├── total_dist[tid] = 0.0
  │       │
  │       └── For start in range(0, n_frames, WINDOW=5):
  │           ├── Tìm first_fn, last_fn có position_transformed
  │           │
  │           ├── [< 2 frame có data] → fill last_speed, skip
  │           │
  │           ├── dist_cm = measure_distance(p_start, p_end)
  │           │
  │           ├── [dist_cm > 300cm] → fill last_speed, skip (nhiễu)
  │           │
  │           ├── elapsed = (last_fn - first_fn) / fps
  │           ├── speed_kh = min(dist_cm * 0.01 / elapsed * 3.6, 38.0)
  │           ├── total_dist[tid] += dist_cm * 0.01
  │           │
  │           └── Ghi speed + distance vào tất cả frame trong window
  │
  └── tracks["players"][fn][tid] có key 'speed' + 'distance'

draw_speed_and_distance(frames, tracks)
  └── Vẽ "xx.x km/h" + "xx.x m" dưới chân mỗi player
```

## 6. Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Frame drop / detection missing** | Window 5 frame, chỉ cần 2 frame có data là tính được; nếu không thì fill `last_speed` |
| **Homography sai → tọa độ nhảy** | `_MAX_DIST_CM=300`: nếu cầu thủ "dịch chuyển" > 3m trong ~0.2s là nhiễu → bỏ qua |
| **Tốc độ phi thực tế** | Clamp `_MAX_SPEED_KMH=38.0` — cao hơn kỷ lục thế giới (37 km/h) |
| **Speed giật giữa các frame** | Ghi cùng speed cho cả window 5 frame thay vì tính per-frame |
| **Overlap** | Window non-overlapping: [0-4], [5-9], [10-14], ... — đơn giản, dễ hiểu |
