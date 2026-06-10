# Đặc tả PlayerBallAssigner

## 1. Tổng quan

`asigners/player_ball_assigner.py` xác định cầu thủ nào đang kiểm soát bóng dựa trên khoảng cách từ bóng đến chân trái/phải của mỗi cầu thủ.

## 2. Import (dòng 1–2)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import numpy as np` | NumPy: xử lý mảng tọa độ (dùng gián tiếp qua `measure_distance`). |
| 2 | `from utils import get_center, measure_distance` | `get_center(bbox)` → `((x1+x2)/2, (y1+y2)/2)`. `measure_distance(p1, p2)` → Euclidean distance `sqrt((x1-x2)² + (y1-y2)²)`. |

## 3. Class PlayerBallAssigner

### 3.1. `__init__(self, max_distance=70)` (dòng 6–7)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 6 | `def __init__(self, max_distance=70):` | `max_distance=70` pixel — khoảng cách tối đa từ bóng đến chân cầu thủ để được coi là "có bóng". 70px ~ 2-3% chiều rộng frame 1920px. |
| 7 | `self.max_distance = max_distance` | Ngưỡng khoảng cách. |


### 3.2. `assign_ball_to_player(self, players, ball_bbox)` (dòng 9–22)

**Mục đích:** Tìm player gần bóng nhất và trả về ID của player đó (hoặc -1 nếu không ai đủ gần).

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 10 | `ball_pos = get_center(ball_bbox)` | Tính tâm bóng: `((x1+x2)/2, (y1+y2)/2)`. |
| 11–12 | `min_dist = float('inf'); assigned = -1` | Khởi tạo: assigned = -1 (không ai có bóng). |
| 13 | `for pid, data in players.items():` | Duyệt tất cả player trong frame hiện tại. players là dict `{track_id: {"bbox": [x1,y1,x2,y2], ...}}`. |
| 14 | `bbox = data['bbox']` | Lấy bounding box của player. |
| 16 | `d_left = measure_distance((bbox[0], bbox[-1]), ball_pos)` | Khoảng cách từ chân trái đến bóng. `bbox[0]` là x1 (trái), `bbox[-1]` là y2 (dưới — foot). |
| 17 | `d_right = measure_distance((bbox[2], bbox[-1]), ball_pos)` | Khoảng cách từ chân phải đến bóng. `bbox[2]` là x2 (phải). |
| 18 | `dist = min(d_left, d_right)` | Lấy khoảng cách gần hơn — bóng có thể ở chân trái hoặc chân phải. |
| 19–21 | Kiểm tra ngưỡng | Nếu `dist < self.max_distance` (70px) và `dist < min_dist` (gần hơn player trước đó), cập nhật assigned. |
| 22 | `return assigned` | Trả về player ID (1, 2, 3...) hoặc -1 nếu không player nào đủ gần. |

**Ví dụ trả về:**
```python
>>> assigner = PlayerBallAssigner(max_distance=70)

>>> # Frame 10: bóng ở (500, 400), player 1 chân trái (495, 420), player 2 chân phải (600, 410)
>>> assigner.assign_ball_to_player(tracks["players"][10], tracks["ball"][10][1]['bbox'])
1   # d_left = sqrt((495-500)² + (420-400)²) = sqrt(25+400) = 20.6 → gần nhất

>>> # Frame 11: bóng ở (300, 350), player 1 chân trái (495, 420)
>>> assigner.assign_ball_to_player(tracks["players"][11], tracks["ball"][11][1]['bbox'])
-1  # dist = sqrt((495-300)² + (420-350)²) = sqrt(38025+4900) = 207.2 > 70 → không ai có bóng
```

## 4. Sơ đồ luồng xử lý

```
PlayerBallAssigner.assign_ball_to_player(players, ball_bbox)
  │
  ├── ball_pos = get_center(ball_bbox)
  │
  ├── For mỗi player (pid, data) trong frame:
  │   ├── bbox = data['bbox']
  │   ├── d_left  = measure_distance((bbox[0], bbox[-1]), ball_pos)
  │   ├── d_right = measure_distance((bbox[2], bbox[-1]), ball_pos)
  │   ├── dist = min(d_left, d_right)
  │   │
  │   └── [dist < max_distance AND dist < min_dist]
  │       ├── min_dist = dist
  │       └── assigned = pid
  │
  └── return assigned (hoặc -1)
```

## 5. Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Bóng ở chân nào?** | Tính khoảng cách đến cả chân trái `(x1, y2)` và chân phải `(x2, y2)`, lấy khoảng cách nhỏ hơn. |
| **Nhiều cầu thủ gần bóng** | Chọn player gần nhất (`dist < min_dist`). |
| **Không ai có bóng** | Trả về -1. `main.py` xử lý `assigned != -1` trước khi cập nhật. |
| **Bbox bóng biến mất** | `ball_bbox` được nội suy từ `interpolate_ball_positions`, luôn có giá trị. |
