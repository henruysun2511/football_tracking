# Đặc tả CameraMovementEstimator

## 1. Tại sao cần ước lượng chuyển động camera?

Trong các trận bóng đá broadcast, camera **không cố định** — nó xoay (pan), nghiêng (tilt), và đôi khi zoom để theo dõi bóng. Điều này gây ra vấn đề:

```
Frame t=0: cầu thủ A ở pixel (500, 300)
Frame t=1: camera xoay sang phải 50px
           A ở pixel (450, 300) — MẶC DÙ A KHÔNG DI CHUYỂN
```

Nếu không bù chuyển động camera:
- Tọa độ pixel của cầu thủ bị sai lệch → homography (chiếu lên sân thực) thiếu chính xác
- Tốc độ cầu thủ bị tính sai (tưởng cầu thủ chạy nhưng thực ra camera xoay)
- Đội hình chiến thuật bị nhiễu

**Giải pháp:** Dùng Lucas-Kanade Optical Flow để đo độ dịch chuyển của nền (background) giữa các frame liên tiếp, sau đó bù trừ khỏi tọa độ cầu thủ.

## 2. Import (dòng 1–4)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import os` | Tạo thư mục stub, kiểm tra đường dẫn. |
| 2 | `import cv2` | OpenCV: `goodFeaturesToTrack` (Shi-Tomasi), `calcOpticalFlowPyrLK` (LK optical flow), vẽ overlay. |
| 3 | `import numpy as np` | Tạo mask numpy, xử lý mảng tọa độ. |
| 4 | `import pickle` | Lưu/đọc camera movement từ file stub .pkl. |

## 3. Class CameraMovementEstimator

### 3.1. `__init__(self, first_frame)` (dòng 8–22)

**Mục đích:** Khởi tạo estimator với tham số optical flow và mask khu vực theo dõi.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 9 | `self.minimum_distance = 5` | Ngưỡng tối thiểu (pixel) để coi là có chuyển động. Nếu max_distance ≤ 5px, frame được coi là tĩnh → movement = [0,0]. Loại bỏ nhiễu do rung camera nhẹ. |
| 11–14 | `self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(..., 10, 0.03))` | Tham số Lucas-Kanade: `winSize=(15,15)` — kích thước cửa sổ tìm kiếm 15×15px. `maxLevel=2` — pyramid 3 tầng (0,1,2) để xử lý chuyển động lớn. `criteria` — điều kiện dừng: tối đa 10 lần lặp hoặc epsilon 0.03. |
| 15 | `h, w = first_frame.shape[:2]` | Lấy kích thước frame. |
| 16–18 | Tạo mask theo dõi | Mask là ma trận nhị phân kích thước (h,w). Cột 0–19 (trái 20px) và cột w-150 → w-1 (phải 150px) được set = 1. **Chỉ track feature ở 2 dải biên** — đây là vùng ít bị cầu thủ che khuất nhất, chủ yếu là khán đài/quảng cáo → đại diện cho chuyển động camera thuần túy. |
| 19–20 | `self.features = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask)` | Tham số Shi-Tomasi corner detector: `maxCorners=100` — tối đa 100 feature points. `qualityLevel=0.3` — ngưỡng chất lượng 30% so với corner tốt nhất. `minDistance=3` — khoảng cách tối thiểu giữa các corner (tránh clustering). `blockSize=7` — kích thước block tính gradient. `mask` — chỉ detect feature trong vùng biên. |
| 21–22 | `self.gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)` | Chuyển frame đầu tiên sang grayscale — lưu làm reference cho frame kế tiếp. Optical flow hoạt động trên ảnh grayscale (1 kênh), nhanh hơn BGR (3 kênh). |

### 3.2. `get_camera_movement(self, frames, read_from_stub=False, stub_path=None)` (dòng 24–71)

**Mục đích:** Tính toán độ dịch chuyển (dx, dy) của camera giữa các frame liên tiếp.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 26–31 | Kiểm tra stub | Nếu `read_from_stub=True` và file stub tồn tại, load và trả về ngay. Dùng try-except (FileNotFoundError) thay vì os.path.exists để tránh race condition. |
| 33 | `movement = [[0, 0]]` | Frame đầu tiên luôn có movement [0,0] (không có frame trước để so sánh). |
| 34 | `old_gray = self.gray_prev.copy()` | Lưu grayscale frame trước. |
| 35 | `old_pts = cv2.goodFeaturesToTrack(old_gray, **self.features)` | Detect corner features trong vùng biên (mask) của frame đầu tiên. Trả về mảng `(N, 1, 2)` — N điểm, mỗi điểm là (x, y). |
| 37 | `for frame in frames[1:]:` | Duyệt từ frame thứ 2 trở đi. |
| 38 | `gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | Chuyển frame hiện tại sang grayscale. |
| 39–43 | Xử lý mất feature | Nếu `old_pts` rỗng hoặc None (không detect được corner nào), ghi nhận [0,0] và re-detect features từ frame hiện tại. |
| 45–46 | `cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, **self.lk_params)` | Lucas-Kanade optical flow: tính vị trí mới của `old_pts` trong `gray`. Trả về: `new_pts` (vị trí mới), `st` (trạng thái — 1=thành công, 0=thất bại), `err` (lỗi). |
| 48–58 | Tìm dominant motion | Duyệt từng cặp (old_pt, new_pt). Chỉ xét điểm có `st[0]==1` (track thành công). Tính khoảng cách Euclidean. Chọn điểm có **khoảng cách lớn nhất** làm dominant motion (giả định: điểm xa nhất là do camera xoay, không phải nhiễu). `dx = ox - nx` (dấu ngược vì camera xoay phải → object dịch trái). |
| 59–63 | Áp dụng ngưỡng | Nếu max_distance > 5px → ghi nhận movement và re-detect features cho frame tiếp theo. Nếu ≤ 5px → coi như tĩnh [0,0] và **giữ nguyên old_pts** (tiếp tục track từ frame trước). |
| 65 | `old_gray = gray` | Cập nhật grayscale frame trước cho lần lặp sau. |
| 67–70 | Lưu stub | Nếu có stub_path, serialize movement list ra file pickle. |
| 71 | `return movement` | Trả về list `[[0,0], [dx1,dy1], [dx2,dy2], ...]` — độ dài = số frame. |

**Ví dụ trả về:**
```python
>>> est = CameraMovementEstimator(first_frame)
>>> mov = est.get_camera_movement(frames)
>>> len(mov)
450  # = số frame
>>> mov[:5]
[[0, 0], [-2.3, 1.1], [-1.8, 0.9], [0, 0], [3.2, -0.5]]
>>> mov[0]   # frame đầu luôn [0, 0]
[0, 0]
>>> mov[3]   # frame không có chuyển động
[0, 0]
>>> mov[4]   # camera xoay phải 3.2px, lên 0.5px
[3.2, -0.5]
```

### 3.3. `add_adjust_positions_to_tracks(self, tracks, movement)` (dòng 73–81)

**Mục đích:** Bù chuyển động camera vào tọa độ cầu thủ.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 74 | `for obj, obj_tracks in tracks.items():` | Duyệt players, referees, ball. |
| 75 | `for frame_num, track in enumerate(obj_tracks):` | Duyệt từng frame. |
| 76 | `dx, dy = movement[frame_num]` | Lấy độ dịch chuyển camera tại frame này. |
| 77–81 | Với mỗi track | Lấy `position` (foot point / center). Tạo `position_adjusted` = `(pos[0] - dx, pos[1] - dy)`. Nếu camera xoay phải 5px (dx=5), thì tọa độ cầu thủ bị dịch trái 5px → cộng 5px để đưa về đúng vị trí. |

**Ví dụ:**
```python
# Frame 10: camera xoay phải 3px (dx=3, dy=0)
# Cầu thủ A: position=(500, 300)
# Thực tế A không di chuyển, nhưng do camera xoay,
# A xuất hiện ở (497, 300) trong frame
# → position_adjusted = (500-3, 300-0) = (497, 300)
# → đây là tọa độ thực tế của A trên ảnh gốc
```

### 3.4. `draw_camera_movement(self, frames, movement)` (dòng 83–99)

**Mục đích:** Vẽ overlay chuyển động camera lên từng frame (dùng để debug).

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 85–86 | Copy frame | Tránh mutate frame gốc. |
| 87–90 | Tạo overlay mờ | Rect trắng (0,0)→(500,100) (góc trên trái) với alpha 0.2 — nền trắng mờ để text dễ đọc. `addWeighted(overlay, 0.2, frame, 0.8, 0)` → 20% overlay + 80% frame gốc. |
| 91–97 | Vẽ text | "Camera Movement" ở dòng 1, "X: -2.3  Y: 1.1" ở dòng 2. Font Hershey Simplex size 1, màu đen, thickness 3. |

## 4. Sơ đồ luồng xử lý

```
CameraMovementEstimator.__init__(first_frame)
  ├── Lưu first_frame grayscale (reference)
  ├── Tạo mask: cột 0-20 + cột w-150-w (vùng biên)
  └── Cấu hình LK params + Shi-Tomasi features

get_camera_movement(frames, read_from_stub, stub_path)
  │
  ├── [stub exists] → pickle.load → return movement
  │
  └── [no stub]
      ├── Frame 0: movement[0] = [0, 0]
      ├── goodFeaturesToTrack → old_pts
      │
      └── For frame 1..N:
          ├── calcOpticalFlowPyrLK(old → new)
          ├── Tìm điểm có distance lớn nhất → (dx, dy)
          ├── Nếu max_distance > 5px → append [dx, dy], re-detect
          └── Nếu ≤ 5px → append [0, 0], giữ nguyên features
              │
              └── pickle.dump(stub)

add_adjust_positions_to_tracks(tracks, movement)
  └── position_adjusted = position - (dx, dy)

draw_camera_movement(frames, movement)
  └── Overlay text frame-by-frame
```

## 5. Tổng kết

CameraMovementEstimator giải quyết vấn đề **nền di động** — một trong những thách thức lớn nhất trong phân tích video thể thao broadcast. Thuật toán dựa trên Lucas-Kanade Optical Flow với 2 cải tiến chính:

1. **Mask biên (dải trái 20px + phải 150px)**: Chỉ track feature ở vùng ít bị cầu thủ che khuất, đảm bảo motion đo được là của camera, không phải của đối tượng chuyển động.

2. **Dominant motion selection**: Chọn feature có khoảng cách lớn nhất thay vì trung bình — vì trên vùng biên, đa số feature là nền tĩnh, chỉ có số ít feature ở rìa khán đài phản ánh đúng chuyển động camera.
