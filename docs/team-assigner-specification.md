# Đặc tả TeamAssigner

## 1. Tổng quan

`asigners/team_assigner.py` (lưu ý: tên thư mục là `asigners`, typo từ `assigners`) gán mỗi cầu thủ vào một trong hai đội dựa trên **màu áo**. Sử dụng K-Means clustering (k=2) trên vùng áo (nửa trên bounding box) để phân cụm.

## 2. Import (dòng 1–2)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import numpy as np` | NumPy: reshape ảnh, xử lý mảng, thống kê corner pixels. |
| 2 | `from sklearn.cluster import KMeans` | K-Means clustering: phân cụm màu áo thành 2 nhóm (đội 1 và đội 2). |

## 3. Class TeamAssigner

### 3.1. `__init__(self)` (dòng 6–8)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 7 | `self.team_colors = {}` | Dictionary `{team_id: BGR_color}` — lưu màu đại diện của mỗi đội. Team 1 và 2 (không phải 0 và 1). |
| 8 | `self.player_team_dict = {}` | Dictionary `{player_id: team_id}` — lưu team đã gán cho từng player. Tránh tính lại mỗi frame. |

### 3.2. `get_clustering_model(self, image)` (dòng 10–15)

**Mục đích:** Tạo K-Means model (k=2) từ ảnh crop vùng áo.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 11 | `img_2d = image.reshape(-1, 3)` | Chuyển ảnh (H, W, 3) thành (H*W, 3) — mỗi pixel là 1 sample với 3 feature (B, G, R). |
| 12–13 | `KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)` | `n_clusters=2`: phân thành 2 cụm — màu áo và màu nền. `k-means++`: khởi tạo thông minh. `n_init=10`: chạy 10 lần, chọn kết quả tốt nhất. `random_state=42`: tái lập kết quả. |
| 14 | `km.fit(img_2d)` | Fit K-Means trên tất cả pixel. |
| 15 | `return km` | Trả về fitted model. |

### 3.3. `get_player_color(self, frame, bbox)` (dòng 17–28)

**Mục đích:** Trích xuất màu áo của cầu thủ từ frame + bbox.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 18 | `x1, y1, x2, y2 = map(int, bbox)` | Chuyển tọa độ bbox về int để slice. |
| 19 | `crop = frame[y1:y2, x1:x2]` | Crop ảnh theo bbox. |
| 20 | `top = crop[:crop.shape[0]//2, :]` | Lấy nửa trên của crop — đây là vùng áo (giả định: nửa dưới là chân/quần). |
| 21 | `km = self.get_clustering_model(top)` | K-Means trên vùng áo → 2 cụm: áo vs nền (da, quần, sân phía sau). |
| 23 | `labels = km.labels_.reshape(top.shape[:2])` | Reshape label (H*W) về dạng 2D (H, W) — mỗi pixel biết thuộc cụm 0 hay 1. |
| 24–26 | Xác định background cluster | Lấy labels của 4 góc ảnh crop. Góc thường là background (sân, khán đài). `max(set(corners), key=corners.count)`: cụm xuất hiện nhiều nhất ở 4 góc = background. |
| 27 | `pl = 1 - bg` | Player cluster = cụm còn lại (1 - bg). Vì K-Means chỉ có 2 cụm: bg ∈ {0,1} → pl = 1 - bg. |
| 28 | `return km.cluster_centers_[pl]` | Trả về BGR color của cụm player — center của cluster (giá trị B, G, R trung bình). |

**Ví dụ trả về:**
```python
>>> assigner.get_player_color(frame, [100, 200, 150, 400])
array([220., 50., 30.])    # BGR: blue=220 → đội màu xanh dương
```

### 3.4. `assign_team_color(self, frame, player_detections)` (dòng 30–40)

**Mục đích:** Gán màu đội cho tất cả player trong frame, phân cụm thành 2 đội.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 31–33 | Thu thập màu | Duyệt tất cả player trong frame hiện tại, gọi `get_player_color` cho mỗi player → list colors. |
| 35–37 | K-Means trên màu | `KMeans(n_clusters=2)` trên list màu BGR của tất cả player → 2 cụm tương ứng 2 đội. `init='k-means++'`, `n_init=10`. |
| 38 | `self.km = km` | Lưu model để dùng trong `get_player_team`. |
| 39–40 | `self.team_colors[1] = km.cluster_centers_[0]; self.team_colors[2] = km.cluster_centers_[1]` | Gán center đầu tiên → team 1, center thứ hai → team 2. Lưu ý: team_id bắt đầu từ 1, không phải 0. |

### 3.5. `get_player_team(self, frame, bbox, player_id)` (dòng 42–48)

**Mục đích:** Lấy team ID cho một player cụ thể.

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 43–44 | Cache check | Nếu player_id đã được gán team trước đó, trả về ngay. |
| 45 | `color = self.get_player_color(frame, bbox)` | Lấy màu áo player. |
| 46 | `team = self.km.predict(color.reshape(1,-1))[0] + 1` | `km.predict()` trả về cluster index (0 hoặc 1). `+1` để chuyển về team 1 hoặc 2. |
| 47 | `self.player_team_dict[player_id] = team` | Cache lại. |
| 48 | `return team` | Trả về 1 hoặc 2. |

**Ví dụ trả về:**
```python
>>> assigner.get_player_team(frame, [100, 200, 150, 400], 1)
1   # player ID=1 thuộc đội 1
>>> assigner.get_player_team(frame, [300, 250, 360, 500], 2)
2   # player ID=2 thuộc đội 2
>>> assigner.get_player_team(frame, [100, 200, 150, 400], 1)  # cached
1   # không chạy K-Means lại
```

## 4. Sơ đồ luồng xử lý

```
TeamAssigner.assign_team_color(frame, player_detections)
  │
  ├── For mỗi player trong frame hiện tại:
  │   └── get_player_color(frame, bbox)
  │       ├── Crop ảnh theo bbox
  │       ├── Lấy nửa trên (vùng áo)
  │       ├── KMeans(k=2) trên pixel vùng áo
  │       ├── Xác định bg = cụm ở 4 góc
  │       └── return km.cluster_centers_[1-bg]  (màu áo)
  │
  ├── KMeans(k=2) trên list màu BGR → 2 đội
  └── Lưu self.km + self.team_colors

TeamAssigner.get_player_team(frame, bbox, player_id)
  │
  ├── [player_id in cache] → return team
  │
  ├── get_player_color(frame, bbox) → BGR
  ├── km.predict(color) + 1 → team
  ├── cache player_team_dict[player_id] = team
  └── return team
```

## 5. Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|---------|-----------|
| **Áo có nhiều màu (sọc, logo)** | 4 góc crop dùng để xác định background. Giả định background chiếm 4 góc. |
| **Player chưa từng thấy** | `get_player_team` gọi K-Means predict (O(1)), không fit lại. Việc fit chỉ xảy ra một lần ở `assign_team_color`. |
| **Team ID thay đổi giữa các frame** | `player_team_dict` cache theo player_id — một khi đã gán, không thay đổi. |
| **Nửa dưới crop không phải áo** | `top = crop[:h//2]` — chỉ lấy nửa trên. Giả định camera quay từ xa, nửa dưới là quần + chân. |
| **K-Means random state** | `random_state=42` đảm bảo kết quả tái lập được. |
