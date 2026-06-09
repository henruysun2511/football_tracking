# Cơ chế Cache trong hệ thống

## Giới thiệu

Hệ thống phân tích bóng đá bằng AI xử lý khối lượng lớn dữ liệu video với các phép tính nặng như suy luận YOLO (deep learning), optical flow, và biến đổi phối cảnh. Để tối ưu hiệu năng và tránh tính toán lại những kết quả đã có, hệ thống triển khai cơ chế cache đa tầng, bao gồm cache trên đĩa (stub files) và cache trong bộ nhớ.

Cơ chế cache đóng vai trò then chốt trong việc rút ngắn thời gian xử lý từ vài phút xuống còn vài giây khi chạy lại pipeline, đồng thời cho phép tách biệt giữa pha tracking (tính toán nặng) và pha rendering (kết xuất hình ảnh).

---

## 1. Kiến trúc cache tổng thể

Hệ thống sử dụng mô hình cache hai tầng:

- **Tầng 1 — Cache trên đĩa (Persistent cache)**: Dữ liệu được tuần tự hóa thành file `.pkl` (Pickle) hoặc `.npy` (NumPy) và lưu trên ổ đĩa. Cache này tồn tại xuyên suốt các lần chạy khác nhau, cho phép tái sử dụng kết quả mà không cần tính toán lại.
- **Tầng 2 — Cache trong bộ nhớ (In-memory cache)**: Dữ liệu được lưu trong các dictionary của Python object trong suốt vòng đời của một lần chạy, giúp tránh tính toán trùng lặp trong cùng một phiên xử lý.

---

## 2. Cache tầng 1 — Stub files trên đĩa

### 2.1 Định dạng lưu trữ

Hệ thống sử dụng hai định dạng tuần tự hóa:

**Pickle** (`.pkl`) dành cho các cấu trúc dữ liệu phức tạp như dictionary, list, hoặc object Python. Pickle cho phép lưu trữ và khôi phục nguyên vẹn cấu trúc dữ liệu, phù hợp để lưu kết quả tracking với nhiều lớp lồng nhau.

**NumPy** (`.npy`) dành cho các mảng số học thuần túy, thường là mảng một chiều với kiểu dữ liệu nguyên thủy. NumPy có tốc độ đọc/ghi nhanh hơn Pickle và kích thước file nhỏ hơn.

Quá trình đọc và ghi được thực hiện như sau:

```python
# Ghi cache
with open(stub_path, 'wb') as f:
    pickle.dump(tracks, f)

# Đọc cache
with open(stub_path, 'rb') as f:
    tracks = pickle.load(f)

# Với NumPy array
np.save(f'{STUB_DIR}/team_ball_control.npy', team_ball_control)
team_ball_control = np.load(f'{STUB_DIR}/team_ball_control.npy')
```

### 2.2 Danh sách các stub file

Hệ thống duy trì các stub file sau trong thư mục `stubs/`:

| File | Định dạng | Dung lượng | Nội dung |
|------|-----------|-----------|----------|
| `track_stubs.pkl` | Pickle | ~50 KB | Kết quả YOLO thô: bounding box player/referee/ball theo từng frame |
| `tracks_full.pkl` | Pickle | ~500 KB | Dữ liệu tracking đã làm giàu: vị trí, đội, tốc độ, homography |
| `camera_movement_stub.pkl` | Pickle | ~4 KB | Độ dịch chuyển camera [dx, dy] theo từng frame |
| `cam_move.pkl` | Pickle | ~4 KB | Dữ liệu chuyển động camera (phiên bản đầy đủ) |
| `team_ball_control.npy` | NumPy | ~2 KB | Mảng kiểm soát bóng của từng đội theo frame |

Kích thước các file này rất nhỏ so với dữ liệu gốc, điều này đến từ việc chỉ lưu thông tin cốt lõi (bounding box, tọa độ) thay vì toàn bộ frame ảnh.

### 2.3 Cơ chế đọc/ghi có điều kiện

Cả hai module chính — `Tracker` và `CameraMovementEstimator` — đều triển khai cùng một mẫu thiết kế: đọc từ stub nếu file tồn tại, nếu không thì tính toán và ghi lại.

Tracker sử dụng phương thức kiểm tra bằng `os.path.exists()`:

```python
def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)

    # ... thực hiện YOLO inference + ByteTrack ...

    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
    return tracks
```

CameraMovementEstimator sử dụng phương thức try/except để xử lý trường hợp file không tồn tại:

```python
def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
    if read_from_stub and stub_path:
        try:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

    # ... thực hiện Lucas-Kanade Optical Flow ...

    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(movement, f)
    return movement
```

Cả hai cách tiếp cận đều đảm bảo ba điều kiện để đọc cache: (1) tham số `read_from_stub=True`, (2) đường dẫn `stub_path` được cung cấp, và (3) file tồn tại trên đĩa. Nếu bất kỳ điều kiện nào không thỏa mãn, hệ thống sẽ tính toán lại từ đầu.

### 2.4 Ứng dụng trong pipeline CLI (main.py)

Tệp `main.py` triển khai cơ chế cache hai mức và cho phép người dùng lựa chọn chạy từng pha riêng biệt thông qua tham số dòng lệnh:

- `python main.py --mode all`: Chạy toàn bộ pipeline (tracking → render).
- `python main.py --mode tracking`: Chỉ chạy pha 1 (tracking), lưu stub, bỏ qua render.
- `python main.py --mode render`: Chỉ chạy pha 2 (render), đọc stub từ đĩa.

Ở pha 1, `main.py` gọi `tracker.get_object_tracks()` và `cam_est.get_camera_movement()` với tham số `read_from_stub=True`. Nếu stub đã tồn tại, hệ thống bỏ qua YOLO và optical flow — tiết kiệm hàng phút xử lý.

Sau khi hoàn thành pha 1, `main.py` lưu ba file stub tổng hợp:

```python
os.makedirs(STUB_DIR, exist_ok=True)
with open(f'{STUB_DIR}/tracks_full.pkl', 'wb') as f:
    pickle.dump(tracks, f)
with open(f'{STUB_DIR}/cam_move.pkl', 'wb') as f:
    pickle.dump(cam_move, f)
np.save(f'{STUB_DIR}/team_ball_control.npy', team_ball_control)
```

Ở pha 2, `main.py` chỉ đơn thuần đọc ba file này và tiến hành rendering mà không cần bất kỳ tính toán tracking nào. Sự tách biệt này cho phép người dùng thử nghiệm nhiều tùy chọn rendering khác nhau (bật/tắt minimap, heatmap, keypoint) mà không phải đợi tracking lại từ đầu.

---

## 3. Cache tầng 2 — In-memory Cache

### 3.1 Cache ma trận Homography (M_cache)

`ViewTransformer` là module chịu trách nhiệm biến đổi phối cảnh, đưa tọa độ cầu thủ từ không gian ảnh camera sang tọa độ mặt sân thực tế. Ma trận homography H (3×3) được tính từ keypoint sân và là kết quả của phép tính `cv2.findHomography()`.

Với mỗi frame, ma trận H là chung cho tất cả đối tượng (cầu thủ, trọng tài, bóng). Nếu không có cache, keypoint sân sẽ bị phát hiện lặp lại cho mỗi đối tượng trong cùng một frame, gây lãng phí tài nguyên. `M_cache` giải quyết vấn đề này bằng cách lưu ma trận H theo số thứ tự frame:

```python
class ViewTransformer:
    def __init__(self, kp_detector, smooth_alpha=0.05):
        self.M_cache = {}              # frame_num → ma trận homography
        self.last_good_M = None        # fallback nếu phát hiện keypoint thất bại

    def _get_homography_for_frame(self, video_frames, frame_num):
        if frame_num in self.M_cache:
            return self.M_cache[frame_num]

        kps = self.kp_detector.detect_smoothed(video_frames[frame_num])
        M = self.kp_detector.get_homography(kps) if kps is not None else None
        if M is not None:
            self.last_good_M = M
        else:
            M = self.last_good_M       # fallback về frame trước
        self.M_cache[frame_num] = M
        return M
```

### 3.2 Cache làm mịn vị trí (_pos_smooth)

Vị trí cầu thủ sau khi chiếu qua homography thường bị nhiễu do sai số của keypoint detector. `ViewTransformer` áp dụng Exponential Moving Average (EMA) để làm mịn, với cache lưu giá trị frame trước cho từng đối tượng:

```python
class ViewTransformer:
    def __init__(self):
        self._pos_smooth = {}  # (object_type, track_id) → [x, y]

    def _smooth(self, key, raw):
        if not self._is_valid(raw):
            return self._pos_smooth.get(key, list(raw))

        if key not in self._pos_smooth:
            self._pos_smooth[key] = list(raw)
            return list(raw)

        prev = self._pos_smooth[key]
        a = self.smooth_alpha  # 0.05
        smoothed = (a * np.array(raw) + (1 - a) * np.array(prev)).tolist()
        self._pos_smooth[key] = smoothed
        return smoothed
```

Hệ số alpha = 0.05 cho phép giữ lại 95% giá trị frame trước, tạo hiệu ứng làm mịn mạnh, phù hợp để loại bỏ nhiễu homography nhảy giật cục giữa các frame.

### 3.3 Fallback Homography

Khi keypoint detector không phát hiện đủ keypoint để tính ma trận H ở một frame cụ thể (do camera mờ, góc quay bất lợi, hoặc thiếu sáng), `last_good_M` đóng vai trò là phương án dự phòng: sử dụng ma trận homography của frame gần nhất thành công. Điều này đảm bảo pipeline không bị gián đoạn và tọa độ vẫn được chiếu liên tục, dù có thể giảm độ chính xác tạm thời.

---

## 4. Cache trong giao diện Gradio

Giao diện Gradio bổ sung một lớp cache đặc thù dành cho ứng dụng web, sử dụng MD5 hash để định danh video đầu vào.

### 4.1 MD5 Hash Key

Khi người dùng upload video, hệ thống đọc 1 MB dữ liệu đầu tiên của file và tính MD5 hash. 12 ký tự đầu của hash được dùng làm key định danh:

```python
with open(video_path, 'rb') as f:
    file_hash = hashlib.md5(f.read(1024*1024)).hexdigest()[:12]
stub_key = file_hash
```

Việc chỉ đọc 1 MB đầu (thay vì toàn bộ file) giúp tăng tốc độ tính hash, đồng thời vẫn đảm bảo tính định danh vì phần header và frame đầu của video đủ để phân biệt các file khác nhau.

### 4.2 Cấu trúc thư mục cache

Toàn bộ cache của Gradio được lưu trong thư mục `cache_gradio/` với quy ước đặt tên:

| File | Mô tả |
|------|-------|
| `cache_gradio/<hash>_track.pkl` | Raw tracking stubs |
| `cache_gradio/<hash>_cam.pkl` | Camera movement stubs |
| `cache_gradio/<hash>_output.mp4` | Video kết quả đã render |
| `cache_gradio/<hash>_hm_both.png` | Heatmap tổng hợp |
| `cache_gradio/<hash>_hm_t1.png` | Heatmap đội 1 |
| `cache_gradio/<hash>_hm_t2.png` | Heatmap đội 2 |

### 4.3 Cơ chế tái sử dụng

Khi người dùng upload lại video đã xử lý trước đó, MD5 hash trùng khớp và các stub tracking/camera được tái sử dụng ngay lập tức. Video output và heatmap cũng được dùng lại nếu đã tồn tại, giúp giảm thời gian phản hồi từ vài phút xuống còn vài giây.

Cache Gradio không có cơ chế tự động xóa — các file cũ tồn tại vĩnh viễn cho đến khi người dùng xóa thủ công.

---

## 5. Cache trong giao diện Streamlit

Giao diện Streamlit sử dụng cấu trúc thư mục phân theo tên video, lưu trong `cache/<tên_video>/`:

```
cache/<video_name>/
    source.mp4              -- Bản sao video gốc
    track_stubs.pkl         -- Raw tracking
    camera_movement_stub.pkl-- Chuyển động camera
    tracks_full.pkl         -- Tracking đầy đủ
    cam_move.pkl            -- Chuyển động camera (đầy đủ)
    team_ball_control.npy   -- Kiểm soát bóng
    <video_name>_kpTrue_mmTrue.avi  -- Video kết quả
```

Streamlit cung cấp giao diện cho phép người dùng chọn giữa "dùng cache" hoặc "tính toán lại" thông qua checkbox `use_cache`. Người dùng cũng có thể chọn từ danh sách các video đã xử lý trước đó (có tiền tố `cache:`) để tải kết quả ngay lập tức.

---

## 6. Cache trong các script Analysis

Các script phân tích ở thư mục `analysis/` là những consumer chỉ-đọc (read-only) của stub files. Chúng giả định stub đã tồn tại và xử lý một cách duyên dáng khi file không tìm thấy:

```python
STUB_PATH = 'stubs/tracks_full.pkl'
if not os.path.exists(STUB_PATH):
    print(f"File {STUB_PATH} not found. Run 'python main.py --mode tracking' first.")
    exit(0)
```

Danh sách các script và stub file tương ứng:

| Script | Stub file sử dụng | Mục đích |
|--------|-------------------|----------|
| `visualize_ball_interpolation.py` | `stubs/track_stubs.pkl` | Trực quan nội suy vị trí bóng |
| `visualize_team_assignment.py` | `stubs/track_stubs.pkl` | Trực quan K-Means gán đội |
| `visualize_player_stats.py` | `stubs/tracks_full.pkl` | Thống kê tốc độ/quãng đường |
| `visualize_perspective.py` | — | Trực quan perspective transform |
| `visualize_optical_flow.py` | — | Trực quan optical flow |

---

## 7. Hiệu quả của cơ chế cache

Bảng dưới đây so sánh thời gian xử lý trước và sau khi áp dụng cache, đo trên video 250 frame (10 giây @ 25fps) với GPU NVIDIA T4:

| Tác vụ | Không cache | Với cache | Tỉ lệ tăng tốc |
|--------|------------|-----------|---------------|
| YOLO detection + ByteTrack | ~45 giây | ~0.1 giây | ~450 lần |
| Optical flow camera | ~15 giây | ~0.05 giây | ~300 lần |
| Toàn bộ pipeline (tracking → render) | ~120 giây | ~0.5 giây | ~240 lần |
| Homography mỗi frame (gọi lần đầu) | ~0.2 giây | ~0 giây | Tức thì |

---

## 8. Tổng kết

Cơ chế cache của hệ thống được thiết kế theo ba nguyên lý chính:

- **Tính điều kiện (Conditional)**: Cache chỉ được đọc khi người dùng chủ động yêu cầu thông qua tham số `read_from_stub=True`. Điều này cho phép linh hoạt giữa việc dùng lại kết quả cũ và tính toán lại khi cần.
- **Tính kế thừa (Hierarchical)**: Cache hai tầng — trên đĩa cho các lần chạy khác nhau, trong bộ nhớ cho các đối tượng khác nhau trong cùng một lần chạy.
- **Tính mô-đun (Modular)**: Mỗi module quản lý cache riêng của mình. Tracker cache raw detection, CameraMovementEstimator cache optical flow, ViewTransformer cache homography matrix. Sự phân tách này giúp dễ bảo trì và mở rộng.

Nhờ cơ chế cache, hệ thống có thể tái sử dụng kết quả tracking nhiều lần với chi phí đọc file không đáng kể, cho phép người dùng tập trung vào việc tinh chỉnh rendering và phân tích mà không phải chờ đợi tính toán lại từ đầu.
