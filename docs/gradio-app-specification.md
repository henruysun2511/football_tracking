# Đặc tả Ứng dụng Gradio — Football AI Analysis

## 1. Giới thiệu

Ứng dụng Gradio là giao diện người dùng trực quan cho hệ thống phân tích bóng đá bằng AI, cho phép người dùng tải lên video trận đấu, thực thi toàn bộ pipeline xử lý và xem kết quả trực tiếp trên trình duyệt. Ứng dụng được xây dựng trên nền tảng Gradio Blocks, chạy trên cổng 7862 với tùy chọn share link cho môi trường Colab và Kaggle.

## 2. Kiến trúc tổng quan

Ứng dụng Gradio đóng vai trò là lớp giao diện (presentation layer) nằm trên pipeline xử lý chính. Luồng xử lý được tổ chức như sau:

```
Video input → UI (Gradio) → process_video() → Pipeline modules → Output files → UI (Gradio)
```

Hàm `process_video()` (dòng 41–270) điều phối toàn bộ quy trình, gọi tuần tự các module:

1. `Tracker` — phát hiện + theo dõi đối tượng
2. `CameraMovementEstimator` — ước lượng chuyển động camera
3. `PitchKeypointDetector` — phát hiện keypoint sân
4. `ViewTransformer` — tính homography + chiếu tọa độ
5. `SpeedDistanceEstimator` — tính tốc độ và quãng đường
6. `TeamAssigner` — gán đội dựa trên màu áo
7. `PlayerBallAssigner` — gán bóng
8. `FormationAnalyzer` — phát hiện đội hình chiến thuật
9. `MinimapRenderer` — vẽ minimap
10. `HeatmapGenerator` — tạo heatmap

## 3. Đặc tả đầu vào (Input Specification)

### 3.1. Video đầu vào

| Thuộc tính | Giá trị |
|---|---|
| Định dạng | MP4 (H.264 recommended) |
| Độ phân giải tối đa | 1920×1080 |
| FPS | Bất kỳ (tự động đọc từ video) |
| Kích thước tối đa | Phụ thuộc vào RAM/VRAM |
| Nguồn | Upload trực tiếp qua giao diện Gradio |

Video được đọc bằng `cv2.VideoCapture` và chuyển thành danh sách frame (NumPy array). FPS được trích xuất tự động; nếu không đọc được, mặc định là 25 FPS.

### 3.2. Model weights

Hai file model weights được yêu cầu tại thư mục `models/`:

| File | Mô hình | Task | Input size |
|---|---|---|---|
| `models/player_detector.pt` | YOLOv8x | Object detection (ball, goalkeeper, player, referee) | 1280×1280 |
| `models/pitch_keypoint_detector.pt` | YOLOv8x-pose | Keypoint detection (32 pitch keypoints) | 640×640 |

### 3.3. Tham số tùy chọn (UI)

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `show_keypoints` | Boolean | True | Hiển thị 32 keypoint sân trên video |
| `show_minimap` | Boolean | True | Hiển thị minimap tactical overlay |
| `show_heatmap` | Boolean | False | Tạo ảnh nhiệt độ di chuyển |

### 3.4. Stub files (đọc tự động từ cache)

Hệ thống tự động kiểm tra và đọc các stub files trong thư mục `cache_gradio/` để tránh tính toán lại:

| File | Nội dung |
|---|---|
| `{hash}_track.pkl` | Kết quả tracking (bounding box, track ID) |
| `{hash}_cam.pkl` | Chuyển động camera (dx, dy per frame) |

Stub key được tạo từ MD5 hash của 1 MB đầu video, đảm bảo tính duy nhất cho từng video đầu vào.

## 4. Đặc tả đầu ra (Output Specification)

### 4.1. Video đã annotation

| Thuộc tính | Giá trị |
|---|---|
| Định dạng | MP4 (H.264) |
| Codec | FFmpeg (ưu tiên) → OpenCV MJPG (fallback) |
| Độ phân giải | Giống video đầu vào |
| FPS | Giống video đầu vào |
| Đường dẫn lưu | `cache_gradio/{hash}_output.mp4` |

Nội dung annotation trên video bao gồm:

1. **Ellipse + track ID** cho mỗi cầu thủ (màu theo đội)
2. **Ellipse màu vàng** cho trọng tài
3. **Triangle màu cyan** cho bóng
4. **Triangle màu xanh lá** cho cầu thủ đang có bóng
5. **Tốc độ (km/h) và quãng đường (m)** bên dưới chân cầu thủ
6. **Overlay chuyển động camera** (dx, dy) — góc trên trái
7. **Overlay tỉ lệ kiểm soát bóng** hai đội — góc trên phải
8. **Overlay đội hình chiến thuật** hai đội — góc dưới trái
9. **32 keypoint sân** (nếu bật `show_keypoints`)
10. **Minimap tactical** ở góc dưới phải (nếu bật `show_minimap`)

### 4.2. Heatmap

| Thuộc tính | Giá trị |
|---|---|
| Định dạng | PNG |
| Kích thước canvas | 630×420 px |
| Sigma Gaussian blur | 15 |
| Số lượng file | 3 (tổng hợp, đội 1, đội 2) |
| Đường dẫn lưu | `cache_gradio/{hash}_hm_*.png` |

Ba ảnh heatmap được tạo:

| File | Colormap | Mô tả |
|---|---|---|
| `{hash}_hm_both.png` | JET | Tổng hợp cả hai đội |
| `{hash}_hm_t1.png` | WINTER | Đội 1 |
| `{hash}_hm_t2.png` | AUTUMN | Đội 2 |

### 4.3. Thống kê trận đấu (Markdown)

Đầu ra dạng Markdown hiển thị trên UI với nội dung:

- **Team 1 Formation:** tên đội hình + độ tin cậy
- **Team 2 Formation:** tên đội hình + độ tin cậy
- **Team 1 Ball Control:** phần trăm kiểm soát bóng
- **Team 2 Ball Control:** phần trăm kiểm soát bóng

### 4.4. Bảng xếp hạng cầu thủ (DataFrame)

Hai bảng dữ liệu dạng dataframe, mỗi bảng 5 dòng:

**Top 5 Players by Speed** — sắp xếp giảm dần theo `Max Speed (km/h)`:

| Cột | Kiểu | Mô tả |
|---|---|---|
| Player ID | int | Mã định danh cầu thủ |
| Max Speed (km/h) | float | Tốc độ tối đa đạt được |
| Distance (m) | float | Tổng quãng đường di chuyển |
| Team | int | Đội (1 hoặc 2) |

**Top 5 Players by Distance** — sắp xếp giảm dần theo `Distance (m)`: cấu trúc tương tự.

### 4.5. Status Log

Hộp log dạng HTML với định dạng monospace, nền đen chữ xanh (`#1e1e1e` background, `#0f0` text), hiển thị tối đa 100 dòng gần nhất, tự động cuộn. Mỗi dòng có timestamp `[HH:MM:SS]`.

## 5. Cơ chế Cache

### 5.1. Disk Cache

Hệ thống sử dụng thư mục `cache_gradio/` để lưu kết quả trung gian và đầu ra. Cache key được tạo từ MD5 hash (12 ký tự đầu) của 1 MB đầu video, đảm bảo:

- Cùng một video upload lại → dùng cache → tiết kiệm thời gian
- Video khác nhau → hash khác nhau → không xung đột

Danh sách file cache:

| File | Kích thước ước tính | Vòng đời |
|---|---|---|
| `{hash}_track.pkl` | ~50 KB | Lưu vĩnh viễn đến khi xóa thủ công |
| `{hash}_cam.pkl` | ~4 KB | Lưu vĩnh viễn đến khi xóa thủ công |
| `{hash}_output.mp4` | ~10–100 MB (tùy video) | Lưu vĩnh viễn đến khi xóa thủ công |
| `{hash}_hm_*.png` | ~100–500 KB mỗi file | Lưu vĩnh viễn đến khi xóa thủ công |

### 5.2. In-Memory Cache

Hệ thống không sử dụng in-memory cache giữa các request (do Gradio không share memory giữa các session). Tuy nhiên, trong một phiên xử lý, các module như `PitchKeypointDetector` sử dụng `M_cache` (dict) để lưu homography matrix đã tính cho từng frame, tránh tính toán lại.

## 6. Luồng xử lý chi tiết

### 6.1. Khởi tạo

```
video_ui(file, show_kp, show_mm, show_hm)
  ├── Kiểm tra file upload
  ├── Gọi process_video()
  └── Trả về kết quả hoặc lỗi
```

### 6.2. Xử lý chính

```
process_video(video_path, show_kp, show_mm, show_hm)
  │
  ├── [0%] Đọc video → frames list
  ├── [0%] Tính MD5 hash → stub_key
  ├── [0%] Đọc FPS từ video
  ├── [0%] Kiểm tra GPU/CPU
  │
  ├── [2%] Tracker.get_object_tracks() → tracks dict
  ├── [2%] Nội suy ball + player positions
  ├── [2%] add_position_to_tracks()
  │
  ├── [15%] CameraMovementEstimator → cam_move list
  ├── [15%] add_adjust_positions_to_tracks()
  │
  ├── [25%] PitchKeypointDetector → homography
  ├── [25%] ViewTransformer → add_transformed_position
  ├── [25%] SpeedDistanceEstimator → speed + distance
  │
  ├── [35%] TeamAssigner → team + team_color per player
  ├── [35%] PlayerBallAssigner → team_ball_control array
  │
  ├── [45%] detect_team_formation() → formation per team
  │
  ├── [50–95%] Render loop (mỗi frame)
  │   ├── Vẽ ellipse, triangle, speed/distance
  │   ├── Vẽ overlay camera movement
  │   ├── Vẽ overlay ball control
  │   ├── Vẽ overlay formation
  │   ├── (nếu show_keypoints) detect + draw keypoints
  │   └── (nếu show_minimap) render + overlay minimap
  │
  ├── [95%] save_video() → output MP4
  ├── [95%] (nếu show_heatmap) HeatmapGenerator → 3 PNG
  ├── [95%] Tính player stats → top_speed, top_dist
  └── [100%] Return output paths + stats
```

### 6.3. Xử lý lỗi

Hàm `video_ui()` bao gồm try/except toàn cục. Khi có lỗi:

1. Log được ghi với nội dung `ERROR: {message}`
2. 500 ký tự cuối của stack trace được log để debug
3. `gr.Error` được raise để Gradio hiển thị thông báo lỗi trên UI

## 7. Giao diện người dùng

### 7.1. Bố cục

```
┌────────────────────────────────────────────────────┐
│              Football AI Analysis                   │
├──────────────┬─────────────────────────────────────┤
│  Upload      │  Result Video                       │
│  [video]     │  [video player]                     │
│              │                                     │
│  Overlay     │  Match Statistics (Markdown)        │
│  ☑ Keypoints │                                     │
│  ☑ Minimap   │  Top 5 Speed    │  Top 5 Distance   │
│  ☐ Heatmap   │  [DataFrame]    │  [DataFrame]      │
│              │                                     │
│  [ Analyze ] │  Heatmaps Gallery (ẩn nếu tắt)      │
│              │                                     │
│              │  Status Log                         │
│              │  [monospace console]                │
└──────────────┴─────────────────────────────────────┘
```

### 7.2. Component

| Component | Loại | Mô tả |
|---|---|---|
| video_input | `gr.Video` | Upload video file, format mp4 |
| show_kp | `gr.Checkbox` | Bật/tắt keypoint overlay |
| show_mm | `gr.Checkbox` | Bật/tắt minimap overlay |
| show_hm | `gr.Checkbox` | Bật/tắt heatmap generation |
| btn | `gr.Button` | Kích hoạt xử lý |
| video_output | `gr.Video` | Hiển thị video kết quả |
| stats_markdown | `gr.Markdown` | Thống kê trận đấu |
| top_speed_df | `gr.Dataframe` | Top 5 tốc độ |
| top_dist_df | `gr.Dataframe` | Top 5 quãng đường |
| heatmap_gallery | `gr.Gallery` | 3 ảnh heatmap (columns=3) |
| status_log | `gr.HTML` | Console log realtime |

## 8. Giới hạn và ràng buộc

| Yếu tố | Giới hạn | Giải pháp |
|---|---|---|
| GPU | Ưu tiên NVIDIA T4+ (Colab) | Tự động fallback CPU với cảnh báo |
| VRAM | Yêu cầu ≥ 1 GB cho batch detection | Tự động chọn CPU nếu VRAM < 1 GB |
| Video dài | 5 phút ~ 15 phút xử lý trên T4 | Cache stub để render lại nhanh |
| Định dạng video | MP4 H.264 được khuyến nghị | Fallback MJPG cho codec khác |
| Kích thước upload | Giới hạn bởi Gradio/Giải pháp hosting | Xử lý streaming không hỗ trợ |

## 9. Triển khai (Deployment)

| Môi trường | Cổng | Share link | Ghi chú |
|---|---|---|---|
| Local | 7862 | Không | `python app/gradio_app.py` |
| Google Colab | 7862 | Có (ngrok) | Tự động phát hiện |
| Kaggle | 7862 | Có (ngrok) | Tự động phát hiện qua biến môi trường `KAGGLE_KERNEL_RUN_TYPE` |

## 10. Ví dụ luồng dữ liệu (Data Flow Example)

```
Input:  sample.mp4 (1280×720, 30 FPS, 450 frames = 15 giây)
        show_keypoints=True, show_minimap=True, show_heatmap=False

Output: cache_gradio/a1b2c3d4e5f6_output.mp4 (15 giây, 1280×720, 30 FPS)
        cache_gradio/a1b2c3d4e5f6_track.pkl
        cache_gradio/a1b2c3d4e5f6_cam.pkl
        + UI: Match Statistics (formations, ball control %)
        + UI: Top 5 Speed, Top 5 Distance DataFrames
        + UI: Status log with timestamps
```

## 11. Kết luận

Ứng dụng Gradio cung cấp một giao diện thân thiện cho toàn bộ hệ thống phân tích bóng đá, cho phép người dùng không chuyên về kỹ thuật có thể sử dụng hệ thống mà không cần dòng lệnh. Kiến trúc cache thông minh giúp tối ưu thời gian xử lý cho các video đã được phân tích trước đó. Hệ thống hỗ trợ đa nền tảng triển khai (local, Colab, Kaggle) với cơ chế tự động phát hiện môi trường.
