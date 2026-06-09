# Mô tả Pipeline xử lý của hệ thống

## Giới thiệu

Hệ thống phân tích bóng đá bằng AI được tổ chức theo kiến trúc pipeline gồm bốn pha chính, bao phủ toàn bộ quy trình từ chuẩn bị dữ liệu đến trực quan hóa kết quả. Mỗi pha đảm nhiệm một nhóm chức năng riêng biệt và có sự phụ thuộc tuần tự rõ ràng. Phần dưới đây mô tả chi tiết từng giai đoạn trong pipeline.

---

## Pha 0 — Data Preparation (Chuẩn bị dữ liệu)

Pha này đóng vai trò nền tảng, đảm bảo chất lượng dữ liệu đầu vào và mô hình trước khi đưa vào inference. Toàn bộ pha 0 chỉ chạy một lần (offline) và không nằm trong pipeline xử lý video thời gian thực.

### Giai đoạn 1: Download dataset

Dữ liệu được tải từ Roboflow — một nền tảng quản lý dataset thị giác máy tính. Hệ thống sử dụng hai bộ dữ liệu riêng biệt:

- **football-players-detection-3zvbc** (version 2): gồm 4 lớp đối tượng gồm ball, goalkeeper, player và referee. Đây là dữ liệu dạng YOLO detection với bounding box chuẩn.
- **football-field-detection-f07vi** (version 14): gồm 1 lớp pitch với 32 keypoint sân, sử dụng format YOLO Pose để lưu tọa độ và trạng thái hiển thị (visible/occluded/invisible) của từng keypoint.

Script `player_dataset.py` và `pitch_keypoint_dataset.py` thực hiện tải dữ liệu tự động thông qua Roboflow API, yêu cầu biến môi trường `ROBOFLOW_API_KEY` được khai báo trong file `.env`.

### Giai đoạn 2: Cleaning dataset

Dữ liệu sau khi tải về tiềm ẩn nhiều lỗi như ảnh corrupt, label thiếu, bounding box ngoài biên hoặc class_id không hợp lệ. Module `player_cleaning.py` và `pitch_keypoint_cleaning.py` thực hiện quét và loại bỏ các mẫu lỗi này qua các bước:

1. **Scan**: duyệt từng ảnh trong tập train/valid/test, kiểm tra ảnh corrupt, label thiếu, label rỗng, bounding box ngoài khoảng [0,1] và class_id ngoài phạm vi cho phép.
2. **Clean**: copy các ảnh và label hợp lệ sang thư mục `cleaned` riêng.
3. **Fix**: clip tọa độ bounding box về khoảng [0,1] bằng numpy.clip(), xóa các dòng có kích thước ≤ 0.001.

Đối với dữ liệu keypoint sân, tiêu chí đặc thù là mỗi ảnh phải có tối thiểu 4 keypoint visible — tương ứng với yêu cầu tối thiểu để tính ma trận homography (4 cặp điểm tương ứng).

### Giai đoạn 3: Huấn luyện mô hình

Hai mô hình YOLOv8 được huấn luyện trên dữ liệu đã làm sạch:

- **YOLOv8x** cho phát hiện đối tượng: input 1280×1280, batch size 8, 100 epochs, early stopping patience 20. Mô hình được fine-tune từ pre-trained weights COCO.
- **YOLOv8x-pose** cho phát hiện keypoint sân: input 640×640, batch size 8, 100 epochs, patience 50. Augmentation được tinh chỉnh riêng — mosaic 1.0 tắt ở 10 epoch cuối, xoay ±10°, scale ±20%, flip ngang 50% và không flip dọc vì sân bóng không bao giờ lộn ngược trong thực tế.

Kết quả đầu ra là file weights `.pt` được lưu tại thư mục `models/`, sẵn sàng cho pha inference.

---

## Pha 1 — Tracking (Theo dõi đối tượng)

Pha này là trái tim của hệ thống, nhận đầu vào là video trận đấu và các model weights đã huấn luyện, thực hiện phát hiện, theo dõi và tính toán các chỉ số vận động cho từng đối tượng trong từng frame.

### Giai đoạn 1: Đọc video → frames

Video đầu vào (định dạng .mp4, codec H.264) được đọc frame-by-frame bằng OpenCV (`cv2.VideoCapture`). Hàm `read_video()` trả về danh sách các frame dưới dạng mảng NumPy. Hệ thống hỗ trợ độ phân giải tối đa 1920×1080 với FPS bất kỳ.

### Giai đoạn 2: Phát hiện + Tracking (YOLOv8 + ByteTrack)

Mỗi frame được đưa qua mô hình YOLOv8x đã fine-tune với batch size 20 và ngưỡng confidence 0.1. Kết quả detection bao gồm bounding box, class ID và confidence score cho bốn lớp: ball, goalkeeper, player và referee.

Các detection sau đó được đưa vào ByteTrack — một thuật toán tracking by detection do thư viện Supervision cung cấp. ByteTrack gán ID duy nhất cho mỗi đối tượng dựa trên IoU matching và kalman filter, với các tham số:

- `track_activation_threshold = 0.25`
- `lost_track_buffer = 30` frames
- `minimum_matching_threshold = 0.8`

Riêng lớp goalkeeper được gộp vào player để đơn giản hóa quá trình xử lý về sau.

### Giai đoạn 3: Nội suy vị trí cầu thủ và bóng

Do YOLO không phát hiện được đối tượng ở mọi frame (do che khuất, mờ, hoặc góc quay), vị trí bounding box bị thiếu được lấp đầy bằng nội suy tuyến tính (linear interpolation) sử dụng Pandas DataFrame.

Quá trình nội suy áp dụng riêng cho từng track_id: tọa độ (x1, y1, x2, y2) được nội suy theo phương pháp `interpolate(method='linear')` kết hợp với `bfill()` để xử lý các frame đầu hoặc cuối còn thiếu.

### Giai đoạn 4: Ước lượng chuyển động camera (Optical Flow)

Camera trong các trận đấu bóng đá broadcast có sự dịch chuyển (pan/tilt) theo diễn biến trên sân. Chuyển động này cần được ước lượng và bù trừ để tọa độ cầu thủ phản ánh đúng vị trí thực tế trên sân.

Thuật toán sử dụng là **Lucas-Kanade Optical Flow** (`cv2.calcOpticalFlowPyrLK`) với feature detection từ `cv2.goodFeaturesToTrack`. Feature points chỉ được track ở dải biên trái (20px) và biên phải (150px) của frame — đây là vùng ít bị ảnh hưởng bởi chuyển động của cầu thủ.

Với mỗi frame, độ dịch chuyển (dx, dy) lớn nhất được ghi nhận. Nếu giá trị ≤ 5px, frame được coi là không có chuyển động camera.

### Giai đoạn 5: Phát hiện keypoint sân → Homography

Mô hình YOLOv8x-pose phát hiện 32 keypoint trên sân bóng, bao gồm góc sân, góc vòng cấm, chấm phạt đền và vòng tròn trung tâm. Ngưỡng confidence được đặt ở 0.3 để cân bằng giữa độ chính xác và tỉ lệ phát hiện.

Keypoint đầu ra được làm mịn bằng Exponential Moving Average (EMA) với hệ số alpha = 0.6, giúp giảm nhiễu giật cục giữa các frame liên tiếp.

### Giai đoạn 6: Chiếu tọa độ lên mặt sân thực (Perspective Transform)

Ma trận homography H (3×3) được tính bằng `cv2.findHomography(src, dst, RANSAC, 5.0)`, trong đó:

- `src`: tọa độ 32 keypoint phát hiện từ ảnh camera
- `dst`: tọa độ chuẩn của các keypoint trên mặt sân thực (được định nghĩa trong `SoccerPitchConfig` với kích thước sân 12000×7000 mm)

Với mỗi đối tượng (player, referee, ball), tọa độ sau khi bù camera được chiếu qua phép `cv2.perspectiveTransform()` để thu được tọa độ thực tế trên sân (đơn vị cm). Kết quả tiếp tục được làm mịn bằng EMA (alpha = 0.15) và loại bỏ outlier — vị trí ngoài biên sân ±10% hoặc khoảng cách nhảy > 600 cm/frame.

### Giai đoạn 7: Tính tốc độ và quãng đường

Dựa trên tọa độ thực tế đã được chiếu, module SpeedDistanceEstimator tính toán các chỉ số vận động cho từng cầu thủ:

- **Tốc độ**: tính trên cửa sổ 5 frame (WINDOW=5). Khoảng cách Euclidean giữa hai điểm đầu và cuối được chia cho thời gian trôi qua để ra tốc độ (m/s), sau đó đổi sang km/h. Kết quả được chặn ở ngưỡng sinh lý 38 km/h để loại nhiễu.
- **Quãng đường**: tích lũy khoảng cách di chuyển giữa các frame liên tiếp, lưu vào key `distance` của từng track_id.

Nếu khoảng cách trong cửa sổ vượt quá 300 cm (thường do homography bị nhiễu), toàn bộ cửa sổ bị loại bỏ.

### Giai đoạn 8: Phân cụm màu áo → gán đội

Module TeamAssigner sử dụng thuật toán K-Means clustering từ thư viện scikit-learn để phân loại cầu thủ vào hai đội dựa trên màu áo:

1. Crop vùng nửa trên của bounding box cầu thủ (vùng áo).
2. Áp dụng K-Means với k=2 để tách màu áo khỏi nền (background).
3. Áp dụng K-Means lần hai với k=2 trên toàn bộ màu áo của tất cả cầu thủ để gán vào đội 1 hoặc đội 2.
4. Kết quả được cache vào `player_team_dict` để tránh tính toán lại.

### Giai đoạn 9: Gán bóng → tính kiểm soát bóng

PlayerBallAssigner xác định cầu thủ đang kiểm soát bóng bằng cách tính khoảng cách từ tâm bóng đến vị trí chân của từng cầu thủ. Cầu thủ có khoảng cách nhỏ nhất và ≤ 70px được gán là người đang có bóng.

Dựa trên thông tin này, mảng `team_ball_control` được xây dựng — mỗi phần tử là ID đội (1 hoặc 2) đang kiểm soát bóng tại frame tương ứng. Dữ liệu sau pha 1 được lưu thành các file stub (pickle/npy) để phục vụ rendering và phân tích sau này.

---

## Pha 2 — Rendering (Kết xuất hình ảnh)

Pha này nhận đầu vào là các stub đã lưu ở pha 1 và thực hiện vẽ các lớp annotation lên video gốc, tạo minimap và heatmap.

### Giai đoạn 10: Phát hiện đội hình chiến thuật

FormationAnalyzer xác định đội hình của mỗi đội (ví dụ 4-3-3, 4-4-2) dựa trên tọa độ thực tế của 10 cầu thủ (loại thủ môn). Các bước thực hiện:

1. Lấy mẫu mỗi 30 frame, chuẩn hóa hướng tấn công (defenders bên trái).
2. Sắp xếp cầu thủ theo trục x từ hậu vệ đến tiền đạo.
3. Thử K-Means với k=3 (3 tuyến) và k=4 (4 tuyến), chọn kết quả phù hợp nhất.
4. Bỏ phiếu trên toàn bộ frame — đội hình có số phiếu cao nhất là kết quả cuối cùng.

### Giai đoạn 11: Vẽ annotation lên từng frame

Với mỗi frame, hệ thống vẽ các thành phần sau:

- Ellipse và track ID cho mỗi cầu thủ (màu theo đội).
- Ellipse màu vàng cho trọng tài, triangle màu cyan cho bóng.
- Triangle màu xanh lá cho cầu thủ đang có bóng.
- Giá trị tốc độ (km/h) và quãng đường (m) bên dưới chân mỗi cầu thủ.
- Overlay chuyển động camera (dx, dy) ở góc trên trái.
- Tỉ lệ kiểm soát bóng của hai đội ở góc trên trái.
- Tên đội hình chiến thuật ở góc dưới trái.
- Keypoint sân (tùy chọn).

### Giai đoạn 12: Vẽ minimap và heatmap

**Minimap**: bản đồ tactical 2D (350×230 px) hiển thị sân từ góc nhìn từ trên xuống. Cầu thủ được biểu diễn bằng các vòng tròn màu theo đội, trọng tài màu vàng, bóng màu xanh cyan. Minimap được overlay lên video gốc ở góc dưới phải với độ trong suốt nền 30%.

**Heatmap**: ảnh nhiệt (630×420 px) thể hiện mật độ di chuyển của cầu thủ, sử dụng JET colormap cho tổng hợp, WINTER cho đội 1 và AUTUMN cho đội 2.

### Giai đoạn 13: Xuất video và ảnh heatmap

Video đầu ra được ghi bằng FFmpeg (codec H.264, định dạng .mp4) — ưu tiên hàng đầu. Nếu FFmpeg không khả dụng, hệ thống fallback sang OpenCV với codec MJPG (.avi). Ba ảnh heatmap được xuất dưới dạng PNG.

---

## Pha 3 — Analysis (Phân tích và trực quan hóa)

Pha này cung cấp các script phân tích và trực quan hóa dữ liệu phục vụ nghiên cứu, debug và báo cáo. Không nằm trong pipeline chính mà chạy độc lập sau khi đã có dữ liệu đầu ra.

### Giai đoạn 14: EDA dataset và kiểm tra chất lượng

Các script phân tích thăm dò dữ liệu (EDA) cung cấp cái nhìn tổng quan về dataset:

- `eda_dataset.py`: thống kê số lượng ảnh mỗi split, phân bố class, kích thước bounding box, tỉ lệ small objects.
- `class_imbalance_analysis.py`: phân tích mất cân bằng dữ liệu giữa các lớp (ball ~3%, player ~75%), box plot và scatter plot kích thước bbox.
- `check_dataset_quality.py`: kiểm tra 6 loại lỗi trên dataset (corrupt, missing label, empty label, malformed line, invalid class, bbox out of bounds).
- `verify_data_split.py`: xác minh phân chia train/val/test có stratified và đồng nhất không.

### Giai đoạn 15: Trực quan hóa pipeline component

Loạt script minh họa trực quan từng bước trong pipeline:

- `visualize_perspective.py`: vẽ frame gốc với keypoint và bird's eye view sau perspective transform.
- `visualize_optical_flow.py`: vẽ feature points, motion vectors và histogram độ lớn chuyển động.
- `visualize_ball_interpolation.py`: so sánh vị trí bóng trước và sau nội suy.
- `visualize_team_assignment.py`: vẽ crop vùng áo, mask K-Means và màu sắc trích xuất.
- `visualize_label_format.py`, `visualize_preprocessing.py`, `visualize_mosaic_augmentation.py`: giải thích format YOLO, augmentation và mosaic.

### Giai đoạn 16: Thống kê cầu thủ và biểu đồ huấn luyện

- `visualize_player_stats.py`: vẽ biểu đồ tốc độ tối đa, tốc độ trung bình và quãng đường di chuyển của 14 cầu thủ đầu tiên.
- `visualize_training_results.py`: vẽ loss curves, mAP curves, precision/recall và mAP@0.5 per class từ file results.csv của quá trình huấn luyện.
- `capture_result_frame.py`: chụp 6 frame cách đều từ video kết quả, ghép vào lưới 2×3 để tạo ảnh showcase.

---

## Tổng kết

Pipeline bốn pha của hệ thống đảm bảo tính mô-đun, tái sử dụng và dễ mở rộng:

- **Pha 0** xây dựng nền tảng dữ liệu và mô hình chất lượng cao.
- **Pha 1** thực hiện toàn bộ quá trình tracking và tính toán chỉ số vận động.
- **Pha 2** kết xuất trực quan kết quả dưới nhiều dạng thức khác nhau.
- **Pha 3** cung cấp công cụ phân tích và đánh giá cho nhà phát triển và nhà nghiên cứu.

Sự tách biệt giữa tracking (pha 1) và rendering (pha 2) thông qua cơ chế stub cho phép người dùng chỉ chạy lại rendering mà không cần track lại từ đầu, tiết kiệm thời gian và tài nguyên tính toán đáng kể.
