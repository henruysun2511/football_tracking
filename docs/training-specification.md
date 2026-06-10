# Đặc tả Quy trình Huấn luyện Mô hình

## 1. Tổng quan

Hệ thống sử dụng hai mô hình YOLOv8 được huấn luyện riêng biệt trên hai bộ dữ liệu khác nhau. Quy trình huấn luyện bao gồm bốn script Python nằm trong thư mục `trainings/`, chia làm hai nhóm:

| Script | Mô hình | Task | Dataset |
|---|---|---|---|
| `football_training.py` | YOLOv8x | Object detection (4 classes) | football-players-detection-3zvbc v2 |
| `pitch_keypoint_training.py` | YOLOv8x-pose | Keypoint detection (32 KPs) | football-field-detection-f07vi v14 |
| `player_dataset.py` | — | Download dataset | football-players-detection-3zvbc v2 |
| `pitch_keypoint_dataset.py` | — | Download dataset | football-field-detection-f07vi v14 |

Cả hai script huấn luyện đều hỗ trợ hai chế độ:
- **Google Colab** (phát hiện qua đường dẫn `/content/drive`): lưu kết quả lên Google Drive
- **Local / Kaggle**: lưu kết quả vào thư mục `models/` trong dự án

---

## 2. Script Download Dataset

### 2.1. `player_dataset.py`

**Mục đích:** Tải bộ dữ liệu phát hiện cầu thủ từ Roboflow về thư mục làm việc.

**Luồng xử lý:**

```mermaid
flowchart LR
    A[.env: ROBOFLOW_API_KEY] --> B[load_dotenv]
    B --> C[Roboflow API]
    C --> D[project.version(2).download]
    D --> E[Dataset on disk]
```

**Giải thích từng dòng:**

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1–3 | `import os; from dotenv import load_dotenv; from roboflow import Roboflow` | Import thư viện: `os` cho biến môi trường, `load_dotenv` đọc file `.env`, `Roboflow` là SDK của Roboflow. |
| 7 | `load_dotenv()` | Load file `.env` từ thư mục hiện tại (chứa `ROBOFLOW_API_KEY=...`). |
| 8 | `api_key = os.getenv("ROBOFLOW_API_KEY")` | Lấy API key từ biến môi trường. Trả về `None` nếu không tìm thấy. |
| 10 | `rf = Roboflow(api_key=api_key)` | Khởi tạo kết nối Roboflow với API key. |
| 11–13 | `project = rf.workspace(...).project(...)` | Truy cập workspace `roboflow-jvuqo` và project `football-players-detection-3zvbc`. |
| 14 | `dataset = project.version(2).download("yolov8")` | Tải version 2 của dataset, format YOLOv8. Hàm `download()` trả về object `Dataset` với thuộc tính `.location` là đường dẫn thư mục chứa dữ liệu. |
| 15 | `print(f"Dataset at: {dataset.location}")` | In đường dẫn nơi dataset được tải về. |

### 2.2. `pitch_keypoint_dataset.py`

Giống `player_dataset.py` nhưng sử dụng project `football-field-detection-f07vi` version 14. Cấu trúc tương tự, chỉ khác tên project và workspace.

---

## 3. Script Huấn luyện Player Detection

### 3.1. Tổng quan

**File:** `trainings/football_training.py` (56 dòng)

**Mô hình:** YOLOv8x (phiên bản largest của YOLOv8 detection)

**Đầu vào:**
- File `.env` chứa `ROBOFLOW_API_KEY`
- Dataset: `datasets/football-players-detection-cleaned/` (nếu có sẵn) hoặc tải từ Roboflow
- Pre-trained weights: `yolov8x.pt` (COCO)

**Đầu ra:**
- Weights: `models/player_detector/player_detector/weights/best.pt`
- (Trên Colab) `/content/drive/MyDrive/football_models/player_detector/weights/best.pt`
- File logs, curves, confusion matrix trong thư mục project

### 3.2. Phân tích chi tiết

#### 3.2.1. Import và cấu hình (dòng 1–5)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 1 | `import os` | Thư viện hệ điều hành: kiểm tra đường dẫn, tạo thư mục, biến môi trường. |
| 2 | `from pathlib import Path` | Path: xử lý đường dẫn Pythonic, hỗ trợ `Path / "subdir"` thay vì `os.path.join()`. |
| 3 | `from dotenv import load_dotenv` | Đọc file `.env` chứa API key (không commit lên Git). |
| 4 | `from roboflow import Roboflow` | SDK Python của Roboflow — tải dataset tự động. |
| 5 | `from ultralytics import YOLO` | Thư viện Ultralytics YOLOv8 — API huấn luyện, validation, inference. |

#### 3.2.2. Xác định dataset (dòng 12–24)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 13 | `dataset_dir = "datasets/football-players-detection-cleaned"` | Đường dẫn tương đối đến dataset đã làm sạch (output từ `cleanings/player_cleaning.py`). |
| 14 | `yaml_path = Path(os.getcwd()) / dataset_dir / "data.yaml"` | Tạo đường dẫn tuyệt đối đến file `data.yaml` bằng `os.getcwd()` + `Path`. |
| 16 | `if not yaml_path.exists():` | Kiểm tra dataset local. Nếu không có → tải từ Roboflow. Nếu có → dùng luôn. |
| 17–21 | Nhánh tải từ Roboflow | `rf = Roboflow(api_key=api_key)` → `project.version(2).download("yolov8")` → `dataset.location` chứa đường dẫn download. `data_yaml_location = f"{dataset.location}/data.yaml"`. |
| 22–24 | Nhánh dùng local | `print("Dataset đã có sẵn...")` → `data_yaml_location = str(yaml_path)`. |

#### 3.2.3. Khởi tạo mô hình (dòng 27)

```python
model = YOLO("yolov8x.pt")
```

Load pre-trained weights YOLOv8x từ COCO dataset. Ultralytics tự động tải file từ GitHub (`https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt`) nếu chưa có trong bộ nhớ cache (`~/.cache/ultralytics/`).

**Kiến trúc YOLOv8x:**
- Backbone: CSPDarknet với multi-scale feature pyramid
- Neck: PAN-FPN (Path Aggregation Network)
- Head: Decoupled detection head (class + bbox + objectness)
- Tham số: ~68.2M
- GFLOPS: ~257.4

#### 3.2.4. Cấu hình đầu ra (dòng 29–35)

| Dòng | Lệnh | Giải thích |
|------|------|------------|
| 29 | `output_project = "models/player_detector"` | Mặc định lưu local. |
| 30 | `if os.path.exists("/content/drive"):` | Phát hiện môi trường Colab (Google Drive được mount tại `/content/drive`). |
| 31 | `output_project = "/content/drive/MyDrive/football_models"` | Trên Colab: lưu vào Google Drive để không mất weights khi runtime ngắt. |
| 32, 35 | `os.makedirs(output_project, exist_ok=True)` | Tạo thư mục nếu chưa tồn tại. |

#### 3.2.5. Huấn luyện (dòng 37–49)

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `data` | `data_yaml_location` | Đường dẫn đến file `data.yaml` chứa cấu hình dataset (số lớp, đường dẫn train/val/test). |
| `epochs` | 100 | Số epoch huấn luyện tối đa. |
| `imgsz` | 1280 | Kích thước ảnh đầu vào. 1280 pixels cho phép phát hiện cầu thủ nhỏ ở xa. Yêu cầu VRAM cao (~8GB+). |
| `batch` | 8 | Số ảnh mỗi batch. Batch nhỏ (8) để vừa VRAM T4 (16GB). |
| `device` | 0 | GPU đầu tiên (CUDA). Trên máy không GPU, cần sửa thành `"cpu"`. |
| `project` | `output_project` | Thư mục gốc lưu kết quả. |
| `name` | `"player_detector"` | Tên experiment. Ultralytics tạo thư mục con `{project}/{name}/`. |
| `patience` | 20 | Early stopping: nếu mAP@0.5:0.95 không cải thiện sau 20 epoch, dừng huấn luyện. |
| `save` | True | Lưu checkpoint cuối cùng (`last.pt`) và tốt nhất (`best.pt`). |
| `plots` | True | Tạo biểu đồ huấn luyện (loss curves, mAP curves, confusion matrix, F1 curve). |
| `workers` | 2 | Số luồng load dữ liệu. Thấp (2) để tránh lỗi hàng đợi trên Colab T4. |

---

## 4. Script Huấn luyện Pitch Keypoint

### 4.1. Tổng quan

**File:** `trainings/pitch_keypoint_training.py` (67 dòng)

**Mô hình:** YOLOv8x-pose (phiên bản largest của YOLOv8 pose estimation)

**Đầu vào:**
- File `.env` chứa `ROBOFLOW_API_KEY`
- Dataset: `datasets/football-field-detection-cleaned/` (nếu có) hoặc tải từ Roboflow
- Pre-trained weights: `yolov8x-pose.pt` (COCO pose)

**Đầu ra:**
- Weights: `models/pitch_keypoint/pitch_keypoint/weights/best.pt`
- (Trên Colab) `/content/drive/MyDrive/football_models/pitch_keypoint/weights/best.pt`

### 4.2. Khác biệt so với Player Detection

#### 4.2.1. Base model (dòng 26)

```python
model = YOLO("yolov8x-pose.pt")
```

Sử dụng `yolov8x-pose.pt` thay vì `yolov8x.pt`. Kiến trúc pose khác detection ở chỗ:
- Head có thêm nhánh keypoint regression: output `(num_classes + 4 + num_kpts * 3)` channels
- Mặc định COCO pose: 17 keypoints người
- Khi fine-tune, Ultralytics tự động thay đổi output head cho `kpt_shape` mới (32 keypoints)

**Kiến trúc YOLOv8x-pose:**
- Backbone & Neck: giống YOLOv8x (CSPDarknet + PAN-FPN)
- Head: Decoupled head + keypoint branch
- Tham số: ~69.8M (nhiều hơn detection ~1.6M do keypoint head)
- GFLOPS: ~264.7

#### 4.2.2. Hyperparameters (dòng 36–61)

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `data` | `f"{dataset.location}/data.yaml"` | **KHÁC BIỆT:** Dùng `dataset.location` trực tiếp thay vì `data_yaml_location`. Đây là một điểm cần lưu ý: nếu dataset đã có sẵn local, biến `dataset` undefined gây lỗi. Trên Colab (nơi không có local dataset) hoạt động bình thường. |
| `epochs` | 100 | Giống player detection. |
| `imgsz` | 640 | **KHÁC BIỆT:** Nhỏ hơn player (1280). Keypoint detection cần ít chi tiết hơn; 640px cho phép batch 8 trên T4. |
| `batch` | 8 | Giống. |
| `device` | 0 | Giống. |
| `patience` | 50 | **KHÁC BIỆT:** Lớn hơn player (20). Keypoint mAP hội tụ chậm hơn, cần patience cao hơn. |
| `project` | `output_project` | Tương tự. |
| `name` | `"pitch_keypoint"` | Tên experiment. |
| `save` | True | Giống. |
| `plots` | True | Giống. |

**Tham số riêng của keypoint training:**

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `lr0` | 0.01 | Learning rate ban đầu. Giá trị mặc định của Ultralytics. |
| `lrf` | 0.1 | Final learning rate factor: `lr_end = lr0 * lrf`. Cosine annealing từ 0.01 → 0.001. |
| `warmup_epochs` | 5 | 5 epoch đầu tăng dần LR từ 0 → `lr0` để ổn định quá trình hội tụ. |
| `mosaic` | 1.0 | Xác suất mosaic augmentation (ghép 4 ảnh thành 1). Giá trị 1.0 = luôn bật. |
| `close_mosaic` | 10 | Tắt mosaic 10 epoch cuối để fine-tune với ảnh gốc (không mosaic), giúp mAP hội tụ tốt hơn. |
| `degrees` | 10.0 | Xoay ảnh ngẫu nhiên trong khoảng ±10°. Nhỏ hơn mặt định (0.0) vì xoay nhiều làm sai lệch keypoint. |
| `scale` | 0.2 | Scale augmentation ±20%. Thấp hơn mặc định (0.5) để keypoint không bị biến dạng quá nhiều. |
| `fliplr` | 0.5 | Flip ngang 50%. Có flip_idx trong data.yaml để mapping keypoint trái-phải. |
| `flipud` | 0.0 | Không flip dọc — sân bóng không bao giờ lộn ngược trong video thực tế. |

---

## 5. So sánh hai quy trình huấn luyện

| Tiêu chí | Player Detection | Pitch Keypoint |
|-----------|-----------------|----------------|
| **Model** | YOLOv8x | YOLOv8x-pose |
| **Input size** | 1280×1280 | 640×640 |
| **Số lớp** | 4 (ball, goalkeeper, player, referee) | 1 (pitch) |
| **Số keypoint** | Không có | 32 (kpt_shape: [32, 3]) |
| **Dataset** | football-players-detection-cleaned (612 train) | football-field-detection-cleaned |
| **Patience** | 20 | 50 |
| **Augmentation** | Mặc định của Ultralytics | Tùy chỉnh: degrees=10, scale=0.2, mosaic=1.0, close_mosaic=10, flipud=0.0 |
| **LR schedule** | Mặc định | lr0=0.01, lrf=0.1, warmup=5 |
| **VRAM ước tính** | ~12–14 GB (T4) | ~6–8 GB (T4) |
| **Thời gian 100 epoch** | ~4–6 giờ (T4) | ~2–3 giờ (T4) |

---

## 6. Bug hiện tại trong pitch_keypoint_training.py

Tại dòng 37:
```python
data=f"{dataset.location}/data.yaml",
```

Biến `dataset` chỉ tồn tại trong nhánh `if not yaml_path.exists()` (dòng 16–21). Khi dataset đã có sẵn ở local (`yaml_path.exists()` = True), nhánh else (dòng 22–24) gán `data_yaml_location` nhưng không định nghĩa `dataset`. Dòng 37 tham chiếu `dataset.location` → **NameError**.

**Cách khắc phục:** Sửa dòng 37 thành:
```python
data=data_yaml_location,
```

File `football_training.py` không mắc lỗi này (dòng 38 đã dùng `data=data_yaml_location`).

---

## 7. So sánh với Player Dataset Detection script

| Tiêu chí | `football_training.py` | `pitch_keypoint_training.py` |
|-----------|----------------------|----------------------------|
| **Dataset cleaning dependency** | `datasets/football-players-detection-cleaned` (từ `player_cleaning.py`) | `datasets/football-field-detection-cleaned` (từ `pitch_keypoint_cleaning.py`) |
| **Base model** | `yolov8x.pt` | `yolov8x-pose.pt` |
| **Data parameter** | `data=data_yaml_location` ✅ | `data=f"{dataset.location}/data.yaml"` ❌ (bug) |
| **Local training support** | Có | Không (chỉ Colab) |
| **Output local** | `models/player_detector/` | `models/pitch_keypoint/` |
| **Output Colab** | `/content/drive/MyDrive/football_models/` | `/content/drive/MyDrive/football_models/` |
| **Augmentation** | Mặc định | Custom: degrees, scale, flipud |

---

## 8. Hướng dẫn sử dụng

### 8.1. Local

```bash
# Chuẩn bị
pip install -r requirements.txt
# Tạo .env với ROBOFLOW_API_KEY

# Chạy training
cd trainings
python football_training.py
python pitch_keypoint_training.py
```

### 8.2. Google Colab

1. Upload file `.env` lên Colab hoặc nhập API key trực tiếp.
2. Chạy từng cell trong notebook `evaluate_models.ipynb` hoặc tạo notebook mới:
```python
!pip install ultralytics roboflow python-dotenv
!git clone https://github.com/henruysun2511/football_tracking.git
%cd football_tracking/trainings
!python football_training.py
```

---

## 9. Kết luận

Hai script huấn luyện `football_training.py` và `pitch_keypoint_training.py` cung cấp quy trình fine-tune tự động cho hai mô hình YOLOv8 khác nhau. Script player detection sử dụng cấu hình mặc định phù hợp cho object detection, trong khi script pitch keypoint sử dụng hyperparameters được tinh chỉnh riêng cho pose estimation (augmentation nhẹ, patience cao). Cả hai script đều hỗ trợ cơ chế tự động tải dataset từ Roboflow khi chưa có sẵn local và tự động phát hiện môi trường Colab để lưu weights lên Google Drive.
