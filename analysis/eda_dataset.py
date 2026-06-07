import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

DATASET_ROOT = "datasets/football-players-detection-2"
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

def parse_labels(label_dir):
    """Đọc tất cả file .txt YOLO format, trả về list (class_id, w, h)"""
    records = []
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"⚠️ Thư mục labels không tồn tại: {label_dir}")
        return records
        
    for f in label_path.glob("*.txt"):
        with open(f) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = parts
                    records.append((int(cls), float(w), float(h)))
    return records

if not Path(DATASET_ROOT).exists():
    print(f"Dataset not found at {DATASET_ROOT}. Skipping.")
    exit(0)

def count_images(split):
    """Đếm số lượng ảnh, hỗ trợ cả đuôi viết hoa/thường (jpg, png, jpeg)"""
    img_dir = Path(DATASET_ROOT) / split / "images"
    if not img_dir.exists():
        return 0
    # Dùng list comprehension kết hợp .suffix.lower() để tránh sót file .JPG, .PNG
    return len([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

# ---- 1. Số ảnh mỗi split ----
split_counts = {s: count_images(s) for s in SPLITS}
print("=== Số ảnh mỗi split ===")
for s, n in split_counts.items():
    print(f"  {s}: {n} ảnh")

# ---- 2. Thống kê phân bố class (train) ----
train_records = parse_labels(os.path.join(DATASET_ROOT, "train", "labels"))

if not train_records:
    print("Không tìm thấy dữ liệu bounding box nào trong tập train. Vui lòng kiểm tra lại đường dẫn dataset!")
else:
    class_counts = Counter(r[0] for r in train_records)

    # TỰ ĐỘNG TẠO THƯ MỤC LƯU FILE (Sửa lỗi sập code)
    output_dir = Path("analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Football Player Detection Dataset – EDA", fontsize=14, fontweight='bold')

    # Biểu đồ 1: Số lượng object theo class
    ax = axes[0]
    counts = [class_counts.get(i, 0) for i in range(4)]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title("Phân bố số lượng object theo class (train)")
    ax.set_ylabel("Số lượng bounding box")
    
    # Tối ưu khoảng cách hiển thị chữ trên đầu cột để không bị dính vào viền
    max_count = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_count * 0.01),
                str(count), ha='center', fontsize=10, fontweight='bold')

    # Biểu đồ 2: Split distribution
    ax = axes[1]
    if sum(split_counts.values()) > 0:
        ax.pie(split_counts.values(), labels=split_counts.keys(),
               autopct='%1.1f%%', colors=['#3498db', '#2ecc71', '#e74c3c'],
               startangle=90)
    ax.set_title("Phân chia Train / Val / Test")

    # Biểu đồ 3: Phân bố kích thước bbox (width × height)
    ax = axes[2]
    widths = [r[1] for r in train_records]
    heights = [r[2] for r in train_records]
    areas = [w * h for w, h in zip(widths, heights)]
    ax.hist(areas, bins=50, color='#9b59b6', edgecolor='white', linewidth=0.3)
    ax.set_title("Phân bố diện tích bbox (normalized)")
    ax.set_xlabel("Diện tích bbox (w×h, normalized)")
    ax.set_ylabel("Tần suất")
    ax.axvline(0.01, color='red', linestyle='--', label='Small object threshold (1%)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_eda.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu: {output_dir / 'dataset_eda.png'}")

    # ---- 3. In bảng thống kê tổng hợp ----
    total = sum(counts)
    print("\n=== Bảng thống kê phân bố class (train) ===")
    print(f"{'Class':<15} {'Số bbox':>10} {'Tỉ lệ':>10}")
    print("-" * 40)
    for i, name in enumerate(CLASS_NAMES):
        n = class_counts.get(i, 0)
        # Phòng ngừa lỗi chia cho 0 nếu total = 0
        percentage = (n / total * 100) if total > 0 else 0
        print(f"{name:<15} {n:>10} {percentage:>9.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<15} {total:>10} {'100.0%':>10}")

    small_objects = sum(1 for a in areas if a < 0.002)
    # Tránh crash nếu total_object bằng 0
    small_percentage = (small_objects / total * 100) if total > 0 else 0
    print(f"\nSmall objects (<0.2% diện tích ảnh): {small_objects} ({small_percentage:.1f}%)")
    
    # Tính tỉ lệ mất cân bằng dữ liệu an toàn
    ball_count = max(class_counts.get(0, 1), 1)
    player_count = class_counts.get(2, 0)
    print(f"Class imbalance ratio (player/ball): {player_count / ball_count:.1f}x")