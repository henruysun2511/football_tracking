# Chạy trước khi train để làm sạch dataset Roboflow
import os, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DATASET_ROOT = "datasets/football-players-detection-2"
CLEANED_ROOT = "datasets/football-players-detection-cleaned"
SPLITS       = ["train", "valid", "test"]
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]
NUM_CLASSES  = 4

# ── BƯỚC 1: SCAN – phát hiện vấn đề ─────────────────────────

def scan_split(split):
    img_dir = Path(DATASET_ROOT) / split / "images"
    lbl_dir = Path(DATASET_ROOT) / split / "labels"

    bad_stems = set()
    stats = dict(total=0, corrupt=0, missing_label=0,
                 empty_label=0, invalid_bbox=0, bad_class=0)
    cls_counter = Counter()

    img_files = sorted(list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.png")))
    stats["total"] = len(img_files)

    for img_path in img_files:
        stem = img_path.stem

        # 1. Ảnh corrupt?
        img = cv2.imread(str(img_path))
        if img is None:
            stats["corrupt"] += 1
            bad_stems.add(stem)
            continue

        # 2. Thiếu label?
        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            stats["missing_label"] += 1
            bad_stems.add(stem)
            continue

        # 3. Label rỗng?
        lines = [l.strip() for l in
                 lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            stats["empty_label"] += 1
            bad_stems.add(stem)
            continue

        # 4. Kiểm tra từng bbox
        has_issue = False
        for line in lines:
            p = line.split()
            if len(p) != 5:
                has_issue = True
                continue
            cls_id = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])

            if cls_id < 0 or cls_id >= NUM_CLASSES:
                stats["bad_class"] += 1
                has_issue = True
                continue
            if not (0 < bw <= 1 and 0 < bh <= 1 and
                    0 <= cx <= 1 and 0 <= cy <= 1):
                stats["invalid_bbox"] += 1
                has_issue = True
                continue
            cls_counter[cls_id] += 1

        if has_issue:
            bad_stems.add(stem)

    stats["good"] = stats["total"] - len(bad_stems)
    return stats, bad_stems, cls_counter


# ── BƯỚC 2: CLEAN – copy ảnh sạch sang thư mục mới ──────────

def clean_split(split, bad_stems):
    for sub in ["images", "labels"]:
        src_dir = Path(DATASET_ROOT) / split / sub
        dst_dir = Path(CLEANED_ROOT) / split / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in src_dir.glob("*"):
            if f.stem not in bad_stems:
                shutil.copy2(f, dst_dir / f.name)
                copied += 1
        print(f"  [{split}/{sub}] copied {copied} files")


# ── BƯỚC 3: FIX BBOX – clip bbox vào [0,1] thay vì xóa ──────

def fix_bbox_labels(split):
    """
    Thay vì loại bỏ ảnh có bbox lỗi nhỏ,
    clip tọa độ về [0,1] và lưu đè.
    """
    lbl_dir = Path(CLEANED_ROOT) / split / "labels"
    fixed_count = 0
    for lbl_path in lbl_dir.glob("*.txt"):
        lines = lbl_path.read_text().strip().splitlines()
        new_lines = []
        changed = False
        for line in lines:
            p = line.split()
            if len(p) != 5:
                continue
            cls_id = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])
            # Clip tọa độ
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            bw = np.clip(bw, 0.001, 1.0)
            bh = np.clip(bh, 0.001, 1.0)
            # Đảm bảo bbox không vượt biên sau clip
            bw = min(bw, 2*cx, 2*(1-cx))
            bh = min(bh, 2*cy, 2*(1-cy))
            if bw > 0.001 and bh > 0.001:
                new_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                changed = True
        if changed:
            lbl_path.write_text("\n".join(new_lines))
            fixed_count += 1
    print(f"  [{split}] fixed {fixed_count} label files")


# ── BƯỚC 4: VISUALIZE trước/sau ──────────────────────────────

def visualize_before_after(split="train", n_samples=6):
    img_dir_before = Path(DATASET_ROOT) / split / "images"
    img_dir_after  = Path(CLEANED_ROOT) / split / "images"
    lbl_dir_before = Path(DATASET_ROOT) / split / "labels"
    lbl_dir_after  = Path(CLEANED_ROOT) / split / "labels"
    COLORS = [(255,50,50),(0,200,255),(50,200,50),(255,160,0)]

    def draw_boxes(img_path, lbl_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((100,100,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if not lbl_path.exists():
            return img
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5: continue
            c = int(p[0]); cx,cy,bw,bh = map(float, p[1:])
            x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
            x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)
            col = COLORS[min(c,3)]
            cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
            cv2.putText(img, CLASS_NAMES[min(c,3)],
                        (x1,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
        return img

    imgs = list(img_dir_before.glob("*.jpg"))[:n_samples]
    fig, axes = plt.subplots(2, len(imgs), figsize=(4*len(imgs), 8))
    fig.suptitle("So sánh Trước / Sau Làm sạch Dữ liệu",
                 fontsize=14, fontweight='bold')

    for i, img_path in enumerate(imgs):
        stem = img_path.stem
        lbl_b = lbl_dir_before / (stem + ".txt")
        img_after_path = img_dir_after / img_path.name
        lbl_a = lbl_dir_after  / (stem + ".txt")

        axes[0,i].imshow(draw_boxes(img_path, lbl_b))
        axes[0,i].set_title(f"TRƯỚC\n{stem[:12]}", fontsize=8)
        axes[0,i].axis('off')

        if img_after_path.exists():
            axes[1,i].imshow(draw_boxes(img_after_path, lbl_a))
            axes[1,i].set_title("SAU làm sạch", fontsize=8)
        else:
            axes[1,i].text(0.5, 0.5, "ĐÃ XÓA\n(ảnh lỗi)",
                           ha='center', va='center',
                           transform=axes[1,i].transAxes,
                           fontsize=12, color='red')
        axes[1,i].axis('off')

    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    plt.savefig("analysis/figures/data_cleaning_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: analysis/figures/data_cleaning_comparison.png")


# ── BƯỚC 5: BÁO CÁO TỔNG KẾT ────────────────────────────────

def print_cleaning_report(before_stats, after_counts):
    print("\n" + "="*60)
    print("  DATA CLEANING REPORT")
    print("="*60)
    print(f"{'Split':<8} {'Trước':>8} {'Sau':>8} "
          f"{'Xóa':>8} {'%giữ':>8}")
    print("-"*60)
    for split in SPLITS:
        b = before_stats[split]["total"]
        a = after_counts.get(split, 0)
        removed = b - a
        pct = a/b*100 if b > 0 else 0
        print(f"{split:<8} {b:>8} {a:>8} {removed:>8} {pct:>7.1f}%")
    print("="*60)


# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Buoc 1: Scan dataset goc...")
    all_before = {}
    all_bad    = {}
    for split in SPLITS:
        stats, bad_stems, cls_cnt = scan_split(split)
        all_before[split] = stats
        all_bad[split]    = bad_stems
        print(f"\n[{split}] total={stats['total']} "
              f"good={stats['good']} "
              f"corrupt={stats['corrupt']} "
              f"missing_label={stats['missing_label']} "
              f"empty={stats['empty_label']} "
              f"invalid_bbox={stats['invalid_bbox']}")

    print("\nBuoc 2: Copy anh sach...")
    after_counts = {}
    for split in SPLITS:
        clean_split(split, all_bad[split])
        after_dir = Path(CLEANED_ROOT) / split / "images"
        after_counts[split] = len(list(after_dir.glob("*.jpg")))

    print("\nBuoc 3: Fix bbox out-of-bounds...")
    for split in SPLITS:
        fix_bbox_labels(split)

    print("\nBuoc 4: Visualize truoc/sau...")
    visualize_before_after("train")

    print_cleaning_report(all_before, after_counts)
    print("\nDataset sach da luu tai:", CLEANED_ROOT)