# analysis/clean_pitch_keypoint_dataset.py
# Dataset pitch keypoint dùng định dạng YOLO Pose:
# mỗi dòng label: class cx cy w h  x1 y1 v1  x2 y2 v2 ... xN yN vN
# v = visibility: 0=không thấy, 1=bị che, 2=thấy rõ

import os, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "datasets/pitch-keypoint-detection-2"
CLEANED_ROOT = "datasets/pitch-keypoint-detection-cleaned"
SPLITS       = ["train", "valid", "test"]
NUM_CLASSES  = 1          # chỉ 1 class: "pitch"
NUM_KEYPOINTS = 32        # số keypoint theo Roboflow sports dataset
MIN_VISIBLE_KPS = 4       # tối thiểu 4 keypoint visible mới giữ lại ảnh

# ── BƯỚC 1: SCAN ─────────────────────────────────────────────

def scan_keypoint_split(split):
    img_dir = Path(DATASET_ROOT) / split / "images"
    lbl_dir = Path(DATASET_ROOT) / split / "labels"

    stats = dict(
        total=0,
        corrupt=0,
        missing_label=0,
        empty_label=0,
        malformed_line=0,
        invalid_bbox=0,
        too_few_keypoints=0,
        invalid_visibility=0,
        good=0,
    )
    bad_stems    = set()
    kp_vis_count = defaultdict(int)   # {kp_index: số lần visible}
    kp_per_image = []                 # số kp visible mỗi ảnh

    img_files = sorted(list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.png")))
    stats["total"] = len(img_files)

    for img_path in img_files:
        stem = img_path.stem

        # 1. Ảnh đọc được không?
        img = cv2.imread(str(img_path))
        if img is None:
            stats["corrupt"] += 1
            bad_stems.add(stem)
            continue

        # 2. Có file label không?
        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            stats["missing_label"] += 1
            bad_stems.add(stem)
            continue

        lines = [l.strip() for l in
                 lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            stats["empty_label"] += 1
            bad_stems.add(stem)
            continue

        image_bad = False
        for line in lines:
            parts = line.split()
            # Format: class cx cy w h [x1 y1 v1 x2 y2 v2 ...]
            # Số phần tử tối thiểu: 5 + NUM_KEYPOINTS*3
            expected = 5 + NUM_KEYPOINTS * 3
            if len(parts) < expected:
                stats["malformed_line"] += 1
                image_bad = True
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # Kiểm tra bbox
            if not (0 < bw <= 1 and 0 < bh <= 1 and
                    0 <= cx <= 1 and 0 <= cy <= 1):
                stats["invalid_bbox"] += 1
                image_bad = True
                continue

            # Đọc keypoints
            kp_data = list(map(float, parts[5:5 + NUM_KEYPOINTS * 3]))
            n_visible = 0
            has_invalid_vis = False

            for i in range(NUM_KEYPOINTS):
                x   = kp_data[i * 3]
                y   = kp_data[i * 3 + 1]
                vis = kp_data[i * 3 + 2]

                # visibility chỉ hợp lệ là 0, 1, 2
                if vis not in (0, 1, 2):
                    has_invalid_vis = True
                    continue

                if vis > 0:
                    n_visible += 1
                    kp_vis_count[i] += 1

                    # Tọa độ kp phải trong [0,1] khi vis > 0
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        has_invalid_vis = True

            if has_invalid_vis:
                stats["invalid_visibility"] += 1
                image_bad = True
                continue

            kp_per_image.append(n_visible)

            # Loại ảnh có quá ít keypoint visible
            if n_visible < MIN_VISIBLE_KPS:
                stats["too_few_keypoints"] += 1
                image_bad = True

        if image_bad:
            bad_stems.add(stem)

    stats["good"] = stats["total"] - len(bad_stems)
    return stats, bad_stems, kp_vis_count, kp_per_image


# ── BƯỚC 2: FIX + COPY ────────────────────────────────────────

def fix_and_copy_keypoint_split(split, bad_stems):
    """
    Copy ảnh sạch, đồng thời fix một số lỗi nhỏ:
    - Clip tọa độ bbox và keypoint về [0, 1]
    - Đặt visibility = 0 cho keypoint có tọa độ nằm ngoài biên
    """
    for sub in ["images", "labels"]:
        dst = Path(CLEANED_ROOT) / split / sub
        dst.mkdir(parents=True, exist_ok=True)

    img_dir = Path(DATASET_ROOT) / split / "images"
    lbl_dir = Path(DATASET_ROOT) / split / "labels"
    fixed = 0

    for img_path in img_dir.glob("*.jpg"):
        stem = img_path.stem
        if stem in bad_stems:
            continue

        # Copy ảnh
        shutil.copy2(img_path,
                     Path(CLEANED_ROOT) / split / "images" / img_path.name)

        # Fix label
        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            continue

        new_lines = []
        changed   = False
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5 + NUM_KEYPOINTS * 3:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # Clip bbox
            cx_c = float(np.clip(cx, 0, 1))
            cy_c = float(np.clip(cy, 0, 1))
            bw_c = float(np.clip(bw, 0.001, 1))
            bh_c = float(np.clip(bh, 0.001, 1))
            if (cx_c, cy_c, bw_c, bh_c) != (cx, cy, bw, bh):
                changed = True

            # Fix keypoints
            kp_raw = list(map(float, parts[5:5 + NUM_KEYPOINTS * 3]))
            kp_new = []
            for i in range(NUM_KEYPOINTS):
                x, y, v = kp_raw[i*3], kp_raw[i*3+1], kp_raw[i*3+2]
                v = int(round(v))
                v = max(0, min(2, v))   # clip visibility về {0,1,2}
                if v > 0:
                    # Clip tọa độ kp, nếu bị đẩy ra ngoài thì set invisible
                    xc = float(np.clip(x, 0, 1))
                    yc = float(np.clip(y, 0, 1))
                    if abs(xc - x) > 0.01 or abs(yc - y) > 0.01:
                        v = 0  # quá lệch → coi như không nhìn thấy
                        changed = True
                    x, y = xc, yc
                kp_new.extend([x, y, v])

            kp_str = " ".join(f"{v:.6f}" for v in kp_new)
            new_lines.append(
                f"{cls_id} {cx_c:.6f} {cy_c:.6f} "
                f"{bw_c:.6f} {bh_c:.6f} {kp_str}"
            )
            if changed:
                fixed += 1

        dst_lbl = Path(CLEANED_ROOT) / split / "labels" / (stem + ".txt")
        dst_lbl.write_text("\n".join(new_lines))

    print(f"  [{split}] fixed labels: {fixed}")


# ── BƯỚC 3: THỐNG KÊ VÀ VISUALIZE ────────────────────────────

def visualize_keypoint_stats(all_kp_vis_count, all_kp_per_image,
                              split="train"):
    """
    Vẽ 2 biểu đồ:
    1. Tần suất visible của từng keypoint index
    2. Phân bố số keypoint visible mỗi ảnh
    """
    os.makedirs("analysis/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Thống kê Pitch Keypoint Dataset [{split}]",
                 fontsize=13, fontweight='bold')

    # Biểu đồ 1: tần suất visible theo kp index
    ax = axes[0]
    indices = range(NUM_KEYPOINTS)
    counts  = [all_kp_vis_count.get(i, 0) for i in indices]
    colors  = ['#2ecc71' if c > np.mean(counts) else '#e74c3c'
               for c in counts]
    ax.bar(indices, counts, color=colors, edgecolor='white', width=0.8)
    ax.axhline(np.mean(counts), color='orange', linestyle='--',
               linewidth=1.5, label=f'Mean = {np.mean(counts):.0f}')
    ax.set_title("Tần suất visible của từng keypoint\n"
                 "(đỏ = ít visible, xanh = nhiều visible)")
    ax.set_xlabel("Keypoint Index")
    ax.set_ylabel("Số lần visible")
    ax.legend(fontsize=9)

    # Biểu đồ 2: phân bố số kp visible mỗi ảnh
    ax = axes[1]
    if all_kp_per_image:
        ax.hist(all_kp_per_image, bins=range(0, NUM_KEYPOINTS + 2),
                color='#3498db', edgecolor='white', linewidth=0.5)
        ax.axvline(MIN_VISIBLE_KPS, color='red', linestyle='--',
                   linewidth=2,
                   label=f'Ngưỡng tối thiểu = {MIN_VISIBLE_KPS}')
        ax.set_title("Phân bố số keypoint visible mỗi ảnh")
        ax.set_xlabel("Số keypoint visible")
        ax.set_ylabel("Số ảnh")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"analysis/figures/pitch_keypoint_stats_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


def visualize_keypoints_on_image(split="train", n=4):
    """Vẽ keypoints lên ảnh để kiểm tra trực quan trước/sau."""
    img_dir_b = Path(DATASET_ROOT)  / split / "images"
    lbl_dir_b = Path(DATASET_ROOT)  / split / "labels"
    img_dir_a = Path(CLEANED_ROOT)  / split / "images"
    lbl_dir_a = Path(CLEANED_ROOT)  / split / "labels"

    KP_COLOR_VIS  = (0, 255, 100)    # visible
    KP_COLOR_OCC  = (255, 200, 0)    # occluded
    KP_COLOR_INV  = (200, 200, 200)  # invisible

    def draw_kps(img_path, lbl_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((200, 300, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if not lbl_path.exists():
            return img
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5 + NUM_KEYPOINTS * 3:
                continue
            kps = list(map(float, parts[5:5 + NUM_KEYPOINTS * 3]))
            for i in range(NUM_KEYPOINTS):
                x, y, v = kps[i*3], kps[i*3+1], int(kps[i*3+2])
                px, py = int(x * w), int(y * h)
                if v == 2:
                    color = KP_COLOR_VIS
                elif v == 1:
                    color = KP_COLOR_OCC
                else:
                    continue  # invisible: không vẽ
                cv2.circle(img, (px, py), 5, color, -1)
                cv2.putText(img, str(i), (px+5, py-4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, color, 1)
        return img

    imgs = list(img_dir_b.glob("*.jpg"))[:n]
    fig, axes = plt.subplots(2, len(imgs),
                              figsize=(5 * len(imgs), 8))
    fig.suptitle("Pitch Keypoint – Trước / Sau Làm sạch",
                 fontsize=13, fontweight='bold')

    for i, img_path in enumerate(imgs):
        stem = img_path.stem
        # Hàng trên: trước
        axes[0, i].imshow(
            draw_kps(img_path, lbl_dir_b / (stem + ".txt")))
        axes[0, i].set_title(f"TRƯỚC\n{stem[:10]}", fontsize=8)
        axes[0, i].axis('off')
        # Hàng dưới: sau
        after_img = img_dir_a / img_path.name
        if after_img.exists():
            axes[1, i].imshow(
                draw_kps(after_img, lbl_dir_a / (stem + ".txt")))
            axes[1, i].set_title("SAU làm sạch", fontsize=8)
        else:
            axes[1, i].text(0.5, 0.5, "ĐÃ LOẠI\n(lỗi)",
                            ha='center', va='center',
                            transform=axes[1, i].transAxes,
                            fontsize=11, color='red')
        axes[1, i].axis('off')

    legend = [
        mpatches.Patch(color='#00ff64', label='Visible (v=2)'),
        mpatches.Patch(color='#ffc800', label='Occluded (v=1)'),
    ]
    fig.legend(handles=legend, loc='lower center',
               ncol=2, fontsize=9)
    plt.tight_layout()
    out = "analysis/figures/pitch_keypoint_before_after.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("=" * 55)
    print("  PITCH KEYPOINT DATASET CLEANING")
    print("=" * 55)

    all_kp_vis = defaultdict(int)
    all_kp_img = []
    before_stats = {}
    all_bad = {}

    # Bước 1: Scan
    for split in SPLITS:
        stats, bad, kp_vis, kp_img = scan_keypoint_split(split)
        before_stats[split] = stats
        all_bad[split]      = bad
        all_kp_vis.update(kp_vis)
        all_kp_img.extend(kp_img)
        print(f"\n[{split}]")
        for k, v in stats.items():
            if v > 0 or k in ('total', 'good'):
                print(f"  {k:<25}: {v}")

    # Bước 2: Fix + copy
    print("\nFix và copy dữ liệu sạch...")
    for split in SPLITS:
        fix_and_copy_keypoint_split(split, all_bad[split])

    # Bước 3: Visualize
    print("\nVisualize thống kê...")
    visualize_keypoint_stats(all_kp_vis, all_kp_img, split="train")
    visualize_keypoints_on_image(split="train", n=4)

    # Tổng kết
    print("\n" + "=" * 55)
    print(f"{'Split':<8} {'Trước':>8} {'Sau':>8} {'Xóa':>8}")
    print("-" * 40)
    for split in SPLITS:
        b = before_stats[split]["total"]
        a = len(list((Path(CLEANED_ROOT)/split/"images").glob("*.jpg")))
        print(f"{split:<8} {b:>8} {a:>8} {b-a:>8}")
    print("=" * 55)
    print(f"Dataset sạch lưu tại: {CLEANED_ROOT}")