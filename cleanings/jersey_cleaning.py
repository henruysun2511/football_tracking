# analysis/clean_jersey_dataset.py
# Jersey number dataset gồm 2 loại tùy cách tổ chức:
# Loại A: Detection (YOLO bbox) - chỉ detect vùng số áo
# Loại B: Classification - mỗi thư mục là 1 số áo (0/, 1/, ..., 99/)
# Script này xử lý CẢ HAI loại

import os, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

# ── CẤU HÌNH ──────────────────────────────────────────────────

# Loại A: YOLO detection format (class = số áo 0-99)
DETECTION_ROOT   = "datasets/jersey-number-detection-1"
DETECTION_CLEAN  = "datasets/jersey-number-detection-cleaned"

# Loại B: Classification format (thư mục = nhãn)
CLASSIFY_ROOT    = "datasets/jersey-number-classification-1"
CLASSIFY_CLEAN   = "datasets/jersey-number-classification-cleaned"

SPLITS       = ["train", "valid", "test"]
VALID_JERSEY = set(range(1, 100))   # số áo hợp lệ: 1-99

# Ngưỡng chất lượng ảnh crop
MIN_CROP_H   = 20   # px, ảnh crop quá nhỏ thì OCR không đọc được
MIN_CROP_W   = 10
MAX_BLUR     = 120  # Laplacian variance, nhỏ hơn = quá mờ


# ════════════════════════════════════════════
# PHẦN A: YOLO DETECTION FORMAT
# ════════════════════════════════════════════

def check_image_quality(img):
    """
    Kiểm tra chất lượng ảnh:
    - Quá tối / quá sáng (mean brightness)
    - Quá mờ (Laplacian variance)
    """
    if img is None:
        return False, "corrupt"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if brightness < 15:
        return False, f"too_dark (mean={brightness:.1f})"
    if brightness > 240:
        return False, f"too_bright (mean={brightness:.1f})"
    if blur_score < MAX_BLUR:
        return False, f"blurry (lap_var={blur_score:.1f})"
    return True, "ok"


def scan_jersey_detection(split):
    img_dir = Path(DETECTION_ROOT) / split / "images"
    lbl_dir = Path(DETECTION_ROOT) / split / "labels"

    stats = dict(
        total=0, corrupt=0, missing_label=0, empty_label=0,
        invalid_class=0, invalid_bbox=0, too_dark_bright=0,
        blurry=0, good=0
    )
    bad_stems     = set()
    class_counter = Counter()
    blur_scores   = []

    img_files = sorted(list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.png")))
    stats["total"] = len(img_files)

    for img_path in img_files:
        stem = img_path.stem
        img  = cv2.imread(str(img_path))

        # 1. Ảnh corrupt?
        if img is None:
            stats["corrupt"] += 1
            bad_stems.add(stem)
            continue

        # 2. Kiểm tra chất lượng ảnh
        ok, reason = check_image_quality(img)
        if not ok:
            if "dark" in reason or "bright" in reason:
                stats["too_dark_bright"] += 1
            else:
                stats["blurry"] += 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_scores.append(
                    cv2.Laplacian(gray, cv2.CV_64F).var())
            bad_stems.add(stem)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. Có label không?
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

        has_issue = False
        for line in lines:
            p = line.split()
            if len(p) != 5:
                has_issue = True
                continue
            cls_id = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])

            # Class ID phải là số áo hợp lệ (1-99)
            if cls_id not in VALID_JERSEY:
                stats["invalid_class"] += 1
                has_issue = True
                continue

            if not (0 < bw <= 1 and 0 < bh <= 1 and
                    0 <= cx <= 1 and 0 <= cy <= 1):
                stats["invalid_bbox"] += 1
                has_issue = True
                continue

            # Kiểm tra kích thước crop thực tế
            h_img, w_img = img.shape[:2]
            crop_w = bw * w_img
            crop_h = bh * h_img
            if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
                has_issue = True  # crop quá nhỏ → OCR không đọc được
                continue

            class_counter[cls_id] += 1

        if has_issue:
            bad_stems.add(stem)

    stats["good"] = stats["total"] - len(bad_stems)
    return stats, bad_stems, class_counter, blur_scores


def fix_jersey_detection(split, bad_stems):
    """Copy ảnh sạch, fix bbox bị clip nhỏ."""
    for sub in ["images", "labels"]:
        Path(DETECTION_CLEAN, split, sub).mkdir(
            parents=True, exist_ok=True)

    img_dir = Path(DETECTION_ROOT) / split / "images"
    lbl_dir = Path(DETECTION_ROOT) / split / "labels"

    for img_path in img_dir.glob("*.jpg"):
        stem = img_path.stem
        if stem in bad_stems:
            continue
        shutil.copy2(img_path,
                     Path(DETECTION_CLEAN)/split/"images"/img_path.name)
        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            continue

        new_lines = []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cls_id = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])
            # Clip về [0,1]
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            bw = np.clip(bw, 0.01, 1)
            bh = np.clip(bh, 0.01, 1)
            if cls_id in VALID_JERSEY:
                new_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        dst = Path(DETECTION_CLEAN)/split/"labels"/(stem+".txt")
        dst.write_text("\n".join(new_lines))


# ════════════════════════════════════════════
# PHẦN B: CLASSIFICATION FORMAT
# ════════════════════════════════════════════

def scan_jersey_classification(split):
    """
    Với classification format, mỗi thư mục = 1 lớp (số áo).
    Kiểm tra: tên thư mục hợp lệ, ảnh đọc được, chất lượng ảnh.
    """
    split_dir = Path(CLASSIFY_ROOT) / split

    stats = dict(
        total_classes=0, invalid_class_name=0,
        total_images=0, corrupt=0, blurry=0,
        too_dark_bright=0, too_small=0, good=0
    )
    bad_files     = []   # [(class_dir, filename)]
    class_counter = Counter()

    if not split_dir.exists():
        print(f"  [{split}] thư mục không tồn tại, bỏ qua.")
        return stats, bad_files, class_counter

    class_dirs = sorted([d for d in split_dir.iterdir()
                         if d.is_dir()])
    stats["total_classes"] = len(class_dirs)

    for cls_dir in class_dirs:
        # Tên thư mục phải là số nguyên hợp lệ 1-99
        try:
            cls_id = int(cls_dir.name)
            if cls_id not in VALID_JERSEY:
                raise ValueError
        except ValueError:
            stats["invalid_class_name"] += 1
            continue

        img_files = list(cls_dir.glob("*.jpg")) + \
                    list(cls_dir.glob("*.png"))
        stats["total_images"] += len(img_files)

        for img_path in img_files:
            img = cv2.imread(str(img_path))

            # Corrupt?
            if img is None:
                stats["corrupt"] += 1
                bad_files.append((cls_dir.name, img_path.name))
                continue

            # Kích thước tối thiểu
            h, w = img.shape[:2]
            if h < MIN_CROP_H or w < MIN_CROP_W:
                stats["too_small"] += 1
                bad_files.append((cls_dir.name, img_path.name))
                continue

            # Chất lượng ảnh
            ok, reason = check_image_quality(img)
            if not ok:
                if "dark" in reason or "bright" in reason:
                    stats["too_dark_bright"] += 1
                else:
                    stats["blurry"] += 1
                bad_files.append((cls_dir.name, img_path.name))
                continue

            class_counter[cls_id] += 1

    stats["good"] = stats["total_images"] - len(bad_files)
    return stats, bad_files, class_counter


def copy_clean_classification(split, bad_files):
    """Copy ảnh sạch giữ nguyên cấu trúc thư mục lớp."""
    bad_set = set((c, f) for c, f in bad_files)

    for cls_dir in (Path(CLASSIFY_ROOT) / split).iterdir():
        if not cls_dir.is_dir():
            continue
        dst_dir = Path(CLASSIFY_CLEAN) / split / cls_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in cls_dir.glob("*.jpg"):
            if (cls_dir.name, img_path.name) not in bad_set:
                shutil.copy2(img_path, dst_dir / img_path.name)


# ════════════════════════════════════════════
# VISUALIZE
# ════════════════════════════════════════════

def visualize_jersey_stats(class_counter, blur_scores,
                            title="Jersey Detection"):
    os.makedirs("analysis/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # 1. Phân bố class (số áo)
    ax = axes[0]
    if class_counter:
        nums   = sorted(class_counter.keys())
        counts = [class_counter[n] for n in nums]
        ax.bar(nums, counts, color='#3498db',
               edgecolor='white', linewidth=0.3)
        ax.set_title("Phân bố số lượng mẫu theo số áo")
        ax.set_xlabel("Số áo")
        ax.set_ylabel("Số ảnh")

        # Highlight số áo phổ biến vs hiếm
        mean_c = np.mean(counts)
        ax.axhline(mean_c, color='red', linestyle='--',
                   linewidth=1.5,
                   label=f'Mean = {mean_c:.0f}')
        ax.legend(fontsize=8)

    # 2. Histogram blur scores
    ax = axes[1]
    if blur_scores:
        ax.hist(blur_scores, bins=40, color='#9b59b6',
                edgecolor='white', linewidth=0.3)
        ax.axvline(MAX_BLUR, color='red', linestyle='--',
                   linewidth=2,
                   label=f'Ngưỡng loại (< {MAX_BLUR})')
        n_blurry = sum(1 for b in blur_scores if b < MAX_BLUR)
        ax.set_title(f"Phân bố độ sắc nét ảnh\n"
                     f"({n_blurry} ảnh mờ sẽ bị loại)")
        ax.set_xlabel("Laplacian Variance (càng cao càng nét)")
        ax.set_ylabel("Số ảnh")
        ax.legend(fontsize=8)

    # 3. Phân bố theo nhóm số áo (1-9, 10-19, ...)
    ax = axes[2]
    if class_counter:
        groups  = [f"{i*10+1}-{(i+1)*10}" for i in range(0, 10)]
        g_counts= []
        for i in range(0, 10):
            lo, hi = i * 10 + 1, (i + 1) * 10
            g_counts.append(sum(class_counter.get(n, 0)
                                for n in range(lo, hi + 1)))
        ax.bar(range(10), g_counts, color='#e67e22',
               edgecolor='white')
        ax.set_xticks(range(10))
        ax.set_xticklabels(groups, rotation=45, fontsize=8)
        ax.set_title("Phân bố theo nhóm số áo")
        ax.set_ylabel("Số mẫu")

        under = sum(1 for c in class_counter.values() if c < 5)
        ax.set_xlabel(
            f"⚠️  {under} số áo có dưới 5 mẫu (cần tăng cường)",
            fontsize=9, color='red')

    plt.tight_layout()
    safe_title = title.replace(" ", "_").lower()
    out = f"analysis/figures/jersey_stats_{safe_title}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {out}")


def visualize_jersey_samples(split="train", n_per_class=3,
                              n_classes=10):
    """
    Hiển thị mẫu ảnh từ dataset classification để kiểm tra
    chất lượng trực quan (trước khi làm sạch).
    """
    split_dir = Path(CLASSIFY_ROOT) / split
    if not split_dir.exists():
        print("Không có dataset classification, bỏ qua.")
        return

    class_dirs = sorted([d for d in split_dir.iterdir()
                          if d.is_dir()])[:n_classes]

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.5, n_classes * 2))
    fig.suptitle(f"Jersey Classification – Mẫu ảnh [{split}]",
                 fontsize=12, fontweight='bold')

    for row, cls_dir in enumerate(class_dirs):
        imgs = list(cls_dir.glob("*.jpg"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(imgs):
                img = cv2.imread(str(imgs[col]))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Scale lên để dễ xem
                    img_rgb = cv2.resize(img_rgb, (60, 80))

                    # Check chất lượng
                    ok, reason = check_image_quality(img)
                    ax.imshow(img_rgb)
                    color = 'green' if ok else 'red'
                    label = f"#{cls_dir.name}"
                    if not ok:
                        label += f"\n✗"
                    ax.set_title(label, fontsize=7,
                                  color=color)

    plt.tight_layout()
    out = "analysis/figures/jersey_classification_samples.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {out}")


def compare_class_distribution(split="train"):
    """
    So sánh phân bố class trước và sau làm sạch
    để chứng minh không bị lệch.
    """
    def count_classes(root, fmt="detection"):
        counter = Counter()
        split_dir = Path(root) / split
        if not split_dir.exists():
            return counter
        if fmt == "detection":
            for lbl in (split_dir / "labels").glob("*.txt"):
                for line in lbl.read_text().strip().splitlines():
                    p = line.split()
                    if p:
                        counter[int(p[0])] += 1
        else:  # classification
            for cls_dir in split_dir.iterdir():
                if cls_dir.is_dir():
                    try:
                        cls_id = int(cls_dir.name)
                        counter[cls_id] = len(
                            list(cls_dir.glob("*.jpg")))
                    except ValueError:
                        pass
        return counter

    before = count_classes(DETECTION_ROOT, "detection")
    after  = count_classes(DETECTION_CLEAN, "detection")

    if not before:
        return

    common_ids = sorted(set(before) | set(after))
    b_vals = [before.get(i, 0) for i in common_ids]
    a_vals = [after.get(i, 0)  for i in common_ids]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"So sánh phân bố class Jersey [{split}]"
                 f" – Trước / Sau làm sạch",
                 fontsize=12, fontweight='bold')

    for ax, vals, label, color in zip(
            axes,
            [b_vals, a_vals],
            ["TRƯỚC làm sạch", "SAU làm sạch"],
            ["#e74c3c", "#2ecc71"]):
        ax.bar(common_ids, vals, color=color,
               edgecolor='white', linewidth=0.3, width=0.8)
        ax.set_title(f"{label}\n"
                     f"(Tổng: {sum(vals)} mẫu, "
                     f"{len([v for v in vals if v > 0])} lớp)")
        ax.set_xlabel("Số áo")
        ax.set_ylabel("Số mẫu")
        ax.axhline(np.mean([v for v in vals if v > 0]),
                   color='black', linestyle='--', linewidth=1,
                   label='Mean')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = "analysis/figures/jersey_class_dist_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  JERSEY NUMBER DATASET CLEANING")
    print("=" * 55)

    # ── PHẦN A: Detection format ─────────────────────────────
    has_detection = Path(DETECTION_ROOT).exists()
    if has_detection:
        print("\n📦 Phần A: YOLO Detection Format")
        all_class_counter = Counter()
        all_blur_scores   = []
        before_stats_det  = {}
        all_bad_det       = {}

        for split in SPLITS:
            stats, bad, cls_cnt, blurs = \
                scan_jersey_detection(split)
            before_stats_det[split] = stats
            all_bad_det[split]      = bad
            all_class_counter.update(cls_cnt)
            all_blur_scores.extend(blurs)
            print(f"\n[{split}]")
            for k, v in stats.items():
                if v > 0 or k in ('total', 'good'):
                    print(f"  {k:<25}: {v}")

        print("\n🧹 Fix và copy dữ liệu detection sạch...")
        for split in SPLITS:
            fix_jersey_detection(split, all_bad_det[split])

        visualize_jersey_stats(
            all_class_counter, all_blur_scores,
            title="Jersey Detection Dataset")
        compare_class_distribution(split="train")

    # ── PHẦN B: Classification format ────────────────────────
    has_classify = Path(CLASSIFY_ROOT).exists()
    if has_classify:
        print("\n📦 Phần B: Classification Format")
        all_cls_cls = Counter()
        before_stats_cls = {}
        all_bad_cls      = {}

        for split in SPLITS:
            stats, bad, cls_cnt = \
                scan_jersey_classification(split)
            before_stats_cls[split] = stats
            all_bad_cls[split]      = bad
            all_cls_cls.update(cls_cnt)
            print(f"\n[{split}]")
            for k, v in stats.items():
                if v > 0 or k in ('total_images', 'good'):
                    print(f"  {k:<25}: {v}")

        print("\n🧹 Copy dữ liệu classification sạch...")
        for split in SPLITS:
            copy_clean_classification(split, all_bad_cls[split])

        visualize_jersey_stats(
            all_cls_cls, [],
            title="Jersey Classification Dataset")
        visualize_jersey_samples(split="train",
                                  n_per_class=3, n_classes=10)

    if not has_detection and not has_classify:
        print("⚠️  Không tìm thấy dataset nào.")
        print(f"   Cần có: {DETECTION_ROOT} hoặc {CLASSIFY_ROOT}")

    print("\n✅ Hoàn thành làm sạch jersey dataset.")