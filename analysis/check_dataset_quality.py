# analysis/check_dataset_quality.py
# Chạy: python analysis/check_dataset_quality.py
# Output: analysis/reports/dataset_quality_report.txt

import os, cv2
import numpy as np
from pathlib import Path

DATASET_ROOT = "training/football-players-detection-1"
SPLITS       = ["train", "valid", "test"]
NUM_CLASSES  = 4
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]

def check_split(split):
    img_dir = Path(DATASET_ROOT) / split / "images"
    lbl_dir = Path(DATASET_ROOT) / split / "labels"

    issues = {
        "corrupt_images"    : [],
        "missing_labels"    : [],
        "empty_labels"      : [],
        "invalid_class"     : [],
        "bbox_out_of_bounds": [],
        "malformed_lines"   : [],
    }

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total = len(img_files)

    for img_path in img_files:
        stem = img_path.stem

        # 1. Ảnh có đọc được không?
        img = cv2.imread(str(img_path))
        if img is None:
            issues["corrupt_images"].append(img_path.name)
            continue

        # 2. Có file label không?
        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            issues["missing_labels"].append(img_path.name)
            continue

        # 3. Label rỗng (background image)?
        lines = lbl_path.read_text().strip().splitlines()
        if len(lines) == 0:
            issues["empty_labels"].append(img_path.name)
            continue

        # 4. Kiểm tra từng dòng label
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                issues["malformed_lines"].append(f"{img_path.name} line {i+1}")
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # Class ID hợp lệ?
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                issues["invalid_class"].append(
                    f"{img_path.name}: class_id={cls_id}")

            # Bbox nằm trong [0,1]?
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and
                   0 <  bw <= 1 and 0 <  bh <= 1):
                issues["bbox_out_of_bounds"].append(
                    f"{img_path.name}: [{cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}]")

    return total, issues

# ---- Chạy kiểm tra tất cả split ----
os.makedirs("analysis/reports", exist_ok=True)
report_lines = ["=" * 60,
                "  FOOTBALL DATASET – QUALITY REPORT",
                "=" * 60, ""]

total_issues = 0
for split in SPLITS:
    total, issues = check_split(split)
    n_issues = sum(len(v) for v in issues.values())
    total_issues += n_issues

    report_lines.append(f"[{split.upper()}]  {total} ảnh  |  {n_issues} vấn đề")
    report_lines.append("-" * 40)
    for issue_type, items in issues.items():
        status = "[OK]" if len(items) == 0 else "[WARN]"
        report_lines.append(f"  {status} {issue_type:<22}: {len(items)}")
        if items:
            for item in items[:3]:  # chỉ in 3 ví dụ đầu
                report_lines.append(f"      → {item}")
            if len(items) > 3:
                report_lines.append(f"      ... và {len(items)-3} trường hợp khác")
    report_lines.append("")

report_lines.append(f"TỔNG VẤN ĐỀ: {total_issues}")
report_text = "\n".join(report_lines)

with open("analysis/reports/dataset_quality_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print("\nSaved: analysis/reports/dataset_quality_report.txt")