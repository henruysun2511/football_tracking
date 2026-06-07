# analysis/class_imbalance_analysis.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "datasets/football-players-detection-2"
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]

def load_all_labels(split):
    lbl_dir = Path(DATASET_ROOT) / split / "labels"
    records = []  # (class_id, cx, cy, w, h)
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            p = line.split()
            if len(p) == 5:
                records.append((int(p[0]), *map(float, p[1:])))
    return records

if not Path(DATASET_ROOT).exists():
    print(f"Dataset not found at {DATASET_ROOT}. Skipping.")
    exit(0)

records = load_all_labels("train")
cls_ids = [r[0] for r in records]
areas   = [r[3] * r[4] for r in records]  # w × h normalized

# Nhóm theo class
cls_areas = defaultdict(list)
for cls, area in zip(cls_ids, areas):
    cls_areas[cls].append(area)

os.makedirs("analysis/figures", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Phân tích Class Imbalance & Kích thước Object",
             fontsize=14, fontweight='bold')
COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

# ---- 1. Số lượng bbox theo class ----
ax = axes[0, 0]
counts = [len(cls_areas.get(i, [])) for i in range(4)]
total  = sum(counts)
bars = ax.bar(CLASS_NAMES, counts, color=COLORS, edgecolor='white')
ax.set_title("Số lượng bbox theo class")
ax.set_ylabel("Số lượng")
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{c}\n({c/total*100:.1f}%)",
            ha='center', fontsize=9, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# ---- 2. Box plot diện tích bbox theo class ----
ax = axes[0, 1]
data_to_plot = [cls_areas.get(i, [0]) for i in range(4)]
bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_title("Phân bố diện tích bbox theo class\n(normalized w×h)")
ax.set_ylabel("Diện tích bbox (w×h)")
ax.set_yscale('log')  # log scale vì ball rất nhỏ
ax.grid(alpha=0.3)

# ---- 3. Scatter: width vs height của bbox ----
ax = axes[1, 0]
for cls_id in range(4):
    sub = [r for r in records if r[0] == cls_id][:300]
    ws  = [r[3] for r in sub]
    hs  = [r[4] for r in sub]
    ax.scatter(ws, hs, c=COLORS[cls_id], alpha=0.4, s=10,
               label=CLASS_NAMES[cls_id])
ax.set_title("Phân bố width × height của bbox")
ax.set_xlabel("Width (normalized)")
ax.set_ylabel("Height (normalized)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)

# ---- 4. Small object analysis ----
THRESHOLDS = [0.001, 0.002, 0.005, 0.01]
ax = axes[1, 1]
x = np.arange(len(THRESHOLDS))
width = 0.2
for i, cls_id in enumerate(range(4)):
    cls_a = cls_areas.get(cls_id, [])
    pcts  = [sum(1 for a in cls_a if a < t) / max(len(cls_a), 1) * 100
             for t in THRESHOLDS]
    ax.bar(x + i * width, pcts, width, color=COLORS[i],
           alpha=0.8, label=CLASS_NAMES[cls_id])
ax.set_title("% bbox là 'small object'\ntheo ngưỡng diện tích")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"<{t*100:.1f}%" for t in THRESHOLDS])
ax.set_ylabel("% trong class")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("analysis/figures/class_imbalance.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- In bảng tóm tắt ----
print("\n=== CLASS IMBALANCE REPORT ===")
print(f"{'Class':<12} {'Count':>7} {'Ratio':>8} {'Avg area':>10} {'% small(<0.2%)':>15}")
print("-" * 58)
for i, name in enumerate(CLASS_NAMES):
    ca  = cls_areas.get(i, [0])
    cnt = len(ca)
    avg = np.mean(ca)
    pct_small = sum(1 for a in ca if a < 0.002) / max(cnt, 1) * 100
    print(f"{name:<12} {cnt:>7} {cnt/total:>7.1%} {avg:>10.5f} {pct_small:>14.1f}%")
print(f"\nImbalance ratio (player/ball): {counts[2]/max(counts[0],1):.1f}x")
print("Gợi ý: Dùng mosaic augmentation + focal loss để giảm ảnh hưởng class imbalance")