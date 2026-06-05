# analysis/verify_data_split.py

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DATASET_ROOT = "training/football-players-detection-1"
SPLITS       = ["train", "valid", "test"]
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]
SPLIT_COLORS = ['#3498db', '#2ecc71', '#e74c3c']

split_data = {}
for split in SPLITS:
    img_dir = Path(DATASET_ROOT) / split / "images"
    lbl_dir = Path(DATASET_ROOT) / split / "labels"
    n_imgs  = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    cls_cnt = Counter()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            p = line.split()
            if p: cls_cnt[int(p[0])] += 1
    split_data[split] = {'n_imgs': n_imgs, 'cls_cnt': cls_cnt}

# ---- Bảng tóm tắt ----
total_imgs = sum(v['n_imgs'] for v in split_data.values())
print("=" * 65)
print("  PHÂN CHIA DỮ LIỆU – DATA SPLIT REPORT")
print("=" * 65)
print(f"{'Split':<8} {'#Images':>8} {'Ratio':>8}"
      + "".join(f"  {c:>10}" for c in CLASS_NAMES))
print("-" * 65)
for split in SPLITS:
    d   = split_data[split]
    row = f"{split:<8} {d['n_imgs']:>8} {d['n_imgs']/total_imgs:>7.1%}"
    for i in range(4):
        row += f"  {d['cls_cnt'].get(i,0):>10}"
    print(row)
print("-" * 65)
print(f"{'TOTAL':<8} {total_imgs:>8} {'100.0%':>8}")

# ---- Visualize ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phân chia dữ liệu Train / Val / Test",
             fontsize=13, fontweight='bold')

# Pie chart số ảnh
ax = axes[0]
sizes  = [split_data[s]['n_imgs'] for s in SPLITS]
labels = [f"{s}\n({n} ảnh)" for s, n in zip(SPLITS, sizes)]
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       colors=SPLIT_COLORS, startangle=90,
       wedgeprops=dict(edgecolor='white', linewidth=2))
ax.set_title("Số lượng ảnh theo split")

# Grouped bar: phân bố class trong mỗi split
ax = axes[1]
x     = np.arange(len(CLASS_NAMES))
width = 0.25
for i, (split, color) in enumerate(zip(SPLITS, SPLIT_COLORS)):
    counts = [split_data[split]['cls_cnt'].get(c, 0) for c in range(4)]
    ax.bar(x + i * width, counts, width, label=split.capitalize(),
           color=color, edgecolor='white', linewidth=0.5)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES)
ax.set_title("Phân bố class theo split\n(kiểm tra stratified split)")
ax.set_ylabel("Số bounding box")
ax.legend(); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/data_split.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis/figures/data_split.png")