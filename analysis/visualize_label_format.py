# analysis/visualize_label_format.py

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import random

DATASET_ROOT = "datasets/football-players-detection-2"
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]
COLORS_CV    = [(255,50,50), (0,200,255), (50,200,50), (255,160,0)]
COLORS_MPL   = ['#ff3232', '#00c8ff', '#32c832', '#ffa000']

def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

# Chọn ảnh có nhiều class nhất
img_dir = Path(DATASET_ROOT) / "train" / "images"
lbl_dir = Path(DATASET_ROOT) / "train" / "labels"

best_img, best_lbl, best_cls_set = None, None, set()
for lf in random.sample(list(lbl_dir.glob("*.txt")), min(200, len(list(lbl_dir.glob("*.txt"))))):
    lines = lf.read_text().strip().splitlines()
    cls_set = {int(l.split()[0]) for l in lines if l.strip()}
    if len(cls_set) > len(best_cls_set):
        best_cls_set = cls_set
        best_lbl = lf
        best_img = img_dir / (lf.stem + ".jpg")

img_bgr = cv2.imread(str(best_img))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W    = img_rgb.shape[:2]
lines   = best_lbl.read_text().strip().splitlines()

# --- Figure 1: Giải thích YOLO format cho 1 bbox ---
# Lấy bbox đầu tiên để minh họa
demo_line = lines[0].split()
cls_id    = int(demo_line[0])
cx, cy, bw, bh = map(float, demo_line[1:])
x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, W, H)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("YOLO Label Format – Phương pháp gán nhãn Bounding Box",
             fontsize=13, fontweight='bold')

# Ảnh annotated đầy đủ
ax = axes[0]
annotated = img_rgb.copy()
for line in lines:
    p = line.split()
    if len(p) != 5: continue
    c = int(p[0])
    ax1_, ay1_, ax2_, ay2_ = yolo_to_pixel(*map(float, p[1:]), W, H)
    col = COLORS_CV[c]
    cv2.rectangle(annotated, (ax1_, ay1_), (ax2_, ay2_), col, 2)
    cv2.putText(annotated, CLASS_NAMES[c], (ax1_, ay1_ - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
ax.imshow(annotated)
legend_patches = [mpatches.Patch(color=COLORS_MPL[i], label=CLASS_NAMES[i]) for i in range(4)]
ax.legend(handles=legend_patches, fontsize=8, loc='upper right')
ax.set_title(f"Ảnh đã gán nhãn ({len(lines)} bounding boxes)")
ax.axis('off')

# Giải thích format YOLO cho 1 bbox
ax = axes[1]
crop_pad = 80
cx1 = max(0, x1 - crop_pad); cy1 = max(0, y1 - crop_pad)
cx2 = min(W, x2 + crop_pad); cy2 = min(H, y2 + crop_pad)
region = img_rgb[cy1:cy2, cx1:cx2].copy()
ax.imshow(region)

# Vẽ bbox trong vùng crop
rx1, ry1 = x1 - cx1, y1 - cy1
rx2, ry2 = x2 - cx1, y2 - cy1
rect = mpatches.FancyBboxPatch((rx1, ry1), rx2 - rx1, ry2 - ry1,
    boxstyle="square,pad=0", linewidth=2,
    edgecolor=COLORS_MPL[cls_id], facecolor='none')
ax.add_patch(rect)

# Đánh dấu center point
rcx = (rx1 + rx2) / 2; rcy = (ry1 + ry2) / 2
ax.plot(rcx, rcy, 'r+', markersize=14, markeredgewidth=2)
ax.annotate(f"center\n(cx={cx:.3f}, cy={cy:.3f})", (rcx, rcy),
    xytext=(rcx + 15, rcy - 20), fontsize=8, color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=1))

title_txt = (
    f"YOLO Format: class cx cy w h\n"
    f"  → {cls_id} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n"
    f"Class: {CLASS_NAMES[cls_id]}  |  "
    f"Pixel: ({x1},{y1})→({x2},{y2})\n"
    f"Size: {x2-x1}×{y2-y1}px  ({bw*100:.1f}%×{bh*100:.1f}% ảnh)"
)
ax.set_title(title_txt, fontsize=8, family='monospace')
ax.axis('off')

plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/label_format_explanation.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis/figures/label_format_explanation.png")