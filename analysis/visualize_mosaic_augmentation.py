# analysis/visualize_mosaic_augmentation.py

import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_ROOT = "datasets/football-players-detection-2"
CLASS_NAMES  = ["ball", "goalkeeper", "player", "referee"]
OUTPUT_SIZE  = 640
COLORS       = [(255,50,50), (0,200,255), (50,200,50), (255,160,0)]

def load_image_and_labels(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    labels = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) == 5:
                labels.append(list(map(float, p)))
    return img, labels

def draw_boxes(img, labels):
    h, w = img.shape[:2]
    out = img.copy()
    for lbl in labels:
        c = int(lbl[0]); cx,cy,bw,bh = lbl[1:]
        x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
        x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
        col = COLORS[min(c, 3)]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        cv2.putText(out, CLASS_NAMES[min(c,3)], (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return out

def create_mosaic(img_paths, lbl_paths, out_size=640):
    """Tái tạo Mosaic Augmentation: ghép 4 ảnh thành 1"""
    half = out_size // 2
    # Điểm giao ngẫu nhiên (không phải chính giữa)
    cx = random.randint(half // 2, half + half // 2)
    cy = random.randint(half // 2, half + half // 2)

    mosaic = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
    all_labels = []

    positions = [(0,0, cx, cy),       # top-left
                 (cx, 0, out_size, cy),  # top-right
                 (0, cy, cx, out_size),  # bottom-left
                 (cx, cy, out_size, out_size)]  # bottom-right

    for i, (ip, lp) in enumerate(zip(img_paths, lbl_paths)):
        img, labels = load_image_and_labels(ip, lp)
        x1d,y1d,x2d,y2d = positions[i]
        tile_w = x2d - x1d; tile_h = y2d - y1d
        tile = cv2.resize(img, (tile_w, tile_h))
        mosaic[y1d:y2d, x1d:x2d] = tile

        # Chuyển đổi labels về tọa độ mosaic
        for lbl in labels:
            c, lcx, lcy, lw, lh = lbl
            # cx,cy trong tile → cx,cy trong mosaic
            new_cx = (x1d + lcx * tile_w) / out_size
            new_cy = (y1d + lcy * tile_h) / out_size
            new_w  = lw * tile_w / out_size
            new_h  = lh * tile_h / out_size
            # Clip về biên ảnh
            new_cx = np.clip(new_cx, new_w/2, 1-new_w/2)
            new_cy = np.clip(new_cy, new_h/2, 1-new_h/2)
            all_labels.append([c, new_cx, new_cy, new_w, new_h])

    return mosaic, all_labels, cx, cy

# Lấy 4 ảnh ngẫu nhiên từ train
if not Path(DATASET_ROOT).exists():
    print(f"Dataset not found at {DATASET_ROOT}. Skipping.")
    exit(0)
img_dir = Path(DATASET_ROOT) / "train" / "images"
lbl_dir = Path(DATASET_ROOT) / "train" / "labels"
all_imgs = list(img_dir.glob("*.jpg"))
selected = random.sample(all_imgs, 4)
img_ps   = selected
lbl_ps   = [lbl_dir / (p.stem + ".txt") for p in selected]

mosaic_img, mosaic_labels, cut_cx, cut_cy = create_mosaic(
    img_ps, lbl_ps, OUTPUT_SIZE)
mosaic_annotated = draw_boxes(mosaic_img, mosaic_labels)

# Visualize: 4 ảnh gốc + mosaic
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Mosaic Augmentation – Ghép 4 ảnh thành 1 training sample",
             fontsize=13, fontweight='bold')

orig_titles = ["Ảnh 1 (top-left)", "Ảnh 2 (top-right)",
               "Ảnh 3 (bot-left)", "Ảnh 4 (bot-right)"]
for i, (ip, lp) in enumerate(zip(img_ps, lbl_ps)):
    ax = fig.add_subplot(2, 4, i + 1)
    img_orig, lbls = load_image_and_labels(ip, lp)
    ax.imshow(cv2.cvtColor(draw_boxes(img_orig, lbls), cv2.COLOR_BGR2RGB))
    ax.set_title(orig_titles[i], fontsize=8)
    ax.axis('off')

ax_mosaic = fig.add_subplot(1, 2, 2)
ax_mosaic.imshow(cv2.cvtColor(mosaic_annotated, cv2.COLOR_BGR2RGB))
# Vẽ đường giao
ax_mosaic.axhline(cut_cy, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
ax_mosaic.axvline(cut_cx, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
ax_mosaic.set_title(
    f"Mosaic Output (640×640)\n"
    f"{len(mosaic_labels)} bbox | cut point ({cut_cx},{cut_cy})",
    fontsize=9)
ax_mosaic.axis('off')

plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/mosaic_augmentation.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis/figures/mosaic_augmentation.png")
print(f"Mosaic labels: {len(mosaic_labels)} bbox từ 4 ảnh")