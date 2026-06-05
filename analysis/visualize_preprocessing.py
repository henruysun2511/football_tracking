# analysis/visualize_preprocessing.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

DATASET_ROOT = "training/football-players-detection-1"
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
COLORS = {0: (255,0,0), 1: (0,255,255), 2: (0,255,0), 3: (255,128,0)}

def draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]
    result = img.copy()
    if not os.path.exists(label_path):
        return result
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            x1 = int((cx - bw/2) * w);  y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w);  y2 = int((cy + bh/2) * h)
            color = COLORS.get(cls, (255,255,255))
            cv2.rectangle(result, (x1,y1), (x2,y2), color, 2)
            cv2.putText(result, CLASS_NAMES[cls], (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return result

def simulate_augmentations(img):
    """Mô phỏng các bước augmentation Roboflow áp dụng"""
    results = {'Original (640×640)': cv2.resize(img, (640, 640))}
    
    # Horizontal Flip
    results['Horizontal Flip'] = cv2.flip(cv2.resize(img, (640,640)), 1)
    
    # HSV shift (brightness/saturation jitter)
    hsv = cv2.cvtColor(cv2.resize(img, (640,640)), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)  # tăng saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.7, 0, 255)  # giảm brightness
    results['HSV Jitter\n(brightness/saturation)'] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Gaussian Noise
    noisy = cv2.resize(img, (640,640)).astype(np.float32)
    noise = np.random.normal(0, 15, noisy.shape)
    results['Gaussian Noise'] = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    
    # Random Crop + Resize (simulate)
    r = cv2.resize(img, (800,800))
    results['Random Crop\n+ Resize'] = cv2.resize(r[80:720, 80:720], (640,640))
    
    return results

# --- Chọn ảnh mẫu ngẫu nhiên từ train ---
import os
img_dir = Path(DATASET_ROOT) / "train" / "images"
sample_img_path = random.choice(list(img_dir.glob("*.jpg")))
label_path = str(sample_img_path).replace("images", "labels").replace(".jpg", ".txt")

img = cv2.imread(str(sample_img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot augmentations
aug_results = simulate_augmentations(img_rgb)
fig, axes = plt.subplots(1, len(aug_results), figsize=(4*len(aug_results), 4))
fig.suptitle("Tiền xử lý & Augmentation – So sánh trước/sau", fontsize=13, fontweight='bold')
for ax, (title, aug_img) in zip(axes, aug_results.items()):
    ax.imshow(aug_img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig("analysis/figures/augmentation_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot labeled image
img_labeled = draw_yolo_boxes(img_rgb, label_path)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img_rgb); ax1.set_title("Ảnh gốc (raw)"); ax1.axis('off')
ax2.imshow(img_labeled); ax2.set_title("Sau gán nhãn (YOLO format)"); ax2.axis('off')
plt.tight_layout()
plt.savefig("analysis/figures/labeled_sample.png", dpi=150, bbox_inches='tight')
plt.show()