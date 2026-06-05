# analysis/visualize_perspective.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video

# 4 điểm góc vùng sân trong ảnh (giống view_transformer.py gốc)
# Đây là tọa độ pixel của 4 góc vùng sân đã định nghĩa thủ công
PIXEL_VERTICES = np.array([
    [115, 550],   # góc trên-trái
    [1280, 550],  # góc trên-phải
    [915, 970],   # góc dưới-phải
    [430, 970],   # góc dưới-trái
], dtype=np.float32)

# Kích thước thực tế vùng sân tương ứng (mét)
# Dựa trên FIFA standard: đường 16.5m, chiều ngang ~68m
TARGET_W, TARGET_H = 68, 23.32  # mét
TARGET_VERTICES = np.array([
    [0, 0],
    [TARGET_W, 0],
    [TARGET_W, TARGET_H],
    [0, TARGET_H],
], dtype=np.float32)

# Tính perspective transform matrix
M, _ = cv2.findHomography(PIXEL_VERTICES, TARGET_VERTICES * 10)

video_frames = read_video('input_videos/08fd33_4.mp4')
frame = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2RGB)

# Warp toàn ảnh (chỉ để visualize)
warped = cv2.warpPerspective(frame, M, (680, 233))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Perspective Transformation – Chuyển view camera → Top-down (bird's eye)",
             fontsize=12, fontweight='bold')

# Frame gốc + polygon vùng sân
ax = axes[0]
ax.imshow(frame)
poly = plt.Polygon(PIXEL_VERTICES, fill=False, edgecolor='yellow', linewidth=3)
ax.add_patch(poly)
for i, (x,y) in enumerate(PIXEL_VERTICES):
    ax.annotate(f'P{i+1}\n({x:.0f},{y:.0f})', (x,y), fontsize=9,
                color='yellow', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
ax.set_title("Ảnh gốc (góc camera nghiêng)\nVùng vàng = vùng sân cần transform")
ax.axis('off')

# Ảnh sau transform
ax = axes[1]
ax.imshow(warped)
ax.set_title(f"Sau Perspective Transform (Bird's eye view)\n"
             f"Tương ứng {TARGET_W}m × {TARGET_H:.1f}m sân thực tế")
for i, (x, y) in enumerate(TARGET_VERTICES * 10):
    ax.annotate(f'P{i+1}', (x,y), fontsize=9, color='yellow', ha='center')
ax.axis('off')

plt.tight_layout()
plt.savefig("analysis/figures/perspective_transform.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis/figures/perspective_transform.png")