# analysis/visualize_perspective.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector

kp_detector = PitchKeypointDetector(
    model_path='models/pitch_keypoint_detector.pt')

video_frames = read_video('input_videos/sample.mp4')
frame = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2RGB)

# Phát hiện keypoints + tính homography
kps = kp_detector.detect_smoothed(video_frames[0])
M = kp_detector.get_homography(kps)

H, W = frame.shape[:2]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Perspective Transformation – Chuyển view camera → Top-down (bird's eye)",
             fontsize=12, fontweight='bold')

# Frame gốc + keypoints
ax = axes[0]
ax.imshow(frame)
if kps is not None:
    xy, confs = kps
    mask = confs > kp_detector.conf
    for (x, y) in xy[mask]:
        ax.plot(x, y, 'yo', markersize=4)
ax.set_title(f"Ảnh gốc ({W}x{H}) — Pitch keypoints dùng để tính homography")
ax.axis('off')

# Ảnh sau transform
length = kp_detector.config.length
width  = kp_detector.config.width
ax = axes[1]
if M is not None:
    # Chiếu keypoints lên field để tìm vùng khả kiến
    xy, confs = kps
    mask = confs > kp_detector.conf
    pts = xy[mask][:, :2]
    if len(pts) >= 4:
        pts_field = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        x_min, y_min = pts_field.min(axis=0)
        x_max, y_max = pts_field.max(axis=0)
        pad_x = (x_max - x_min) * 0.3
        pad_y = (y_max - y_min) * 0.3
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(float(length), x_max + pad_x)
        y_max = min(float(width), y_max + pad_y)
        # Ma trận crop: field -> output pixel
        scale = 10
        T = np.array([[1/scale, 0, -x_min/scale],
                      [0, 1/scale, -y_min/scale],
                      [0, 0, 1]], dtype=np.float32)
        M_crop = T @ M
        out_w = int((x_max - x_min) / scale)
        out_h = int((y_max - y_min) / scale)
        warped = cv2.warpPerspective(frame, M_crop, (max(out_w, 1), max(out_h, 1)))
        ax.imshow(warped)
        ax.set_title(f"Sau Perspective Transform (Bird's eye view)\n"
                     f"Vùng sân {x_min/100:.0f}m-{x_max/100:.0f}m × {y_min/100:.0f}m-{y_max/100:.0f}m")
    else:
        ax.text(0.5, 0.5, 'Không đủ keypoints để tính homography\n(cần >= 4 keypoints)',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
else:
    ax.text(0.5, 0.5, 'Không đủ keypoints để tính homography\n(cần >= 4 keypoints)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
ax.axis('off')

plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/perspective_transform.png", dpi=150, bbox_inches='tight')
print("Saved: analysis/figures/perspective_transform.png")

if os.environ.get('DISPLAY') or os.name == 'nt':
    plt.show()