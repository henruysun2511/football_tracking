# analysis/visualize_perspective.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

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
    warped = cv2.warpPerspective(frame, M, (length // 10, width // 10))
    ax.imshow(warped)
    ax.set_title(f"Sau Perspective Transform (Bird's eye view)\n"
                 f"Sân {length/100:.0f}m × {width/100:.0f}m (tỉ lệ 1:10)")
else:
    ax.text(0.5, 0.5, 'Không đủ keypoints để tính homography\n(cần >= 4 keypoints)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig("analysis/figures/perspective_transform.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: analysis/figures/perspective_transform.png")