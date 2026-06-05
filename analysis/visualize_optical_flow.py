# analysis/visualize_optical_flow.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video

# Tham số giống camera_movement_estimator.py gốc
FEATURES_PARAMS = dict(
    maxCorners=100, qualityLevel=0.3,
    minDistance=3, blockSize=7,
    mask=None  # sẽ được tạo dưới
)
LK_PARAMS = dict(
    winSize=(15, 15), maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

video_frames = read_video('input_videos/08fd33_4.mp4')
frame1 = video_frames[0]
frame2 = video_frames[1]

# Tạo mask che vùng giữa sân (tập trung vào cột trái/phải – ít cầu thủ, nhiều đặc trưng sân)
h, w = frame1.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
mask[:40, :] = 255          # dải trên
mask[h-40:, :] = 255       # dải dưới

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Tìm feature points trong frame 1
FEATURES_PARAMS['mask'] = mask
old_pts = cv2.goodFeaturesToTrack(gray1, **FEATURES_PARAMS)

# Lucas-Kanade Optical Flow
new_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, old_pts, None, **LK_PARAMS)

# Lọc các điểm track thành công
good_old = old_pts[status == 1]
good_new = new_pts[status == 1]

# Tính vector chuyển động camera
dx = np.median(good_new[:, 0] - good_old[:, 0])
dy = np.median(good_new[:, 1] - good_old[:, 1])

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Lucas-Kanade Optical Flow – Ước lượng chuyển động Camera", fontsize=13, fontweight='bold')

# Frame 1 với feature points
vis1 = cv2.cvtColor(frame1.copy(), cv2.COLOR_BGR2RGB)
for pt in good_old:
    cv2.circle(vis1, (int(pt[0]),int(pt[1])), 4, (0,255,0), -1)
axes[0].imshow(vis1); axes[0].set_title(f'Frame t – {len(good_old)} feature points'); axes[0].axis('off')

# Frame 2 với optical flow vectors
vis2 = cv2.cvtColor(frame2.copy(), cv2.COLOR_BGR2RGB)
for pt_old, pt_new in zip(good_old, good_new):
    x0,y0 = int(pt_old[0]),int(pt_old[1])
    x1,y1 = int(pt_new[0]),int(pt_new[1])
    cv2.arrowedLine(vis2, (x0,y0), (x1,y1), (255,50,50), 2, tipLength=0.4)
axes[1].imshow(vis2); axes[1].set_title(f'Frame t+1 – Motion vectors\n(Camera: Δx={dx:.1f}px, Δy={dy:.1f}px)'); axes[1].axis('off')

# Histogram độ lớn chuyển động
magnitudes = np.sqrt((good_new[:,0]-good_old[:,0])**2 + (good_new[:,1]-good_old[:,1])**2)
axes[2].hist(magnitudes, bins=20, color='#e67e22', edgecolor='white')
axes[2].axvline(np.median(magnitudes), color='red', linestyle='--', label=f'Median = {np.median(magnitudes):.1f}px')
axes[2].set_title('Phân bố độ lớn chuyển động (magnitude)')
axes[2].set_xlabel('Magnitude (pixels)')
axes[2].legend()

plt.tight_layout()
plt.savefig("analysis/figures/optical_flow.png", dpi=150, bbox_inches='tight')
plt.show()