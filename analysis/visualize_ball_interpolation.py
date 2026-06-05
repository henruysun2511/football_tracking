# analysis/visualize_ball_interpolation.py

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load stub tracks đã được tính sẵn
with open('stubs/track_stubs.pkl', 'rb') as f:
    tracks = pickle.load(f)

# Trích xuất bbox trung tâm của bóng theo từng frame
ball_positions_raw = []
for frame_num, frame_ball in enumerate(tracks['ball']):
    if 1 in frame_ball:
        bbox = frame_ball[1]['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        ball_positions_raw.append((float(cx), float(cy)))
    else:
        ball_positions_raw.append((np.nan, np.nan))  # frame thiếu detection

# Thực hiện interpolation (giống tracker.interpolate_ball_positions)
df = pd.DataFrame(ball_positions_raw, columns=['cx', 'cy'])
df_interp = df.interpolate(method='linear').bfill()

# Lấy 150 frame đầu để visualize
N = 150
frames = np.arange(N)
raw_cx = df['cx'].values[:N]
interp_cx = df_interp['cx'].values[:N]
missing_mask = np.isnan(raw_cx)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle("Tiền xử lý vị trí bóng: Trước & Sau Interpolation", fontsize=13, fontweight='bold')

# Trước interpolation
ax = axes[0]
ax.plot(frames[~missing_mask], raw_cx[~missing_mask], 'b-o', markersize=3, label='Detected')
ax.scatter(frames[missing_mask], np.zeros(missing_mask.sum()),
           color='red', marker='x', s=50, zorder=5, label='Missing (không detect được)')
ax.set_title(f"TRƯỚC interpolation – {missing_mask.sum()} frames thiếu ({missing_mask.sum()/N*100:.1f}%)")
ax.set_ylabel("Tọa độ X trung tâm bóng (pixel)")
ax.legend(); ax.grid(alpha=0.3)

# Sau interpolation
ax = axes[1]
ax.plot(frames, interp_cx, 'g-', linewidth=1.5, label='Sau interpolation')
ax.scatter(frames[~missing_mask], raw_cx[~missing_mask],
           color='blue', s=15, zorder=5, label='Detected (giữ nguyên)')
ax.scatter(frames[missing_mask], interp_cx[missing_mask],
           color='orange', s=20, marker='D', zorder=5, label='Interpolated (lấp đầy)')
ax.set_title("SAU interpolation – tất cả frames có vị trí bóng liên tục")
ax.set_xlabel("Frame số"); ax.set_ylabel("Tọa độ X trung tâm bóng (pixel)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("analysis/figures/ball_interpolation.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Đã lưu: analysis/figures/ball_interpolation.png")
print(f"   Tổng frames: {N} | Frames thiếu: {missing_mask.sum()} ({missing_mask.sum()/N*100:.1f}%)")