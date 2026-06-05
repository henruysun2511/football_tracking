# analysis/visualize_team_assignment.py

import cv2, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_player_crop(frame, bbox, top_half_only=True):
    """Crop vùng cầu thủ, lấy nửa trên (vùng áo)"""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if top_half_only and crop.shape[0] > 10:
        crop = crop[:crop.shape[0]//2, :]
    return crop

def segment_player_color(crop):
    """Tách màu áo khỏi nền bằng KMeans 2 cụm"""
    img_2d = crop.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(img_2d)
    labels = kmeans.labels_.reshape(crop.shape[:2])
    
    # Xác định cluster nào là "nền" (góc ảnh thường là nền)
    corner_labels = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
    bg_cluster = max(set(corner_labels), key=corner_labels.count)
    player_cluster = 1 - bg_cluster
    
    player_color = kmeans.cluster_centers_[player_cluster].astype(np.uint8)
    mask = (labels == player_cluster).astype(np.uint8) * 255
    return player_color, mask, kmeans.cluster_centers_

# Load video frame và track data
from utils import read_video
video_frames = read_video('input_videos/08fd33_4.mp4')
with open('stubs/track_stubs.pkl', 'rb') as f:
    tracks = pickle.load(f)

# Lấy 4 cầu thủ đầu từ frame 0
frame = video_frames[0]
player_ids = list(tracks['players'][0].keys())[:4]

fig, axes = plt.subplots(4, 4, figsize=(14, 10))
fig.suptitle("KMeans – Phân tách màu áo cầu thủ để nhận diện đội", fontsize=13, fontweight='bold')

for i, pid in enumerate(player_ids):
    bbox = tracks['players'][0][pid]['bbox']
    crop = get_player_crop(frame, bbox)
    if crop.size == 0: continue
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    player_color, mask, centers = segment_player_color(crop)
    
    # Col 0: ảnh crop gốc
    axes[i,0].imshow(crop_rgb); axes[i,0].set_title(f'Player {pid} – Crop', fontsize=8)
    # Col 1: mask phân tách
    axes[i,1].imshow(mask, cmap='gray'); axes[i,1].set_title('Mask (KMeans)', fontsize=8)
    # Col 2: màu áo trích xuất được
    color_img = np.ones((60,60,3), dtype=np.uint8) * player_color[[2,1,0]]
    axes[i,2].imshow(color_img)
    axes[i,2].set_title(f'Màu áo RGB{tuple(player_color[[2,1,0]])}', fontsize=7)
    # Col 3: 2 cluster centers trong không gian màu
    axes[i,3].bar(['Cluster 0\n(nền)','Cluster 1\n(áo)'],
                 [np.mean(centers[0]), np.mean(centers[1])],
                 color=[centers[0]/[255], centers[1]/[255]])
    axes[i,3].set_title('Cluster Centers', fontsize=8)
    for ax in axes[i]: ax.axis('off')

plt.tight_layout()
plt.savefig("analysis/figures/kmeans_team_assignment.png", dpi=150, bbox_inches='tight')
plt.show()