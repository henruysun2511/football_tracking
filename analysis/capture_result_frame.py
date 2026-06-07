# analysis/capture_result_frame.py
# Chạy sau khi đã có output_videos/output_enhanced.avi

import cv2
import matplotlib.pyplot as plt
import numpy as np

def capture_best_frames(video_path, output_path, n_frames=6):
    """Lấy n_frames đại diện từ output video để đưa vào báo cáo"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps
    
    print(f"Video: {total} frames, {fps:.1f} FPS, {duration:.1f}s")
    
    # Lấy frames trải đều qua video
    frame_indices = np.linspace(10, total-10, n_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    
    # Grid display
    rows = (n_frames + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    fig.suptitle("Football Analysis – Kết quả Demo trên Video",
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten() if rows > 1 else axes
    for i, (idx, frame) in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {idx} ({idx/fps:.1f}s)", fontsize=10)
        axes[i].axis('off')
    for j in range(len(frames), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")

import os

if not os.path.exists('output_videos/output_enhanced.avi'):
    print("No output video found. Run main.py first to generate output_videos/output_enhanced.avi")
    exit(0)

capture_best_frames(
    'output_videos/output_enhanced.avi',
    'analysis/figures/result_showcase.png',
    n_frames=6
)

# Lưu 1 frame chất lượng cao để đưa vào báo cáo
cap = cv2.VideoCapture('output_videos/output_enhanced.avi')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
if ret:
    cv2.imwrite('analysis/figures/best_frame.png', frame)
    print("Saved high-quality frame: analysis/figures/best_frame.png")
cap.release()