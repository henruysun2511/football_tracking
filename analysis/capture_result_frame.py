# analysis/capture_result_frame.py
# Chay sau khi da co output_videos/output_enhanced.avi

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

VIDEO_PATH = 'output_videos/output-5.mp4'
if not os.path.exists(VIDEO_PATH):
    print(f"No output video found at {VIDEO_PATH}. Run main.py first.")
    exit(0)


def capture_best_frames(video_path, output_path, n_frames=6):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps

    print(f"Video: {total} frames, {fps:.1f} FPS, {duration:.1f}s")

    frame_indices = np.linspace(10, total - 10, n_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    rows = (n_frames + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    fig.suptitle("Football Analysis - Ket qua Demo", fontsize=14, fontweight='bold')

    axes = axes.flatten() if rows > 1 else axes
    for i, (idx, frame) in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {idx} ({idx / fps:.1f}s)", fontsize=10)
        axes[i].axis('off')
    for j in range(len(frames), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")


capture_best_frames(VIDEO_PATH, 'analysis/figures/result_showcase.png', n_frames=6)

# Luu 1 frame chat luong cao
cap = cv2.VideoCapture(VIDEO_PATH)
mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
ret, frame = cap.read()
if ret:
    cv2.imwrite('analysis/figures/best_frame.png', frame)
    print("Saved high-quality frame: analysis/figures/best_frame.png")
cap.release()
