# analysis/visualize_training_results.py
# Yêu cầu: YOLO phải đã train xong, có file runs/detect/train/results.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Đường dẫn file results.csv từ YOLO training
RESULTS_CSV = "runs/detect/train/results.csv"

if not os.path.exists(RESULTS_CSV):
    print("Chưa có file training results. Tạo dữ liệu demo...")
    # Tạo dữ liệu mẫu để demo khi chưa train
    epochs = 50
    df = pd.DataFrame({
        'epoch': range(1, epochs+1),
        'train/box_loss': 1.5 * np.exp(-np.linspace(0,3,epochs)) + 0.05*np.random.randn(epochs).clip(-0.1,0.1),
        'val/box_loss':   1.7 * np.exp(-np.linspace(0,2.5,epochs)) + 0.08*np.random.randn(epochs).clip(-0.1,0.1),
        'metrics/mAP50':   np.clip(0.92 * (1-np.exp(-np.linspace(0,4,epochs))), 0, 1),
        'metrics/mAP50-95': np.clip(0.68 * (1-np.exp(-np.linspace(0,4,epochs))), 0, 1),
        'metrics/precision': np.clip(0.89 * (1-np.exp(-np.linspace(0,3.5,epochs))), 0, 1),
        'metrics/recall': np.clip(0.87 * (1-np.exp(-np.linspace(0,3,epochs))), 0, 1),
    })
else:
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("YOLO Fine-tuning – Training Results & Model Evaluation", fontsize=14, fontweight='bold')

# 1. Loss curves
ax = axes[0,0]
ax.plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Loss')
ax.plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Loss')
ax.set_title("Bounding Box Regression Loss"); ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

# 2. mAP curves
ax = axes[0,1]
ax.plot(df['epoch'], df['metrics/mAP50'], 'g-', linewidth=2, label='mAP@0.5')
ax.plot(df['epoch'], df['metrics/mAP50-95'], 'orange', linewidth=2, linestyle='--', label='mAP@0.5:0.95')
best_map50 = df['metrics/mAP50'].max()
ax.axhline(best_map50, color='green', linestyle=':', alpha=0.7, label=f'Best mAP@0.5 = {best_map50:.3f}')
ax.set_title("mAP (Mean Average Precision)"); ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.set_ylim(0, 1.05)

# 3. Precision & Recall
ax = axes[1,0]
ax.plot(df['epoch'], df['metrics/precision'], 'purple', label='Precision')
ax.plot(df['epoch'], df['metrics/recall'], 'teal', linestyle='--', label='Recall')
ax.set_title("Precision & Recall"); ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Epoch"); ax.set_ylim(0, 1.05)

# 4. Bảng kết quả cuối (per-class nếu có)
# Chạy: model.val() để lấy per-class metrics
ax = axes[1,1]
classes = ['ball', 'goalkeeper', 'player', 'referee']
# Thay bằng giá trị thực từ model.val() sau khi train
map50_per_class = [0.71, 0.88, 0.94, 0.85]
colors_cls = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
bars = ax.barh(classes, map50_per_class, color=colors_cls, edgecolor='white')
for bar, val in zip(bars, map50_per_class):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontweight='bold')
ax.set_xlim(0, 1.1); ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title("mAP@0.5 theo từng class"); ax.set_xlabel("mAP@0.5")
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("analysis/figures/training_results.png", dpi=150, bbox_inches='tight')
plt.show()

# In bảng tổng hợp metrics cuối
print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Epoch cuối) ===")
last = df.iloc[-1]
print(f"  mAP@0.5        : {last['metrics/mAP50']:.4f}")
print(f"  mAP@0.5:0.95   : {last['metrics/mAP50-95']:.4f}")
print(f"  Precision      : {last['metrics/precision']:.4f}")
print(f"  Recall         : {last['metrics/recall']:.4f}")