import os
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# Khởi tạo model
model = YOLO('models/pitch_keypoint_detector.pt')

head = model.model.model[-1]
if not hasattr(head, 'detect'):
    head.detect = Detect.forward

results_generator = model.predict(
    'input_videos/sample2.mp4',
    save=True,
    project='output_videos',
    name='inference',
    exist_ok=True,
    stream=True,
    device=0 if torch.cuda.is_available() else 'cpu',
    verbose=False
)

print("Đang tiến hành xử lý video bằng GPU (CUDA)...")

for frame_idx, r in enumerate(results_generator):
   
    if (frame_idx + 1) % 50 == 0:
        print(f"Đã xử lý thành công: {frame_idx + 1} khung hình...")
        
print("\n=====================================")
print("Đã xử lý và lưu video thành công vào thư mục output_videos")