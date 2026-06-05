import os
from ultralytics import YOLO 

# Khởi tạo model
model = YOLO('models/player_detector.pt')

results_generator = model.predict(
    'input_videos/sample.mp4',
    save=True,
    project='output_videos',
    name='.',
    exist_ok=True,
    stream=True,
    device=0,
    verbose=False 
)

print("Đang tiến hành xử lý video bằng GPU (CUDA)...")

for frame_idx, r in enumerate(results_generator):
   
    if (frame_idx + 1) % 50 == 0:
        print(f"Đã xử lý thành công: {frame_idx + 1} khung hình...")
        
print("\n=====================================")
print("Đã xử lý và lưu video thành công vào thư mục output_videos")