import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

def main():
 
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    

    dataset_dir = "datasets/football-players-detection-2"
    yaml_path = Path(os.getcwd()) / dataset_dir / "data.yaml"

    if not yaml_path.exists():
        print("Không tìm thấy dataset cục bộ. Bắt đầu tải từ Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
        dataset = project.version(2).download("yolov8")
        data_yaml_location = f"{dataset.location}/data.yaml"
    else:
        print(f"Dataset đã có sẵn tại thư mục: {dataset_dir}")
        data_yaml_location = str(yaml_path)


    model = YOLO("yolov8x.pt")

    output_project = "models/player_detector"
    if os.path.exists("/content/drive"):
       output_project = "/content/drive/MyDrive/football_models"
       os.makedirs(output_project, exist_ok=True)
    else:
       output_project = "models/player_detector"
       os.makedirs(output_project, exist_ok=True)

    results = model.train(
    data=data_yaml_location,
    epochs=100,
    imgsz=1280,
    batch=8,
    device=0,
    project=output_project,  
    name="player_detector",
    patience=20,
    save=True,
    plots=True,
    workers=2,
)
   
    print("\n==================================================")
    print("Quá trình huấn luyện hoàn tất!")
    

if __name__ == "__main__":
    main()
