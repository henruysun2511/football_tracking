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
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
        dataset = project.version(2).download("yolov8")
        data_yaml_location = f"{dataset.location}/data.yaml"
    else:
        data_yaml_location = str(yaml_path)

    model = YOLO("yolov8s.pt")

    os.makedirs("models/player_detector_s", exist_ok=True)

    results = model.train(
        data=data_yaml_location,
        epochs=100,
        imgsz=640,
        batch=2,
        device=0,
        project="models/player_detector_s",
        name=".",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        workers=0,
    )

    print("Training complete!")


if __name__ == '__main__':
    main()
