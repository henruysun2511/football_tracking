import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


def main():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    dataset_dir = "datasets/football-shirt-number-1"
    yaml_path = Path(os.getcwd()) / dataset_dir / "data.yaml"

    if not yaml_path.exists():
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("lellosss").project("football-shirt-number")
        dataset = project.version(1).download("yolov8")
        data_yaml_location = f"{dataset.location}/data.yaml"
    else:
        data_yaml_location = str(yaml_path)

    model = YOLO("yolov8n.pt")

    os.makedirs("models/jersey_detector", exist_ok=True)

    results = model.train(
        data=data_yaml_location,
        epochs=100,
        imgsz=640,
        batch=2,
        device=0,
        project="models/jersey_detector",
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
