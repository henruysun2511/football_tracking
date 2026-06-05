import os
from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project(
        "football-players-detection-3zvbc"
    )
    dataset = project.version(2).download("yolov8")
    print(f"Dataset at: {dataset.location}")


if __name__ == '__main__':
    main()
