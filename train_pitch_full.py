"""
Train pitch keypoint detector fine-tuned on your video.
Usage on Colab:
  !python train_pitch_full.py --roboflow-key YOUR_KEY --video input_videos/sample.mp4
"""
import argparse, os, subprocess, sys, shutil, random
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HOME = os.getcwd()
DATASET_DIR = f'{HOME}/training_dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

# ── SoccerPitchConfig (must match dataset keypoint order) ──────────────
# The 32 keypoints in order expected by the model (Roboflow / sports lib)
# DO NOT CHANGE unless you verify the dataset's keypoint order
KEYPOINT_NAMES = [
    "top_left_corner", "left_penalty_box_top", "left_goal_box_top",
    "left_goal_box_bottom", "left_penalty_box_bottom", "bottom_left_corner",
    "left_goal_box_line_top", "left_goal_box_line_bottom",
    "left_penalty_spot", "left_penalty_box_line_top",
    "left_penalty_box_inner_top", "left_penalty_box_inner_bottom",
    "left_penalty_box_line_bottom", "halfway_line_top",
    "center_circle_top", "center_circle_bottom", "halfway_line_bottom",
    "right_penalty_box_line_top", "right_penalty_box_inner_top",
    "right_penalty_box_inner_bottom", "right_penalty_box_line_bottom",
    "right_penalty_spot", "right_goal_box_line_top",
    "right_goal_box_line_bottom", "top_right_corner",
    "right_penalty_box_top", "right_goal_box_top",
    "right_goal_box_bottom", "right_penalty_box_bottom",
    "bottom_right_corner", "center_circle_left", "center_circle_right"
]


def step_download_roboflow(api_key):
    """Download football-field-detection dataset v12 from Roboflow."""
    print("=" * 60)
    print("Step 1: Download Roboflow football-field-detection dataset")
    print("=" * 60)
    os.chdir(DATASET_DIR)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', 'roboflow'
    ], capture_output=True)
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
    version = project.version(12)
    dataset = version.download("yolov8")
    os.chdir(HOME)

    # Fix data.yaml paths
    data_yaml = f'{dataset.location}/data.yaml'
    with open(data_yaml, 'r') as f:
        content = f.read()
    content = content.replace('train: ../train/images', f'train: {dataset.location}/train/images')
    content = content.replace('val: ../valid/images', f'val: {dataset.location}/valid/images')
    with open(data_yaml, 'w') as f:
        f.write(content)
    print(f"Dataset at: {dataset.location}")
    return dataset.location


def step_extract_frames(video_path, output_dir, interval=15):
    """Extract frames from user's video at regular intervals."""
    print("=" * 60)
    print(f"Step 2: Extract frames from {video_path}")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.0f} FPS")

    count = 0
    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        path = f'{output_dir}/frame_{i:06d}.jpg'
        cv2.imwrite(path, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {output_dir}")
    return count


def step_pseudo_label(model_path, image_dir, label_dir, conf_thresh=0.3):
    """Generate pseudo-labels using existing model."""
    print("=" * 60)
    print("Step 3: Generate pseudo-labels with existing model")
    print("=" * 60)
    os.makedirs(label_dir, exist_ok=True)

    model = YOLO(model_path)
    images = sorted(Path(image_dir).glob('*.jpg'))
    total = len(images)
    print(f"Running inference on {total} images...")

    labeled = 0
    for img_path in images:
        results = model(str(img_path), conf=conf_thresh, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            continue
        kps = results[0].keypoints
        if kps is None or len(kps.data) == 0:
            continue

        xy = kps.data[0].cpu().numpy()
        confs = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

        h, w = cv2.imread(str(img_path)).shape[:2]

        # YOLO pose format: class_id x1 y1 x2 y2 kp1_x kp1_y kp1_v ...
        # Use full image as bbox (class_id=0, bbox covering whole image)
        lines = []
        kp_values = []
        for (x, y, v), c in zip(xy, confs):
            visibility = 2 if c > conf_thresh else 1
            kp_values.extend([x / w, y / h, visibility])

        if len(kp_values) > 0:
            # class_id=0, bbox=0,0,1,1 (normalized, covering full image)
            line = f"0 0.5 0.5 1.0 1.0 " + " ".join(f"{v:.6f}" for v in kp_values)
            lines.append(line)

        if lines:
            label_path = f'{label_dir}/{img_path.stem}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))
            labeled += 1

        if labeled % 10 == 0:
            print(f"  Labeled {labeled}/{total}...")

    print(f"Total labeled: {labeled}/{total}")
    return labeled


def step_merge_datasets(roboflow_dir, user_image_dir, user_label_dir,
                        merged_dir):
    """Merge Roboflow dataset + pseudo-labeled user frames."""
    print("=" * 60)
    print("Step 4: Merge datasets")
    print("=" * 60)

    # Copy Roboflow dataset train
    train_img_src = f'{roboflow_dir}/train/images'
    train_lbl_src = f'{roboflow_dir}/train/labels'

    merged_train_img = f'{merged_dir}/train/images'
    merged_train_lbl = f'{merged_dir}/train/labels'
    merged_val_img = f'{merged_dir}/valid/images'
    merged_val_lbl = f'{merged_dir}/valid/labels'

    os.makedirs(merged_train_img, exist_ok=True)
    os.makedirs(merged_train_lbl, exist_ok=True)
    os.makedirs(merged_val_img, exist_ok=True)
    os.makedirs(merged_val_lbl, exist_ok=True)

    # Copy 80% of Roboflow train to merged train, 20% to merged val
    roboflow_images = list(Path(train_img_src).glob('*'))
    random.shuffle(roboflow_images)
    split = int(len(roboflow_images) * 0.8)
    for img in roboflow_images[:split]:
        shutil.copy(str(img), merged_train_img)
        lbl = Path(train_lbl_src) / f'{img.stem}.txt'
        if lbl.exists():
            shutil.copy(str(lbl), merged_train_lbl)

    for img in roboflow_images[split:]:
        shutil.copy(str(img), merged_val_img)
        lbl = Path(train_lbl_src) / f'{img.stem}.txt'
        if lbl.exists():
            shutil.copy(str(lbl), merged_val_lbl)

    # Copy Roboflow validation set
    val_img_src = f'{roboflow_dir}/valid/images'
    val_lbl_src = f'{roboflow_dir}/valid/labels'
    for img in Path(val_img_src).glob('*'):
        shutil.copy(str(img), merged_val_img)
        lbl = Path(val_lbl_src) / f'{img.stem}.txt'
        if lbl.exists():
            shutil.copy(str(lbl), merged_val_lbl)

    # Copy pseudo-labeled user frames to train set
    for img in Path(user_image_dir).glob('*'):
        shutil.copy(str(img), merged_train_img)
        lbl = Path(user_label_dir) / f'{img.stem}.txt'
        if lbl.exists():
            shutil.copy(str(lbl), merged_train_lbl)

    train_count = len(os.listdir(merged_train_img))
    val_count = len(os.listdir(merged_val_img))
    print(f"Training images: {train_count}, Validation: {val_count}")

    # Create data.yaml
    data_yaml = f'{merged_dir}/data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"train: {merged_train_img}\n")
        f.write(f"val: {merged_val_img}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['football-field']\n")
        f.write(f"kpt_shape: [32, 3]\n")
    print(f"data.yaml -> {data_yaml}")
    return data_yaml


def step_train(data_yaml, base_model='yolov8x-pose.pt', epochs=50):
    """Fine-tune the model."""
    print("=" * 60)
    print(f"Step 5: Fine-tune YOLOv8x-pose ({epochs} epochs)")
    print("=" * 60)

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        project='training',
        name='pitch_finetune',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        workers=2,
    )

    best = f'{HOME}/training/pitch_finetune/weights/best.pt'
    last = f'{HOME}/training/pitch_finetune/weights/last.pt'
    out = f'{HOME}/models/pitch_keypoint_detector.pt'
    os.makedirs(f'{HOME}/models', exist_ok=True)
    if os.path.exists(best):
        shutil.copy(best, out)
        print(f"Best model saved to: {out}")
    elif os.path.exists(last):
        shutil.copy(last, out)
        print(f"Last model saved to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roboflow-key', required=True,
                        help='Roboflow API key from https://app.roboflow.com/settings/api')
    parser.add_argument('--video', default='input_videos/sample.mp4',
                        help='Path to input video')
    parser.add_argument('--interval', type=int, default=15,
                        help='Extract every N-th frame for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Keypoint confidence threshold for pseudo-labels')
    parser.add_argument('--base-model', default='yolov8x-pose.pt',
                        help='Base model (yolov8x-pose.pt or existing fine-tuned)')
    args = parser.parse_args()

    # 1. Download Roboflow dataset
    roboflow_dir = step_download_roboflow(args.roboflow_key)

    # 2. Extract frames from user's video
    frame_dir = f'{DATASET_DIR}/user_frames'
    label_dir = f'{DATASET_DIR}/user_labels'
    step_extract_frames(args.video, frame_dir, args.interval)

    # 3. Generate pseudo-labels
    #    If we already have a fine-tuned model, use it; otherwise use base
    existing = f'{HOME}/models/pitch_keypoint_detector.pt'
    pseudo_model = existing if os.path.exists(existing) else args.base_model
    if not os.path.exists(pseudo_model):
        # Download base model
        print(f"Downloading {args.base_model}...")
        YOLO(args.base_model)  # triggers download
        pseudo_model = args.base_model

    step_pseudo_label(pseudo_model, frame_dir, label_dir, args.conf_thresh)

    # 4. Merge datasets
    merged_dir = f'{DATASET_DIR}/merged'
    data_yaml = step_merge_datasets(roboflow_dir, frame_dir, label_dir, merged_dir)

    # 5. Train
    best_model = step_train(data_yaml, args.base_model, args.epochs)

    print("\n" + "=" * 60)
    print("DONE! Model saved to:", best_model)
    print("Update main.py or pitch_keypoint_detector.py to use this path.")
    print("=" * 60)


if __name__ == '__main__':
    main()
