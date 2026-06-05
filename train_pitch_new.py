"""
Train a pitch keypoint detector from scratch.
Can use Roboflow dataset OR your own labeled images.

Usage:
  1) Label frames from your video (recommended):
     python train_pitch_new.py --mode label --video input_videos/sample.mp4
     # Click the 32 keypoints in order on each frame, press 'n' for next

  2) Train using Roboflow dataset (no manual labeling):
     python train_pitch_new.py --mode train --roboflow-key YOUR_KEY

  3) Train on your own labeled data:
     python train_pitch_new.py --mode train --data-dir ./my_dataset
"""
import argparse, os, sys, json, shutil, random, subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HOME = os.getcwd()
NUM_KEYPOINTS = 32
KP_NAMES = [
    "top-left corner", "left penalty box top", "left goal box top",
    "left goal box bottom", "left penalty box bottom", "bottom-left corner",
    "left goal box line top", "left goal box line bottom",
    "left penalty spot", "left penalty box line top",
    "left penalty box inner top", "left penalty box inner bottom",
    "left penalty box line bottom", "halfway line top",
    "center circle top", "center circle bottom", "halfway line bottom",
    "right penalty box line top", "right penalty box inner top",
    "right penalty box inner bottom", "right penalty box line bottom",
    "right penalty spot", "right goal box line top",
    "right goal box line bottom", "top-right corner",
    "right penalty box top", "right goal box top",
    "right goal box bottom", "right penalty box bottom",
    "bottom-right corner", "center circle left", "center circle right"
]


def mode_label(args):
    """Interactive labeling: click 32 keypoints per frame."""
    import cv2

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Select frames to label (every N frames)
    frame_indices = list(range(0, total, args.interval))
    print(f"Video: {total} frames. Will label {len(frame_indices)} frames " +
          f"(every {args.interval}th frame).")
    print(f"Frame size: {w}x{h}")
    print()
    print("INSTRUCTIONS:")
    print(f"  1. Click on each of the {NUM_KEYPOINTS} keypoints IN ORDER")
    print("  2. Press 'n' to save and go to next frame")
    print("  3. Press 'r' to redo current frame")
    print("  4. Press 'q' to quit")
    print("  Keypoint order:")
    for i, name in enumerate(KP_NAMES):
        print(f"    {i:2d}: {name}")
    print()

    window_name = "Label Keypoints - click in order, 'n'=next, 'r'=redo, 'q'=quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, lambda e, x, y, f, p: _on_click(e, x, y, f, p))

    labeled_count = 0
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        kps = []
        current_kp = 0
        saved = False

        while True:
            # Show instructions
            img = display.copy()
            for i, (x, y) in enumerate(kps):
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img,
                f"Frame {fi}  KP {current_kp}/{NUM_KEYPOINTS}: {KP_NAMES[current_kp]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img,
                "[click] mark  [n] next  [r] redo  [q] quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(window_name, img)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nSaved {labeled_count} labeled frames.")
                return

            elif key == ord('r'):
                kps = []
                current_kp = 0

            elif key == ord('n'):
                if len(kps) == NUM_KEYPOINTS:
                    # Save YOLO format label
                    label_path = f'{args.out_dir}/frame_{fi:06d}.txt'
                    img_path = f'{args.out_dir}/frame_{fi:06d}.jpg'
                    cv2.imwrite(img_path, frame)
                    with open(label_path, 'w') as f:
                        vals = []
                        for (x, y) in kps:
                            vals.extend([x / w, y / h, 2])
                        f.write(f"0 0.5 0.5 1.0 1.0 " +
                                " ".join(f"{v:.6f}" for v in vals))
                    labeled_count += 1
                    saved = True
                    print(f"  Frame {fi}: saved {NUM_KEYPOINTS} keypoints")
                    break
                else:
                    print(f"  Only {len(kps)}/{NUM_KEYPOINTS} keypoints placed!")

            # Handle click events via global var
            if _click_event is not None:
                x, y = _click_event
                _click_event = None
                kps.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display, str(current_kp), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                current_kp += 1
                if current_kp >= NUM_KEYPOINTS:
                    print(f"  All {NUM_KEYPOINTS} points placed. Press 'n' to save.")

        if not saved:
            # Save anyway if all points placed and user pressed something else
            pass

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Labeled {labeled_count} frames -> {args.out_dir}/")

    # Generate dataset info
    info = {
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_names": KP_NAMES,
        "labeled_frames": labeled_count,
        "label_dir": args.out_dir,
        "image_size": [w, h]
    }
    with open(f'{args.out_dir}/dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Dataset info: {args.out_dir}/dataset_info.json")


_click_event = None
def _on_click(event, x, y, flags, param):
    global _click_event
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_event = (x, y)


def download_roboflow_dataset(api_key, out_dir):
    """Download football-field-detection dataset from Roboflow."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'roboflow'],
                   capture_output=True)
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
    version = project.version(12)
    dataset = version.download("yolov8", location=out_dir)
    return dataset.location


def prepare_dataset_yaml(data_dir, output_yaml):
    """Create data.yaml for YOLOv8 pose training."""
    train_img = os.path.abspath(f'{data_dir}/train/images')
    val_img = os.path.abspath(f'{data_dir}/valid/images')

    # If Roboflow format has different paths
    if not os.path.exists(train_img):
        train_img = os.path.abspath(f'{data_dir}/train')
        val_img = os.path.abspath(f'{data_dir}/valid')

    with open(output_yaml, 'w') as f:
        f.write(f"train: {train_img}\n")
        f.write(f"val: {val_img}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['football-field']\n")
        f.write(f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n")
    print(f"Created: {output_yaml}")
    return output_yaml


def mode_train(args):
    """Train the model."""
    print("=" * 60)
    print("Training Pitch Keypoint Detector")
    print("=" * 60)

    # Prepare dataset
    if args.roboflow_key:
        print("\n[1/4] Downloading Roboflow dataset...")
        rf_dir = download_roboflow_dataset(args.roboflow_key,
                                           f'{HOME}/rf_dataset')
        data_dir = rf_dir
        # Fix Roboflow paths
        for split in ['train', 'valid']:
            old_img = os.path.join(rf_dir, split, 'images')
            old_lbl = os.path.join(rf_dir, split, 'labels')
            new_img = os.path.join(rf_dir, split)
            # Roboflow downloads to {split}/images/ and {split}/labels/
            # We need {split}/image1.jpg and {split}/image1.txt
            if os.path.exists(old_img):
                for f in os.listdir(old_img):
                    shutil.move(os.path.join(old_img, f),
                                os.path.join(new_img, f))
                os.rmdir(old_img)
            if os.path.exists(old_lbl):
                for f in os.listdir(old_lbl):
                    shutil.move(os.path.join(old_lbl, f),
                                os.path.join(new_img, f))
                os.rmdir(old_lbl)
    else:
        data_dir = args.data_dir

    print(f"\n[2/4] Preparing data.yaml...")
    data_yaml = prepare_dataset_yaml(data_dir, f'{HOME}/training_data.yaml')

    # Count images
    train_imgs = [f for f in os.listdir(os.path.dirname(data_yaml) + '/train/images')
                  if f.endswith(('.jpg', '.png', '.jpeg'))] if False else []
    train_dir = None
    with open(data_yaml) as f:
        for line in f:
            if line.startswith('train:'):
                train_dir = line.split(':')[1].strip()
                break
    if train_dir and os.path.exists(train_dir):
        train_imgs = [f for f in os.listdir(train_dir)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        val_dir = None
        with open(data_yaml) as f:
            for line in f:
                if line.startswith('val:'):
                    val_dir = line.split(':')[1].strip()
                    break
        val_imgs = [f for f in os.listdir(val_dir)
                    if f.endswith(('.jpg', '.png', '.jpeg'))] if val_dir and os.path.exists(val_dir) else []
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val:   {len(val_imgs)} images")

    # Verify a label file
    if train_dir:
        label_files = list(Path(train_dir).parent / 'labels' if train_dir else Path('.'))
    label_dir = os.path.join(os.path.dirname(train_dir) if train_dir else '.', 'labels')
    if os.path.exists(label_dir):
        lbls = os.listdir(label_dir)
        if lbls:
            with open(os.path.join(label_dir, lbls[0])) as f:
                sample = f.read().strip()
                kp_count = (len(sample.split()) - 5) // 3  # cls x y w h kp*3
                print(f"  Keypoints per annotation: {kp_count}")
                print(f"  Sample: {sample[:100]}...")

    # Download base model
    print(f"\n[3/4] Loading base model...")
    model = YOLO('yolov8x-pose.pt')
    print(f"  Base: YOLOv8x-pose (69.8M params)")

    # Train
    print(f"\n[4/4] Training ({args.epochs} epochs)...")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        patience=30,
        project=HOME,
        name='pitch_model',
        exist_ok=True,
        pretrained=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        workers=2,
        verbose=True,
    )

    # Save final model
    best = f'{HOME}/pitch_model/weights/best.pt'
    out = f'{HOME}/models/pitch_keypoint_detector.pt'
    os.makedirs(f'{HOME}/models', exist_ok=True)
    if os.path.exists(best):
        shutil.copy(best, out)
        print(f"\nModel saved to: {out}")
    else:
        last = f'{HOME}/pitch_model/weights/last.pt'
        if os.path.exists(last):
            shutil.copy(last, out)
            print(f"\nModel saved to: {out} (last epoch)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train pitch keypoint detector")
    sub = parser.add_subparsers(dest='mode', required=True)

    # --- LABEL subcommand ---
    p_label = sub.add_parser('label',
        help="Interactively label keypoints on video frames")
    p_label.add_argument('--video', default='input_videos/sample.mp4')
    p_label.add_argument('--out-dir', default='labeled_frames')
    p_label.add_argument('--interval', type=int, default=15,
                         help='Label every N-th frame')

    # --- TRAIN subcommand ---
    p_train = sub.add_parser('train', help="Train the model")
    src = p_train.add_mutually_exclusive_group(required=True)
    src.add_argument('--roboflow-key', help='Roboflow API key')
    src.add_argument('--data-dir',
                     help='Local dataset dir with train/ and valid/ folders')
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--imgsz', type=int, default=640)
    p_train.add_argument('--batch', type=int, default=16)

    args = parser.parse_args()

    if args.mode == 'label':
        mode_label(args)
    elif args.mode == 'train':
        mode_train(args)


if __name__ == '__main__':
    main()
