import os, warnings, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import torch

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── COCO baselines from Ultralytics docs ───
COCO_BASELINES = {
    "YOLOv8x":       {"map50": 0.685, "map50_95": 0.539},
    "YOLOv8x-pose":  {"kp_map50": None, "kp_map50_95": None},
}

# ─── Paths ───
PLAYER_MODEL = str(ROOT / "models" / "player_detector.pt")
PITCH_MODEL  = str(ROOT / "models" / "pitch_keypoint_detector.pt")

PLAYER_DATA = ROOT / "datasets" / "football-players-detection-2"
PITCH_DATA  = ROOT / "datasets" / "football-field-detection-14"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: Running on CPU — validation may be slow for YOLOv8x @ imgsz=1280")

# ─── Helper ───
def make_fixed_yaml(base):
    import yaml
    with open(base / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    # Map yaml keys (train/val/test) to actual directory names
    DIR_MAP = {"train": "train", "val": "valid", "test": "test"}
    for key, subdir in DIR_MAP.items():
        if key not in cfg:
            continue
        img_dir = base / subdir / "images"
        if not img_dir.is_dir():
            img_dir = base / key / "images"
        if img_dir.is_dir():
            cfg[key] = str(img_dir)
    fixed = str(base / "_data_fixed.yaml")
    with open(fixed, "w") as f:
        yaml.dump(cfg, f)
    return fixed

# =====================================================================
# 1. Player Detection Comparison
# =====================================================================
print("\n" + "="*60)
print("  PLAYER DETECTION: COCO Pretrained vs Fine-tuned")
print("="*60)

player_yaml = make_fixed_yaml(PLAYER_DATA)

# Fine-tuned model
print("\n[1/2] Evaluating fine-tuned model...")
ft_model = YOLO(PLAYER_MODEL)
ft_results = ft_model.val(data=player_yaml, imgsz=1280, batch=4, device=DEVICE, plots=False, verbose=False)
ft_map50 = float(ft_results.box.map50)
ft_map50_95 = float(ft_results.box.map)
ft_ap50 = ft_results.box.ap50.tolist()
ft_ap = ft_results.box.ap.tolist()

# COCO pretrained — use hardcoded baselines
print("[2/2] COCO baseline (from Ultralytics docs)")
pt_map50 = COCO_BASELINES["YOLOv8x"]["map50"]
pt_map50_95 = COCO_BASELINES["YOLOv8x"]["map50_95"]
pt_ap50 = [pt_map50] * 4
pt_ap = [pt_map50_95] * 4

# Print comparison
CLASSES = ["ball", "goalkeeper", "player", "referee"]
print(f"\n{'Metric':<25} {'COCO Pretrained':<18} {'Fine-tuned':<18} {'Delta':<10}")
print("-"*71)
print(f"{'mAP@0.5':<25} {pt_map50:<18.4f} {ft_map50:<18.4f} {ft_map50 - pt_map50:<+10.4f}")
print(f"{'mAP@0.5:0.95':<25} {pt_map50_95:<18.4f} {ft_map50_95:<18.4f} {ft_map50_95 - pt_map50_95:<+10.4f}")
print("-"*71)
for i, name in enumerate(CLASSES):
    print(f"{'AP@0.5 - ' + name:<25} {pt_ap50[i]:<18.4f} {ft_ap50[i]:<18.4f} {ft_ap50[i] - pt_ap50[i]:<+10.4f}")

os.remove(player_yaml) if os.path.exists(player_yaml) else None

# =====================================================================
# 2. Pitch Keypoint Comparison
# =====================================================================
print("\n" + "="*60)
print("  PITCH KEYPOINT: Fine-tuned model only")
print("="*60)
print("  NOTE: COCO pretrained YOLOv8x-pose detects 17 person keypoints,")
print("        not 32 football field keypoints — direct comparison invalid.")
print("="*60)

pitch_yaml = make_fixed_yaml(PITCH_DATA)

print("\nEvaluating fine-tuned pitch model...")
kp_model = YOLO(PITCH_MODEL)
kp_results = kp_model.val(data=pitch_yaml, imgsz=640, batch=4, device=DEVICE, plots=False, verbose=False)

kp_map50 = float(kp_results.pose.map50) if hasattr(kp_results, 'pose') and kp_results.pose is not None else 0
kp_map50_95 = float(kp_results.pose.map) if hasattr(kp_results, 'pose') and kp_results.pose is not None else 0
box_map50 = float(kp_results.box.map50)
box_map50_95 = float(kp_results.box.map)

print(f"\n{'Metric':<25} {'Value':<12}")
print("-"*37)
print(f"{'Box mAP@0.5':<25} {box_map50:<12.4f}")
print(f"{'Box mAP@0.5:0.95':<25} {box_map50_95:<12.4f}")
print(f"{'Pose mAP@0.5':<25} {kp_map50:<12.4f}")
print(f"{'Pose mAP@0.5:0.95':<25} {kp_map50_95:<12.4f}")

os.remove(pitch_yaml) if os.path.exists(pitch_yaml) else None

# =====================================================================
# 3. Visualization
# =====================================================================
print("\nGenerating comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("So sánh COCO Pretrained vs Fine-tuned trên Dataset Bóng đá",
             fontsize=13, fontweight="bold")

# Left: Player detection
ax = axes[0]
x = np.arange(len(CLASSES))
w = 0.35
bars1 = ax.bar(x - w/2, pt_ap50, w, label="COCO Pretrained", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + w/2, ft_ap50, w, label="Fine-tuned", color="#e74c3c", alpha=0.8)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=9)
ax.set_ylabel("AP@0.5")
ax.set_title("Player Detection — AP@0.5 per Class")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
ax.grid(axis="y", alpha=0.3)

# Right: Overall mAP comparison
ax = axes[1]
metrics = ["mAP@0.5", "mAP@0.5:0.95"]
pt_vals = [pt_map50, pt_map50_95]
ft_vals = [ft_map50, ft_map50_95]

x2 = np.arange(len(metrics))
bars3 = ax.bar(x2 - w/2, pt_vals, w, label="COCO Pretrained", color="#3498db", alpha=0.8)
bars4 = ax.bar(x2 + w/2, ft_vals, w, label="Fine-tuned", color="#e74c3c", alpha=0.8)

# Annotate deltas
for i, (pv, fv) in enumerate(zip(pt_vals, ft_vals)):
    delta = fv - pv
    mid = (bars3[i].get_x() + bars4[i].get_x() + bars4[i].get_width()) / 2
    ax.text(mid, max(pv, fv) + 0.03, f"{delta:+.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#2ecc71" if delta > 0 else "#e74c3c")

for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)

ax.set_xticks(x2)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("mAP")
ax.set_title("Player Detection — Overall mAP")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = str(OUTPUT_DIR / "compare_pretrained_trained.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
