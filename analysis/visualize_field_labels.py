import cv2, os, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATASET_ROOT = "datasets/football-field-detection-14"
N_KP = 32

# Keypoint connections forming the football field topology
# Based on flip_idx: 0↔24, 1↔25, ... 30↔31
EDGES = [
    # --- Outer boundary ---
    # Left sideline: top-to-bottom
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    # Right sideline: top-to-bottom
    (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    # Top goal line & bottom goal line
    (0, 24), (5, 29),
    # Horizontal lines connecting left-right pairs at same y-level
    (1, 25), (2, 26), (3, 27), (4, 28),
    # --- Penalty areas ---
    (2, 6), (6, 7), (7, 3),        # left penalty box
    (26, 22), (22, 23), (23, 27),  # right penalty box
    # --- Goal areas ---
    (1, 8), (2, 8),                # left goal box
    (25, 21), (26, 21),            # right goal box
    # --- Center / inner markings ---
    (9, 10),                        # top-center marking
    (13, 14), (14, 15), (15, 16),  # right inner vertical
    (17, 18), (18, 19), (19, 20),  # right-center vertical
    (11, 12),                       # left-center vertical
    # --- Penalty spots ---
    (30, 31),
]

def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

if not Path(DATASET_ROOT).exists():
    print(f"Dataset not found at {DATASET_ROOT}. Skipping.")
    exit(0)

img_dir = Path(DATASET_ROOT) / "train" / "images"
lbl_dir = Path(DATASET_ROOT) / "train" / "labels"

# Pick a sample with at least several visible keypoints (v > 0)
best_lbl, best_img, best_visible = None, None, 0
all_txt = list(lbl_dir.glob("*.txt"))
random.shuffle(all_txt)
for lf in all_txt[:100]:
    nums = list(map(float, lf.read_text().strip().split()))
    if len(nums) < 5 + 3 * N_KP:
        continue
    kps = np.array(nums[5:]).reshape(-1, 3)
    n_vis = int((kps[:, 2] > 0).sum())
    if n_vis > best_visible:
        best_visible = n_vis
        best_lbl = lf
        best_img = img_dir / (lf.stem + ".jpg")

img_bgr = cv2.imread(str(best_img))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
nums = list(map(float, best_lbl.read_text().strip().split()))
cls_id = int(nums[0])
cx, cy, bw, bh = nums[1:5]
x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, W, H)

# Parse keypoints
kps = np.array(nums[5:]).reshape(-1, 3)
kps[:, 0] *= W
kps[:, 1] *= H

vis_colors = {0: "#aaaaaa", 1: "#ffaa00", 2: "#e74c3c"}
vis_labels = {0: "not labeled", 1: "occluded", 2: "visible"}

# --- Build two-panel figure ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("YOLO Keypoint Format — Football Field Detection (32 keypoints)",
             fontsize=13, fontweight="bold")

# ─── Left panel: full image with keypoints + field outline ───
ax = axes[0]
ax.imshow(img_rgb)
ax.set_title(f"Sample: {best_img.name}  |  {best_visible}/{N_KP} keypoints visible")
ax.axis("off")

# Draw edges
for i, j in EDGES:
    if i < N_KP and j < N_KP:
        xi, yi = kps[i, 0], kps[i, 1]
        xj, yj = kps[j, 0], kps[j, 1]
        # Only draw edge if at least one endpoint is visible
        vi, vj = kps[i, 2], kps[j, 2]
        style = ":" if (vi == 0 or vj == 0) else "-"
        alpha_v = 0.5 if (vi == 0 or vj == 0) else 0.8
        ax.plot([xi, xj], [yi, yj], color="#2ecc71", linestyle=style,
                linewidth=1.2, alpha=alpha_v)

# Draw bounding box
rect = mpatches.FancyBboxPatch((x1, y1), x2 - x1, y2 - y1,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor="#3498db", facecolor="none", linestyle="--")
ax.add_patch(rect)

# Draw keypoints
for i in range(N_KP):
    x, y, v = kps[i]
    col = vis_colors.get(int(v), "#888")
    ms = 5 if v > 0 else 3
    ax.plot(x, y, "o", color=col, markersize=ms, markeredgecolor="white", markeredgewidth=0.5)
    if v > 0:
        ax.annotate(str(i), (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=5, color=col, fontweight="bold")

# Legend
legend_elements = [
    mpatches.Patch(color="#e74c3c", label="Visible (v=2)"),
    mpatches.Patch(color="#ffaa00", label="Occluded (v=1)"),
    mpatches.Patch(color="#aaaaaa", label="Not labeled (v=0)"),
    plt.Line2D([0], [0], color="#2ecc71", linewidth=1.5, label="Field edge"),
    mpatches.Patch(edgecolor="#3498db", facecolor="none", linewidth=1.5, label="Bounding box"),
]
ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

# ─── Right panel: zoomed crop + YOLO format explanation ───
ax = axes[1]
# Show full image but zoom into the keypoint area
margin = 0.05
x_min = max(0, kps[:, 0].min() - margin * W)
x_max = min(W, kps[:, 0].max() + margin * W)
y_min = max(0, kps[:, 1].min() - margin * H)
y_max = min(H, kps[:, 1].max() + margin * H)
ax.imshow(img_rgb)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_max, y_min)  # flip y for image coords

ax.set_title("YOLO Keypoint Label Format", fontsize=10)
ax.axis("off")

# Draw edges (same as left panel)
for i, j in EDGES:
    if i < N_KP and j < N_KP:
        xi, yi = kps[i, 0], kps[i, 1]
        xj, yj = kps[j, 0], kps[j, 1]
        vi, vj = kps[i, 2], kps[j, 2]
        style = ":" if (vi == 0 or vj == 0) else "-"
        alpha_v = 0.5 if (vi == 0 or vj == 0) else 0.8
        ax.plot([xi, xj], [yi, yj], color="#2ecc71", linestyle=style,
                linewidth=1, alpha=alpha_v)

# Draw keypoints with larger labels
for i in range(N_KP):
    x, y, v = kps[i]
    col = vis_colors.get(int(v), "#888")
    ms = 6 if v > 0 else 4
    ax.plot(x, y, "o", color=col, markersize=ms, markeredgecolor="white", markeredgewidth=0.5)
    ax.annotate(str(i), (x, y), xytext=(4, 4),
                textcoords="offset points", fontsize=5.5, color=col, fontweight="bold")

# Format the label as YOLO keypoint string
kp_str = " ".join(f"{kps[i,0]/W:.4f} {kps[i,1]/H:.4f} {int(kps[i,2])}" for i in range(N_KP))
label_str = (
    f"YOLO Format:  class  cx  cy  w  h  "
    f"(kp1_x  kp1_y  kp1_v  ...  kp{N_KP}_x  kp{N_KP}_y  kp{N_KP}_v)\n\n"
    f"  {nums[0]:.0f}  {cx:.4f}  {cy:.4f}  {bw:.4f}  {bh:.4f}  "
    f"  ...  (1 class × 1 bbox × {N_KP} keypoints × 3 = {N_KP*3} values)\n\n"
    f"Keypoints:\n"
)
# Show first 8 and last 4 keypoints as example
preview = []
for i in range(min(8, N_KP)):
    x, y, v = kps[i, 0] / W, kps[i, 1] / H, int(kps[i, 2])
    preview.append(f"  KP{i:2d}: x={x:.4f} y={y:.4f} v={v}")
preview.append("  ...")
for i in range(N_KP - 4, N_KP):
    x, y, v = kps[i, 0] / W, kps[i, 1] / H, int(kps[i, 2])
    preview.append(f"  KP{i:2d}: x={x:.4f} y={y:.4f} v={v}")

# Place format explanation below the zoomed image
fig.text(0.52, 0.02, label_str + "\n".join(preview),
         fontsize=7, fontfamily="monospace", va="bottom",
         bbox=dict(boxstyle="round", facecolor="#f9f9f9", alpha=0.9))

plt.tight_layout()
os.makedirs("analysis/figures", exist_ok=True)
plt.savefig("analysis/figures/field_label_format.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: analysis/figures/field_label_format.png")
