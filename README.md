# Football AI Analysis

Detect, track, and analyze football players/ball/referees from broadcast video — with team assignment, speed/distance, formation detection, pitch keypoints, minimap, and heatmap.

## Features

| Feature | Description |
|---------|-------------|
| **Detection & Tracking** | YOLOv8 + ByteTrack — players, ball, referees, goalkeepers merged |
| **Team Assignment** | KMeans on jersey color (first frame), cached per track ID |
| **Ball Possession** | Nearest-player-to-ball assigner (70px threshold) |
| **Camera Movement** | Lucas-Kanade optical flow on side strips, signed dx/dy |
| **Pitch Keypoints** | YOLOv8-pose 32 keypoints → per-frame homography |
| **Speed & Distance** | Windowed (5 frames), transformed pitch coords → km/h + m |
| **Formation** | KMeans x-clustering of 10 outfield players → 20 formations |
| **Jersey OCR** | EasyOCR on upper-body crop (CLAHE, cached) |
| **Minimap** | Top-down player positions with team colors |
| **Heatmap** | Position rasterization per team |
| **Web UI** | Streamlit (local) / Gradio (Colab-shareable) |

## Project Structure

```
├── main.py                          CLI entry point
├── app/
│   ├── streamlit_app.py             Streamlit UI
│   └── gradio_app.py                Gradio UI
├── trackers/tracker.py              YOLO + ByteTrack
├── estimators/
│   ├── camera_movement_estimator.py Optical flow
│   ├── view_transformer.py          Per-frame homography
│   └── speed_distance_estimator.py  Speed & distance
├── asigners/
│   ├── team_assigner.py             KMeans jersey color
│   └── player_ball_assigner.py      Nearest-ball assigner
├── pitch_keypoint_detector/         32 keypoints + config
├── heatmap_generator/               Position → heatmap
├── minimap/                         Top-down overlay
├── formations/                      Formation detection
├── ocr/jersey_ocr.py                EasyOCR (disabled)
├── utils/                           Video I/O + bbox helpers
├── models/                          YOLO weights
├── stubs/                           Pickled tracks cache
├── analysis/                        14 EDA/viz scripts
└── cleanings/                       Dataset cleaning tools
```

## Models

Download to `models/`:

| Model | Source |
|-------|--------|
| `player_detector.pt` | [football-players-detection-2](https://universe.roboflow.com/...) — YOLOv8, 4 classes |
| `pitch_keypoint_detector.pt` | YOLOv8x-pose, 32 keypoints, trained on [football-field-detection-f07vi/12](https://universe.roboflow.com/...) |

> Files >100MB not on GitHub — upload via Google Drive, then `!gdown <id>` on Colab.

## Installation

```bash
# Local
pip install -r requirements.txt

# Colab
!git clone https://github.com/henruysun2511/football_tracking.git
%cd football_tracking
!pip install -r requirements.txt
```

## Usage

### CLI (full pipeline)

```bash
python main.py                           # tracking + render
python main.py --mode tracking           # phase 1 only
python main.py --mode render             # phase 2 only (from stubs)
```

### Web UI

```bash
# Streamlit
streamlit run app/streamlit_app.py
# → http://localhost:8501

# Gradio (Colab shareable)
python app/gradio_app.py
```

### YOLO raw inference

```bash
python yolo_inference.py
```

## Notes

- **GPU**: Tested on Colab T4 (16GB VRAM). Local GTX 960M (2GB) works for editing but run full pipeline on Colab.
- **Memory**: Phase 1 stores all frames in RAM (~1080p × 2min ≈ 22GB). Phase 2 streams from disk frame-by-frame.
- **Stubs**: Delete `stubs/` after pipeline changes: `rm -rf stubs && python main.py --mode tracking`
- **ViewTransformer**: Uses per-frame homography from PitchKeypointDetector (not hardcoded 4-point). Raw `position`, not `position_adjusted`.
- **Windows**: `if __name__ == '__main__'` guard required. `workers=0` for multiprocessing.
- **ASCII only**: All log messages in plain ASCII (no emoji, no Unicode).
