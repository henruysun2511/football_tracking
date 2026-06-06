# Pipeline Football AI Analysis

```
input_videos/sample.mp4
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 1: TRACKING  (──mode tracking)               │
│  python main.py --mode tracking                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Read video (utils.read_video)                    │
│     → list of frames                                 │
│                                                      │
│  2. Player/Ball/Referee Detection (trackers/)        │
│     • YOLOv8 (models/player_detector.pt)             │
│     • ByteTrack (sv.ByteTrack)                       │
│     • Rename goalkeeper → player                     │
│     • Output: tracks dict {players, referees, ball}  │
│     • Lưu cache: stubs/track_stubs.pkl               │
│                                                      │
│  3. Add position to tracks                           │
│     • foot position (center_x, y_bottom) cho players │
│     • center point cho ball                          │
│                                                      │
│  4. Camera Movement Estimation (estimators/)         │
│     • Lucas-Kanade Optical Flow                     │
│     • Feature points: top/bottom of frame            │
│     • Output: camera_movement_per_frame (dx, dy)     │
│     • add_adjust_positions: position_adjusted         │
│     • Lưu cache: stubs/camera_movement_stub.pkl      │
│                                                      │
│  5. View Transformation (estimators/)                │
│     • PitchKeypointDetector: detect 32 keypoints     │
│       (models/old/pitch_keypoint_detector.pt)        │
│     • Per-frame homography (not hardcoded)           │
│     • Temporal smoothing (avg 5 frames)              │
│     • Fallback to previous homography on failure     │
│     • Uses raw position (not position_adjusted)      │
│     • Output: position_transformed (real-world m)    │
│                                                      │
│  6. Ball Interpolation (trackers/)                   │
│     • pandas interpolate + bfill                     │
│                                                      │
│  7. Speed & Distance (estimators/)                   │
│     • Window: 5 frames                               │
│     • position_transformed → distance in meters      │
│     • speed = distance / time → km/h                 │
│     • Skip referees, ball                            │
│                                                      │
│  8. Team Assignment (asigners/)                      │
│     • KMeans clustering trên màu áo (first frame)   │
│     • Lưu team + team_color vào tracks               │
│                                                      │
│  9. Ball Possession (asigners/)                      │
│     • Nearest player to ball (threshold 70px)        │
│     • has_ball flag + team_ball_control array        │
│                                                      │
│  10. Save stubs:                                     │
│      • stubs/tracks_full.pkl                         │
│      • stubs/cam_move.pkl                            │
│      • stubs/team_ball_control.npy                   │
│                                                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 2: RENDER  (──mode render)                    │
│  python main.py --mode render                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Load stubs (tracks_full.pkl, cam_move.pkl, ...) │
│                                                      │
│  2. Formation Detection (formations/)                │
│     • KMeans clustering: 10 outfield players         │
│       by x-position → 3 or 4 lines                   │
│     • Map to ~20 known formations (4-4-2, 4-3-3...)  │
│     • Confidence score per team                      │
│                                                      │
│  3. Heatmap Generation (heatmap_generator/)          │
│     • Rasterize position_transformed lên pitch map   │
│     • Both teams + per-team heatmaps                 │
│                                                      │
│  For EACH frame:                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  4. Draw Annotations                        │    │
│  │     • Player ellipse + track_id             │    │
│  │     • Ball triangle                         │    │
│  │     • Team colors (đỏ/xanh)                 │    │
│  │     • Jersey OCR (ocr/):                    │    │
│  │       - EasyOCR                             │    │
│  │       - Crop upper-body (5-55% bbox)        │    │
│  │       - CLAHE + adaptive threshold          │    │
│  │       - Cached per track_id                 │    │
│  │                                             │    │
│  │  5. Camera Movement Overlay                 │    │
│  │     • dx, dy in top-left corner             │    │
│  │                                             │    │
│  │  6. Speed & Distance Text                   │    │
│  │     • Below foot position                   │    │
│  │     • "xx.xx km/h" + "xx.xx m"             │    │
│  │                                             │    │
│  │  7. Team Ball Control %                     │    │
│  │     • Bottom-right corner                   │    │
│  │                                             │    │
│  │  8. Formation Name                          │    │
│  │     • Bottom-left (e.g. "Team 1: 4-4-2")   │    │
│  │                                             │    │
│  │  9. Pitch Keypoints (pitch_keypoint_detector/)│   │
│  │     • 32 green dots + index                 │    │
│  │                                             │    │
│  │  10. Minimap (minimap/)                     │    │
│  │      • Bottom-right overlay                 │    │
│  │      • Player dots, ball, team colors       │    │
│  │                                             │    │
│  │  11. Write frame → VideoWriter             │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  12. Save:                                           │
│      • output_videos/output_enhanced.avi             │
│      • output_videos/heatmap_both.png                │
│      • output_videos/heatmap_team1.png               │
│      • output_videos/heatmap_team2.png               │
│                                                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  WEB UI (app/)                                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  streamlit run app/streamlit_app.py                  │
│  • Upload video or select from cache                 │
│  • Toggle overlays (keypoints, minimap, heatmap)     │
│  • Preview + download                                │
│                                                      │
│  python app/gradio_app.py                            │
│  • Gradio interface (shareable on Colab)             │
│  • Upload → process → show video + heatmaps          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Module Structure

| Module | File(s) | Role |
|--------|---------|------|
| `trackers/` | `tracker.py` | YOLO detection + ByteTrack tracking |
| `estimators/` | `camera_movement_estimator.py` | Optical flow camera estimation |
| | `view_transformer.py` | Per-frame homography via pitch keypoints |
| | `speed_distance_estimator.py` | Speed (km/h) + distance (m) from transformed positions |
| `asigners/` | `team_assigner.py` | KMeans jersey color clustering |
| | `player_ball_assigner.py` | Nearest-player ball assigner |
| `pitch_keypoint_detector/` | `pitch_keypoint_detector.py` | YOLOv8-pose 32 keypoints + SoccerPitchConfig |
| `heatmap_generator/` | `heatmap_generator.py` | Position rasterization to pitch heatmap |
| `minimap/` | `minimap_renderer.py` | Top-down minimap overlay |
| `formations/` | `formation_analyzer.py` | KMeans formation detection (20 formations) |
| `ocr/` | `jersey_ocr.py` | EasyOCR jersey number reader |
| `utils/` | `video_util.py`, `bbox_util.py` | Video I/O + bbox helpers |
| `app/` | `streamlit_app.py`, `gradio_app.py` | Web interfaces |
| `cleanings/` | `player_cleaning.py`, `jersey_cleaning.py` | Dataset cleaning scripts |
| `trainings/` | Various | Training scripts (football, jersey, pitch keypoint) |
| `analysis/` | Various (14 scripts) | Dataset EDA, quality check, visualization |

## Models

| Model | Path | Task |
|-------|------|------|
| player_detector.pt | `models/player_detector.pt` | YOLOv8 player/ball/referee detection |
| pitch_keypoint_detector.pt | `models/old/pitch_keypoint_detector.pt` | YOLOv8-pose 32 pitch keypoints |
| pitch_keypoint_detector.pt | `models/pitch_keypoint_detector.pt` | Fine-tuned version |

## Running

```bash
# Full pipeline (tracking + render)
python main.py

# Separate phases
python main.py --mode tracking
python main.py --mode render

# YOLO inference (raw detection)
python yolo_inference.py

# Web UI
streamlit run app/streamlit_app.py
python app/gradio_app.py
```
