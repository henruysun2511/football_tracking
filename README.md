# Football AI Tracking & Analysis

An advanced AI-powered computer vision project for tracking and analyzing football (soccer) matches. This project detects players, referees, and the ball, projects their coordinates onto a 2D tactical minimap, and calculates various match statistics such as player speed, distance covered, ball possession, and team formations.

## 🚀 Features

- **Object Detection & Tracking:** Uses YOLO for accurate detection and ByteTrack for robust tracking of players, referees, and the football. Missing player tracks are smoothly interpolated.
- **Pitch Keypoint Detection & 2D Minimap:** Detects keypoints on the football pitch and uses homography transformation to project 3D camera coordinates onto a 2D tactical minimap.
- **Camera Movement Estimation:** Calculates and compensates for camera panning/zooming to stabilize player tracking and speed calculations.
- **Speed & Distance Estimation:** Measures real-world speed (km/h) and total distance covered (m) for each tracked player based on pixel-to-meter coordinate transformation.
- **Team Assignment:** Automatically assigns players to their respective teams using K-Means clustering based on their jersey colors.
- **Ball Possession Analysis:** Assigns ball control to the nearest player and calculates real-time team ball possession percentages.
- **Formation Detection:** Analyzes player positions to determine the tactical formation (e.g., 4-3-3, 4-4-2) of each team.
- **Heatmap Generation:** Generates positional heatmaps for the match and individual teams to analyze player movement density.

## 📁 Project Structure

- `app/` & `main.py`: Main execution scripts integrating the complete tracking and rendering pipeline.
- `trackers/`: YOLO and ByteTrack wrappers for detection, tracking, and track interpolation.
- `estimators/`: 
  - `camera_movement_estimator.py`: Compensates for video camera movements.
  - `speed_distance_estimator.py`: Computes real-world speed and distance.
  - `view_transformer.py`: Handles perspective transformation from the video frame to the 2D pitch map.
- `pitch_keypoint_detector/`: Custom model to identify critical soccer pitch landmarks.
- `minimap/`: Renders the 2D tactical map and plots player/ball positions.
- `asigners/`: Team color clustering and player-ball assignment logic.
- `heatmap_generator/`: Processes tracking data to output density heatmaps.
- `formations/`: Algorithms to detect team playing formations.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd football_tracking
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required pre-trained weights in the `models/` directory:
   - `player_detector.pt` (YOLO model for tracking)
   - `pitch_keypoint_detector.pt` (or equivalent in `models/old/`)

## 🎮 Usage

Run the main pipeline to process a video and generate the annotated output.

```bash
# Run the complete pipeline (Tracking + Rendering)
python main.py --mode all --video input_videos/sample.mp4

# Run only the tracking phase (saves tracking data to stubs/)
python main.py --mode tracking --video input_videos/sample.mp4

# Run only the rendering phase (requires existing stubs)
python main.py --mode render --video input_videos/sample.mp4
```

### Outputs
The output video with all overlays (minimap, speed, distances, camera movement, ball possession) will be saved to `output_videos/output_enhanced.avi`.
Density heatmaps will also be saved as PNG images in the `output_videos/` folder.

## 📝 Recent Improvements
- Optimized `ViewTransformer` to eliminate chaotic 2D map projections and fix speed calculation spikes.
- Integrated `interpolate_player_positions` using Pandas DataFrame linear interpolation for buttery smooth player bounding boxes.
- Improved UI overlay layout to prevent minimap and team statistics from overlapping on smaller resolutions.

## 🤝 Acknowledgments
Inspired by various football computer vision research projects and tutorials (e.g. `football_analysis`).
