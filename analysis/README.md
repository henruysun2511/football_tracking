# Analysis Scripts

## Da chay duoc (10 file)
| Script | Output | Ghi chu |
|--------|--------|---------|
| `check_dataset_quality.py` | `reports/dataset_quality_report.txt` | Dataset sach, 0 van de |
| `class_imbalance_analysis.py` | `figures/class_imbalance.png` | player/ball = 23.6x |
| `eda_dataset.py` | `figures/dataset_eda.png` | 612 train / 38 val / 13 test |
| `verify_data_split.py` | `figures/data_split.png` | Pie + bar chart |
| `visualize_label_format.py` | `figures/label_format_explanation.png` | YOLO format |
| `visualize_mosaic_augmentation.py` | `figures/mosaic_augmentation.png` | Mosaic augmentation |
| `visualize_preprocessing.py` | `figures/augmentation_comparison.png`, `labeled_sample.png` | Augmentation samples |
| `visualize_training_results.py` | `figures/training_results.png` | Demo metrics (fake data) |
| `visualize_perspective.py` | `figures/perspective_transform.png` | Bird's eye tu pitch keypoints |
| `visualize_optical_flow.py` | `figures/optical_flow.png` | Optical flow vectors |

## Chua chay duoc (4 file)
| Script | Thieu | Cach chay |
|--------|-------|-----------|
| `visualize_ball_interpolation.py` | `stubs/track_stubs.pkl` | Chay `python main.py --mode tracking` truoc de tao stub |
| `visualize_player_stats.py` | `stubs/track_stubs.pkl` | Nhu tren |
| `visualize_team_assignment.py` | `stubs/track_stubs.pkl` + `input_videos/08fd33_4.mp4` | Can stub + video mau cu |
| `capture_result_frame.py` | `output_videos/output_video.avi` | Can output video tu main.py hoac doi ten `sample.avi` thanh `output_video.avi` |
