# Sơ Đồ Pipeline Dự Án — Định Dạng UML (Mermaid)

> File này sử dụng cú pháp Mermaid.js để hiển thị sơ đồ UML.
> Xem trên GitHub, GitLab, hoặc dùng `mermaid-cli` để render ra PNG/SVG.

---

## 1. Flowchart Tổng Thể

```mermaid
flowchart TD
    %% ========== DATA PREPARATION ==========
    subgraph DATA_PREP ["DATA PREPARATION"]
        direction TB
        A1["Roboflow API<br/>football-players-detection-3zvbc"] --> A2["player_dataset.py<br/>Download Dataset"]
        A3["Roboflow API<br/>football-field-detection-f07vi"] --> A4["pitch_keypoint_dataset.py<br/>Download Dataset"]
        A2 --> A5["player_cleaning.py<br/>Scan + Fix + Copy"]
        A4 --> A6["pitch_keypoint_cleaning.py<br/>Scan + Fix + Copy"]
        A5 --> A7["football_training.py<br/>YOLOv8x epochs=100 imgsz=1280"]
        A6 --> A8["pitch_keypoint_training.py<br/>YOLOv8x-pose epochs=100 imgsz=640"]
        A7 --> A9["models/player_detector.pt"]
        A8 --> A10["models/pitch_keypoint_detector.pt"]
    end

    %% ========== MAIN INFERENCE ==========
    subgraph MAIN ["MAIN INFERENCE PIPELINE (main.py)"]
        direction TB

        %% Phase 1: Tracking
        subgraph P1 ["PHA 1 — TRACKING"]
            direction TB
            B0["Input Video (.mp4/.avi)"]
            B0 --> B1["read_video()"]
            B1 --> B2["Tracker.get_object_tracks()<br/>YOLOv8x detect_frames(batch=20, conf=0.1)<br/>ByteTrack(activation=0.25, lost=30)"]
            B2 --> B3["CameraMovementEstimator<br/>Lucas-Kanade Optical Flow<br/>feature_params(maxCorners=100)"]
            B3 --> B4["PitchKeypointDetector<br/>32 keypoints<br/>conf_threshold=0.3"]
            B4 --> B5["ViewTransformer<br/>cv2.findHomography + RANSAC<br/>EMA smooth alpha=0.15"]
            B5 --> B6["SpeedDistanceEstimator<br/>window=5 frames<br/>max_speed=38km/h"]
            B6 --> B7["TeamAssigner<br/>K-Means n_clusters=2 (màu áo)"]
            B7 --> B8["PlayerBallAssigner<br/>khoảng cách <= 70px"]
            B8 --> B9["Save stubs<br/>tracks_full.pkl + cam_move.pkl + team_ball_control.npy"]
        end

        %% Phase 2: Rendering
        subgraph P2 ["PHA 2 — RENDERING"]
            direction TB
            C0["Load stubs"]
            C0 --> C1["FormationAnalyzer<br/>K-Means (3-4 cụm)<br/>bỏ phiếu -> sơ đồ"]
            C1 --> C2["Với mỗi frame:"]
            C2 --> C3["Vẽ ellipse cầu thủ (màu đội) + ID"]
            C2 --> C4["Vẽ triangle bóng (cyan) + người cầm bóng (xanh)"]
            C2 --> C5["Vẽ ellipse trọng tài (vàng)"]
            C2 --> C6["Vẽ camera movement overlay (góc trên trái)"]
            C2 --> C7["Vẽ speed + distance text (dưới chân)"]
            C2 --> C8["Vẽ team ball control %"]
            C2 --> C9["Vẽ formation labels"]
            C2 --> C10["Vẽ pitch keypoints (tùy chọn)"]
            C2 --> C11["MinimapRenderer overlay (góc dưới phải, tùy chọn)"]
            C11 --> C12["save_video()<br/>ffmpeg H.264 (.mp4)<br/>fallback OpenCV (.avi)"]
            C12 --> C13["HeatmapGenerator<br/>3 ảnh: both + team1 + team2"]
        end
    end

    %% ========== WEB UI ==========
    subgraph WEB ["WEB UI"]
        D1["Gradio app/gradio_app.py"] --> D2["Upload video"]
        D2 --> D3["Cache stub (MD5 hash)"]
        D3 --> D4["Pipeline main.py"]
        D4 --> D5["Hiển thị video output"]
        D4 --> D6["Hiển thị heatmap gallery"]
        D4 --> D7["Status log real-time"]
    end

    %% ========== ANALYSIS ==========
    subgraph ANALYSIS ["ANALYSIS & VISUALIZATION"]
        direction TB
        
        subgraph EDA ["Dataset EDA"]
            E1["eda_dataset.py"] --> E1O["figures/dataset_eda.png"]
            E2["class_imbalance_analysis.py"] --> E2O["figures/class_imbalance.png"]
            E3["check_dataset_quality.py"] --> E3O["reports/dataset_quality_report.txt"]
            E4["verify_data_split.py"] --> E4O["figures/data_split.png"]
        end

        subgraph LA ["Labeling & Augmentation"]
            L1["visualize_label_format.py"] --> L1O["figures/label_format_explanation.png"]
            L2["visualize_preprocessing.py"] --> L2O["figures/augmentation_comparison.png + labeled_sample.png"]
            L3["visualize_mosaic_augmentation.py"] --> L3O["figures/mosaic_augmentation.png"]
        end

        subgraph PC ["Pipeline Component"]
            P1_["visualize_perspective.py"] --> P1O["figures/perspective_transform.png"]
            P2_["visualize_optical_flow.py"] --> P2O["figures/optical_flow.png"]
            P3_["visualize_ball_interpolation.py"] --> P3O["figures/ball_interpolation.png"]
            P4_["visualize_team_assignment.py"] --> P4O["figures/kmeans_team_assignment.png"]
        end

        subgraph RS ["Kết quả & Stats"]
            R1["visualize_player_stats.py"] --> R1O["figures/player_stats.png"]
            R2["visualize_training_results.py"] --> R2O["figures/training_results.png"]
            R3["capture_result_frame.py"] --> R3O["figures/result_showcase.png + best_frame.png"]
        end
    end

    %% ========== CONNECTIONS ==========
    A9 --> P1
    A10 --> P1
    P1 --> P2
    P2 --> D5
    P2 --> D6
```

---

## 2. Sequence Diagram — Luồng Xử Lý Chính

```mermaid
sequenceDiagram
    participant User as Người dùng
    participant UI as Web UI (Gradio)
    participant Main as main.py
    participant Tracker as Tracker (YOLOv8 + ByteTrack)
    participant Cam as CameraMovementEstimator
    participant KP as PitchKeypointDetector
    participant View as ViewTransformer
    participant Speed as SpeedDistanceEstimator
    participant Team as TeamAssigner
    participant Ball as PlayerBallAssigner
    participant Formation as FormationAnalyzer
    participant Render as Rendering Engine
    participant Video as save_video()

    User->>UI: Upload video
    UI->>Main: start_pipeline(video_path)
    
    Note over Main,Video: PHA 1 — TRACKING
    
    Main->>Main: read_video() -> frames
    Main->>Tracker: get_object_tracks(frames)
    Tracker->>Tracker: YOLOv8x detect_frames()
    Tracker->>Tracker: ByteTrack.update_with_detections()
    Tracker-->>Main: tracks (bbox only)
    
    Main->>Cam: get_camera_movement(frames)
    Cam->>Cam: calcOpticalFlowPyrLK()
    Cam-->>Main: cam_move [[dx, dy]]
    Main->>Cam: add_adjust_positions_to_tracks()
    
    Main->>KP: detect_smoothed(frame)
    KP-->>Main: keypoints (xy + conf)
    Main->>View: get_homography(keypoints)
    View-->>Main: homography matrix H
    Main->>View: add_transformed_position_to_tracks()
    
    Main->>Speed: add_speed_and_distance_to_tracks()
    Main->>Team: assign_team_color() + get_player_team()
    Main->>Ball: assign_ball_to_player()
    
    Main->>Main: Save stubs (tracks_full.pkl, cam_move.pkl, team_ball_control.npy)
    
    Note over Main,Video: PHA 2 — RENDERING
    
    Main->>Main: Load stubs
    Main->>Formation: detect_team_formation()
    Formation-->>Main: (formation1, formation2)
    
    loop Mỗi frame
        Main->>Render: draw_ellipse() + triangle() + text overlays
        Render->>Render: Draw camera movement, speed, ball control, formation
        Render->>Render: MinimapRenderer.overlay() (tùy chọn)
        Render-->>Main: frame đã annotation
    end
    
    Main->>Video: save_video(output_frames)
    Video-->>Main: output_videos/output_enhanced.mp4
    
    Main->>Main: HeatmapGenerator.render() -> 3 PNG
    Main-->>UI: return result paths
    UI-->>User: Hiển thị video + heatmaps + log
```

---

## 3. Class Diagram — Quan Hệ Giữa Các Module

```mermaid
classDiagram
    class Tracker {
        +model_path: str
        +tracker: ByteTrack
        +detect_frames(frames, batch_size)
        +get_object_tracks(frames, read_from_stub, stub_path)
        +interpolate_ball_positions(ball_positions)
        +interpolate_player_positions(player_positions)
        +add_position_to_tracks(tracks)
        +draw_ellipse(frame, bbox, color, track_id)
        +draw_triangle(frame, bbox, color)
        +draw_team_ball_control(frame, frame_num, team_ball_control)
        +draw_annotations(video_frames, tracks, team_ball_control)
    }

    class CameraMovementEstimator {
        +first_frame: np.array
        +mask: np.array
        +get_camera_movement(frames, read_from_stub, stub_path)
        +add_adjust_positions_to_tracks(tracks, movement)
        +draw_camera_movement(frames, movement)
    }

    class ViewTransformer {
        +kp_detector: PitchKeypointDetector
        +smooth_alpha: float
        +add_transformed_position_to_tracks(tracks, video_frames)
    }

    class SpeedDistanceEstimator {
        +add_speed_and_distance_to_tracks(tracks, fps)
        +draw_speed_and_distance(frames, tracks)
    }

    class TeamAssigner {
        +team_colors: dict
        +player_team_dict: dict
        +assign_team_color(frame, player_detections)
        +get_player_team(frame, bbox, player_id)
        +get_player_color(frame, bbox)
    }

    class PlayerBallAssigner {
        +max_distance: int
        +assign_ball_to_player(players, ball_bbox)
    }

    class PitchKeypointDetector {
        +model_path: str
        +conf_threshold: float
        +config: SoccerPitchConfig
        +detect(frame)
        +detect_smoothed(frame, alpha)
        +get_homography(frame_keypoints)
        +transform_point(point, M)
    }

    class SoccerPitchConfig {
        +width: int
        +length: int
        +vertices: np.array
        +edges: list
    }

    class MinimapRenderer {
        +render(tracks, frame_num, team_colors)
        +overlay(frame, minimap, pos)
    }

    class HeatmapGenerator {
        +team1_heatmap: np.array
        +team2_heatmap: np.array
        +update_from_tracks(tracks)
        +render_both()
        +render_team(team, colormap)
    }

    class FormationAnalyzer {
        +detect_team_formation(tracks, team_id)
        +detect_formation(positions)
    }

    class GradioApp {
        +video_input: gr.Video
        +chk_keypoints: gr.Checkbox
        +chk_minimap: gr.Checkbox
        +chk_heatmap: gr.Checkbox
        +btn_analyze: gr.Button
        +video_output: gr.Video
        +gallery_heatmap: gr.Gallery
        +status_log: gr.HTML
        +start_pipeline(video_path, show_kp, show_minimap, show_heatmap)
    }

    class PlayerCleaning {
        +scan_split(split)
        +clean_split(split, bad_stems)
        +fix_bbox_labels(split)
        +visualize_before_after(split, n_samples)
    }

    class PitchKeypointCleaning {
        +scan_keypoint_split(split)
        +fix_and_copy_keypoint_split(split, bad_stems)
        +visualize_keypoint_stats(kp_vis_count, kp_per_image, split)
    }

    class FootballTraining {
        +model: YOLO
        +train(data, epochs, imgsz, batch, device)
    }

    class PitchKeypointTraining {
        +model: YOLO
        +train(data, epochs, imgsz, batch, device)
    }

    %% Relationships
    ViewTransformer --> PitchKeypointDetector : uses
    ViewTransformer ..> SoccerPitchConfig : reads config
    MinimapRenderer ..> SoccerPitchConfig : uses dimensions
    GradioApp --> Tracker : calls
    GradioApp --> CameraMovementEstimator : calls
    GradioApp --> ViewTransformer : calls
    GradioApp --> SpeedDistanceEstimator : calls
    GradioApp --> TeamAssigner : calls
    GradioApp --> PlayerBallAssigner : calls
    GradioApp --> FormationAnalyzer : calls
    GradioApp --> MinimapRenderer : calls
    GradioApp --> HeatmapGenerator : calls
    FootballTraining --> PlayerCleaning : uses cleaned dataset
    PitchKeypointTraining --> PitchKeypointCleaning : uses cleaned dataset
```

---

## 4. State Diagram — Trạng Thái Pipeline

```mermaid
stateDiagram-v2
    [*] --> IDLE
    
    IDLE --> DOWNLOADING : User chạy dataset script
    DOWNLOADING --> CLEANING : Dataset ready
    CLEANING --> TRAINING : Dataset cleaned
    TRAINING --> READY : Model trained (best.pt)
    
    IDLE --> TRACKING : python main.py --mode all
    
    TRACKING --> CAMERA_ESTIMATION : tracks raw
    CAMERA_ESTIMATION --> HOMOGRAPHY : cam_move
    HOMOGRAPHY --> SPEED_CALC : position_transformed
    SPEED_CALC --> TEAM_CLUSTERING : speed + distance
    TEAM_CLUSTERING --> BALL_ASSIGNMENT : team assigned
    BALL_ASSIGNMENT --> STUB_SAVED : ball control
    STUB_SAVED --> FORMATION_DETECTION : stubs ready
    
    FORMATION_DETECTION --> RENDERING : formation
    RENDERING --> VIDEO_SAVED : annotated frames
    VIDEO_SAVED --> HEATMAP_GENERATION : video done
    HEATMAP_GENERATION --> COMPLETED : all outputs ready
    
    TRACKING --> ERROR_TRACK : detection fail
    ERROR_TRACK --> IDLE
    
    RENDERING --> ERROR_RENDER : insufficient stubs
    ERROR_RENDER --> IDLE
    
    COMPLETED --> IDLE : User uploads new video
```

---

## 5. Dependency Graph

```mermaid
flowchart LR
    subgraph Deps ["Phụ thuộc giữa các Module"]
        direction LR

        %% Data layer
        RF[Roboflow API] --> DD[Download Dataset]
        DD --> CL[Cleaning Dataset]
        CL --> TR[Training]
        TR --> MW[Model Weights .pt]
        
        %% Pipeline layer
        VI[Video Input] --> RD[read_video OpenCV]
        RD --> DT[YOLOv8 Detection]
        DT --> BT[ByteTrack]
        BT --> CI[Camera Movement<br/>Optical Flow]
        CI --> KP[Pitch Keypoint<br/>YOLOv8-pose]
        KP --> HV[ViewTransformer<br/>Homography]
        HV --> SD[Speed & Distance]
        SD --> TA[Team Assigner<br/>K-Means]
        TA --> BA[Ball Assigner]
        BA --> ST[Save Stubs .pkl]
        
        ST --> LO[Load Stubs]
        LO --> FA[Formation Analyzer<br/>K-Means]
        FA --> DR[Drawing Engine<br/>OpenCV + Matplotlib]
        DR --> MR[MinimapRenderer]
        DR --> HG[HeatmapGenerator]
        DR --> SV[save_video<br/>ffmpeg / OpenCV]
        
        %% Web UI layer
        UI[Gradio App] --> MV[Model Weights .pt]
        UI --> ST
    end

    style RF fill:#f9f,stroke:#333
    style MW fill:#9cf,stroke:#333
    style ST fill:#9cf,stroke:#333
    style SV fill:#6c6,stroke:#333
    style UI fill:#fc6,stroke:#333
```

---

## 6. Component Diagram — Kiến Trúc Module

```mermaid
graph TB
    subgraph Input ["Đầu vào"]
        V[Video File .mp4]
        M[Model Weights .pt]
    end

    subgraph Core ["Core Pipeline"]
        T[Trackers]
        E[Estimators]
        A[Asigners]
        KP2[PitchKeypoint Detector]
    end

    subgraph Output ["Đầu ra"]
        VOut[Video Output .mp4]
        Stub[Stubs .pkl]
        HMap[Heatmap .png]
        Stats[Statistics]
    end

    subgraph Support ["Hỗ trợ"]
        CL2[Cleanings]
        TR2[Trainings]
        AN[Analysis]
        WEB2[Web UI]
    end

    V --> T
    M --> T
    M --> KP2
    
    T --> E
    KP2 --> E
    E --> A
    A --> Output
    
    CL2 --> TR2
    TR2 --> M
    
    AN --> Stub
    AN --> VOut
    
    WEB2 --> Core
    WEB2 --> Output

    style Core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Input fill:#fff3e0,stroke:#ff6f00
    style Output fill:#e8f5e9,stroke:#2e7d32
    style Support fill:#f3e5f5,stroke:#7b1fa2
```

---

*Tài liệu UML được tạo ngày 07/06/2026, tương ứng commit `139139c`.*
*Render: dùng `mermaid-cli` (`mmdc -i docs/pipeline-uml.md -o docs/pipeline-uml.png`) hoặc xem trực tiếp trên GitHub.*
