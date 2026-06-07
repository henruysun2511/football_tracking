import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pickle
import threading
import time
import traceback
import hashlib

import gradio as gr
import numpy as np

from utils import read_video, save_video, get_foot_position
from trackers import Tracker
from asigners import TeamAssigner, PlayerBallAssigner
from estimators import (CameraMovementEstimator,
                        ViewTransformer,
                        SpeedDistanceEstimator)
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector
from heatmap_generator.heatmap_generator import HeatmapGenerator
from minimap.minimap_renderer import MinimapRenderer
from formations import detect_team_formation

CACHE_DIR = Path("cache_gradio")
CACHE_DIR.mkdir(exist_ok=True)

_log_lines = []
_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        ts = time.strftime("%H:%M:%S")
        _log_lines.append(f"[{ts}] {msg}")
        return "\n".join(_log_lines[-100:])


def process_video(video_path, show_keypoints, show_minimap, show_heatmap,
                  progress=gr.Progress()):
    log("Reading video...")
    progress(0, desc="Reading video...")
    video_frames = read_video(video_path)
    with open(video_path, 'rb') as f:
        file_hash = hashlib.md5(f.read(1024*1024)).hexdigest()[:12]
    stub_key = file_hash
    log(f"Loaded {len(video_frames)} frames from {Path(video_path).name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    import torch
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)} VRAM free: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    else:
        log("WARNING: No GPU detected, running on CPU")

    progress(0.02, desc="Tracking players...")
    tracker = Tracker("models/player_detector.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True,
        stub_path=str(CACHE_DIR / f"{stub_key}_track.pkl"))
    log(f"Tracking done — {len(tracks['players'])} frames")

    log("Interpolating ball & player positions...")
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracks['players'] = tracker.interpolate_player_positions(tracks['players'])
    tracker.add_position_to_tracks(tracks)

    log("Estimating camera movement...")
    progress(0.15, desc="Camera estimation...")
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_move = cam_est.get_camera_movement(
        video_frames, read_from_stub=True,
        stub_path=str(CACHE_DIR / f"{stub_key}_cam.pkl"))
    cam_est.add_adjust_positions_to_tracks(tracks, cam_move)

    log("Computing homography...")
    progress(0.25, desc="Pitch keypoints -> homography...")
    kp_detector = PitchKeypointDetector(
        model_path="models/pitch_keypoint_detector.pt")
    vt = ViewTransformer(kp_detector)
    vt.add_transformed_position_to_tracks(tracks, video_frames)

    sde = SpeedDistanceEstimator()
    sde.add_speed_and_distance_to_tracks(tracks, fps=fps)

    log("Assigning teams...")
    progress(0.35, desc="Assigning teams by jersey color...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    for fn, pt in enumerate(tracks["players"]):
        for pid, pd in pt.items():
            team = team_assigner.get_player_team(video_frames[fn], pd["bbox"], pid)
            tracks["players"][fn][pid]["team"] = team
            tracks["players"][fn][pid]["team_color"] = team_assigner.team_colors[team]

    ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for fn, pt in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][fn][1]["bbox"]
        assigned = ball_assigner.assign_ball_to_player(pt, ball_bbox)
        if assigned != -1:
            tracks["players"][fn][assigned]["has_ball"] = True
            team_ball_control.append(tracks["players"][fn][assigned]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    log("Detecting formations...")
    progress(0.45, desc="Detecting formations...")
    total_frames = len(video_frames)
    f1, n1, c1 = detect_team_formation(
        tracks, 1, frame_nums=range(0, total_frames, 30), method='kmeans')
    f2, n2, c2 = detect_team_formation(
        tracks, 2, frame_nums=range(0, total_frames, 30), method='kmeans')
    log(f"Team 1 formation: {n1} (conf={c1:.0%})")
    log(f"Team 2 formation: {n2} (conf={c2:.0%})")

    log("Rendering frames...")
    progress(0.5, desc="Rendering frames...")
    h, w = video_frames[0].shape[:2]
    out_frames = []
    total = len(video_frames)

    kp_detector2 = PitchKeypointDetector(
        model_path="models/pitch_keypoint_detector.pt")
    minimap_renderer = MinimapRenderer(w=350, h=230)
    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors[1]),
        2: tuple(int(c) for c in team_assigner.team_colors[2]),
    }

    for fn in range(total):
        if fn % 30 == 0:
            log(f"Rendering frame {fn}/{total}...")
            progress(0.5 + 0.45 * fn / total,
                     desc=f"Rendering frame {fn}/{total}...")
        frame = video_frames[fn].copy()

        # Draw players, referees, ball
        for tid, data in tracks["players"][fn].items():
            color = data.get("team_color", (0, 255, 0))
            frame = tracker.draw_ellipse(frame, data["bbox"], color, tid)
            if data.get("has_ball"):
                frame = tracker.draw_triangle(frame, data["bbox"], (0, 255, 0))
        for tid, data in tracks["referees"][fn].items():
            frame = tracker.draw_ellipse(frame, data["bbox"], (255, 255, 0), tid)
        if 1 in tracks["ball"][fn]:
            frame = tracker.draw_triangle(
                frame, tracks["ball"][fn][1]["bbox"], (0, 255, 255))

        # Camera movement overlay (top-left)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        dx, dy = cam_move[fn]
        cv2.putText(frame, "Camera Movement", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"X: {dx:.1f}  Y: {dy:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # Speed / distance per player
        for tid, data in tracks["players"][fn].items():
            spd = data.get("speed")
            dst = data.get("distance")
            bbox = data.get("bbox")
            if spd is None or bbox is None:
                continue
            pos = list(get_foot_position(bbox))
            pos[1] += 40
            pos = tuple(map(int, pos))
            cv2.putText(frame, f"{spd:.1f} km/h", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if dst is not None:
                cv2.putText(frame, f"{dst:.1f} m",
                            (pos[0], pos[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Team ball control (top-right)
        t1 = int(np.sum(team_ball_control[:fn + 1] == 1))
        t2 = int(np.sum(team_ball_control[:fn + 1] == 2))
        tot = t1 + t2 + 1e-6
        bc_ov = frame.copy()
        cv2.rectangle(bc_ov, (w - 310, 10), (w - 10, 100),
                      (255, 255, 255), -1)
        cv2.addWeighted(bc_ov, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"Team1: {t1 / tot * 100:.0f}%",
                    (w - 290, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team2: {t2 / tot * 100:.0f}%",
                    (w - 290, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        # Formation overlay (bottom-left)
        cv2.putText(frame, f"Team 1: {n1}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {n2}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Pitch keypoints + minimap
        if show_keypoints:
            kps = kp_detector2.detect_smoothed(video_frames[fn])
            if kps is not None:
                frame = kp_detector2.draw_keypoints(frame, kps)
        if show_minimap:
            mm = minimap_renderer.render(tracks, fn, team_colors=team_colors)
            frame = minimap_renderer.overlay(frame, mm, pos="bottom_right")

        out_frames.append(frame)
    log(f"Rendered {total} frames — saving video...")

    progress(0.95, desc="Saving output video...")
    out_path = str(CACHE_DIR / f"{stub_key}_output.mp4")
    save_video(out_frames, out_path, fps=fps)
    log(f"Saved: {out_path}")

    heatmap_paths = []
    if show_heatmap:
        log("Generating heatmaps...")
        hm = HeatmapGenerator(w=630, h=420)
        hm.update_from_tracks(tracks)
        hm_path = str(CACHE_DIR / f"{stub_key}_hm_both.png")
        cv2.imwrite(hm_path, hm.render_both())
        heatmap_paths.append(hm_path)
        hm1_path = str(CACHE_DIR / f"{stub_key}_hm_t1.png")
        cv2.imwrite(hm1_path, hm.render_team(1, cv2.COLORMAP_WINTER))
        heatmap_paths.append(hm1_path)
        hm2_path = str(CACHE_DIR / f"{stub_key}_hm_t2.png")
        cv2.imwrite(hm2_path, hm.render_team(2, cv2.COLORMAP_AUTUMN))
        heatmap_paths.append(hm2_path)
        log("Saved heatmaps")

    # Extract stats
    import pandas as pd
    player_stats = {}
    for pt in tracks["players"]:
        for pid, pd_data in pt.items():
            if pid not in player_stats:
                player_stats[pid] = {"Max Speed (km/h)": 0.0, "Distance (m)": 0.0, "Team": pd_data.get("team", 1)}
            spd = pd_data.get("speed")
            dst = pd_data.get("distance")
            if spd is not None and spd > player_stats[pid]["Max Speed (km/h)"]:
                player_stats[pid]["Max Speed (km/h)"] = spd
            if dst is not None and dst > player_stats[pid]["Distance (m)"]:
                player_stats[pid]["Distance (m)"] = dst

    df_players = pd.DataFrame.from_dict(player_stats, orient='index').reset_index()
    df_players.rename(columns={"index": "Player ID"}, inplace=True)
    df_players["Max Speed (km/h)"] = df_players["Max Speed (km/h)"].round(1)
    df_players["Distance (m)"] = df_players["Distance (m)"].round(1)

    top_speed = df_players.sort_values(by="Max Speed (km/h)", ascending=False).head(5)
    top_dist = df_players.sort_values(by="Distance (m)", ascending=False).head(5)

    match_stats = f"""
### Match Statistics
- **Team 1 Formation:** {n1} | **Ball Control:** {t1/tot*100:.0f}%
- **Team 2 Formation:** {n2} | **Ball Control:** {t2/tot*100:.0f}%
    """

    progress(1.0, desc="Done!")
    log("Processing complete!")
    return out_path, heatmap_paths if heatmap_paths else None, match_stats, top_speed, top_dist


def video_ui(file, show_kp, show_mm, show_hm, progress=gr.Progress()):
    if file is None:
        return None, None, "", None, None, log("No file uploaded")
    video_path = file
    try:
        out_vid, hm_paths, match_stats, top_speed, top_dist = process_video(
            video_path, show_kp, show_mm, show_hm, progress)
        
        return (out_vid, 
                gr.update(value=hm_paths, visible=True) if hm_paths else gr.update(visible=False), 
                match_stats, top_speed, top_dist, log("Done!"))
    except Exception as e:
        err = traceback.format_exc()
        log(f"ERROR: {e}")
        log(err[-500:])
        raise gr.Error(str(e))


with gr.Blocks(title="Football AI Analysis", css="""
    footer { display: none !important; }
    .gradio-container { max-width: 1100px !important; margin: auto; }
    #log-box { font-size: 12px; font-family: monospace; background: #1e1e1e; color: #0f0; padding: 10px; border-radius: 4px; height: 300px; overflow-y: auto; white-space: pre-wrap; }
""") as demo:
    gr.Markdown("# Football AI Analysis")
    gr.Markdown("Detect players, ball, referees -- draw pitch keypoints, minimap, heatmap")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload video", format="mp4")
            with gr.Accordion("Overlay options", open=True):
                show_kp = gr.Checkbox(label="Pitch Keypoints", value=True)
                show_mm = gr.Checkbox(label="Minimap", value=True)
                show_hm = gr.Checkbox(label="Heatmap", value=False)
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            video_output = gr.Video(label="Result", format="mp4")
            
            # --- New Statistics Section ---
            stats_markdown = gr.Markdown("### Match Statistics\n*(Waiting for analysis...)*")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Top 5 Players by Speed")
                    top_speed_df = gr.Dataframe(headers=["Player ID", "Max Speed (km/h)", "Distance (m)", "Team"])
                with gr.Column():
                    gr.Markdown("#### Top 5 Players by Distance")
                    top_dist_df = gr.Dataframe(headers=["Player ID", "Max Speed (km/h)", "Distance (m)", "Team"])
            
            heatmap_gallery = gr.Gallery(label="Heatmaps", columns=3, visible=False)
            status_log = gr.HTML(value="<div id='log-box'>Waiting...</div>", label="Status Log")

    btn.click(
        fn=video_ui,
        inputs=[video_input, show_kp, show_mm, show_hm],
        outputs=[video_output, heatmap_gallery, stats_markdown, top_speed_df, top_dist_df, status_log],
    )


if __name__ == "__main__":
    import sys
    is_notebook = any(
        mod in sys.modules for mod in
        ["google.colab", "kaggle"])
    demo.launch(share=is_notebook or "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
                server_port=7862)
