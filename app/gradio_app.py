import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pickle
import threading
import time
import traceback

import gradio as gr
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from asigners import TeamAssigner, PlayerBallAssigner
from estimators import (CameraMovementEstimator,
                        ViewTransformer,
                        SpeedDistanceEstimator)
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector
from heatmap_generator.heatmap_generator import HeatmapGenerator
from minimap.minimap_renderer import MinimapRenderer

CACHE_DIR = Path("cache_gradio")
CACHE_DIR.mkdir(exist_ok=True)

_log_lines = []
_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        ts = time.strftime("%H:%M:%S")
        _log_lines.append(f"[{ts}] {msg}")
        return "\n".join(_log_lines[-50:])


def process_video(video_path, show_keypoints, show_minimap, show_heatmap,
                  progress=gr.Progress()):
    log("Reading video...")
    progress(0, desc="Reading video...")
    video_frames = read_video(video_path)
    stub_key = Path(video_path).stem
    log(f"Loaded {len(video_frames)} frames")

    log("Tracking players...")
    progress(0.05, desc="Tracking players...")
    tracker = Tracker("models/player_detector.pt")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True,
        stub_path=str(CACHE_DIR / f"{stub_key}_track.pkl"))
    tracker.add_position_to_tracks(tracks)
    log(f"Players: {len(tracks['players'])} frames tracked")

    log("Camera estimation...")
    progress(0.15, desc="Camera estimation...")
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_move = cam_est.get_camera_movement(
        video_frames, read_from_stub=True,
        stub_path=str(CACHE_DIR / f"{stub_key}_cam.pkl"))
    cam_est.add_adjust_positions_to_tracks(tracks, cam_move)

    log("View transform & interpolation...")
    progress(0.25, desc="Pitch keypoints → homography...")
    kp_detector = PitchKeypointDetector(
        model_path="models/pitch_keypoint_detector.pt")
    vt = ViewTransformer(kp_detector)
    vt.add_transformed_position_to_tracks(tracks, video_frames)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    sde = SpeedDistanceEstimator()
    sde.add_speed_and_distance_to_tracks(tracks)

    log("Assigning teams by jersey color...")
    progress(0.35, desc="Assigning teams...")
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

    log("Rendering annotations...")
    progress(0.5, desc="Rendering annotations...")
    annotated = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    annotated = cam_est.draw_camera_movement(annotated, cam_move)
    sde.draw_speed_and_distance(annotated, tracks)

    kp_detector = PitchKeypointDetector(
        model_path="models/pitch_keypoint_detector.pt")
    minimap_renderer = MinimapRenderer(w=350, h=230)
    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors[1]),
        2: tuple(int(c) for c in team_assigner.team_colors[2]),
    }

    log("Rendering frames with overlays...")
    progress(0.7, desc=f"Rendering frames (0/{len(annotated)})...")
    out_frames = []
    total = len(annotated)
    for fn, frame in enumerate(annotated):
        if fn % 30 == 0:
            progress(0.7 + 0.25 * fn / total,
                     desc=f"Rendering frames ({fn}/{total})...")
        frame = frame.copy()
        if show_keypoints:
            kps = kp_detector.detect_smoothed(video_frames[fn])
            if kps is not None:
                frame = kp_detector.draw_keypoints(frame, kps)
        if show_minimap:
            mm = minimap_renderer.render(tracks, fn, team_colors=team_colors)
            frame = minimap_renderer.overlay(frame, mm, pos="bottom_right")
        out_frames.append(frame)
    log(f"Rendered {total} frames")

    progress(0.95, desc="Saving output video...")
    out_path = str(CACHE_DIR / f"{stub_key}_output.mp4")
    save_video(out_frames, out_path)
    log("Saved output video")

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
        log("Heatmaps saved")

    progress(1.0, desc="Done!")
    log("Processing complete!")
    return out_path, heatmap_paths if heatmap_paths else None


def video_ui(file, show_kp, show_mm, show_hm, progress=gr.Progress()):
    if file is None:
        return None, None, log("No file uploaded")
    video_path = file
    try:
        out_vid, hm_paths = process_video(
            video_path, show_kp, show_mm, show_hm, progress)
        return out_vid, hm_paths, log("Done!")
    except Exception as e:
        err = traceback.format_exc()
        log(f"ERROR: {e}")
        log(err[-500:])
        raise gr.Error(str(e))


with gr.Blocks(title="Football AI Analysis", css="""
    footer { display: none !important; }
    .gradio-container { max-width: 1100px !important; margin: auto; }
    #log-box { font-size: 12px; font-family: monospace; background: #1e1e1e; color: #0f0; padding: 10px; border-radius: 4px; height: 180px; overflow-y: auto; white-space: pre-wrap; }
""") as demo:
    gr.Markdown("# Football AI Analysis")
    gr.Markdown("Detect players, ball, referees — draw pitch keypoints, minimap, heatmap")

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
            heatmap_gallery = gr.Gallery(label="Heatmaps", columns=3,
                                         visible=False)
            status_log = gr.HTML(value="<div id='log-box'>Waiting...</div>",
                                 label="Status Log")

    btn.click(
        fn=video_ui,
        inputs=[video_input, show_kp, show_mm, show_hm],
        outputs=[video_output, heatmap_gallery, status_log],
    )


if __name__ == "__main__":
    import sys
    is_notebook = any(
        mod in sys.modules for mod in
        ["google.colab", "kaggle"])
    demo.launch(share=is_notebook or "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
                server_port=7862)
