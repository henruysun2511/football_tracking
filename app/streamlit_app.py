import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pickle
import tempfile
import shutil

import streamlit as st
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

st.set_page_config(layout="wide", page_title="Football AI Analysis")

with open(Path(__file__).parent / "styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

CACHE_DIR = Path("cache")
INPUT_DIR = Path("input_videos")
CACHE_DIR.mkdir(exist_ok=True)


def get_available_videos():
    files = []
    if INPUT_DIR.is_dir():
        files += [str(f) for f in sorted(INPUT_DIR.iterdir())
                  if f.suffix in (".mp4", ".avi", ".mov")]
    if CACHE_DIR.is_dir():
        for d in CACHE_DIR.iterdir():
            if d.is_dir() and (d / "tracks_full.pkl").exists():
                files.append(f"cache:{d.name}")
    return files


def run_tracking(video_path, stub_key):
    stub_dir = CACHE_DIR / stub_key
    stub_dir.mkdir(parents=True, exist_ok=True)

    video_frames = read_video(video_path)

    tracker = Tracker("models/player_detector.pt")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=str(stub_dir / "track_stubs.pkl"))
    tracker.add_position_to_tracks(tracks)

    cam_est = CameraMovementEstimator(video_frames[0])
    cam_move = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=str(stub_dir / "camera_movement_stub.pkl"))
    cam_est.add_adjust_positions_to_tracks(tracks, cam_move)

    kp_detector = PitchKeypointDetector(
        model_path="models/old/pitch_keypoint_detector.pt")
    vt = ViewTransformer(kp_detector)
    vt.add_transformed_position_to_tracks(tracks, video_frames)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    sde = SpeedDistanceEstimator()
    sde.add_speed_and_distance_to_tracks(tracks)

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

    with open(stub_dir / "tracks_full.pkl", "wb") as f:
        pickle.dump(tracks, f)
    with open(stub_dir / "cam_move.pkl", "wb") as f:
        pickle.dump(cam_move, f)
    np.save(stub_dir / "team_ball_control.npy", team_ball_control)

    return video_frames, tracks, cam_move, team_ball_control, team_assigner


def load_from_cache(stub_key):
    video_frames = read_video(str(CACHE_DIR / stub_key / "source.mp4"))
    with open(CACHE_DIR / stub_key / "tracks_full.pkl", "rb") as f:
        tracks = pickle.load(f)
    with open(CACHE_DIR / stub_key / "cam_move.pkl", "rb") as f:
        cam_move = pickle.load(f)
    team_ball_control = np.load(CACHE_DIR / stub_key / "team_ball_control.npy")

    team_assigner = TeamAssigner()
    team_assigner.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
    for tid, data in tracks["players"][0].items():
        tc = data.get("team_color")
        if tc is not None:
            team_assigner.team_colors[data.get("team", 1)] = tuple(int(c) for c in tc)

    return video_frames, tracks, cam_move, team_ball_control, team_assigner


def render_video(video_frames, tracks, cam_move, team_ball_control,
                 team_assigner, show_keypoints, show_minimap):
    tracker = Tracker("models/player_detector.pt")
    cam_est = CameraMovementEstimator(video_frames[0])
    sde = SpeedDistanceEstimator()

    kp_detector = PitchKeypointDetector(
        model_path="models/old/pitch_keypoint_detector.pt")
    minimap_renderer = MinimapRenderer(w=350, h=230)

    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors[1]),
        2: tuple(int(c) for c in team_assigner.team_colors[2]),
    }

    annotated = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    annotated = cam_est.draw_camera_movement(annotated, cam_move)
    sde.draw_speed_and_distance(annotated, tracks)

    out = []
    for fn, frame in enumerate(annotated):
        frame = frame.copy()
        if show_keypoints:
            kps = kp_detector.detect_smoothed(video_frames[fn])
            if kps is not None:
                frame = kp_detector.draw_keypoints(frame, kps)
        if show_minimap:
            mm = minimap_renderer.render(tracks, fn, team_colors=team_colors)
            frame = minimap_renderer.overlay(frame, mm, pos="bottom_right")
        out.append(frame)
    return out


# ── UI ────────────────────────────────────────────────────

st.title("Football AI Analysis")
st.caption("Detect players, ball, referees -- draw pitch keypoints, minimap, heatmap")

with st.sidebar:
    st.header("Video Source")
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    available = get_available_videos()
    selected = st.selectbox("Or select from input/cache", [""] + available)
    use_cache = st.checkbox("Use cache if available", value=True)
    analyze_btn = st.button("Analyze", type="primary")

    st.divider()
    st.header("Overlay")
    show_kp = st.checkbox("Pitch Keypoints", value=True)
    show_mm = st.checkbox("Minimap", value=True)
    show_hm = st.checkbox("Heatmap", value=False)
    render_btn = st.button("Re-render", disabled=not show_hm)

st.divider()

if "video_frames" not in st.session_state:
    st.session_state.video_frames = None
if "tracks" not in st.session_state:
    st.session_state.tracks = None
if "processed" not in st.session_state:
    st.session_state.processed = False

if analyze_btn:
    src_path = None
    stub_key = None

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            src_path = tmp.name
            stub_key = Path(uploaded.name).stem
    elif selected:
        if selected.startswith("cache:"):
            name = selected.split(":", 1)[1]
            stub_key = name
            src_path = str(CACHE_DIR / name / "source.mp4")
            if use_cache and (CACHE_DIR / name / "tracks_full.pkl").exists():
                with st.spinner("Loading from cache..."):
                    vf, tr, cm, tbc, ta = load_from_cache(name)
                    st.session_state.video_frames = vf
                    st.session_state.tracks = tr
                    st.session_state.cam_move = cm
                    st.session_state.team_ball_control = tbc
                    st.session_state.team_assigner = ta
                    st.session_state.processed = True
                    st.success("Loaded from cache!")
                stub_key = None
            else:
                stub_key = Path(selected).stem
        else:
            src_path = selected
            stub_key = Path(selected).stem

    if stub_key and src_path:
        with st.spinner("Running analysis (may take a while)..."):
            vf, tr, cm, tbc, ta = run_tracking(src_path, stub_key)
            cache_dir = CACHE_DIR / stub_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not (cache_dir / "source.mp4").exists():
                shutil.copy2(src_path, cache_dir / "source.mp4")
            st.session_state.video_frames = vf
            st.session_state.tracks = tr
            st.session_state.cam_move = cm
            st.session_state.team_ball_control = tbc
            st.session_state.team_assigner = ta
            st.session_state.processed = True
            st.success("Analysis complete!")

if st.session_state.processed:
    vf = st.session_state.video_frames
    tr = st.session_state.tracks
    cm = st.session_state.cam_move
    tbc = st.session_state.team_ball_control
    ta = st.session_state.team_assigner

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames", len(vf))
    with col2:
        players = len(set(tid for f in tr["players"] for tid in f.keys()))
        st.metric("Players Detected", players)
    with col3:
        team1_pct = np.mean(tbc == 1) * 100 if len(tbc) > 0 else 0
        st.metric("Possession Team 1", f"{team1_pct:.0f}%")
    with col4:
        team2_pct = np.mean(tbc == 2) * 100 if len(tbc) > 0 else 0
        st.metric("Possession Team 2", f"{team2_pct:.0f}%")

    st.divider()
    st.subheader("Result")

    if render_btn and show_hm:
        with st.spinner("Rendering heatmap..."):
            hm = HeatmapGenerator(w=630, h=420)
            hm.update_from_tracks(tr)
            col1, col2, col3 = st.columns(3)
            col1.image(hm.render_both(), caption="Both Teams",
                       use_container_width=True)
            col2.image(hm.render_team(1, cv2.COLORMAP_WINTER),
                       caption="Team 1", use_container_width=True)
            col3.image(hm.render_team(2, cv2.COLORMAP_AUTUMN),
                       caption="Team 2", use_container_width=True)

    with st.spinner("Rendering video..."):
        out_frames = render_video(vf, tr, cm, tbc, ta, show_kp, show_mm)
        out_path = CACHE_DIR / "preview.avi"
        save_video(out_frames, str(out_path))

    st.video(str(out_path))
    with open(str(out_path), "rb") as f:
        st.download_button("Download video", f.read(), file_name="output.avi")

else:
    st.info("Select a video and click Analyze to start.")
