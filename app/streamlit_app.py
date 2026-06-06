import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import pickle

import streamlit as st
import numpy as np
from utils import read_video
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

CACHE = Path("cache")
INPUT = Path("input_videos")
CACHE.mkdir(exist_ok=True)


# ── pipeline ───────────────────────────────────────────────

def _available():
    items = []
    if INPUT.is_dir():
        items += [str(f) for f in sorted(INPUT.iterdir())
                  if f.suffix in (".mp4", ".avi", ".mov")]
    if CACHE.is_dir():
        for d in sorted(CACHE.iterdir()):
            if d.is_dir() and (d / "tracks_full.pkl").exists() and (d / "source.mp4").exists():
                items.append(f"cache:{d.name}")
    return items


def _analyze(video_path, key):
    sd = CACHE / key
    sd.mkdir(parents=True, exist_ok=True)

    frames = read_video(video_path)
    tr = Tracker("models/player_detector.pt")
    tracks = tr.get_object_tracks(
        frames, read_from_stub=True,
        stub_path=str(sd / "track_stubs.pkl"))
    tr.add_position_to_tracks(tracks)

    cam = CameraMovementEstimator(frames[0])
    cm = cam.get_camera_movement(
        frames, read_from_stub=True,
        stub_path=str(sd / "camera_movement_stub.pkl"))
    cam.add_adjust_positions_to_tracks(tracks, cm)

    kpd = PitchKeypointDetector("models/old/pitch_keypoint_detector.pt")
    ViewTransformer(kpd).add_transformed_position_to_tracks(tracks, frames)

    tracks["ball"] = tr.interpolate_ball_positions(tracks["ball"])

    SpeedDistanceEstimator().add_speed_and_distance_to_tracks(tracks)

    ta = TeamAssigner()
    ta.assign_team_color(frames[0], tracks["players"][0])
    for fn, pt in enumerate(tracks["players"]):
        for pid, pd in pt.items():
            team = ta.get_player_team(frames[fn], pd["bbox"], pid)
            tracks["players"][fn][pid]["team"] = team
            tracks["players"][fn][pid]["team_color"] = ta.team_colors[team]

    ba = PlayerBallAssigner()
    tbc = []
    for fn, pt in enumerate(tracks["players"]):
        bb = tracks["ball"][fn][1]["bbox"]
        a = ba.assign_ball_to_player(pt, bb)
        if a != -1:
            tracks["players"][fn][a]["has_ball"] = True
            tbc.append(tracks["players"][fn][a]["team"])
        else:
            tbc.append(tbc[-1] if tbc else 0)
    tbc = np.array(tbc)

    with open(sd / "tracks_full.pkl", "wb") as f:
        pickle.dump(tracks, f)
    with open(sd / "cam_move.pkl", "wb") as f:
        pickle.dump(cm, f)
    np.save(sd / "team_ball_control.npy", tbc)

    del frames
    return tracks, cm, tbc, ta


def _load(key):
    sd = CACHE / key
    with open(sd / "tracks_full.pkl", "rb") as f:
        tracks = pickle.load(f)
    with open(sd / "cam_move.pkl", "rb") as f:
        cm = pickle.load(f)
    tbc = np.load(sd / "team_ball_control.npy")
    ta = TeamAssigner()
    ta.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
    for _, d in tracks["players"][0].items():
        tc = d.get("team_color")
        if tc is not None:
            ta.team_colors[d.get("team", 1)] = tuple(int(c) for c in tc)
    return tracks, cm, tbc, ta


def _render(video_path, tracks, cm, tbc, ta, show_kp, show_mm, out):
    tr = Tracker()
    kpd = PitchKeypointDetector("models/old/pitch_keypoint_detector.pt")
    mm = MinimapRenderer(w=350, h=230)
    tc = {1: tuple(int(c) for c in ta.team_colors[1]),
          2: tuple(int(c) for c in ta.team_colors[2])}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))

    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, d in tracks["players"][fn].items():
            color = d.get("team_color", (0, 255, 0))
            frame = tr.draw_ellipse(frame, d["bbox"], color, tid)
            if d.get("has_ball"):
                frame = tr.draw_triangle(frame, d["bbox"], (0, 255, 0))
        for _, d in tracks["referees"][fn].items():
            frame = tr.draw_ellipse(frame, d["bbox"], (255, 255, 0))
        if 1 in tracks["ball"][fn]:
            frame = tr.draw_triangle(
                frame, tracks["ball"][fn][1]["bbox"], (0, 255, 255))

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (500, 100), (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
        dx, dy = cm[fn]
        cv2.putText(frame, "Camera Movement", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"X: {dx:.1f}  Y: {dy:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        for _, d in tracks["players"][fn].items():
            spd = d.get("speed")
            dst = d.get("distance")
            bb = d.get("bbox")
            if spd is None or bb is None:
                continue
            fx = int((bb[0] + bb[2]) / 2)
            fy = int(bb[3]) + 40
            cv2.putText(frame, f"{spd:.2f} km/h", (fx, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if dst is not None:
                cv2.putText(frame, f"{dst:.2f} m", (fx, fy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        t1 = int(np.sum(tbc[:fn + 1] == 1))
        t2 = int(np.sum(tbc[:fn + 1] == 2))
        tot = t1 + t2 + 1e-6
        ov = frame.copy()
        cv2.rectangle(ov, (w - 550, h - 120), (w - 370, h - 30),
                      (255, 255, 255), -1)
        cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"Team1: {t1 / tot * 100:.0f}%",
                    (w - 530, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team2: {t2 / tot * 100:.0f}%",
                    (w - 530, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)

        if show_kp or show_mm:
            kps = kpd.detect_smoothed(frame)
        if show_kp and kps is not None:
            frame = kpd.draw_keypoints(frame, kps)
        if show_mm:
            m = mm.render(tracks, fn, team_colors=tc)
            frame = mm.overlay(frame, m, pos="bottom_right")

        writer.write(frame)
        fn += 1

    cap.release()
    writer.release()


# ── UI ─────────────────────────────────────────────────────

st.title("Football AI Analysis")
st.caption("Detection → tracking → team assign → speed/distance → minimap → heatmap")

if "ready" not in st.session_state:
    st.session_state.ready = False

with st.sidebar:
    st.header("Video")
    uploaded = st.file_uploader("Upload", type=["mp4", "avi", "mov"])
    available = _available()
    selected = st.selectbox("Or pick", [""] + available)
    use_cache = st.checkbox("Use cache", value=True)
    analyze = st.button("Analyze", type="primary")

    if analyze:
        src = None
        key = None

        if uploaded:
            key = Path(uploaded.name).stem
            cd = CACHE / key
            cd.mkdir(parents=True, exist_ok=True)
            src = str(cd / "source.mp4")
            with open(src, "wb") as f:
                f.write(uploaded.read())
        elif selected:
            if selected.startswith("cache:"):
                key = selected.split(":", 1)[1]
                src = str(CACHE / key / "source.mp4")
                if use_cache and (CACHE / key / "tracks_full.pkl").exists():
                    with st.spinner("Loading cache..."):
                        tr, cm, tbc, ta = _load(key)
                        st.session_state.update(
                            ready=True, stub=key, src=src,
                            tracks=tr, cam_move=cm,
                            team_ball_control=tbc, team_assigner=ta)
                        st.success("Loaded!")
                    key = None
            else:
                key = Path(selected).stem
                src = selected

        if key and src:
            with st.spinner("Analyzing..."):
                tr, cm, tbc, ta = _analyze(src, key)
                st.session_state.update(
                    ready=True, stub=key, src=src,
                    tracks=tr, cam_move=cm,
                    team_ball_control=tbc, team_assigner=ta)
                st.success("Done!")

    if st.session_state.ready:
        st.divider()
        st.header("Overlay")
        show_kp = st.checkbox("Pitch Keypoints", value=True)
        show_mm = st.checkbox("Minimap", value=True)

if not st.session_state.ready:
    st.info("Select a video and click **Analyze**.")
    st.stop()

tr = st.session_state.tracks
cm = st.session_state.cam_move
tbc = st.session_state.team_ball_control
ta = st.session_state.team_assigner

# ── collapsible sections ───────────────────────────────────

with st.expander("Summary", True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", len(tbc))
    col2.metric("Players", len(set(
        tid for f in tr["players"] for tid in f.keys())))
    col3.metric("Team 1 Possession",
                f"{np.mean(tbc == 1) * 100:.0f}%" if len(tbc) else "0%")
    col4.metric("Team 2 Possession",
                f"{np.mean(tbc == 2) * 100:.0f}%" if len(tbc) else "0%")

with st.expander("Video", False):
    show_heat = st.checkbox("Show Heatmap below video")
    render = st.button("Render Video")
    out_key = f"{st.session_state.stub}_kp{show_kp}_mm{show_mm}.avi"
    out_path = CACHE / out_key

    if render:
        with st.spinner("Rendering video..."):
            _render(st.session_state.src, tr, cm, tbc, ta,
                    show_kp, show_mm, out_path)
        st.success("Rendered!")

    if out_path.exists():
        st.video(str(out_path))
        with open(str(out_path), "rb") as f:
            st.download_button("Download", f.read(), file_name="output.avi")

    if show_heat:
        with st.spinner("Generating heatmap..."):
            hm = HeatmapGenerator(w=630, h=420)
            hm.update_from_tracks(tr)
        c1, c2, c3 = st.columns(3)
        c1.image(hm.render_both(), caption="Both", use_container_width=True)
        c2.image(hm.render_team(1, cv2.COLORMAP_WINTER),
                 caption="Team 1", use_container_width=True)
        c3.image(hm.render_team(2, cv2.COLORMAP_AUTUMN),
                 caption="Team 2", use_container_width=True)

with st.expander("Heatmap Only", False):
    if st.button("Generate Heatmap"):
        with st.spinner("..."):
            hm = HeatmapGenerator(w=630, h=420)
            hm.update_from_tracks(tr)
        c1, c2, c3 = st.columns(3)
        c1.image(hm.render_both(), caption="Both", use_container_width=True)
        c2.image(hm.render_team(1, cv2.COLORMAP_WINTER),
                 caption="Team 1", use_container_width=True)
        c3.image(hm.render_team(2, cv2.COLORMAP_AUTUMN),
                 caption="Team 2", use_container_width=True)
