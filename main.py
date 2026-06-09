import os
import cv2
import numpy as np
import pickle
import argparse

from utils import read_video, get_foot_position
from trackers import Tracker
from asigners import TeamAssigner, PlayerBallAssigner
from estimators import (CameraMovementEstimator,
                        ViewTransformer,
                        SpeedDistanceEstimator)
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector
from heatmap_generator.heatmap_generator import HeatmapGenerator
from minimap.minimap_renderer import MinimapRenderer
from formations import detect_team_formation

# ==============================================================================
# 🎯 CẤU HÌNH ĐƯỜNG DẪN CHUẨN TRÊN KAGGLE (ĐÃ ĐỔI TỪ DATASET CỦA BẠN)
# ==============================================================================
KAGGLE_DATASET_DIR = '/kaggle/input/datasets/huysun'
PLAYER_MODEL_PATH = os.path.join(KAGGLE_DATASET_DIR, 'models/player_detector.pt')
PITCH_KP_MODEL_PATH = os.path.join(KAGGLE_DATASET_DIR, 'models/pitch_keypoint_detector.pt')
DEFAULT_VIDEO_PATH = os.path.join(KAGGLE_DATASET_DIR, 'input-videos/sample.mp4')

STUB_DIR = 'stubs'  # Lưu tại /kaggle/working/football_tracking/stubs


def phase1_tracking(video_path=DEFAULT_VIDEO_PATH):
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    tracker = Tracker(PLAYER_MODEL_PATH)  # 🛠️ Đã sửa đường dẫn
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=f'{STUB_DIR}/track_stubs.pkl')

    # ✅ Thứ tự đúng:
    # 1. Interpolate trước để fill gap bbox
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracks['players'] = tracker.interpolate_player_positions(tracks['players'])

    # 2. Add position (foot point) sau khi đã có đủ bbox
    tracker.add_position_to_tracks(tracks)

    # 3. Camera movement → position_adjusted
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_move = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=f'{STUB_DIR}/camera_movement_stub.pkl')
    cam_est.add_adjust_positions_to_tracks(tracks, cam_move)

    # 4. Homography dùng position_adjusted
    kp_detector = PitchKeypointDetector(model_path=PITCH_KP_MODEL_PATH)  # 🛠️ Đã sửa đường dẫn
    vt = ViewTransformer(kp_detector)
    vt.add_transformed_position_to_tracks(tracks, video_frames)

    # 5. Speed sau khi đã có position_transformed đầy đủ
    sde = SpeedDistanceEstimator()
    sde.add_speed_and_distance_to_tracks(tracks, fps=fps)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for pid, pdata in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], pdata['bbox'], pid)
            tracks['players'][frame_num][pid]['team'] = team
            tracks['players'][frame_num][pid]['team_color'] = \
                team_assigner.team_colors[team]

    ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned]['team'])
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    os.makedirs(STUB_DIR, exist_ok=True)
    with open(f'{STUB_DIR}/tracks_full.pkl', 'wb') as f:
        pickle.dump(tracks, f)
    with open(f'{STUB_DIR}/cam_move.pkl', 'wb') as f:
        pickle.dump(cam_move, f)
    np.save(f'{STUB_DIR}/team_ball_control.npy', team_ball_control)
    print("Saved stubs to stubs/")

    return video_frames, tracks, cam_move, team_ball_control, team_assigner


def phase2_render(video_frames, tracks, cam_move,
                  team_ball_control, team_assigner, fps=25):
    print("Phase 2: Rendering...")
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"VRAM free: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    else:
        print("WARNING: No GPU detected, running on CPU")
    tracker = Tracker()

    kp_detector = PitchKeypointDetector(model_path=PITCH_KP_MODEL_PATH)  # 🛠️ Đã sửa đường dẫn
    heatmap_gen = HeatmapGenerator(w=630, h=420)
    heatmap_gen.update_from_tracks(tracks)
    minimap_renderer = MinimapRenderer(w=350, h=230)

    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors[1]),
        2: tuple(int(c) for c in team_assigner.team_colors[2]),
    }

    os.makedirs('output_videos', exist_ok=True)

    h, w = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_writer = cv2.VideoWriter(
        'output_videos/output_enhanced.avi', fourcc, fps, (w, h))

    total = len(video_frames)

    f1, n1, c1 = detect_team_formation(
        tracks, 1, frame_nums=range(0, total, 30), method='kmeans')
    f2, n2, c2 = detect_team_formation(
        tracks, 2, frame_nums=range(0, total, 30), method='kmeans')
    print(f"Team 1 formation: {n1} (conf={c1:.0%})")
    print(f"Team 2 formation: {n2} (conf={c2:.0%})")

    for frame_num in range(total):
        if frame_num % 30 == 0:
            print(f"Rendering frame {frame_num}/{total}...")
        frame = video_frames[frame_num].copy()

        for tid, data in tracks["players"][frame_num].items():
            color = data.get("team_color", (0, 255, 0))
            frame = tracker.draw_ellipse(frame, data["bbox"], color, tid)
            if data.get("has_ball"):
                frame = tracker.draw_triangle(frame, data["bbox"], (0, 255, 0))
        for tid, data in tracks["referees"][frame_num].items():
            frame = tracker.draw_ellipse(frame, data["bbox"], (255, 255, 0), tid)
        if 1 in tracks["ball"][frame_num]:
            frame = tracker.draw_triangle(
                frame, tracks["ball"][frame_num][1]["bbox"], (0, 255, 255))

        # Camera movement
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        dx, dy = cam_move[frame_num]
        cv2.putText(frame, "Camera Movement", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"X: {dx:.1f}  Y: {dy:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # Speed / distance — white text với outline đen
        for tid, data in tracks["players"][frame_num].items():
            spd = data.get("speed")
            dst = data.get("distance")
            bbox = data.get("bbox")
            if spd is None or bbox is None:
                continue
            pos = list(get_foot_position(bbox))
            pos[1] += 40
            pos = tuple(map(int, pos))
            # Vẽ outline đen trước, chữ trắng sau
            cv2.putText(frame, f"{spd:.1f} km/h", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, f"{spd:.1f} km/h", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if dst is not None:
                pos2 = (pos[0], pos[1] + 20)
                cv2.putText(frame, f"{dst:.1f} m", pos2,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame, f"{dst:.1f} m", pos2,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Team ball control
        t1 = int(np.sum(team_ball_control[:frame_num + 1] == 1))
        t2 = int(np.sum(team_ball_control[:frame_num + 1] == 2))
        tot = t1 + t2 + 1e-6
        bc_ov = frame.copy()
        cv2.rectangle(bc_ov, (10, 110), (310, 200), (255, 255, 255), -1)
        cv2.addWeighted(bc_ov, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"Team1: {t1/tot*100:.0f}%",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team2: {t2/tot*100:.0f}%",
                    (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Formation
        fm_ov = frame.copy()
        cv2.rectangle(fm_ov, (10, h - 110), (310, h - 20), (255, 255, 255), -1)
        cv2.addWeighted(fm_ov, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"Team1 Formation: {n1}",
                    (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Team2 Formation: {n2}",
                    (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Keypoints + minimap
        kps = kp_detector.detect_smoothed(video_frames[frame_num])
        if kps is not None:
            frame = kp_detector.draw_keypoints(frame, kps)
        minimap = minimap_renderer.render(
            tracks, frame_num, team_colors=team_colors)
        frame = minimap_renderer.overlay(frame, minimap, pos='bottom_right')

        out_writer.write(frame)

    out_writer.release()
    print("Saved: output_videos/output_enhanced.avi")

    cv2.imwrite('output_videos/heatmap_both.png', heatmap_gen.render_both())
    cv2.imwrite('output_videos/heatmap_team1.png',
                heatmap_gen.render_team(1, cv2.COLORMAP_WINTER))
    cv2.imwrite('output_videos/heatmap_team2.png',
                heatmap_gen.render_team(2, cv2.COLORMAP_AUTUMN))
    print("Saved heatmaps to output_videos/")


def phase2_render_from_stubs(video_path=DEFAULT_VIDEO_PATH):
    with open(f'{STUB_DIR}/tracks_full.pkl', 'rb') as f:
        tracks = pickle.load(f)
    with open(f'{STUB_DIR}/cam_move.pkl', 'rb') as f:
        cam_move = pickle.load(f)
    team_ball_control = np.load(f'{STUB_DIR}/team_ball_control.npy')

    video_frames = read_video(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    team_assigner = TeamAssigner()
    team_assigner.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
    for tid, data in tracks['players'][0].items():
        tc = data.get('team_color')
        if tc is not None:
            team_assigner.team_colors[data.get('team', 1)] = \
                tuple(int(c) for c in tc)

    phase2_render(video_frames, tracks, cam_move,
                  team_ball_control, team_assigner, fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tracking', 'render', 'all'],
                        default='all')
    parser.add_argument('--video', default=DEFAULT_VIDEO_PATH,  # 🛠️ Đã sửa đường dẫn mặc định
                        help='Path to input video')
    args = parser.parse_args()

    if args.mode == 'tracking':
        phase1_tracking(args.video)
    elif args.mode == 'render':
        phase2_render_from_stubs(args.video)
    else:
        vf, tr, cm, tbc, ta = phase1_tracking(args.video)
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        phase2_render(vf, tr, cm, tbc, ta, fps=fps)