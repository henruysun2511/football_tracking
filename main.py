import os
import cv2
import numpy as np
import pickle
import argparse

from utils import read_video
from trackers import Tracker
from asigners import TeamAssigner, PlayerBallAssigner
from estimators import (CameraMovementEstimator,
                        ViewTransformer,
                        SpeedDistanceEstimator)
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector
from heatmap_generator.heatmap_generator import HeatmapGenerator
from minimap.minimap_renderer import MinimapRenderer
from formations import detect_team_formation
from ocr import read_jersey_number, draw_jersey_number


STUB_DIR = 'stubs'


def phase1_tracking():
    video_frames = read_video('input_videos/sample.mp4')
    print(f"Loaded {len(video_frames)} frames")

    tracker = Tracker('models/player_detector.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=f'{STUB_DIR}/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    cam_est = CameraMovementEstimator(video_frames[0])
    cam_move = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=f'{STUB_DIR}/camera_movement_stub.pkl')
    cam_est.add_adjust_positions_to_tracks(tracks, cam_move)

    kp_detector = PitchKeypointDetector(
        model_path='models/old/pitch_keypoint_detector.pt')
    vt = ViewTransformer(kp_detector)
    vt.add_transformed_position_to_tracks(tracks, video_frames)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    sde = SpeedDistanceEstimator()
    sde.add_speed_and_distance_to_tracks(tracks)

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
                  team_ball_control, team_assigner):
    print("Phase 2: Rendering...")
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"VRAM free: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    else:
        print("WARNING: No GPU detected, running on CPU")
    tracker = Tracker()

    kp_detector = PitchKeypointDetector(
        model_path='models/old/pitch_keypoint_detector.pt')
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
        'output_videos/output_enhanced.avi', fourcc, 30, (w, h))

    total = len(video_frames)

    # Detect formations
    f1, n1, c1 = detect_team_formation(
        tracks, 1, frame_nums=range(0, total, 30), method='kmeans')
    f2, n2, c2 = detect_team_formation(
        tracks, 2, frame_nums=range(0, total, 30), method='kmeans')
    print(f"Team 1 formation: {n1} (conf={c1:.0%})")
    print(f"Team 2 formation: {n2} (conf={c2:.0%})")

    jersey_cache = {}
    print("Jersey OCR ready (lazy init on first use)")
    for frame_num in range(total):
        if frame_num % 30 == 0:
            print(f"Rendering frame {frame_num}/{total}...")
        frame = video_frames[frame_num].copy()

        # draw annotations (inline để không tạo list trung gian)
        for tid, data in tracks["players"][frame_num].items():
            color = data.get("team_color", (0,255,0))
            frame = tracker.draw_ellipse(frame, data["bbox"], color, tid)
            if data.get("has_ball"):
                frame = tracker.draw_triangle(frame, data["bbox"], (0,255,0))
            # Jersey number OCR (lazy, cached per track_id)
            if tid not in jersey_cache:
                num = read_jersey_number(
                    video_frames[frame_num], data['bbox'],
                    cache=jersey_cache, track_id=tid)
                if num is not None:
                    jersey_cache[tid] = num
            if tid in jersey_cache:
                draw_jersey_number(
                    frame, data['bbox'], jersey_cache[tid], tid)
        for tid, data in tracks["referees"][frame_num].items():
            frame = tracker.draw_ellipse(frame, data["bbox"], (255,255,0), tid)
        if 1 in tracks["ball"][frame_num]:
            frame = tracker.draw_triangle(frame,
                tracks["ball"][frame_num][1]["bbox"], (0,255,255))

        # camera movement
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        dx, dy = cam_move[frame_num]
        cv2.putText(frame, "Camera Movement", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"X: {dx:.1f}  Y: {dy:.1f}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        # speed / distance (below foot — reference style)
        for tid, data in tracks["players"][frame_num].items():
            spd = data.get("speed")
            dst = data.get("distance")
            bbox = data.get("bbox")
            if spd is None or bbox is None:
                continue
            x1, _, x2, y2 = map(int, bbox)
            foot_x = (x1 + x2) // 2
            foot_y = y2
            cv2.putText(frame, f"{spd:.2f} km/h",
                        (foot_x, foot_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2)
            if dst is not None:
                cv2.putText(frame, f"{dst:.2f} m",
                            (foot_x, foot_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2)

        # team ball control
        team1_ct = np.sum(team_ball_control[:frame_num+1] == 1)
        team2_ct = np.sum(team_ball_control[:frame_num+1] == 2)
        bc_total = team1_ct + team2_ct + 1e-6
        cv2.putText(frame, f"Team 1: {team1_ct/bc_total*100:.0f}%",
                    (w-530, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        cv2.putText(frame, f"Team 2: {team2_ct/bc_total*100:.0f}%",
                    (w-530, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # formation overlay
        cv2.putText(frame, f"Team 1: {n1}", (10, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Team 2: {n2}", (10, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # keypoints + minimap
        kps = kp_detector.detect_smoothed(video_frames[frame_num])
        if kps is not None:
            frame = kp_detector.draw_keypoints(frame, kps)
        minimap = minimap_renderer.render(
            tracks, frame_num, team_colors=team_colors)
        frame = minimap_renderer.overlay(
            frame, minimap, pos='bottom_right')

        out_writer.write(frame)

    out_writer.release()
    print("Saved: output_videos/output_enhanced.avi")

    cv2.imwrite('output_videos/heatmap_both.png',
                heatmap_gen.render_both())
    cv2.imwrite('output_videos/heatmap_team1.png',
                heatmap_gen.render_team(1, cv2.COLORMAP_WINTER))
    cv2.imwrite('output_videos/heatmap_team2.png',
                heatmap_gen.render_team(2, cv2.COLORMAP_AUTUMN))
    print("Saved heatmaps to output_videos/")


def phase2_render_from_stubs():
    with open(f'{STUB_DIR}/tracks_full.pkl', 'rb') as f:
        tracks = pickle.load(f)
    with open(f'{STUB_DIR}/cam_move.pkl', 'rb') as f:
        cam_move = pickle.load(f)
    team_ball_control = np.load(f'{STUB_DIR}/team_ball_control.npy')

    video_frames = read_video('input_videos/sample.mp4')

    team_assigner = TeamAssigner()
    team_assigner.team_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
    for tid, data in tracks['players'][0].items():
        tc = data.get('team_color')
        if tc is not None:
            team_assigner.team_colors[data.get('team', 1)] = \
                tuple(int(c) for c in tc)

    phase2_render(video_frames, tracks, cam_move,
                  team_ball_control, team_assigner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tracking', 'render', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.mode == 'tracking':
        phase1_tracking()
    elif args.mode == 'render':
        phase2_render_from_stubs()
    else:
        vf, tr, cm, tbc, ta = phase1_tracking()
        phase2_render(vf, tr, cm, tbc, ta)
