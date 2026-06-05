import os
import cv2
import numpy as np
import pickle
import argparse

from utils import read_video, save_video
from trackers import Tracker
from asigners import TeamAssigner, PlayerBallAssigner
from estimators import (CameraMovementEstimator,
                        ViewTransformer,
                        SpeedDistanceEstimator)
from pitch_keypoint_detector.pitch_keypoint_detector import PitchKeypointDetector
from heatmap_generator.heatmap_generator import HeatmapGenerator
from minimap.minimap_renderer import MinimapRenderer


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

    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)

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
    tracker = Tracker()  # no model needed — only drawing
    cam_est = CameraMovementEstimator(video_frames[0])
    sde = SpeedDistanceEstimator()

    kp_detector = PitchKeypointDetector(
        model_path='models/old/pitch_keypoint_detector.pt')
    heatmap_gen = HeatmapGenerator(w=630, h=420)
    heatmap_gen.update_from_tracks(tracks)
    minimap_renderer = MinimapRenderer(w=350, h=230)

    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors[1]),
        2: tuple(int(c) for c in team_assigner.team_colors[2]),
    }

    output_frames = []
    annotated = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)
    annotated = cam_est.draw_camera_movement(annotated, cam_move)
    sde.draw_speed_and_distance(annotated, tracks)

    total = len(annotated)
    for frame_num, frame in enumerate(annotated):
        if frame_num % 30 == 0:
            print(f"Rendering frame {frame_num}/{total}...")
        frame = frame.copy()
        kps = kp_detector.detect_smoothed(video_frames[frame_num])
        if kps is not None:
            frame = kp_detector.draw_keypoints(frame, kps)
        minimap = minimap_renderer.render(
            tracks, frame_num, team_colors=team_colors)
        frame = minimap_renderer.overlay(
            frame, minimap, pos='bottom_right')
        output_frames.append(frame)

    os.makedirs('output_videos', exist_ok=True)
    save_video(output_frames, 'output_videos/output_enhanced.avi')
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
