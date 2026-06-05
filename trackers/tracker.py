import warnings
warnings.filterwarnings("ignore", message=".*ByteTrack.*deprecated.*")

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO
import supervision as sv


class Tracker:
    def __init__(self, model_path):
        self.model   = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ── Detection ────────────────────────────────────────────
    def detect_frames(self, frames, batch_size=20):
        import torch
        detections = []
        kwargs = dict(conf=0.1, device='cpu')
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1024**3
            if free_gb > 1.0:
                kwargs['device'] = 0
                if total / 1024**3 <= 4:
                    kwargs['half'] = True
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model.predict(batch, **kwargs)
            detections.extend(results)
        return detections

    # ── Tracking ─────────────────────────────────────────────
    def get_object_tracks(self, frames,
                          read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass

        detections = self.detect_frames(frames)

        tracks = {
            "players":  [{} for _ in frames],
            "referees": [{} for _ in frames],
            "ball":     [{} for _ in frames],
        }

        for frame_num, detection in enumerate(detections):
            cls_names     = detection.names           # {id: name}
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Supervision detection object
            det_sv = sv.Detections.from_ultralytics(detection)

            # Goalkeeper → player (merge class)
            for i, class_id in enumerate(det_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    det_sv.class_id[i] = cls_names_inv["player"]

            # ByteTrack
            det_with_tracks = self.tracker.update_with_detections(det_sv)
            if det_with_tracks.tracker_id is None:
                continue

            for i in range(len(det_with_tracks)):
                bbox     = det_with_tracks.xyxy[i].tolist()
                cls_id   = det_with_tracks.class_id[i]
                track_id = det_with_tracks.tracker_id[i]
                name     = cls_names[cls_id]

                if name == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif name == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Ball — không track, lấy detection conf cao nhất
            ball_boxes = det_sv.xyxy[det_sv.class_id == cls_names_inv["ball"]]
            if len(ball_boxes) > 0:
                bbox = ball_boxes[0].tolist()
                tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ── Ball interpolation ───────────────────────────────────
    def interpolate_ball_positions(self, ball_positions):
        ball_list = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_list,
                          columns=['x1','y1','x2','y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    # ── Position helpers ─────────────────────────────────────
    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for tid, tdata in track.items():
                    bbox = tdata['bbox']
                    if obj == 'ball':
                        pos = ((bbox[0]+bbox[2])/2,
                               (bbox[1]+bbox[3])/2)
                    else:
                        pos = ((bbox[0]+bbox[2])/2, bbox[3])
                    tracks[obj][frame_num][tid]['position'] = pos

    # ── Draw annotations ─────────────────────────────────────
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2   = int(bbox[3])
        x_c  = int((bbox[0]+bbox[2])/2)
        w    = int(bbox[2]-bbox[0])
        cv2.ellipse(frame,
                    center=(x_c, y2),
                    axes=(w, int(0.35*w)),
                    angle=0, startAngle=-45, endAngle=235,
                    color=color, thickness=2, lineType=cv2.LINE_4)
        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1r = x_c - rect_w//2
            y1r = y2 - rect_h//2 + 15
            cv2.rectangle(frame, (x1r, y1r),
                          (x1r+rect_w, y1r+rect_h), color, -1)
            cv2.putText(frame, str(track_id),
                        (x1r+2, y1r+rect_h-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y1 = int(bbox[1])
        xc = int((bbox[0]+bbox[2])/2)
        pts = np.array([[xc, y1],
                        [xc-10, y1-20],
                        [xc+10, y1-20]], np.int32)
        cv2.drawContours(frame, [pts], 0, color, -1)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num,
                                team_ball_control):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 550, h - 120), (w - 50, h - 30),
                      (255,255,255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        team1 = np.sum(team_ball_control[:frame_num+1] == 1)
        team2 = np.sum(team_ball_control[:frame_num+1] == 2)
        total = team1 + team2 + 1e-6

        x = w - 530
        cv2.putText(frame,
                    f"Team 1: {team1/(total)*100:.0f}%",
                    (x, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 3)
        cv2.putText(frame,
                    f"Team 2: {team2/(total)*100:.0f}%",
                    (x, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)
        return frame

    def draw_annotations(self, video_frames, tracks,
                          team_ball_control):
        output = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for tid, data in tracks["players"][frame_num].items():
                color = data.get("team_color", (0,255,0))
                frame = self.draw_ellipse(frame, data["bbox"],
                                          color, tid)
                if data.get("has_ball"):
                    frame = self.draw_triangle(frame,
                                               data["bbox"],
                                               (0,255,0))

            for tid, data in tracks["referees"][frame_num].items():
                frame = self.draw_ellipse(frame, data["bbox"],
                                          (255,255,0), tid)

            if 1 in tracks["ball"][frame_num]:
                frame = self.draw_triangle(frame,
                    tracks["ball"][frame_num][1]["bbox"],
                    (0,255,255))

            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control)
            output.append(frame)
        return output