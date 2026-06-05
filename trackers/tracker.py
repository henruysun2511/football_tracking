from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2


class Tracker:
    def __init__(self, model_path=None):
        self.model   = YOLO(model_path) if model_path else None
        self.tracker = sv.ByteTrack() if model_path else None

    # ── Detection ────────────────────────────────────────────
    def detect_frames(self, frames, batch_size=20):
        if self.model is None:
            raise RuntimeError("Tracker created without model_path")
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        if device == 0 and torch.cuda.mem_get_info()[0] < 1*1024**3:
            device = 'cpu'
        if device == 0:
            self.model.to('cuda')
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model.predict(batch, conf=0.1, device=device)
            detections.extend(results)
        return detections

    # ── Tracking ─────────────────────────────────────────────
    def get_object_tracks(self, frames,
                          read_from_stub=False, stub_path=None):
        if self.model is None:
            raise RuntimeError("Tracker created without model_path")
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        print(f"DEBUG: {len(detections)} detection results from {len(frames)} frames")

        tracks = {
            "players":  [],
            "referees": [],
            "ball":     [],
        }

        for frame_num, detection in enumerate(detections):
            if frame_num == 0:
                print(f"DEBUG frame 0: type={type(detection)}")
                try:
                    print(f"  boxes={len(detection.boxes)} obb={hasattr(detection, 'obb')}")
                except:
                    print("  no boxes attr")
            cls_names     = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            det_sv = sv.Detections.from_ultralytics(detection)

            for i, class_id in enumerate(det_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    det_sv.class_id[i] = cls_names_inv["player"]

            det_with_tracks = self.tracker.update_with_detections(det_sv)

            if frame_num == 0:
                print(f"DEBUG ByteTrack: type={type(det_with_tracks)} len={len(det_with_tracks)}")
                if len(det_with_tracks) > 0:
                    first = det_with_tracks[0]
                    print(f"  first item type={type(first)} len={len(first)} items={first}")
                else:
                    print(f"  EMPTY! det_sv had {len(det_sv)} objects")

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in det_with_tracks:
                bbox     = frame_detection[0].tolist()
                cls_id   = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in det_sv:
                bbox   = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
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
