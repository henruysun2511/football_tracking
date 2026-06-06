import os
import cv2
import numpy as np
import pickle


class CameraMovementEstimator:
    def __init__(self, first_frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15,15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,
                      10, 0.03))
        h, w = first_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :20] = 1
        mask[:, -150:] = 1
        self.features = dict(maxCorners=100, qualityLevel=0.3,
                             minDistance=3, blockSize=7, mask=mask)
        self.gray_prev = cv2.cvtColor(first_frame,
                                       cv2.COLOR_BGR2GRAY)

    def get_camera_movement(self, frames,
                            read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass

        movement = [[0, 0]]
        old_gray  = self.gray_prev.copy()
        old_pts   = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if old_pts is None or len(old_pts) == 0:
                movement.append([0, 0])
                old_pts = cv2.goodFeaturesToTrack(gray, **self.features)
                old_gray = gray
                continue

            new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, old_pts, None, **self.lk_params)

            max_distance = 0
            dx = dy = 0
            for old, new, s in zip(old_pts, new_pts, st):
                if s[0] == 1:
                    ox, oy = old.ravel()
                    nx, ny = new.ravel()
                    dist = ((nx - ox)**2 + (ny - oy)**2) ** 0.5
                    if dist > max_distance:
                        max_distance = dist
                        dx = nx - ox
                        dy = ny - oy
            if max_distance > self.minimum_distance:
                movement.append([dx, dy])
                old_pts = cv2.goodFeaturesToTrack(gray, **self.features)
            else:
                movement.append([0, 0])

            old_gray = gray

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(movement, f)
        return movement

    def add_adjust_positions_to_tracks(self, tracks, movement):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                dx, dy = movement[frame_num]
                for tid in track:
                    pos = track[tid].get('position')
                    if pos:
                        track[tid]['position_adjusted'] = (
                            pos[0] - dx, pos[1] - dy)

    def draw_camera_movement(self, frames, movement):
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100),
                          (255,255,255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            dx, dy = movement[frame_num]
            cv2.putText(frame, f"Camera Movement",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,0), 3)
            cv2.putText(frame, f"X: {dx:.1f}  Y: {dy:.1f}",
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,0), 3)
            output.append(frame)
        return output