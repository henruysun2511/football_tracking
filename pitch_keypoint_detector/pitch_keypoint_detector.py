import cv2
import numpy as np
from ultralytics import YOLO


class SoccerPitchConfig:
    width = 7000
    length = 12000
    penalty_box_width = 4100
    penalty_box_length = 2015
    goal_box_width = 1832
    goal_box_length = 550
    centre_circle_radius = 915
    penalty_spot_distance = 1100

    @property
    def vertices(self):
        w, l = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        ccr = self.centre_circle_radius
        psd = self.penalty_spot_distance
        return np.array([
            (0, 0),
            (0, (w - pbw) / 2),
            (0, (w - gbw) / 2),
            (0, (w + gbw) / 2),
            (0, (w + pbw) / 2),
            (0, w),
            (gbl, (w - gbw) / 2),
            (gbl, (w + gbw) / 2),
            (psd, w / 2),
            (pbl, (w - pbw) / 2),
            (pbl, (w - gbw) / 2),
            (pbl, (w + gbw) / 2),
            (pbl, (w + pbw) / 2),
            (l / 2, 0),
            (l / 2, w / 2 - ccr),
            (l / 2, w / 2 + ccr),
            (l / 2, w),
            (l - pbl, (w - pbw) / 2),
            (l - pbl, (w - gbw) / 2),
            (l - pbl, (w + gbw) / 2),
            (l - pbl, (w + pbw) / 2),
            (l - psd, w / 2),
            (l - gbl, (w - gbw) / 2),
            (l - gbl, (w + gbw) / 2),
            (l, 0),
            (l, (w - pbw) / 2),
            (l, (w - gbw) / 2),
            (l, (w + gbw) / 2),
            (l, (w + pbw) / 2),
            (l, w),
            (l / 2 - ccr, w / 2),
            (l / 2 + ccr, w / 2),
        ], dtype=np.float32)


class PitchKeypointDetector:
    def __init__(self, model_path='models/pitch_keypoint.pt',
                 conf_threshold=0.3):
        self.model = YOLO(model_path)
        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        if self.device != 'cpu':
            self.model.to('cuda')
            print(f"PitchKeypointDetector: using GPU")
        else:
            print("PitchKeypointDetector: using CPU")
        self.conf = conf_threshold
        self.config = SoccerPitchConfig()
        self.prev_keypoints = None

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False, device=self.device)
        kps = results[0].keypoints
        if kps is None or len(kps.data) == 0:
            return None
        xy = kps.data[0].cpu().numpy()
        confs = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
        return xy, confs

    def detect_smoothed(self, frame, alpha=0.6):
        detected = self.detect(frame)
        if detected is None:
            return self.prev_keypoints
        xy, confs = detected
        mask = confs > self.conf
        if self.prev_keypoints is not None:
            prev_xy, _ = self.prev_keypoints
            smoothed = xy.copy()
            smoothed[mask] = (alpha * xy[mask] +
                              (1 - alpha) * prev_xy[mask])
            self.prev_keypoints = (smoothed, confs)
        else:
            self.prev_keypoints = (xy, confs)
        return self.prev_keypoints

    def get_homography(self, frame_keypoints):
        if frame_keypoints is None:
            return None
        xy, confs = frame_keypoints
        mask = confs > self.conf
        if mask.sum() < 4:
            return None
        src = xy[mask][:, :2].astype(np.float32)
        dst = self.config.vertices[mask]
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return M

    def transform_point(self, point, M):
        if M is None:
            return None
        p = np.array([[point]], dtype=np.float32)
        tp = cv2.perspectiveTransform(p, M)
        return tp[0][0]

    def draw_keypoints(self, frame, frame_keypoints):
        if frame_keypoints is None:
            return frame
        xy, confs = frame_keypoints
        for i, ((x, y), c) in enumerate(zip(xy, confs)):
            if c < self.conf:
                continue
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame
