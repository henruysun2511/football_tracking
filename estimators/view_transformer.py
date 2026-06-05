from collections import deque

import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, kp_detector):
        self.kp_detector = kp_detector
        self.length = kp_detector.config.length
        self.width = kp_detector.config.width
        self.M_cache = {}
        self.M_history = deque(maxlen=5)
        self.last_good_M = None

    def _get_homography_for_frame(self, video_frames, frame_num):
        if frame_num in self.M_cache:
            return self.M_cache[frame_num]

        kps = self.kp_detector.detect_smoothed(video_frames[frame_num])
        M = self.kp_detector.get_homography(kps) if kps is not None else None

        if M is not None:
            self.M_history.append(M)
            self.last_good_M = M
            if len(self.M_history) >= 2:
                M = np.mean(np.array(self.M_history), axis=0)
        else:
            M = self.last_good_M

        self.M_cache[frame_num] = M
        return M

    def add_transformed_position_to_tracks(self, tracks, video_frames):
        for obj in ['players', 'referees', 'ball']:
            for frame_num, frame_track in enumerate(tracks[obj]):
                M = self._get_homography_for_frame(video_frames, frame_num)
                if M is None:
                    continue
                for tid, data in frame_track.items():
                    pos = data.get('position')
                    if pos is None:
                        continue
                    try:
                        p = np.array([[pos]], dtype=np.float32)
                        tp = cv2.perspectiveTransform(p, M)[0][0]
                        tracks[obj][frame_num][tid][
                            'position_transformed'] = tp.tolist()
                    except Exception:
                        pass
