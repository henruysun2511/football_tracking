from collections import deque

import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, kp_detector, smooth_alpha=0.15):
        self.kp_detector = kp_detector
        self.length = kp_detector.config.length   # 12000 cm
        self.width  = kp_detector.config.width    # 7000 cm
        self.M_cache = {}
        self.last_good_M = None
        self.smooth_alpha = smooth_alpha
        self._pos_smooth = {}

        # Max jump per frame (cm). Config: 12000cm sân, 25fps
        # Cầu thủ nhanh nhất ~12 m/s = 1200 cm/s → 48 cm/frame @25fps
        # Dùng ngưỡng rộng 400cm/frame để không reject lúc tracking gap
        self._max_jump = {
            'players':  400,
            'referees': 400,
            'ball':     1200,
        }

    def _get_homography_for_frame(self, video_frames, frame_num):
        if frame_num in self.M_cache:
            return self.M_cache[frame_num]

        kps = self.kp_detector.detect_smoothed(video_frames[frame_num])
        M   = self.kp_detector.get_homography(kps) if kps is not None else None

        if M is not None:
            self.last_good_M = M
        else:
            M = self.last_good_M

        self.M_cache[frame_num] = M
        return M

    def _is_valid(self, pos):
        """
        Chấp nhận tọa độ nằm trong sân + margin 20%.
        Margin rộng để không reject cầu thủ đứng gần đường biên.
        """
        margin_x = self.length * 0.20   # 2400 cm
        margin_y = self.width  * 0.20   # 1400 cm
        return (-margin_x <= pos[0] <= self.length + margin_x and
                -margin_y <= pos[1] <= self.width  + margin_y)

    def _smooth(self, key, raw, obj_type='players'):
        """
        EMA với outlier rejection.
        KHÔNG clamp tọa độ — giữ nguyên giá trị thật để tính speed đúng.
        Chỉ reject nếu hoàn toàn out-of-bounds hoặc jump vô lý.
        """
        if key not in self._pos_smooth:
            if self._is_valid(raw):
                self._pos_smooth[key] = list(raw)
            return list(raw)

        prev = self._pos_smooth[key]

        # Reject nếu hoàn toàn out-of-bounds (homography sai nặng)
        if not self._is_valid(raw):
            return list(prev)

        # Reject nếu jump quá lớn trong 1 frame
        max_j = self._max_jump.get(obj_type, 400)
        dist  = np.linalg.norm(np.array(raw) - np.array(prev))
        if dist > max_j:
            return list(prev)

        # EMA bình thường — KHÔNG clamp kết quả
        a        = self.smooth_alpha
        smoothed = (a * np.array(raw) + (1 - a) * np.array(prev)).tolist()
        self._pos_smooth[key] = smoothed
        return smoothed

    def add_transformed_position_to_tracks(self, tracks, video_frames):
        for obj in ['players', 'referees', 'ball']:
            for frame_num, frame_track in enumerate(tracks[obj]):
                M = self._get_homography_for_frame(video_frames, frame_num)
                if M is None:
                    # Giữ vị trí frame trước để minimap không nhảy
                    for tid in frame_track:
                        key = (obj, tid)
                        if key in self._pos_smooth:
                            tracks[obj][frame_num][tid][
                                'position_transformed'] = self._pos_smooth[key]
                    continue

                for tid, data in frame_track.items():
                    pos = data.get('position')
                    if pos is None:
                        continue
                    try:
                        p  = np.array([[pos]], dtype=np.float32)
                        tp = cv2.perspectiveTransform(p, M)[0][0]
                        raw = tp.tolist()
                        smoothed = self._smooth((obj, tid), raw, obj_type=obj)
                        tracks[obj][frame_num][tid][
                            'position_transformed'] = smoothed
                    except Exception:
                        pass