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

        # Max jump per frame cho từng loại object (cm)
        # Cầu thủ nhanh nhất ~12 m/s ≈ 1200 cm / (60fps) ≈ 50 cm/frame @ 24fps
        # Dùng ngưỡng rộng hơn để không reject lúc tracking miss nhiều frame
        self._max_jump = {
            'players':  600,   # ~14 m/s @ 24fps, đủ rộng cho mọi trường hợp
            'referees': 600,
            'ball':     1500,  # bóng có thể bay nhanh hơn
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

    def _is_inbounds(self, pos):
        """Kiểm tra tọa độ nằm trong bounds sân (có margin 10%)."""
        margin_x = self.length * 0.10
        margin_y = self.width  * 0.10
        return (-margin_x <= pos[0] <= self.length + margin_x and
                -margin_y <= pos[1] <= self.width  + margin_y)

    def _clamp(self, pos):
        """Clamp về bounds sân thực tế."""
        x = max(0.0, min(float(pos[0]), float(self.length)))
        y = max(0.0, min(float(pos[1]), float(self.width)))
        return [x, y]

    def _smooth(self, key, raw, obj_type='players'):
        """
        EMA với outlier rejection.
        - Nếu raw out-of-bounds (kể cả margin) → reject, giữ prev
        - Nếu jump quá lớn → reject, giữ prev
        - Nếu là lần đầu → init với giá trị clamped
        """
        clamped = self._clamp(raw)

        if key not in self._pos_smooth:
            if self._is_inbounds(raw):
                self._pos_smooth[key] = clamped
            return clamped

        prev = self._pos_smooth[key]

        # Reject nếu out of bounds (homography sai hoàn toàn)
        if not self._is_inbounds(raw):
            return prev  # Giữ vị trí cũ, không update state

        # Reject nếu jump quá lớn (1 frame không thể đi xa vậy)
        max_j = self._max_jump.get(obj_type, 600)
        dist  = np.linalg.norm(np.array(clamped) - np.array(prev))
        if dist > max_j:
            return prev  # Giữ vị trí cũ, không update state

        # EMA bình thường
        a        = self.smooth_alpha
        smoothed = (a * np.array(clamped) + (1 - a) * np.array(prev)).tolist()
        self._pos_smooth[key] = smoothed
        return smoothed

    def add_transformed_position_to_tracks(self, tracks, video_frames):
        for obj in ['players', 'referees', 'ball']:
            for frame_num, frame_track in enumerate(tracks[obj]):
                M = self._get_homography_for_frame(video_frames, frame_num)
                if M is None:
                    # Dùng vị trí frame trước nếu có (giữ minimap ổn định)
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