import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, kp_detector, smooth_alpha=0.05):
        # smooth_alpha nhỏ = smoothing mạnh = ổn định hơn
        # 0.05 nghĩa là 95% giữ giá trị cũ, 5% cập nhật mới
        self.kp_detector  = kp_detector
        self.length       = kp_detector.config.length   # 12000 cm
        self.width        = kp_detector.config.width    # 7000 cm
        self.M_cache      = {}
        self.last_good_M  = None
        self.smooth_alpha = smooth_alpha
        self._pos_smooth  = {}

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
        margin_x = self.length * 0.25
        margin_y = self.width  * 0.25
        return (-margin_x <= pos[0] <= self.length + margin_x and
                -margin_y <= pos[1] <= self.width  + margin_y)

    def _smooth(self, key, raw):
        if not self._is_valid(raw):
            # Hoàn toàn out-of-bounds: giữ prev nếu có, không update state
            return self._pos_smooth.get(key, list(raw))

        if key not in self._pos_smooth:
            self._pos_smooth[key] = list(raw)
            return list(raw)

        prev     = self._pos_smooth[key]
        a        = self.smooth_alpha
        smoothed = (a * np.array(raw) + (1 - a) * np.array(prev)).tolist()
        self._pos_smooth[key] = smoothed
        return smoothed

    def add_transformed_position_to_tracks(self, tracks, video_frames):
        for obj in ['players', 'referees', 'ball']:
            for frame_num, frame_track in enumerate(tracks[obj]):
                M = self._get_homography_for_frame(video_frames, frame_num)
                if M is None:
                    # Giữ vị trí frame trước
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
                        raw      = tp.tolist()
                        smoothed = self._smooth((obj, tid), raw)
                        tracks[obj][frame_num][tid][
                            'position_transformed'] = smoothed
                    except Exception:
                        pass