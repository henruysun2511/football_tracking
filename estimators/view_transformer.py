import cv2
import numpy as np


class ViewTransformer:
    # Tọa độ pixel 4 điểm góc vùng sân (chỉnh theo video của bạn)
    PIXEL_VERTICES = np.array([
        [115, 550], [1280, 550],
        [915, 970], [430, 970]
    ], dtype=np.float32)

    # Kích thước thực tế tương ứng (mét)
    TARGET_W = 68.0
    TARGET_H = 23.32

    TARGET_VERTICES = np.array([
        [0, 0], [TARGET_W, 0],
        [TARGET_W, TARGET_H], [0, TARGET_H]
    ], dtype=np.float32)

    def __init__(self):
        self.M, _ = cv2.findHomography(
            self.PIXEL_VERTICES, self.TARGET_VERTICES)

    def transform_point(self, point):
        p = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.M)
        return transformed[0][0]

    def add_transformed_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for tid, data in track.items():
                    pos = data.get('position_adjusted',
                                   data.get('position'))
                    if pos is None:
                        continue
                    try:
                        tp = self.transform_point(pos)
                        if (0 <= tp[0] <= self.TARGET_W and
                                0 <= tp[1] <= self.TARGET_H):
                            tracks[obj][frame_num][tid][
                                'position_transformed'] = tp.tolist()
                    except Exception:
                        pass