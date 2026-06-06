import cv2
import numpy as np
from utils import measure_distance

# Config units: SoccerPitchConfig dùng cm (length=12000, width=7000)
# → position_transformed đơn vị là cm
# → dist từ measure_distance cũng là cm
_UNIT_TO_METER = 0.01   # 1 cm = 0.01 m

# Giới hạn sinh lý học cầu thủ bóng đá
_MAX_SPEED_KMH  = 42.0  # Kylian Mbappé ~38 km/h, buffer an toàn
_MAX_DIST_JUMP  = 15.0  # Tối đa 15m trong 1 window (5 frame @ 24fps ≈ 0.21s)


class SpeedDistanceEstimator:
    WINDOW = 5

    def add_speed_and_distance_to_tracks(self, tracks, fps=24):
        total_dist = {}
        for obj, obj_tracks in tracks.items():
            if obj in ('ball', 'referees'):
                continue

            n_frames = len(obj_tracks)
            for start in range(0, n_frames, self.WINDOW):
                end = min(start + self.WINDOW, n_frames - 1)
                if end == start:
                    continue

                tids = set()
                for fn in range(start, end + 1):
                    tids.update(obj_tracks[fn].keys())

                for tid in tids:
                    p_start = obj_tracks[start].get(tid, {}).get(
                        'position_transformed')
                    p_end   = obj_tracks[end].get(tid, {}).get(
                        'position_transformed')

                    if p_start is None or p_end is None:
                        continue

                    dist_cm = measure_distance(p_start, p_end)
                    dist_m  = dist_cm * _UNIT_TO_METER

                    # Sanity check: reject nếu distance phi thực tế
                    if dist_m > _MAX_DIST_JUMP:
                        continue

                    elapsed  = (end - start) / fps
                    if elapsed <= 0:
                        continue

                    speed_ms = dist_m / elapsed
                    speed_kh = min(speed_ms * 3.6, _MAX_SPEED_KMH)

                    if obj not in total_dist:
                        total_dist[obj] = {}
                    total_dist[obj][tid] = (
                        total_dist[obj].get(tid, 0.0) + dist_m)

                    for fn in range(start, end + 1):
                        if tid in obj_tracks[fn]:
                            obj_tracks[fn][tid]['speed']    = speed_kh
                            obj_tracks[fn][tid]['distance'] = \
                                total_dist[obj][tid]

    def draw_speed_and_distance(self, frames, tracks):
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj in ('ball', 'referees'):
                    continue
                for tid, data in obj_tracks[frame_num].items():
                    speed = data.get('speed')
                    dist  = data.get('distance')
                    bbox  = data.get('bbox')
                    if speed is None or bbox is None:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    foot_x = (x1 + x2) // 2
                    foot_y = y2
                    cv2.putText(frame, f"{speed:.1f} km/h",
                                (foot_x, foot_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    if dist is not None:
                        cv2.putText(frame, f"{dist:.1f} m",
                                    (foot_x, foot_y + 58),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)