import cv2
import numpy as np
from utils import measure_distance

_UNIT_TO_METER = 0.01   # cm → m
_MAX_SPEED_KMH = 38.0
_MAX_DIST_CM   = 300.0


class SpeedDistanceEstimator:
    WINDOW = 5

    def add_speed_and_distance_to_tracks(self, tracks, fps=25):
        total_dist = {}
        for obj, obj_tracks in tracks.items():
            if obj in ('ball', 'referees'):
                continue

            n_frames = len(obj_tracks)

            # Tập hợp tất cả tid xuất hiện trong toàn bộ video
            all_tids = set()
            for fd in obj_tracks:
                all_tids.update(fd.keys())

            for tid in all_tids:
                if obj not in total_dist:
                    total_dist[obj] = {}
                total_dist[obj].setdefault(tid, 0.0)
                last_speed = None

                # Duyệt từng window
                for start in range(0, n_frames, self.WINDOW):
                    end = min(start + self.WINDOW, n_frames - 1)
                    if end == start:
                        continue

                    # Tìm frame đầu tiên và cuối cùng trong window có data
                    first_fn, last_fn = None, None
                    for fn in range(start, end + 1):
                        if tid in obj_tracks[fn] and \
                           obj_tracks[fn][tid].get('position_transformed') is not None:
                            if first_fn is None:
                                first_fn = fn
                            last_fn = fn

                    # Cần ít nhất 2 frame khác nhau để tính speed
                    if first_fn is None or last_fn is None or first_fn == last_fn:
                        if last_speed is not None:
                            for fn in range(start, end + 1):
                                if tid in obj_tracks[fn]:
                                    obj_tracks[fn][tid]['speed'] = last_speed
                                    obj_tracks[fn][tid]['distance'] = total_dist[obj][tid]
                        continue

                    p_start = obj_tracks[first_fn][tid]['position_transformed']
                    p_end   = obj_tracks[last_fn][tid]['position_transformed']

                    dist_cm = measure_distance(p_start, p_end)
                    if dist_cm > _MAX_DIST_CM:
                        if last_speed is not None:
                            for fn in range(start, end + 1):
                                if tid in obj_tracks[fn]:
                                    obj_tracks[fn][tid]['speed'] = last_speed
                                    obj_tracks[fn][tid]['distance'] = total_dist[obj][tid]
                        continue

                    elapsed  = (last_fn - first_fn) / fps
                    if elapsed <= 0:
                        continue

                    speed_ms = (dist_cm * _UNIT_TO_METER) / elapsed
                    speed_kh = min(speed_ms * 3.6, _MAX_SPEED_KMH)

                    total_dist[obj][tid] += dist_cm * _UNIT_TO_METER
                    last_speed = speed_kh

                    # Ghi speed vào tất cả frame trong window có tid
                    for fn in range(start, end + 1):
                        if tid in obj_tracks[fn]:
                            obj_tracks[fn][tid]['speed']    = speed_kh
                            obj_tracks[fn][tid]['distance'] = total_dist[obj][tid]

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