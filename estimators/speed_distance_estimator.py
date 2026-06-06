import cv2
import numpy as np
from utils import measure_distance


class SpeedDistanceEstimator:
    WINDOW = 5    # số frame mỗi cửa sổ tính tốc độ
    FPS    = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_dist = {}
        for obj, obj_tracks in tracks.items():
            if obj in ('ball', 'referees'):
                continue
            n_frames = len(obj_tracks)
            for start in range(0, n_frames, self.WINDOW):
                end = min(start + self.WINDOW, n_frames - 1)
                # Tập hợp tất cả track_id trong cửa sổ
                tids = set()
                for fn in range(start, end+1):
                    tids.update(obj_tracks[fn].keys())

                for tid in tids:
                    p_start = obj_tracks[start].get(tid, {}).get(
                        'position_transformed')
                    p_end   = obj_tracks[end].get(tid, {}).get(
                        'position_transformed')
                    if p_start is None or p_end is None:
                        continue

                    dist_cm = measure_distance(p_start, p_end)
                    dist_m = dist_cm / 100
                    elapsed = (end - start) / self.FPS
                    speed_ms = dist_m / elapsed if elapsed > 0 else 0
                    speed_kh = speed_ms * 3.6

                    if obj not in total_dist:
                        total_dist[obj] = {}
                    total_dist[obj][tid] = (
                        total_dist[obj].get(tid, 0) + dist_m)

                    for fn in range(start, end+1):
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
                    text_x = foot_x
                    text_y = foot_y + 40
                    cv2.putText(frame, f"{speed:.2f} km/h",
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0,0), 2)
                    cv2.putText(frame, f"{dist:.2f} m",
                                (text_x, text_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,0,0), 2)